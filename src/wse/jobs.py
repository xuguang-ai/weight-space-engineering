"""The job queue: submission, polling, cancellation, leases, and crash settlement.

Rules implemented here (Sections 3, 4.1, 4.3, 6.2, and 7.2 of the plan):

* ``submit_evaluation`` validates the experiment (OPEN, ``bounded_eval``),
  ownership of the checkpoint, the suite (registered for the task;
  development split only), and ``eval_config`` (a preregistered sample count
  and seed). In one ``BEGIN IMMEDIATE`` transaction it then re-checks the
  experiment state, reserves the planned usage, queues the job, and records
  the idempotent result, and it returns the job immediately. The same key and
  input return the same job without reserving again; the same key with other
  input is an ``IDEMPOTENCY_CONFLICT``.
* A suite whose evaluator generates text (``generation@1``) is evaluated as a
  generation pass (M2 plan Section 5.4). It needs a policy that grants
  generation passes and a checkpoint that can generate; the pass may hold at
  most ``problems_per_generation_pass`` problems, each output at most
  ``new_tokens_per_output`` new tokens. It reserves one ``generation_passes``,
  problems x samples x ``max_new_tokens`` ``generated_tokens`` (the most it can
  generate), and its compute seconds, and no forward-batch counter; the worker
  settles the tokens it generated. A pass's size comes from the suite's
  parameters and the model's profile, so submission never reads the
  problems, which stay in evaluator storage.
* ``submit_probe`` queues a probe (M2 plan Section 5.3; ``wse.probes``,
  **Managed probes**) of a checkpoint the experiment owns. It needs an OPEN
  ``bounded_eval`` experiment whose policy grants probes, a checkpoint whose
  adapter declares ``probe.hidden_states``, at most
  ``texts_per_probe_call`` texts, sites that resolve against the
  checkpoint's manifest (residual boundaries, canonical module paths, and
  role sites through the family descriptor), and an inline summary that
  can fit the policy's response limit. Text sources are checked before
  anything is reserved: an output URI must name a generation output that
  this experiment owns (any other URI fails exactly like one that does not
  exist), a suite item must be an item of a development generation suite
  of the run's task (a final suite is refused as its evaluation is), and a
  string must encode to at least 2 tokens with the checkpoint's stored
  tokenizer (its window starts after its first token).
  It reserves one ``probe_calls``, texts x ``tokens_per_probe_text``
  ``teacher_forced_tokens`` (the most its forwards can read), and its
  compute seconds; the worker settles the forward lengths it read. Retries
  and conflicts follow the same idempotency rules as evaluations.
* ``get_job`` can hold the call for up to 10 seconds until the job is
  terminal, so polling does not consume agent turns.
* ``cancel_job`` releases a queued job's whole reservation. For a running job
  it only records the request: cancelling a request does not stop work by
  itself, so the worker stops the job and settles the work already executed.
  Cancelling a finished job changes nothing and refunds nothing.
* A running job holds a lease that its worker renews with heartbeats. A job
  whose lease expired, or whose local worker process no longer exists, was
  lost in a crash: its usage cannot be verified, so it fails with its whole
  reservation charged and is never rerun automatically. Lost jobs are settled
  when an engine opens, before a worker claims work, and before new jobs are
  accepted.
* Settled usage leaves no field silently empty: counters a job never uses
  are 0, and a field that cannot be measured is ``None`` with the reason in
  ``Usage.unavailable``.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import sqlite3
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from wse.audit import record_event
from wse.budget import budget_error
from wse.database import utc_now
from wse.experiments import _check_key, _hashable
from wse.schemas import (
    MAX_PROBE_STRING_CHARACTERS,
    OBJECT_ID_PATTERNS,
    TERMINAL_JOB_STATES,
    DomainError,
    ErrorCode,
    EvalConfig,
    EvaluationReport,
    EvaluationSuite,
    GenerationOutput,
    Job,
    JobProgress,
    ModelManifest,
    ProbeGroups,
    ProbeOutputText,
    ProbeRoleSite,
    ProbeStringText,
    ProbeSuiteText,
    TextDatasetManifest,
    TextSuiteData,
    Usage,
    WSEError,
    canonical_json,
    json_hash,
)

if TYPE_CHECKING:
    from wse.api import Engine

MAX_WAIT_SECONDS = 10.0
WAIT_POLL_SECONDS = 0.05


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _wait_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0 <= value <= MAX_WAIT_SECONDS:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"wait_seconds must be a number between 0 and {MAX_WAIT_SECONDS:g}")
    return float(value)


# The ledger counters that jobs charge: evaluations, generation passes, and probes, each 0 unless the job uses it.
JOB_COUNTERS = ("evaluation_batches", "evaluated_examples", "model_forward_calls", "generation_passes",
                "generated_tokens", "probe_calls", "teacher_forced_tokens")


def job_counters() -> dict[str, int]:
    """Zero for every ledger counter a job may charge (``JOB_COUNTERS``)."""
    return dict.fromkeys(JOB_COUNTERS, 0)


def unused_counters() -> dict[str, int]:
    """Zero for the ledger counters that no job charges: those of synchronous operations."""
    return {"analysis_calls": 0, "tensor_read_calls": 0, "candidate_commits": 0, "patch_previews": 0}


def queued_cancel_usage() -> Usage:
    """Usage of a job cancelled before any worker started it."""
    return Usage(**job_counters(), compute_seconds=0.0, **unused_counters(), artifact_bytes=0, peak_memory_bytes=None,
                 unavailable={"peak_memory_bytes": "no worker executed this job, so no memory was measured"})


def lost_job_usage(requested: Mapping[str, Any]) -> Usage:
    """Usage charged for a lost job: its whole reservation, with unmeasured fields explained."""
    return Usage.model_validate({
        **job_counters(), "compute_seconds": 0.0,
        **requested,
        **unused_counters(), "artifact_bytes": 0,
        "peak_memory_bytes": None,
        "unavailable": {"peak_memory_bytes": "the worker was lost before reporting usage"},
    })


def process_exists(pid: Any) -> bool:
    """Whether a local process with this ID exists (unknown IDs count as existing)."""
    if not isinstance(pid, int) or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # it exists but belongs to someone else
    return True


def lost_job_error() -> WSEError:
    return WSEError(
        ErrorCode.WORKER_FAILED,
        "the worker stopped without settling this job; its usage could not be verified, so the whole "
        "reservation was charged, and the job is not rerun automatically",
        details={"reason": "lost"},
    )


def _text_source(text: Mapping[str, Any]) -> str:
    """The kind of a stored probe text: ``output``, ``text``, or ``suite_item``."""
    return "output" if "output_uri" in text else ("text" if "text" in text else "suite_item")


def _decoder_layer_count(manifest: ModelManifest, registry: Any) -> int | None:
    """The number of decoder layers in the manifest's configuration, or ``None`` when it does not say."""
    from wse.adapters.families import model_family

    config = manifest.config.get("model_config")
    if not isinstance(config, Mapping):
        return None
    family = model_family(manifest, registry=registry)
    if family is not None and family.text_config_key and isinstance(config.get(family.text_config_key), Mapping):
        config = config[family.text_config_key]
    for key in ("num_hidden_layers", "n_layer"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
    return None


def _probe_sites(manifest: ModelManifest, sites: Any, registry: Any, limit: int) -> list[dict[str, Any]]:
    """Each requested site with the canonical site name it resolves to against the manifest (no weight is read).

    Names follow ``wse.probes``; a role site ``{"role", "layer", "point"}``
    names the input or output of the module that holds the role
    (``TensorRole.site``). The worker checks again what each site captures.
    """
    import difflib

    from wse.adapters.families import role_tensor
    from wse.probes import CAPTURE_POINTS, MAX_SITE_CANDIDATES, canonical_module_paths, residual_boundary

    if isinstance(sites, (str, bytes, Mapping)) or not isinstance(sites, (list, tuple)) or not sites:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "sites must be a non-empty list of site names ('residual:<i>', "
                       "'<canonical module path>:input' or ':output') or role sites {\"role\", \"layer\", \"point\"}")
    if len(sites) > limit:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"a probe reads at most {limit} sites, not {len(sites)}",
                       details={"sites": len(sites)})
    layers = _decoder_layer_count(manifest, registry)
    paths: dict[str, str] | None = None
    resolved = []
    for index, site in enumerate(sites):
        if isinstance(site, Mapping):
            try:
                role = ProbeRoleSite.model_validate(dict(site))
            except (ValidationError, TypeError, ValueError):
                raise WSEError(ErrorCode.INVALID_ARGUMENT,
                               f"site {index} must be a name or a role site {{\"role\": \"mlp.down\", \"layer\": "
                               "15, \"point\": \"input\" or \"output\"}} (no layer for embed, lm_head, and "
                               "final_norm)", details={"site": index}) from None
            name = f"{role_tensor(manifest, role.role, role.layer, registry=registry).site}:{role.point}"
            resolved.append({"site": name, "requested": role.model_dump(mode="json")})
            continue
        if not isinstance(site, str) or not site:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"site {index} must be a site name or a role site",
                           details={"site": index})
        boundary = residual_boundary(site, index)  # refuses overlong names and boundaries of more than 9 digits
        if boundary is not None:
            if layers is not None and boundary > layers:
                raise WSEError(ErrorCode.NOT_FOUND,
                               f"site {site!r} does not exist: the model has {layers} decoder layers, so residual "
                               f"sites run from residual:0 to residual:{layers}",
                               details={"site": site, "candidates": [f"residual:{layers}"]})
            resolved.append({"site": site, "requested": site})
            continue
        path, colon, point = site.rpartition(":")
        if not colon or not path or point not in CAPTURE_POINTS:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"site {site[:200]!r} is neither 'residual:<layer boundary>' nor '<canonical module "
                           "path>:input' or ':output'", details={"site": index})
        paths = paths if paths is not None else canonical_module_paths(manifest.tensors)
        if path not in paths:
            candidates = difflib.get_close_matches(path, list(paths), n=MAX_SITE_CANDIDATES, cutoff=0.0)
            raise WSEError(ErrorCode.NOT_FOUND,
                           f"site {site[:200]!r} names no module: a module path is a dotted prefix of canonical "
                           f"tensor names; the closest are {candidates[:3]}",
                           details={"site": index, "candidates": candidates})
        resolved.append({"site": site, "requested": site})
    names = [site["site"] for site in resolved]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"sites {repeated[:10]} are requested more than once",
                       details={"sites": repeated[:10]})
    return resolved


def _probe_groups(groups: Any, count: int) -> dict[str, list[int]] | None:
    """Checked groups of text indices: disjoint, non-empty, within the texts, and at least three texts together."""
    if groups is None:
        return None
    try:
        parsed = ProbeGroups.model_validate(dict(groups)) if isinstance(groups, Mapping) else None
    except (ValidationError, TypeError, ValueError):
        parsed = None
    if parsed is None:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "groups must be {\"a\": [text indices], \"b\": [text indices]}, "
                       "two non-empty lists of integers")
    for name, indices in (("a", parsed.a), ("b", parsed.b)):
        if any(not 0 <= index < count for index in indices) or len(set(indices)) != len(indices):
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"group {name} must list distinct text indices in [0, {count})", details={"group": name})
    shared = sorted(set(parsed.a) & set(parsed.b))
    if shared:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"the groups share texts {shared[:10]}; a contrast compares "
                       "disjoint groups", details={"shared": shared[:10]})
    if len(parsed.a) + len(parsed.b) < 3:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "the groups need at least three texts together, so that the pooled "
                       "standard deviation of a contrast has a degree of freedom")
    return {"a": list(parsed.a), "b": list(parsed.b)}


class JobService:
    """Evaluation and probe jobs of an ``Engine``, scoped like its experiment service."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.db = engine.store.db
        self.experiments = engine.experiments
        self.ledger = engine.experiments.ledger
        self._tokenizer_cache: tuple[Any, Any] | None = None  # (the stored files' hashes, their tokenizer)

    # -- suites (operator) ------------------------------------------------------

    def register_suite(self, suite: EvaluationSuite | Mapping[str, Any]) -> EvaluationSuite:
        """Register an immutable evaluation suite (operator only) and return its registered form.

        Examples of a final suite are always moved into evaluator storage, so
        the registered suite carries only their ``DatasetManifest``; a
        development suite may keep inline examples or reference stored ones.
        A suite that references stored examples is verified against them. A
        final suite must preregister its full size as a sample count (final
        evaluation always covers every example), and each task has at most
        one final suite.
        """
        self.experiments._require_operator("registering an evaluation suite")
        try:
            parsed = suite if isinstance(suite, EvaluationSuite) else EvaluationSuite.model_validate(dict(suite))
        except (ValidationError, TypeError) as exc:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid evaluation suite: {exc}") from exc
        evaluator = self.engine.registry.resolve_ref("evaluator", parsed.evaluator)
        if (parsed.split == "final" or isinstance(parsed.data, TextSuiteData)) and parsed.dataset is None:
            parsed = parsed.model_copy(update={"dataset": self.engine.datasets.put(parsed.data)})
        elif parsed.dataset is not None and parsed.data is not None:
            self.engine.datasets.put(parsed.data)
        resolved = parsed
        if parsed.dataset is not None:
            try:
                resolved = parsed.resolved(self.engine.datasets.load(parsed.dataset))  # verifies data and preregistration
            except ValidationError as exc:
                raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid evaluation suite: {exc}") from exc
        validate = getattr(evaluator, "validate_suite", None)
        if callable(validate):
            validate(resolved, self.engine.registry)
        if isinstance(resolved.data, TextSuiteData):
            self._check_text_overlap(resolved)
        if parsed.split == "final" and parsed.example_count not in parsed.sample_counts:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"a final suite must preregister its full size ({parsed.example_count}) as a sample count")
        parsed = parsed.registered()
        with self.db.transaction() as conn:
            row = conn.execute("SELECT suite_hash FROM suites WHERE suite_id = ?", (parsed.suite_id,)).fetchone()
            other_final = conn.execute(
                "SELECT suite_id FROM suites WHERE task_id = ? AND split = 'final' AND suite_id != ?",
                (parsed.task_id, parsed.suite_id),
            ).fetchone() if parsed.split == "final" else None
            if other_final is not None:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"task {parsed.task_id} already has the final suite {other_final['suite_id']}; a task's final "
                    "set is fixed, so new final data needs a new task",
                    details={"task_id": parsed.task_id, "final_suite_id": other_final["suite_id"]},
                )
            if row is None:
                conn.execute(
                    "INSERT INTO suites (suite_id, task_id, split, suite_hash, suite_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (parsed.suite_id, parsed.task_id, parsed.split, parsed.suite_hash, parsed.canonical_json(),
                     utc_now()),
                )
                record_event(conn, "suite.registered", parsed.suite_id,
                             {"task_id": parsed.task_id, "split": parsed.split, "suite_hash": parsed.suite_hash,
                              "evaluator": str(parsed.evaluator), "data_hash": parsed.data_hash})
            elif row["suite_hash"] != parsed.suite_hash:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"suite {parsed.suite_id} is already registered with different content; suites are "
                    "immutable, so register a new suite ID instead",
                    details={"suite_id": parsed.suite_id, "registered_hash": row["suite_hash"]},
                )
        return parsed

    def _check_text_overlap(self, suite: EvaluationSuite) -> None:
        """Refuse development problems that occur in a final suite (M2 plan Section 4.3), in either order.

        Problems compare by normalized text (``wse.evaluators.text_suites``).
        Every registered suite of the other split counts, in every task, so a
        practice set can never contain a final-suite problem.
        """
        from wse.evaluators.text_suites import overlapping_items

        rows = self.db.conn.execute("SELECT suite_id FROM suites WHERE split != ? AND suite_id != ? "
                                    "ORDER BY created_at, rowid", (suite.split, suite.suite_id)).fetchall()
        for row in rows:
            other = self._read_suite(row["suite_id"])
            if not isinstance(other.dataset, TextDatasetManifest):
                continue
            other_data = self.engine.datasets.load(other.dataset)
            dev_data, final_data = (suite.data, other_data) if suite.split == "development" else \
                (other_data, suite.data)
            overlap = overlapping_items(dev_data, final_data)
            if overlap:
                final_id = other.suite_id if suite.split == "development" else suite.suite_id
                dev_id = suite.suite_id if suite.split == "development" else other.suite_id
                raise WSEError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"{len(overlap)} problems of development suite {dev_id} also occur (normalized) in final suite "
                    f"{final_id}; remove them from the practice set (wse.evaluators.text_suites.exclude_overlap)",
                    details={"overlapping_items": overlap[:50], "overlap_count": len(overlap),
                             "development_suite_id": dev_id, "final_suite_id": final_id},
                )

    def get_suite(self, suite_id: Any) -> EvaluationSuite:
        """A registered suite. A run-bound service sees only development suites of its run's task."""
        return self.load_suite(suite_id)

    def load_suite(self, suite_id: Any, conn: sqlite3.Connection | None = None) -> EvaluationSuite:
        """A registered suite; a run-bound service sees only development suites of its run's task.

        Final suites and other tasks' suites are reported exactly like missing
        ones to a run-bound service. The worker and operator use an unbound
        service, which sees every suite.
        """
        suite = self._read_suite(suite_id, conn)
        bound = self.experiments.bound_run_id
        if bound is not None:
            grant = self.experiments._grant_row(bound)
            if suite.split != "development" or suite.task_id != grant["task_id"]:
                raise WSEError(ErrorCode.NOT_FOUND, f"suite {suite_id!r} does not exist",
                               details={"suite_id": str(suite_id)})
        return suite

    def list_suites(self, task_id: str) -> list[EvaluationSuite]:
        """Registered suites of ``task_id``, oldest first.

        A run-bound service lists only the development suites of its run's
        task; for any other task it lists nothing, exactly as for a task
        without suites.
        """
        bound = self.experiments.bound_run_id
        if bound is not None and task_id != self.experiments._grant_row(bound)["task_id"]:
            return []
        rows = self.db.conn.execute("SELECT suite_id, split FROM suites WHERE task_id = ? ORDER BY created_at, rowid",
                                    (task_id,)).fetchall()
        return [self._read_suite(row["suite_id"]) for row in rows
                if bound is None or row["split"] == "development"]

    def _read_suite(self, suite_id: Any, conn: sqlite3.Connection | None = None) -> EvaluationSuite:
        """Any registered suite, unscoped (internal use)."""
        row = None
        if isinstance(suite_id, str) and len(suite_id) <= 128:
            row = (conn or self.db.conn).execute("SELECT * FROM suites WHERE suite_id = ?", (suite_id,)).fetchone()
        if row is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"suite {suite_id!r} does not exist",
                           details={"suite_id": str(suite_id)})
        suite = EvaluationSuite.model_validate_json(row["suite_json"])
        if suite.suite_hash != row["suite_hash"]:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored suite {suite_id} does not match its hash")
        return suite

    def _development_suite(self, suite_id: Any, task_id: str, *,
                           use: str = "agents may evaluate development suites only") -> EvaluationSuite:
        try:
            suite = self._read_suite(suite_id)
        except WSEError as exc:
            if exc.code != ErrorCode.NOT_FOUND:
                raise
            suite = None
        if suite is None or suite.task_id != task_id:
            # Suites of other tasks are reported exactly like missing ones.
            raise WSEError(ErrorCode.NOT_FOUND, f"suite {suite_id!r} does not exist for task {task_id}",
                           details={"suite_id": str(suite_id), "task_id": task_id})
        if suite.split != "development":
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"suite {suite.suite_id} is not a development suite; {use}, and final evaluation is an operator "
                "action after freezing",
                details={"suite_id": suite.suite_id},
            )
        return suite

    # -- submission -------------------------------------------------------------

    def submit_evaluation(self, experiment_id: str, checkpoint_id: str, suite_id: str,
                          config: EvalConfig | Mapping[str, Any], key: str) -> Job:
        """Queue a development evaluation and return the job immediately."""
        _check_key(key)
        self.experiments._experiment_row(experiment_id)
        raw_config = config.model_dump(mode="json") if isinstance(config, EvalConfig) else config
        request_hash = json_hash({"operation": "submit_evaluation", "checkpoint_id": _hashable(checkpoint_id),
                                  "suite_id": _hashable(suite_id), "config": _hashable(raw_config)})
        recorded = self.experiments._lookup(self.db.conn, experiment_id, key, "submit_evaluation", request_hash)
        if recorded is not None:
            return self._job(experiment_id, json.loads(recorded)["job_id"])
        return self.experiments._guarded(
            "job.submit", experiment_id, request_hash, None,
            lambda: self._submit(experiment_id, checkpoint_id, suite_id, raw_config, key, request_hash),
        )

    def _submit(self, experiment_id: str, checkpoint_id: Any, suite_id: Any, raw_config: Any, key: str,
                request_hash: str) -> Job:
        self.recover_lost_jobs()
        row = self.experiments._experiment_row(experiment_id)
        self.experiments._require_open(row)
        if row["mode"] != "bounded_eval":
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"evaluation is unavailable in {row['mode']} mode: the experiment's policy permits no forward "
                "passes of the target network",
                details={"mode": row["mode"]},
            )
        self.experiments._owned_checkpoint(self.db.conn, experiment_id, checkpoint_id)
        suite = self._development_suite(suite_id, row["task_id"])
        try:
            config = EvalConfig.model_validate(raw_config)
        except (ValidationError, TypeError, ValueError) as exc:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                "eval_config must be an object with exactly sample_count (integer) and seed (integer); the server "
                "fixes the device and batch size",
                details={"preregistered": {"sample_count": suite.sample_counts, "seed": suite.seeds}},
            ) from exc
        if config.sample_count not in suite.sample_counts or config.seed not in suite.seeds:
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"eval_config {config.model_dump()} is not preregistered for suite {suite.suite_id}; use one of "
                f"sample_count {suite.sample_counts} and seed {suite.seeds}",
                details={"preregistered": {"sample_count": suite.sample_counts, "seed": suite.seeds}},
            )
        policy = self.experiments.experiment_policy(experiment_id)
        if self.generates_text(suite):
            planned, batches = self._generation_request(policy, suite, config, checkpoint_id)
        else:
            per_batch = policy.budget.examples_per_batch
            if per_batch < 1:
                raise WSEError(ErrorCode.POLICY_DENIED, "the experiment's policy permits no evaluation batches")
            batches = -(-config.sample_count // per_batch)
            planned = {"evaluation_batches": batches, "evaluated_examples": config.sample_count,
                       "model_forward_calls": batches}
        job_id = _new_id("job")
        with self.db.transaction() as conn:
            again = self.experiments._lookup(conn, experiment_id, key, "submit_evaluation", request_hash)
            if again is not None:
                job_id = json.loads(again)["job_id"]
            else:
                # The same transactional state check that freezing uses.
                self.experiments._require_open(self.experiments._experiment_row(experiment_id, conn))
                compute = min(policy.budget.seconds_per_operation,
                              self.ledger.remaining(conn, experiment_id, "compute_seconds"))
                if compute <= 0:
                    raise budget_error(conn, experiment_id, "compute_seconds", policy.budget.seconds_per_operation)
                reservation = self.ledger.reserve(experiment_id, job_id, {**planned, "compute_seconds": compute},
                                                  conn=conn)
                now = utc_now()
                conn.execute(
                    "INSERT INTO jobs (job_id, experiment_id, kind, checkpoint_id, suite_id, config_json, state, "
                    "reservation_id, progress_json, created_at) VALUES (?, ?, 'evaluation', ?, ?, ?, 'queued', ?, ?, ?)",
                    (job_id, experiment_id, checkpoint_id, suite.suite_id, config.canonical_json(),
                     reservation.reservation_id, JobProgress(batches_total=batches).canonical_json(), now),
                )
                record_event(conn, "job.submitted", job_id,
                             {"checkpoint_id": checkpoint_id, "suite_id": suite.suite_id,
                              "suite_hash": suite.suite_hash, "config": config.model_dump(mode="json"),
                              "reservation_id": reservation.reservation_id, "requested": reservation.requested},
                             experiment_id=experiment_id)
                self.experiments._remember(conn, experiment_id, key, "submit_evaluation", request_hash,
                                           canonical_json({"job_id": job_id}))
        return self._job(experiment_id, job_id)

    def generates_text(self, suite: EvaluationSuite) -> bool:
        """Whether ``suite`` runs as a generation pass: its evaluator generates text (``generation@1``)."""
        from wse.evaluators.generation import generates_text

        return generates_text(self.engine.registry.resolve_ref("evaluator", suite.evaluator))

    def _generation_request(self, policy: Any, suite: EvaluationSuite, config: EvalConfig,
                            checkpoint_id: str) -> tuple[dict[str, int], int]:
        """The allowance a generation pass reserves, after checking it against the policy, and its batch count.

        One pass and its most tokens: problems x samples x ``max_new_tokens``
        (``generation_pass_size``). A policy that leaves the generation
        counters unset permits no pass; a pass of more problems than
        ``problems_per_generation_pass``, or a suite whose outputs may be longer
        than ``new_tokens_per_output``, is refused before anything is reserved.
        """
        from wse.evaluators.generation import DEFAULT_BATCH_SIZE, generation_pass_size

        budget = policy.budget
        if budget.limit("generation_passes") < 1:
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"suite {suite.suite_id} is a generation suite, and the experiment's policy {policy.policy_id} "
                "permits no generation passes",
                details={"suite_id": suite.suite_id, "policy_id": policy.policy_id},
            )
        manifest = self.engine.store.load_manifest(checkpoint_id)
        if not self.engine._capabilities(manifest).supports("generate.text"):
            raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                           f"checkpoint {checkpoint_id} ({manifest.adapter}) cannot generate text, so it cannot be "
                           f"evaluated on generation suite {suite.suite_id}",
                           details={"checkpoint_id": checkpoint_id, "adapter": str(manifest.adapter)})
        size = generation_pass_size(suite, config, source=manifest.source, registry=self.engine.registry)
        problems_cap, tokens_cap = budget.limit("problems_per_generation_pass"), budget.limit("new_tokens_per_output")
        if size.problems > problems_cap:
            permitted = [count for count in suite.sample_counts if count <= problems_cap]
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"a generation pass evaluates at most {problems_cap} problems under the experiment's policy, and "
                f"sample_count {size.problems} asks for more; use one of the preregistered sample counts "
                f"{permitted} of suite {suite.suite_id}",
                details={"problems_per_generation_pass": problems_cap, "sample_count": size.problems,
                         "permitted_sample_counts": permitted},
            )
        if size.max_new_tokens > tokens_cap:
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"suite {suite.suite_id} generates up to {size.max_new_tokens} new tokens per output, and the "
                f"experiment's policy permits at most {tokens_cap}; evaluate a suite whose max_new_tokens is at most "
                f"{tokens_cap}",
                details={"new_tokens_per_output": tokens_cap, "max_new_tokens": size.max_new_tokens,
                         "suite_id": suite.suite_id},
            )
        return ({"generation_passes": 1, "generated_tokens": size.max_generated_tokens},
                -(-size.outputs // DEFAULT_BATCH_SIZE))

    # -- probes -------------------------------------------------------------------

    def submit_probe(self, experiment_id: str, checkpoint_id: str, texts: Any, sites: Any, key: str, *,
                     groups: Any = None, top: Any = None) -> Job:
        """Queue a probe of a checkpoint the experiment owns and return the job immediately (module docstring).

        ``texts`` are ``{"output_uri": ..., "last": n (optional)}``,
        ``{"text": ...}``, or ``{"suite_id": ..., "item": i}``; ``sites`` are
        site names (``residual:<i>``, ``<canonical module path>:input`` or
        ``:output``) or role sites ``{"role", "layer", "point"}``;
        ``groups`` ``{"a": [...], "b": [...]}`` gives a contrast per site;
        ``top`` (1-64, default 8) the top standardized dimensions per site in
        the inline summary. The job's result is that summary
        (``wse.probes``).
        """
        _check_key(key)
        self.experiments._experiment_row(experiment_id)
        request_hash = json_hash({"operation": "submit_probe", "checkpoint_id": _hashable(checkpoint_id),
                                  "texts": _hashable(texts), "sites": _hashable(sites), "groups": _hashable(groups),
                                  "top": _hashable(top)})
        recorded = self.experiments._lookup(self.db.conn, experiment_id, key, "submit_probe", request_hash)
        if recorded is not None:
            return self._job(experiment_id, json.loads(recorded)["job_id"])
        return self.experiments._guarded(
            "job.submit", experiment_id, request_hash, None,
            lambda: self._submit_probe(experiment_id, checkpoint_id, texts, sites, groups, top, key, request_hash),
        )

    def _submit_probe(self, experiment_id: str, checkpoint_id: Any, texts: Any, sites: Any, groups: Any, top: Any,
                      key: str, request_hash: str) -> Job:
        from wse.adapters.base import CAP_PROBE_HIDDEN_STATES

        self.recover_lost_jobs()
        row = self.experiments._experiment_row(experiment_id)
        self.experiments._require_open(row)
        if row["mode"] != "bounded_eval":
            raise WSEError(ErrorCode.POLICY_DENIED,
                           f"probes are unavailable in {row['mode']} mode: the experiment's policy permits no forward "
                           "passes of the target network", details={"mode": row["mode"]})
        self.experiments._owned_checkpoint(self.db.conn, experiment_id, checkpoint_id)
        policy = self.experiments.experiment_policy(experiment_id)
        if policy.budget.limit("probe_calls") < 1:
            raise WSEError(ErrorCode.POLICY_DENIED,
                           f"the experiment's policy {policy.policy_id} permits no probes",
                           details={"policy_id": policy.policy_id})
        manifest = self.engine.store.load_manifest(checkpoint_id)
        if not self.engine._capabilities(manifest).supports(CAP_PROBE_HIDDEN_STATES):
            raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                           f"checkpoint {checkpoint_id} ({manifest.adapter}) cannot be probed: its adapter does not "
                           f"support {CAP_PROBE_HIDDEN_STATES}",
                           details={"checkpoint_id": checkpoint_id, "adapter": str(manifest.adapter),
                                    "missing_capabilities": [CAP_PROBE_HIDDEN_STATES]})
        request = self.probe_request(experiment_id, row["task_id"], manifest, policy, texts, sites, groups, top)
        count = len(request["texts"])
        planned = {"probe_calls": 1, "teacher_forced_tokens": count * request["tokens_per_text"]}
        job_id = _new_id("job")
        with self.db.transaction() as conn:
            again = self.experiments._lookup(conn, experiment_id, key, "submit_probe", request_hash)
            if again is not None:
                job_id = json.loads(again)["job_id"]
            else:
                self.experiments._require_open(self.experiments._experiment_row(experiment_id, conn))
                compute = min(policy.budget.seconds_per_operation,
                              self.ledger.remaining(conn, experiment_id, "compute_seconds"))
                if compute <= 0:
                    raise budget_error(conn, experiment_id, "compute_seconds", policy.budget.seconds_per_operation)
                reservation = self.ledger.reserve(experiment_id, job_id, {**planned, "compute_seconds": compute},
                                                  conn=conn)
                conn.execute(
                    "INSERT INTO jobs (job_id, experiment_id, kind, checkpoint_id, suite_id, config_json, state, "
                    "reservation_id, progress_json, created_at) VALUES (?, ?, 'probe', ?, NULL, ?, 'queued', ?, ?, ?)",
                    (job_id, experiment_id, checkpoint_id, canonical_json(request), reservation.reservation_id,
                     JobProgress(batches_total=count).canonical_json(), utc_now()),
                )
                record_event(conn, "job.submitted", job_id,
                             {"kind": "probe", "checkpoint_id": checkpoint_id, "texts": count,
                              "sources": sorted({_text_source(text) for text in request["texts"]}),
                              "sites": [site["site"] for site in request["sites"]],
                              "groups": None if request["groups"] is None else
                              {name: len(indices) for name, indices in request["groups"].items()},
                              "reservation_id": reservation.reservation_id, "requested": reservation.requested},
                             experiment_id=experiment_id)
                self.experiments._remember(conn, experiment_id, key, "submit_probe", request_hash,
                                           canonical_json({"job_id": job_id}))
        return self._job(experiment_id, job_id)

    def probe_request(self, experiment_id: str, task_id: str, manifest: ModelManifest, policy: Any, texts: Any,
                      sites: Any, groups: Any, top: Any) -> dict[str, Any]:
        """The checked, normalized request of a probe: its texts, resolved sites, groups, summary size, and limit.

        Every check that needs no forward runs here, before anything is
        reserved; the worker checks again what it reads.
        """
        from wse.probes import DEFAULT_SUMMARY_TOP, MAX_PROBE_SITES, MAX_PROBE_TOKENS, MAX_SUMMARY_TOP, \
            PROBE_JOB_BYTES, summary_size_bound

        budget = policy.budget
        if isinstance(texts, (str, bytes, Mapping)) or not isinstance(texts, (list, tuple)) or not texts:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "texts must be a non-empty list of texts, each "
                           "{\"output_uri\": ...}, {\"text\": ...}, or {\"suite_id\": ..., \"item\": ...}")
        cap = budget.limit("texts_per_probe_call")
        if len(texts) > cap:
            raise WSEError(ErrorCode.POLICY_DENIED,
                           f"a probe teacher-forces at most {cap} texts under the experiment's policy, not {len(texts)}",
                           details={"texts_per_probe_call": cap, "texts": len(texts)})
        parsed = [self._probe_text(experiment_id, task_id, index, text) for index, text in enumerate(texts)]
        self._check_probe_strings(manifest, parsed)
        resolved = _probe_sites(manifest, sites, self.engine.registry, MAX_PROBE_SITES)
        shown_groups = _probe_groups(groups, len(parsed))
        if top is None:
            top = DEFAULT_SUMMARY_TOP
        elif isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= MAX_SUMMARY_TOP:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"top must be an integer from 1 to {MAX_SUMMARY_TOP}")
        tokens_per_text = min(budget.limit("tokens_per_probe_text"), MAX_PROBE_TOKENS)
        if tokens_per_text < 2:
            raise WSEError(ErrorCode.POLICY_DENIED, "the experiment's policy permits probe texts of fewer than 2 "
                           "tokens, which hold no scored window", details={"tokens_per_probe_text": tokens_per_text})
        limit = policy.operations.limits.max_result_json_bytes - PROBE_JOB_BYTES
        bounds = summary_size_bound(texts=len(parsed), sites=[site["site"] for site in resolved],
                                    grouped=shown_groups is not None, tokens_per_text=tokens_per_text)
        if bounds[-1] > limit:
            fitting = max([count for count, size in enumerate(bounds) if size <= limit], default=0)
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"the inline summary of {len(parsed)} texts at {len(resolved)} sites could exceed the "
                           f"{limit}-byte inline-result limit of the experiment's policy; probe at most {fitting} "
                           "such sites, or fewer texts",
                           details={"limit_bytes": limit, "texts": len(parsed), "sites": len(resolved),
                                    "max_sites": fitting})
        return {"texts": parsed, "sites": resolved, "groups": shown_groups, "top": top,
                "tokens_per_text": tokens_per_text}

    def _check_probe_strings(self, manifest: ModelManifest, texts: list[dict[str, Any]]) -> None:
        """Refuse a string text that encodes to fewer than 2 tokens, as the worker would after loading the model.

        The string is encoded as the worker encodes it, with the checkpoint's
        stored tokenizer (no weights are read); the tokenizer of the latest
        checkpoint is kept for the next probe.
        """
        strings = [(index, text["text"]) for index, text in enumerate(texts) if "text" in text]
        if not strings:
            return
        from wse.engines.hf_torch import load_tokenizer
        from wse.hf_snapshot import MODEL_TEXT_FILES

        key = None if manifest.source is None else tuple(sorted(
            (name, digest) for name, digest in manifest.source.files.items() if name in MODEL_TEXT_FILES))
        if self._tokenizer_cache is None or self._tokenizer_cache[0] != key:
            self._tokenizer_cache = (key, load_tokenizer(self.engine._model_files(manifest)))
        tokenizer = self._tokenizer_cache[1]
        for index, string in strings:
            count = len(tokenizer(string)["input_ids"])
            if count < 2:
                raise WSEError(ErrorCode.INVALID_ARGUMENT,
                               f"text {index} encodes to {count} token(s); a probed string needs at least 2, since "
                               "its window starts after its first token", details={"text": index})

    def _probe_text(self, experiment_id: str, task_id: str, index: int, text: Any) -> dict[str, Any]:
        """One checked probe text, as stored in the request."""
        models = (ProbeOutputText, ProbeStringText, ProbeSuiteText)
        parsed = None
        if isinstance(text, Mapping):
            for model in models:
                try:
                    parsed = model.model_validate(dict(text))
                    break
                except (ValidationError, TypeError, ValueError):
                    continue
        if parsed is None:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"text {index} must be exactly one of {{\"output_uri\": \"wse://analyses/<sha256>\", \"last\": n}} (an "
                "output of this run's generation passes), {\"text\": \"...\"} (1 to "
                f"{MAX_PROBE_STRING_CHARACTERS} characters), or "
                "{\"suite_id\": \"...\", \"item\": i} (an item of a development suite of this run's task)",
                details={"text": index})
        if isinstance(parsed, ProbeOutputText):
            output = self.probe_output(experiment_id, parsed.output_uri, index)
            del output  # checked here; the worker reads it again
        elif isinstance(parsed, ProbeSuiteText):
            suite = self._development_suite(parsed.suite_id, task_id,
                                            use="probes read items of development suites only")
            if not self.generates_text(suite):
                raise WSEError(ErrorCode.INVALID_ARGUMENT,
                               f"text {index}: suite {suite.suite_id} is not a generation suite; a suite item is probed "
                               "as the prompt a generation pass renders", details={"text": index})
            if parsed.item >= suite.example_count:
                raise WSEError(ErrorCode.NOT_FOUND,
                               f"text {index}: suite {suite.suite_id} has {suite.example_count} items, so item "
                               f"{parsed.item} does not exist", details={"text": index, "items": suite.example_count})
        return parsed.model_dump(mode="json", exclude_none=True)

    def probe_output(self, experiment_id: str, uri: str, index: int) -> GenerationOutput:
        """The generation output ``uri`` names, which ``experiment_id`` must own; others fail like missing ones."""
        try:
            document = self.experiments.experiment_results(experiment_id)(uri)
        except WSEError as exc:
            if exc.code != ErrorCode.NOT_FOUND:
                raise
            raise WSEError(ErrorCode.NOT_FOUND, f"text {index}: {exc.message}",
                           details={"text": index, "uri": uri}) from None
        if not isinstance(document, Mapping) or document.get("kind") != "generation_output":
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"text {index}: {uri} is a stored result of this run but not a generation output; give "
                           "the output_uri of a sample of a generation pass's report", details={"text": index})
        output = GenerationOutput.model_validate(document)
        if output.experiment_id != experiment_id:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"text {index}: generation output {uri} names another "
                           "experiment", details={"text": index})
        return output

    # -- reading ----------------------------------------------------------------

    def _job_row(self, conn: sqlite3.Connection, experiment_id: str, job_id: Any) -> sqlite3.Row:
        row = None
        if isinstance(job_id, str) and OBJECT_ID_PATTERNS["job"].fullmatch(job_id):
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ? AND experiment_id = ?",
                               (job_id, experiment_id)).fetchone()
        if row is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"job {job_id!r} does not exist in experiment {experiment_id}",
                           details={"job_id": str(job_id), "experiment_id": experiment_id})
        return row

    def job_from_row(self, row: sqlite3.Row, conn: sqlite3.Connection | None = None) -> Job:
        """A job from its row; a probe's ``config_json`` holds its request, which the job does not repeat."""
        return Job(
            job_id=row["job_id"],
            experiment_id=row["experiment_id"],
            kind=row["kind"],
            checkpoint_id=row["checkpoint_id"],
            suite_id=row["suite_id"],
            config=EvalConfig.model_validate_json(row["config_json"]) if row["kind"] == "evaluation" else None,
            state=row["state"],
            cancel_requested=row["cancel_requested_at"] is not None,
            reservation=self.ledger.get(row["reservation_id"], conn),
            progress=JobProgress.model_validate_json(row["progress_json"]),
            usage=Usage.model_validate_json(row["usage_json"]) if row["usage_json"] else None,
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            report_id=row["report_id"],
            error=DomainError.model_validate_json(row["error_json"]) if row["error_json"] else None,
            attempts=row["attempts"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def _job(self, experiment_id: str, job_id: Any) -> Job:
        return self.job_from_row(self._job_row(self.db.conn, experiment_id, job_id))

    def get_job(self, experiment_id: str, job_id: str, wait_seconds: float = 0) -> Job:
        """The job's state, progress, usage, and inline result.

        ``wait_seconds`` (0-10) holds the call until the job is terminal or
        the wait expires.
        """
        wait = _wait_seconds(wait_seconds)
        self.experiments._experiment_row(experiment_id)
        deadline = time.monotonic() + wait
        while True:
            self.recover_lost_jobs()
            job = self._job(experiment_id, job_id)
            left = deadline - time.monotonic()
            if job.terminal or left <= 0:
                return job
            time.sleep(min(WAIT_POLL_SECONDS, left))

    def list_jobs(self, experiment_id: str) -> list[Job]:
        self.experiments._experiment_row(experiment_id)
        rows = self.db.conn.execute(
            "SELECT * FROM jobs WHERE experiment_id = ? ORDER BY created_at, rowid", (experiment_id,)
        ).fetchall()
        return [self.job_from_row(row) for row in rows]

    # -- cancellation -----------------------------------------------------------

    def cancel_job(self, experiment_id: str, job_id: str, key: str) -> Job:
        """Request a stop; computation already performed remains charged."""
        _check_key(key)
        self.experiments._experiment_row(experiment_id)
        request_hash = json_hash({"operation": "cancel_job", "job_id": _hashable(job_id)})
        recorded = self.experiments._lookup(self.db.conn, experiment_id, key, "cancel_job", request_hash)
        if recorded is not None:
            return self._job(experiment_id, json.loads(recorded)["job_id"])
        return self.experiments._guarded("job.cancel", experiment_id, request_hash, str(job_id),
                                         lambda: self._cancel(experiment_id, job_id, key, request_hash))

    def _cancel(self, experiment_id: str, job_id: Any, key: str, request_hash: str) -> Job:
        with self.db.transaction() as conn:
            if self.experiments._lookup(conn, experiment_id, key, "cancel_job", request_hash) is None:
                row = self._job_row(conn, experiment_id, job_id)
                if row["state"] == "queued":
                    self.finish(conn, job_id, "cancelled", queued_cancel_usage(), from_state="queued",
                                error=WSEError(ErrorCode.WORKER_FAILED, "the job was cancelled before it started; "
                                               "its whole reservation was released",
                                               details={"reason": "cancelled"}))
                elif row["state"] == "running" and row["cancel_requested_at"] is None:
                    conn.execute("UPDATE jobs SET cancel_requested_at = ? WHERE job_id = ? AND state = 'running'",
                                 (utc_now(), job_id))
                    record_event(conn, "job.cancel_requested", job_id, {}, experiment_id=experiment_id)
                self.experiments._remember(conn, experiment_id, key, "cancel_job", request_hash,
                                           canonical_json({"job_id": job_id}))
        return self._job(experiment_id, job_id)

    # -- settlement (shared with the worker) -----------------------------------

    def finish(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        state: str,
        usage: Usage,
        *,
        result: dict[str, Any] | None = None,
        error: WSEError | None = None,
        progress: JobProgress | None = None,
        from_state: str = "running",
        lease_owner: str | None = None,
        lease_expired_before: float | None = None,
        report: EvaluationReport | None = None,
    ) -> bool:
        """Move a job to a terminal state and settle its reservation, in the caller's transaction.

        ``report``, for a successful evaluation, is persisted and linked to
        the job in the same transaction.

        The update is conditional on the job still being in ``from_state``
        (and, when given, held by ``lease_owner`` or with a lease expired
        before ``lease_expired_before``), so exactly one of a worker, its
        hard stop, a cancellation, or crash recovery settles a job. Returns
        whether this call did.
        """
        if state not in TERMINAL_JOB_STATES:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{state!r} is not a terminal job state")
        assignments = ["state = ?", "usage_json = ?", "result_json = ?", "error_json = ?", "finished_at = ?",
                       "lease_owner = NULL", "lease_expires_at = NULL", "lease_pid = NULL"]
        values: list[Any] = [state, usage.canonical_json(), canonical_json(result) if result is not None else None,
                             canonical_json(error.to_domain_error().model_dump(mode="json")) if error is not None else None, utc_now()]
        if progress is not None:
            assignments.append("progress_json = ?")
            values.append(progress.canonical_json())
        conditions, arguments = ["job_id = ?", "state = ?"], [job_id, from_state]
        if lease_owner is not None:
            conditions.append("lease_owner = ?")
            arguments.append(lease_owner)
        if lease_expired_before is not None:
            conditions.append("(lease_expires_at IS NULL OR lease_expires_at < ?)")
            arguments.append(lease_expired_before)
        updated = conn.execute(
            f"UPDATE jobs SET {', '.join(assignments)} WHERE {' AND '.join(conditions)}", (*values, *arguments)
        ).rowcount
        if not updated:
            return False
        row = conn.execute("SELECT experiment_id, reservation_id FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        reservation = self.ledger.settle(row["reservation_id"], usage, state, conn=conn)
        if report is not None:
            if state != "succeeded" or report.job_id != job_id or report.experiment_id != row["experiment_id"]:
                raise WSEError(ErrorCode.INVALID_ARGUMENT, "only a successful job stores its own report")
            self.engine.evaluation.insert_report(conn, report.model_copy(update={"usage": usage}))
            conn.execute("UPDATE jobs SET report_id = ? WHERE job_id = ?", (report.report_id, job_id))
        record_event(conn, f"job.{state}", job_id,
                     {"charged": reservation.charged, "error": error.code.value if error else None,
                      "reason": error.details.get("reason") if error else None},
                     experiment_id=row["experiment_id"])
        return True

    def recover_lost_jobs(self) -> list[str]:
        """Settle lost running jobs: fail them and charge the whole reservation.

        A running job is lost when its lease expired or when the local worker
        process holding the lease no longer exists, so a restarted engine
        settles the job of a crashed worker before it accepts new jobs,
        without waiting for the lease to expire.
        """
        now = time.time()
        running = self.db.conn.execute(
            "SELECT job_id, lease_owner, lease_expires_at, lease_pid FROM jobs WHERE state = 'running'"
        ).fetchall()
        candidates = []
        for row in running:
            if row["lease_expires_at"] is None or row["lease_expires_at"] < now:
                candidates.append((row["job_id"], {"lease_expired_before": now}))
            elif row["lease_owner"] is not None and not process_exists(row["lease_pid"]):
                candidates.append((row["job_id"], {"lease_owner": row["lease_owner"]}))
        if not candidates:
            return []
        lost = []
        with self.db.transaction() as conn:
            for job_id, condition in candidates:
                row = conn.execute("SELECT reservation_id FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                requested = self.ledger.get(row["reservation_id"], conn).requested
                if self.finish(conn, job_id, "failed", lost_job_usage(requested), error=lost_job_error(),
                               **condition):
                    lost.append(job_id)
        return lost


__all__ = ["JobService", "MAX_WAIT_SECONDS", "lost_job_error", "lost_job_usage", "process_exists",
           "queued_cancel_usage"]
