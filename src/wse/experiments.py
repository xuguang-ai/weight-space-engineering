"""Experiment state machine: run grants, idempotency, ownership, patches, CAS selection.

Rules implemented here (Sections 4.1, 5.1-5.3, and 6.2 of the plan):

* An operator-issued run grant fixes the task, the immutable policy (and so
  the mode), and the base checkpoint, and permits at most one experiment.
  Retries return the existing experiment; changing the task, policy, or
  idempotency key cannot create another one or reset its budget.
* Every mutation takes an idempotency key. The same key and input return the
  recorded result without creating another version or charging again; the
  same key with different input returns ``IDEMPOTENCY_CONFLICT``. Keys are
  scoped to the run (creation) or the experiment (everything else).
* Every operation checks object ownership: an experiment sees its own base
  checkpoint, its own candidates, and its own patches. Objects of other
  experiments are reported as ``NOT_FOUND``. A service bound to a run sees
  only that run's experiment.
* ``preview_patch`` freezes the spec with the server-resolved base hash,
  materializes the patch through the registered edit operators, and stores an
  immutable preview without changing the selected checkpoint. It runs only the
  edit operators that the experiment policy lists, whether an operation names
  an alias or an exact reference (``scale_slice@1``). Vector references among
  the operators' parameters are read through the run-scoped reader of
  ``find_analysis_result`` before any weights are read, so another run's
  result fails exactly like a result that does not exist.
* ``apply_patch`` accepts only a ``patch_id``, replays the stored materialized
  patch (no editing operator runs), and commits the candidate, its ownership,
  the candidate-commit charge, the idempotent result, and audit events in one
  SQLite transaction. The charge is a conditional ``UPDATE`` that cannot
  exceed the policy's limit.
* ``select_checkpoint`` is a compare-and-swap on the selected ID; selecting an
  ancestor is a rollback, which deletes no history and refunds nothing.
* Experiment-scoped reads and analyses (``read_tensor``, ``analyze``) check
  ownership, use the experiment policy's operations and limits, and charge
  ``tensor_read_calls`` or the operator's usage-contract counter plus compute
  seconds through ``wse.budget``. Each new preview charges one
  ``patch_previews`` (its deltas reveal the edited region's base values) plus
  compute seconds; applies charge compute seconds; inspection statistics
  charge compute seconds only.
* A run sees a patch without its materialized after-values, and with each
  inline vector of an operator's reference parameters (for example a
  2,048-value direction) as ``{length, sha256}``; the exact patch, which keeps
  every value, is an operator view. A run reads a large analysis result only
  when its own experiment produced it.
* Rejected mutations (creation, previews, applies, selections, freezing) are
  recorded as audit events and leave state unchanged.
* ``freeze`` (operator only) is a compare-and-swap on the selected ID that
  moves an OPEN experiment to FROZEN. In one ``BEGIN IMMEDIATE`` transaction
  it checks that no job is queued or running and no reservation is held;
  otherwise it returns ``VERSION_CONFLICT`` so the caller can wait for or
  cancel that work and retry. Job submission re-checks the state in its own
  transaction, so freezing and submission cannot race. A frozen experiment
  accepts no previews, candidates, selections, jobs, or metered reads.
"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from wse.audit import experiment_events, record_event, record_failure
from wse.budget import BudgetLedger, budget_error
from wse.database import utc_now
from wse.operators.base import MAX_REPORTED_PROBLEMS
from wse.patch_blocks import read_patch_file, write_patch_file
from wse.patches import materialize_patch, replay_patch, resolve_references, summarize_vector_parameters
from wse.policy import LEDGER_COUNTERS, ExperimentPolicy, builtin_experiment_policies
from wse.schemas import (
    CHECKPOINT_ID_PATTERN,
    IDEMPOTENCY_KEY_PATTERN,
    OBJECT_ID_PATTERNS,
    BudgetEntry,
    ErrorCode,
    Experiment,
    ExperimentCheckpoint,
    FrozenOperation,
    FrozenPatchSpec,
    ModelInspection,
    ModelManifest,
    PatchArtifact,
    PatchDocument,
    PatchOperationSummary,
    PatchPreview,
    PatchSpec,
    PatchStepSummary,
    Provenance,
    RunGrant,
    StorageCost,
    TensorInfo,
    TensorStatistics,
    Usage,
    WSEError,
    canonical_json,
    json_hash,
)
from wse.tensors import blob_id

if TYPE_CHECKING:
    from wse.api import Engine, ModelHandle
    from wse.schemas import AnalysisResult, TensorSlice

PATCH_URI_PREFIX = "wse://patches/"
TASK_ID_PATTERN = r"^[A-Za-z0-9_.\-]{1,128}$"


class _Recorded(Exception):
    """Raised inside a transaction when another request already recorded the result."""

    def __init__(self, result_json: str) -> None:
        self.result_json = result_json


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _check_id(kind: str, value: Any, *, what: str) -> str:
    if not isinstance(value, str) or not OBJECT_ID_PATTERNS[kind].fullmatch(value):
        raise WSEError(ErrorCode.NOT_FOUND, f"{what} {value!r} does not exist", details={what.replace(" ", "_"): str(value)})
    return value


def _check_key(key: Any) -> str:
    if not isinstance(key, str) or not IDEMPOTENCY_KEY_PATTERN.fullmatch(key):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"idempotency_key must match {IDEMPOTENCY_KEY_PATTERN.pattern}; reuse the same key to retry "
            "a request and a new key for a new request",
        )
    return key


def _hashable(value: Any) -> Any:
    """``value`` for request hashing; non-JSON input hashes by type and text."""
    try:
        return json.loads(canonical_json(value))
    except WSEError:
        return {"unhashable": type(value).__name__, "text": repr(value)[:256]}


def _number(value: Any) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _candidate_tensor_infos(
    base_manifest: ModelManifest, base: ModelHandle, candidate: ModelHandle
) -> list[TensorInfo]:
    """The base manifest's tensor entries with the blob IDs of the candidate's tensors.

    Tensors a patch did not change are shared with the base handle and keep
    their recorded blob IDs; changed tensors are hashed.
    """
    infos = []
    for info in base_manifest.tensors:
        tensor = candidate._tensors[info.name]
        if info.blob_id is not None and tensor is not base._tensors[info.name]:
            info = info.model_copy(update={"blob_id": blob_id(tensor)})
        infos.append(info)
    return infos


class ExperimentService:
    """Managed experiments over an ``Engine``'s store and registry.

    ``bound_run_id`` restricts the service to one run grant, as for a server
    process started for one agent trial.
    """

    def __init__(self, engine: Engine, *, bound_run_id: str | None = None) -> None:
        self.engine = engine
        self.store = engine.store
        self.db = engine.store.db
        self.bound_run_id = bound_run_id
        self.ledger = BudgetLedger(self.db, bound_run_id=bound_run_id)
        for policy in builtin_experiment_policies().values():
            self._ensure_policy(policy)
        if bound_run_id is not None:
            self._grant_row(bound_run_id)

    # -- policies and grants (operator) -------------------------------------

    def _ensure_policy(self, policy: ExperimentPolicy) -> ExperimentPolicy:
        payload = canonical_json(policy.model_dump(mode="json"))
        with self.db.transaction() as conn:
            row = conn.execute("SELECT policy_hash FROM policies WHERE policy_id = ?", (policy.policy_id,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO policies (policy_id, mode, policy_hash, policy_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (policy.policy_id, policy.mode, policy.policy_hash, payload, utc_now()),
                )
                record_event(conn, "policy.registered", policy.policy_id,
                             {"policy_hash": policy.policy_hash, "mode": policy.mode})
            elif row["policy_hash"] != policy.policy_hash:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"policy {policy.policy_id} is already registered with different content; policies are "
                    "immutable, so register a new policy ID instead",
                    details={"policy_id": policy.policy_id, "registered_hash": row["policy_hash"]},
                )
        return policy

    def _require_operator(self, action: str) -> None:
        if self.bound_run_id is not None:
            raise WSEError(ErrorCode.POLICY_DENIED, f"{action} is an operator action and is not available to a run")

    def register_policy(self, policy: ExperimentPolicy | Mapping[str, Any]) -> ExperimentPolicy:
        """Register an immutable experiment policy (operator only)."""
        self._require_operator("registering a policy")
        try:
            parsed = policy if isinstance(policy, ExperimentPolicy) else ExperimentPolicy.model_validate(dict(policy))
        except ValidationError as exc:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid experiment policy: {exc}") from exc
        return self._ensure_policy(parsed)

    def get_policy(self, policy_id: str) -> ExperimentPolicy:
        row = self.db.conn.execute("SELECT * FROM policies WHERE policy_id = ?", (policy_id,)).fetchone()
        if row is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"policy {policy_id!r} is not registered",
                           details={"policy_id": str(policy_id)})
        policy = ExperimentPolicy.model_validate_json(row["policy_json"])
        if policy.policy_hash != row["policy_hash"]:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored policy {policy_id} does not match its hash")
        return policy

    def grant_run(self, task_id: str, policy_id: str, base_checkpoint_id: str) -> RunGrant:
        """Issue a run grant for one experiment (operator only)."""
        self._require_operator("granting a run")
        if not isinstance(task_id, str) or not re.fullmatch(TASK_ID_PATTERN, task_id):
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"task_id must match {TASK_ID_PATTERN}")
        self.get_policy(policy_id)
        self.store.load_manifest(base_checkpoint_id)  # READY and hash-verified
        grant = RunGrant(run_id=_new_id("run"), task_id=task_id, policy_id=policy_id,
                         base_checkpoint_id=base_checkpoint_id, created_at=utc_now())
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO run_grants (run_id, task_id, policy_id, base_checkpoint_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (grant.run_id, task_id, policy_id, base_checkpoint_id, grant.created_at),
            )
            record_event(conn, "run.granted", grant.run_id, grant.model_dump(mode="json"))
        return grant

    def _grant_row(self, run_id: Any) -> sqlite3.Row:
        row = None
        if isinstance(run_id, str) and OBJECT_ID_PATTERNS["run"].fullmatch(run_id):
            row = self.db.conn.execute("SELECT * FROM run_grants WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"run grant {run_id!r} does not exist; the operator issues run grants",
                           details={"run_id": str(run_id)})
        return row

    def get_run_grant(self, run_id: str) -> RunGrant:
        self._check_bound(run_id)
        row = self._grant_row(run_id)
        return RunGrant(run_id=row["run_id"], task_id=row["task_id"], policy_id=row["policy_id"],
                        base_checkpoint_id=row["base_checkpoint_id"], created_at=row["created_at"])

    def _check_bound(self, run_id: Any) -> None:
        if self.bound_run_id is not None and run_id != self.bound_run_id:
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                "this service is bound to a different run grant; the run_id must match it",
                details={"run_id": str(run_id)},
            )

    # -- idempotency ----------------------------------------------------------

    @staticmethod
    def _lookup(conn: sqlite3.Connection, scope: str, key: str, operation: str, request_hash: str) -> str | None:
        row = conn.execute(
            "SELECT operation, request_hash, result_json FROM idempotency_records WHERE scope = ? AND idempotency_key = ?",
            (scope, key),
        ).fetchone()
        if row is None:
            return None
        if row["operation"] != operation or row["request_hash"] != request_hash:
            raise WSEError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                f"idempotency key {key!r} was already used for a different request "
                f"({row['operation']}); use a new key for a new request",
                details={"idempotency_key": key, "recorded_operation": row["operation"]},
            )
        return row["result_json"]

    @staticmethod
    def _remember(conn: sqlite3.Connection, scope: str, key: str, operation: str, request_hash: str,
                  result_json: str) -> None:
        conn.execute(
            "INSERT INTO idempotency_records (scope, idempotency_key, operation, request_hash, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (scope, key, operation, request_hash, result_json, utc_now()),
        )

    def _guarded(
        self,
        operation: str,
        experiment_id: str | None,
        request_hash: str,
        subject_id: str | None,
        body: Callable[[], Any],
    ) -> Any:
        """Run a mutation; record a rejected one as an audit event and re-raise."""
        try:
            return body()
        except WSEError as exc:
            record_failure(self.db, operation, exc, experiment_id=experiment_id,
                           subject_id=subject_id, request_hash=request_hash)
            raise

    # -- experiments ------------------------------------------------------------

    def _experiment_row(self, experiment_id: Any, conn: sqlite3.Connection | None = None) -> sqlite3.Row:
        _check_id("experiment", experiment_id, what="experiment")
        conn = conn or self.db.conn
        row = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        if row is None or (self.bound_run_id is not None and row["run_id"] != self.bound_run_id):
            raise WSEError(ErrorCode.NOT_FOUND, f"experiment {experiment_id!r} does not exist",
                           details={"experiment_id": str(experiment_id)})
        return row

    @staticmethod
    def _require_open(row: sqlite3.Row) -> None:
        if row["state"] != "OPEN":
            raise WSEError(
                ErrorCode.VERSION_CONFLICT,
                f"experiment {row['experiment_id']} is {row['state']}; only OPEN experiments accept changes",
                details={"experiment_id": row["experiment_id"], "state": row["state"]},
            )

    def _experiment(self, row: sqlite3.Row, conn: sqlite3.Connection | None = None) -> Experiment:
        conn = conn or self.db.conn
        ledger = {
            entry["counter"]: entry
            for entry in conn.execute(
                "SELECT counter, limit_value, consumed, reserved FROM budget_ledger WHERE experiment_id = ?",
                (row["experiment_id"],),
            )
        }
        budget = {}
        for counter in LEDGER_COUNTERS:
            if counter in ledger:
                entry = ledger[counter]
                limit, consumed = _number(entry["limit_value"]), _number(entry["consumed"])
                reserved = _number(entry["reserved"])
                budget[counter] = BudgetEntry(limit=limit, consumed=consumed, reserved=reserved,
                                              remaining=_number(max(limit - consumed - reserved, 0)))
        return Experiment(
            experiment_id=row["experiment_id"],
            run_id=row["run_id"],
            task_id=row["task_id"],
            policy_id=row["policy_id"],
            policy_hash=row["policy_hash"],
            mode=row["mode"],
            base_checkpoint_id=row["base_checkpoint_id"],
            selected_checkpoint_id=row["selected_checkpoint_id"],
            state=row["state"],
            budget=budget,
            created_at=row["created_at"],
            frozen_at=row["frozen_at"],
            ended_at=row["ended_at"],
        )

    def create_experiment(self, run_id: str, task_id: str, policy_id: str, key: str) -> Experiment:
        """Create the run's single experiment, or return it if it already exists.

        A rejected request is recorded against the run's experiment when it
        exists, and against the run grant otherwise. A service bound to
        another run never writes into that run's experiment trace.
        """
        request_hash = json_hash({"operation": "create_experiment", "run_id": _hashable(run_id),
                                  "task_id": _hashable(task_id), "policy_id": _hashable(policy_id)})
        try:
            return self._create(run_id, task_id, policy_id, key, request_hash)
        except WSEError as exc:
            existing = None
            if isinstance(run_id, str) and OBJECT_ID_PATTERNS["run"].fullmatch(run_id) \
                    and self.bound_run_id in (None, run_id):
                existing = self.db.conn.execute(
                    "SELECT experiment_id FROM experiments WHERE run_id = ?", (run_id,)).fetchone()
            record_failure(self.db, "experiment.create", exc,
                           experiment_id=existing["experiment_id"] if existing else None,
                           subject_id=str(run_id), request_hash=request_hash)
            raise

    def _create(self, run_id: Any, task_id: Any, policy_id: Any, key: Any, request_hash: str) -> Experiment:
        _check_key(key)
        self._check_bound(run_id)
        scope = f"run:{run_id}"
        with self.db.transaction() as conn:
            self._lookup(conn, scope, key, "create_experiment", request_hash)
            grant = self._grant_row(run_id)
            if (task_id, policy_id) != (grant["task_id"], grant["policy_id"]):
                raise WSEError(
                    ErrorCode.POLICY_DENIED,
                    "the run grant fixes the task and policy of this run; they cannot be changed",
                    details={"granted_task_id": grant["task_id"], "granted_policy_id": grant["policy_id"]},
                )
            existing = conn.execute("SELECT * FROM experiments WHERE run_id = ?", (run_id,)).fetchone()
            if existing is None:
                policy = self.get_policy(policy_id)
                experiment_id, now = _new_id("exp"), utc_now()
                conn.execute(
                    "INSERT INTO experiments (experiment_id, run_id, task_id, policy_id, policy_hash, mode, "
                    "base_checkpoint_id, selected_checkpoint_id, state, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)",
                    (experiment_id, run_id, task_id, policy_id, policy.policy_hash, policy.mode,
                     grant["base_checkpoint_id"], grant["base_checkpoint_id"], now),
                )
                for counter, limit in policy.budget.ledger_limits().items():
                    conn.execute(
                        "INSERT INTO budget_ledger (experiment_id, counter, limit_value, consumed) VALUES (?, ?, ?, 0)",
                        (experiment_id, counter, limit),
                    )
                conn.execute(
                    "INSERT INTO experiment_checkpoints (experiment_id, checkpoint_id, role, patch_id, created_at) "
                    "VALUES (?, ?, 'base', NULL, ?)",
                    (experiment_id, grant["base_checkpoint_id"], now),
                )
                record_event(conn, "experiment.created", experiment_id,
                             {"run_id": run_id, "task_id": task_id, "policy_id": policy_id,
                              "policy_hash": policy.policy_hash, "mode": policy.mode,
                              "base_checkpoint_id": grant["base_checkpoint_id"]},
                             experiment_id=experiment_id)
                existing = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
            if self._lookup(conn, scope, key, "create_experiment", request_hash) is None:
                self._remember(conn, scope, key, "create_experiment", request_hash,
                               canonical_json({"experiment_id": existing["experiment_id"]}))
            return self._experiment(existing, conn)

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self._experiment(self._experiment_row(experiment_id))

    def experiment_policy(self, experiment_id: str) -> ExperimentPolicy:
        row = self._experiment_row(experiment_id)
        policy = self.get_policy(row["policy_id"])
        if policy.policy_hash != row["policy_hash"]:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"policy of experiment {experiment_id} changed")
        return policy

    # -- ownership --------------------------------------------------------------

    def _owned_checkpoint(self, conn: sqlite3.Connection, experiment_id: str, checkpoint_id: Any) -> sqlite3.Row:
        row = None
        if isinstance(checkpoint_id, str) and CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id):
            row = conn.execute(
                "SELECT c.*, e.role, e.patch_id FROM experiment_checkpoints e JOIN checkpoints c "
                "ON c.checkpoint_id = e.checkpoint_id WHERE e.experiment_id = ? AND e.checkpoint_id = ?",
                (experiment_id, checkpoint_id),
            ).fetchone()
        if row is None:
            raise WSEError(
                ErrorCode.NOT_FOUND,
                f"checkpoint {checkpoint_id!r} does not exist in experiment {experiment_id}",
                details={"checkpoint_id": str(checkpoint_id), "experiment_id": experiment_id},
            )
        if row["state"] != "READY":
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"checkpoint {checkpoint_id} is {row['state']} and cannot be used",
                           details={"checkpoint_id": checkpoint_id})
        return row

    def list_checkpoints(self, experiment_id: str) -> list[ExperimentCheckpoint]:
        """The experiment's READY base and candidates, oldest first."""
        self._experiment_row(experiment_id)
        rows = self.db.conn.execute(
            "SELECT e.checkpoint_id, e.role, e.patch_id, e.created_at, c.parent_checkpoint_id, c.weights_hash "
            "FROM experiment_checkpoints e JOIN checkpoints c ON c.checkpoint_id = e.checkpoint_id "
            "WHERE e.experiment_id = ? AND c.state = 'READY' ORDER BY e.created_at, e.rowid",
            (experiment_id,),
        ).fetchall()
        return [
            ExperimentCheckpoint(checkpoint_id=r["checkpoint_id"], role=r["role"], patch_id=r["patch_id"],
                                 parent_checkpoint_id=r["parent_checkpoint_id"], weights_hash=r["weights_hash"],
                                 created_at=r["created_at"])
            for r in rows
        ]

    def load_manifest(self, experiment_id: str, checkpoint_id: str) -> ModelManifest:
        """The manifest of a checkpoint this experiment owns."""
        self._experiment_row(experiment_id)
        self._owned_checkpoint(self.db.conn, experiment_id, checkpoint_id)
        return self.store.load_manifest(checkpoint_id)

    # -- experiment-scoped reads (metered) ---------------------------------------

    def inspect_model(self, experiment_id: str, checkpoint_id: str) -> ModelManifest:
        """The manifest of an owned checkpoint (no tensor data; not charged)."""
        return self.load_manifest(experiment_id, checkpoint_id)

    def inspect_details(self, experiment_id: str, checkpoint_id: str) -> ModelInspection:
        """The manifest of an owned checkpoint with whole-tensor statistics by layer.

        Statistics come from the experiment policy's registered ``stats``
        analysis, so no separate algorithm runs here. Computing them is a
        managed computation: it needs an OPEN experiment with compute time
        left, stops at the policy's per-operation time limit, and is charged
        as compute seconds (not as analysis calls). The manifest itself is
        free. A tensor without statistics reports why: the experiment is not
        OPEN, compute time is exhausted, the time limit was reached, the
        tensor exceeds ``max_analysis_elements``, or the policy has no
        ``stats`` kind.
        """
        manifest = self.load_manifest(experiment_id, checkpoint_id)
        row = self._experiment_row(experiment_id)
        role = self._owned_checkpoint(self.db.conn, experiment_id, checkpoint_id)["role"]
        experiment_policy = self.experiment_policy(experiment_id)
        policy = experiment_policy.operations
        limit = experiment_policy.budget.seconds_per_operation
        blocked: str | None = None
        if row["state"] != "OPEN":
            blocked = f"statistics are computed only while the experiment is OPEN; it is {row['state']}"
        else:
            try:
                self.ledger.require(self.db.conn, experiment_id, {"compute_seconds": 0})
            except WSEError as exc:
                if exc.code != ErrorCode.BUDGET_EXCEEDED:
                    raise
                blocked = "the experiment's compute_seconds budget is exhausted"
        metered = blocked is None
        start = time.perf_counter()
        statistics = []
        for info in manifest.tensors:
            entry: dict[str, Any] = {"tensor": info.name, "shape": list(info.shape), "dtype": info.dtype}
            if blocked is None and time.perf_counter() - start >= limit:
                blocked = (f"the per-operation time limit of {limit:g} s was reached; analyze the remaining "
                           "tensors individually")
            if blocked is not None:
                entry.update(statistics=None, unavailable=blocked)
                statistics.append(TensorStatistics.model_validate(entry))
                continue
            selection = [{"start": 0, "stop": extent, "step": 1} for extent in info.shape]
            try:
                result = self.engine._analyze(checkpoint_id, "stats", info.name, selection, policy=policy)
            except WSEError as exc:
                if exc.code not in (ErrorCode.POLICY_DENIED, ErrorCode.INVALID_ARGUMENT,
                                    ErrorCode.UNSUPPORTED_CAPABILITY):
                    raise
                entry.update(statistics=None, unavailable=exc.message)
            else:
                entry.update(statistics=result.result, operator=result.operator,
                             unavailable=None if result.result is not None else "stored by reference")
            statistics.append(TensorStatistics.model_validate(entry))
        usage = None
        if metered:
            elapsed = time.perf_counter() - start
            with self.db.transaction() as conn:
                self._require_open(self._experiment_row(experiment_id, conn))
                charged = self.ledger.charge_compute(conn, experiment_id, elapsed)
            usage = Usage(compute_seconds=charged, analysis_calls=0, tensor_read_calls=0, model_forward_calls=0)
        return ModelInspection(experiment_id=experiment_id, checkpoint_id=checkpoint_id, role=role,
                               manifest=manifest, tensor_statistics=statistics,
                               manifest_uri=f"wse://models/{checkpoint_id}/manifest", usage=usage)

    def find_patch(self, patch_id: Any) -> PatchDocument:
        """A patch by ID alone (for resource reads), scoped like every other lookup.

        Patches of experiments this service may not see are reported exactly
        like patches that do not exist. The document omits the materialized
        after-values (see ``PatchDocument``).
        """
        row = None
        if isinstance(patch_id, str) and OBJECT_ID_PATTERNS["patch"].fullmatch(patch_id):
            row = self.db.conn.execute("SELECT experiment_id FROM patches WHERE patch_id = ?", (patch_id,)).fetchone()
        missing = WSEError(ErrorCode.NOT_FOUND, f"patch {patch_id!r} does not exist", details={"patch_id": str(patch_id)})
        if row is None:
            raise missing
        try:
            self._experiment_row(row["experiment_id"])
        except WSEError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                raise missing from None
            raise
        artifact = self._artifact(row["experiment_id"], patch_id)
        steps = [PatchStepSummary(op=step.op, operator=step.operator, tensor=step.tensor, selection=step.selection,
                                  dtype=step.dtype, before_hash=step.before_hash, after_hash=step.after_hash,
                                  changed_elements=step.changed_elements)
                 for step in artifact.patch.steps]
        return PatchDocument(preview=self.get_patch(row["experiment_id"], patch_id), spec=self._shown_spec(artifact.spec),
                             steps=steps, result_weights_hash=artifact.patch.result_weights_hash)

    def _shown_spec(self, spec: FrozenPatchSpec) -> FrozenPatchSpec:
        """The frozen spec as a run reads it: each inline vector of an operator's reference parameters as
        ``{length, sha256}``, so that a 2,048-value direction stays within the inline-result limit. The stored
        spec keeps the values."""
        operations = []
        for operation in spec.operations:
            try:
                operator = self.engine.registry.resolve_ref("operator", operation.operator)
            except WSEError:
                operator = None  # not registered here: shown as recorded
            names = tuple(getattr(operator, "reference_parameters", ()) or ())
            operations.append(operation.model_copy(update={
                "parameters": summarize_vector_parameters(operation.parameters, names)}) if names else operation)
        return spec.model_copy(update={"operations": operations})

    def _metered(self, experiment_id: str, checkpoint_id: Any, counter: str, units: int,
                 compute: Callable[[ExperimentPolicy], Any],
                 record: Callable[[sqlite3.Connection, Any], None] | None = None) -> Any:
        """Run a synchronous read or analysis and charge it to the experiment.

        Invalid requests fail before anything is charged. The charge is a
        conditional UPDATE after the computation, so concurrent requests can
        never overdraw: a request that loses the race returns
        ``BUDGET_EXCEEDED`` and its result is discarded.
        """
        # A frozen ledger stays fixed: metered reads need an OPEN experiment.
        self._require_open(self._experiment_row(experiment_id))
        self._owned_checkpoint(self.db.conn, experiment_id, checkpoint_id)
        policy = self.experiment_policy(experiment_id)
        self.ledger.require(self.db.conn, experiment_id, {counter: units, "compute_seconds": 0})
        start = time.perf_counter()
        result = compute(policy)
        with self.db.transaction() as conn:
            self.ledger.charge(conn, experiment_id, counter, units)
            self.ledger.charge_compute(conn, experiment_id, time.perf_counter() - start)
            if record is not None:
                record(conn, result)
        return result

    def read_tensor(self, experiment_id: str, checkpoint_id: str, tensor: str, selection: Any) -> TensorSlice:
        """Bounded read of an owned checkpoint; charges one ``tensor_read_calls``."""
        return self._metered(
            experiment_id, checkpoint_id, "tensor_read_calls", 1,
            lambda policy: self.engine._bounded_read(checkpoint_id, tensor, selection, policy.operations)[0],
        )

    def analyze(self, experiment_id: str, checkpoint_id: str, kind: str, tensor: str, selection: Any,
                k: int | None = None, *, parameters: Mapping[str, Any] | None = None) -> AnalysisResult:
        """A permitted analysis of an owned checkpoint, charged under its operator's usage contract."""
        policy = self.experiment_policy(experiment_id)
        spec = self.engine.registry.resolve_ref("operator", policy.operations.analysis_operator(kind)).spec
        def record(conn: sqlite3.Connection, result: AnalysisResult) -> None:
            if result.result_uri:
                conn.execute("INSERT OR IGNORE INTO analysis_results (uri, experiment_id, created_at) VALUES (?, ?, ?)",
                             (result.result_uri, experiment_id, utc_now()))

        return self._metered(
            experiment_id, checkpoint_id, spec.usage.counter, spec.usage.units_per_call,
            lambda policy: self.engine._analyze(checkpoint_id, kind, tensor, selection, k, parameters=parameters,
                                                policy=policy.operations),
            record,
        )

    def find_analysis_result(self, uri: Any) -> dict[str, Any]:
        """A stored large analysis result (``AnalysisResult.result_uri``).

        A service bound to a run reads only results its own experiment
        produced; any other URI is reported exactly like one that does not
        exist. The operator reads any stored result.
        """
        missing = WSEError(ErrorCode.NOT_FOUND, f"analysis result {uri!r} does not exist", details={"uri": str(uri)})
        if self.bound_run_id is not None:
            owned = isinstance(uri, str) and self.db.conn.execute(
                "SELECT 1 FROM analysis_results a JOIN experiments e ON e.experiment_id = a.experiment_id "
                "WHERE a.uri = ? AND e.run_id = ?", (uri, self.bound_run_id)).fetchone()
            if not owned:
                raise missing
        try:
            return json.loads(self.store.read_analysis_result(uri))
        except WSEError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                raise missing from None
            raise

    def _patch_row(self, experiment_id: str, patch_id: Any) -> sqlite3.Row:
        row = None
        if isinstance(patch_id, str) and OBJECT_ID_PATTERNS["patch"].fullmatch(patch_id):
            row = self.db.conn.execute(
                "SELECT * FROM patches WHERE patch_id = ? AND experiment_id = ?", (patch_id, experiment_id)
            ).fetchone()
        if row is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"patch {patch_id!r} does not exist in experiment {experiment_id}",
                           details={"patch_id": str(patch_id), "experiment_id": experiment_id})
        return row

    def get_patch(self, experiment_id: str, patch_id: str) -> PatchPreview:
        self._experiment_row(experiment_id)
        return PatchPreview.model_validate_json(self._patch_row(experiment_id, patch_id)["preview_json"])

    def read_patch(self, experiment_id: str, patch_id: str) -> PatchArtifact:
        """The stored frozen spec and exact materialized patch (operator view; hash-verified).

        Not available to a run: the after-values reproduce base weights, so a
        run sees patches through ``find_patch`` instead.
        """
        self._require_operator("reading the exact materialized patch")
        return self._artifact(experiment_id, patch_id)

    def _artifact(self, experiment_id: str, patch_id: str) -> PatchArtifact:
        self._experiment_row(experiment_id)
        return self._artifact_and_blocks(self._patch_row(experiment_id, patch_id))[0]

    def _artifact_and_blocks(self, patch_row: sqlite3.Row) -> tuple[PatchArtifact, dict[str, bytes]]:
        """A stored patch artifact and its binary blocks (none for a JSON patch), hash-verified."""
        document, blocks = read_patch_file(self.store.read_patch(patch_row))
        return PatchArtifact.model_validate(document), blocks

    def audit_log(self, experiment_id: str) -> list[dict[str, Any]]:
        self._experiment_row(experiment_id)
        return experiment_events(self.db, experiment_id)

    # -- patches ----------------------------------------------------------------

    @staticmethod
    def _parse_spec(spec: PatchSpec | Mapping[str, Any]) -> PatchSpec:
        if isinstance(spec, PatchSpec):
            return spec
        if not isinstance(spec, Mapping):
            raise WSEError(ErrorCode.INVALID_PATCH, "patch_spec must be an object")
        try:
            return PatchSpec.model_validate(dict(spec))
        except ValidationError as exc:
            # The message and details list the first problems and count the rest, so that a spec with many invalid
            # operations cannot produce an unbounded error (or audit record).
            errors = exc.errors(include_url=False, include_input=False, include_context=False)
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'patch_spec'}: {error['msg']}"
                for error in errors[:MAX_REPORTED_PROBLEMS]
            )
            if len(errors) > MAX_REPORTED_PROBLEMS:
                problems += f"; and {len(errors) - MAX_REPORTED_PROBLEMS} more problems"
            extra = [error["loc"][0] for error in errors if error["type"] == "extra_forbidden"
                     and len(error["loc"]) == 1]
            hint = (" The server resolves the base content hash itself; do not send it."
                    if "base_weights_hash" in extra else "")
            raise WSEError(
                ErrorCode.INVALID_PATCH,
                f"invalid patch_spec: {problems}.{hint}",
                details={"errors": json.loads(exc.json(include_url=False, include_input=False))[:MAX_REPORTED_PROBLEMS],
                         "error_count": len(errors)},
            ) from exc

    def preview_patch(self, experiment_id: str, spec: PatchSpec | Mapping[str, Any], key: str) -> PatchPreview:
        """Validate and materialize a patch; store an immutable preview.

        The selected checkpoint does not change. Returns the recorded preview
        for a repeated key with the same spec.
        """
        _check_key(key)
        row = self._experiment_row(experiment_id)
        if not isinstance(spec, (PatchSpec, Mapping)):
            raise WSEError(ErrorCode.INVALID_PATCH, "patch_spec must be an object")
        try:
            raw = spec.model_dump(mode="json") if isinstance(spec, PatchSpec) else json.loads(canonical_json(dict(spec)))
        except (WSEError, TypeError, ValueError) as exc:
            raise WSEError(ErrorCode.INVALID_PATCH, f"patch_spec must be plain JSON: {exc}") from exc
        request_hash = json_hash({"operation": "preview_patch", "spec": raw})
        recorded = self._lookup(self.db.conn, experiment_id, key, "preview_patch", request_hash)
        if recorded is not None:
            return PatchPreview.model_validate_json(recorded)
        return self._guarded("patch.preview", experiment_id, request_hash, None,
                             lambda: self._preview(row, raw, key, request_hash))

    def _preview(self, row: sqlite3.Row, raw: dict[str, Any], key: str, request_hash: str) -> PatchPreview:
        start = time.perf_counter()
        experiment_id = row["experiment_id"]
        self._require_open(row)
        self.ledger.require(self.db.conn, experiment_id, {"patch_previews": 1, "compute_seconds": 0})
        parsed = self._parse_spec(raw)
        policy = self.experiment_policy(experiment_id)
        self._owned_checkpoint(self.db.conn, experiment_id, parsed.base_checkpoint_id)
        operations = [{"op": operation.op, **operation.operator_request()} for operation in parsed.operations]
        # Vector references resolve through the run-scoped reader, as analyses do, before any weights are read.
        references = resolve_references(operations, self.find_analysis_result, registry=self.engine.registry,
                                        policy=policy.operations)
        base = self.engine._open_checkpoint(parsed.base_checkpoint_id)
        edit = materialize_patch(
            base,
            operations,
            registry=self.engine.registry,
            policy=policy.operations,
            references=references,
            managed=True,
        )
        frozen = FrozenPatchSpec(
            base_checkpoint_id=parsed.base_checkpoint_id,
            base_weights_hash=base.weights_hash,
            hypothesis=parsed.hypothesis,
            expected_behavior=parsed.expected_behavior,
            development_suite_id=parsed.development_suite_id,
            operations=[
                FrozenOperation(op=step.op, operator=step.operator, tensor=step.tensor,
                                selection=step.selection, parameters=step.parameters)
                for step in edit.patch.steps
            ],
        )
        artifact = PatchArtifact(spec=frozen, patch=edit.patch)
        if edit.blocks:  # an open-weight model's steps: one binary patch file with their blocks
            data = write_patch_file(artifact.model_dump(mode="json"), edit.blocks)
            relpath, file_sha, size = self.store.publish_patch(data, binary=True)
        else:
            data = artifact.canonical_json().encode("utf-8")
            relpath, file_sha, size = self.store.publish_patch(data)
        patch_id = _new_id("patch")
        diffs = list(edit.diff)
        changed = sum(diff.changed_elements for diff in diffs)
        l2 = sum(diff.l2_delta ** 2 for diff in diffs) ** 0.5
        base_l2 = sum(diff.before_l2 ** 2 for diff in diffs) ** 0.5
        candidate_bytes = sum(t.numel() * t.element_size() for t in base._tensors.values())
        preview = PatchPreview(
            patch_id=patch_id,
            experiment_id=experiment_id,
            spec_hash=json_hash(frozen.model_dump(mode="json")),
            base_checkpoint_id=parsed.base_checkpoint_id,
            base_weights_hash=base.weights_hash,
            result_weights_hash=edit.patch.result_weights_hash,
            hypothesis=frozen.hypothesis,
            expected_behavior=frozen.expected_behavior,
            development_suite_id=frozen.development_suite_id,
            operations=[
                PatchOperationSummary(index=i, op=step.op, operator=step.operator, tensor=step.tensor,
                                      selection=step.selection, changed_elements=step.changed_elements)
                for i, step in enumerate(edit.patch.steps)
            ],
            tensors=diffs,
            changed_elements=changed,
            parameter_count=base.parameter_count,
            changed_fraction=changed / base.parameter_count,
            l2_delta=l2,
            max_abs_delta=max((diff.max_abs_delta for diff in diffs), default=0.0),
            relative_l2_delta=(l2 / base_l2) if base_l2 > 0 else None,
            all_finite=all(diff.all_finite for diff in diffs),
            storage=StorageCost(patch_bytes=size, candidate_bytes=candidate_bytes),
            patch_uri=PATCH_URI_PREFIX + patch_id,
            usage=Usage(model_forward_calls=0, candidate_commits=0, patch_previews=1, artifact_bytes=size,
                        compute_seconds=round(time.perf_counter() - start, 6)),
        )
        preview_json = preview.canonical_json()
        with self.db.transaction() as conn:
            recorded = self._lookup(conn, experiment_id, key, "preview_patch", request_hash)
            if recorded is not None:
                return PatchPreview.model_validate_json(recorded)
            self._require_open(self._experiment_row(experiment_id, conn))
            self.store.require_published(relpath)
            self.ledger.charge(conn, experiment_id, "patch_previews", 1)
            self.ledger.charge_compute(conn, experiment_id, preview.usage.compute_seconds or 0.0)
            conn.execute(
                "INSERT INTO patches (patch_id, experiment_id, state, spec_hash, base_checkpoint_id, "
                "result_weights_hash, relpath, file_sha256, size_bytes, preview_json, created_at) "
                "VALUES (?, ?, 'READY', ?, ?, ?, ?, ?, ?, ?, ?)",
                (patch_id, experiment_id, preview.spec_hash, parsed.base_checkpoint_id,
                 preview.result_weights_hash, relpath, file_sha, size, preview_json, utc_now()),
            )
            record_event(conn, "patch.previewed", patch_id,
                         {"spec_hash": preview.spec_hash, "base_checkpoint_id": parsed.base_checkpoint_id,
                          "result_weights_hash": preview.result_weights_hash,
                          "operators": [str(step.operator) for step in edit.patch.steps],
                          "changed_elements": changed},
                         experiment_id=experiment_id)
            self._remember(conn, experiment_id, key, "preview_patch", request_hash, preview_json)
        return preview

    def apply_patch(self, experiment_id: str, patch_id: str, key: str) -> ModelManifest:
        """Create a READY candidate from a validated preview, charging one candidate commit."""
        _check_key(key)
        self._experiment_row(experiment_id)
        request_hash = json_hash({"operation": "apply_patch", "patch_id": _hashable(patch_id)})
        recorded = self._lookup(self.db.conn, experiment_id, key, "apply_patch", request_hash)
        if recorded is not None:
            return self._recorded_manifest(recorded)
        return self._guarded("patch.apply", experiment_id, request_hash, str(patch_id),
                             lambda: self._apply(experiment_id, patch_id, key, request_hash))

    def _recorded_manifest(self, result_json: str) -> ModelManifest:
        return self.store.load_manifest(json.loads(result_json)["checkpoint_id"])

    def _remaining(self, conn: sqlite3.Connection, experiment_id: str, counter: str) -> int | float:
        return self.ledger.remaining(conn, experiment_id, counter)

    def _charge(self, conn: sqlite3.Connection, experiment_id: str, counter: str, amount: int) -> None:
        """Atomically consume ``amount``; never exceeds the limit (single conditional UPDATE)."""
        self.ledger.charge(conn, experiment_id, counter, amount)

    def _budget_error(self, conn: sqlite3.Connection, experiment_id: str, counter: str, amount: int) -> WSEError:
        return budget_error(conn, experiment_id, counter, amount)

    def _apply(self, experiment_id: str, patch_id: Any, key: str, request_hash: str) -> ModelManifest:
        row = self._experiment_row(experiment_id)
        self._require_open(row)
        patch_row = self._patch_row(experiment_id, patch_id)
        if patch_row["applied_checkpoint_id"] is not None:
            # The candidate already exists: no new version, no new charge.
            manifest = self.store.load_manifest(patch_row["applied_checkpoint_id"])
            with self.db.transaction() as conn:
                if self._lookup(conn, experiment_id, key, "apply_patch", request_hash) is None:
                    self._remember(conn, experiment_id, key, "apply_patch", request_hash,
                                   canonical_json({"checkpoint_id": manifest.checkpoint_id}))
            return manifest
        start = time.perf_counter()
        if self._remaining(self.db.conn, experiment_id, "candidate_commits") < 1:
            raise self._budget_error(self.db.conn, experiment_id, "candidate_commits", 1)
        self.ledger.require(self.db.conn, experiment_id, {"compute_seconds": 0})
        artifact, blocks = self._artifact_and_blocks(patch_row)
        self._owned_checkpoint(self.db.conn, experiment_id, artifact.spec.base_checkpoint_id)
        base = self.engine._open_checkpoint(artifact.spec.base_checkpoint_id)
        if base.weights_hash != artifact.spec.base_weights_hash:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"base of patch {patch_id} no longer matches its frozen hash")
        candidate = replay_patch(base, artifact.patch, blocks=blocks)  # stored values only; no operator runs
        base_manifest = self.store.load_manifest(artifact.spec.base_checkpoint_id)
        manifest = base_manifest.model_copy(
            update={
                "checkpoint_id": self.store.new_checkpoint_id(),
                "weights_hash": candidate.weights_hash,
                "tensors": _candidate_tensor_infos(base_manifest, base, candidate),
                "provenance": Provenance(kind="patch", source=patch_id, parent_checkpoint_id=base_manifest.checkpoint_id),
            }
        )
        manifest = ModelManifest.model_validate(manifest.model_dump(mode="json"))

        def on_commit(conn: sqlite3.Connection, committed: ModelManifest) -> None:
            recorded = self._lookup(conn, experiment_id, key, "apply_patch", request_hash)
            if recorded is not None:
                raise _Recorded(recorded)
            current = conn.execute(
                "SELECT state, applied_checkpoint_id FROM patches WHERE patch_id = ?", (patch_id,)
            ).fetchone()
            if current["applied_checkpoint_id"] is not None:
                raise _Recorded(canonical_json({"checkpoint_id": current["applied_checkpoint_id"]}))
            self._require_open(self._experiment_row(experiment_id, conn))
            self._charge(conn, experiment_id, "candidate_commits", 1)
            self.ledger.charge_compute(conn, experiment_id, time.perf_counter() - start)
            conn.execute(
                "INSERT INTO experiment_checkpoints (experiment_id, checkpoint_id, role, patch_id, created_at) "
                "VALUES (?, ?, 'candidate', ?, ?)",
                (experiment_id, committed.checkpoint_id, patch_id, utc_now()),
            )
            conn.execute("UPDATE patches SET applied_checkpoint_id = ? WHERE patch_id = ?",
                         (committed.checkpoint_id, patch_id))
            record_event(conn, "patch.applied", patch_id,
                         {"checkpoint_id": committed.checkpoint_id, "weights_hash": committed.weights_hash,
                          "parent_checkpoint_id": base_manifest.checkpoint_id},
                         experiment_id=experiment_id)
            self._remember(conn, experiment_id, key, "apply_patch", request_hash,
                           canonical_json({"checkpoint_id": committed.checkpoint_id}))

        try:
            return self.store.commit_checkpoint(
                manifest,
                dict(candidate._tensors),
                dedup_key=None,
                event_type="checkpoint.candidate",
                on_commit=on_commit,
                experiment_id=experiment_id,
            )
        except _Recorded as done:
            if self._lookup(self.db.conn, experiment_id, key, "apply_patch", request_hash) is None:
                with self.db.transaction() as conn:
                    if self._lookup(conn, experiment_id, key, "apply_patch", request_hash) is None:
                        self._remember(conn, experiment_id, key, "apply_patch", request_hash, done.result_json)
            return self._recorded_manifest(done.result_json)

    # -- selection --------------------------------------------------------------

    def _ancestors(self, conn: sqlite3.Connection, checkpoint_id: str) -> list[str]:
        chain, current = [], checkpoint_id
        while current is not None and len(chain) < 10_000:
            row = conn.execute("SELECT parent_checkpoint_id FROM checkpoints WHERE checkpoint_id = ?",
                               (current,)).fetchone()
            current = row["parent_checkpoint_id"] if row else None
            if current is not None:
                chain.append(current)
        return chain

    def select_checkpoint(
        self, experiment_id: str, checkpoint_id: str, expected_selected_id: str, key: str
    ) -> Experiment:
        """Compare-and-swap the selected checkpoint; selecting an ancestor is a rollback."""
        _check_key(key)
        self._experiment_row(experiment_id)
        request_hash = json_hash({"operation": "select_checkpoint", "checkpoint_id": _hashable(checkpoint_id),
                                  "expected_selected_id": _hashable(expected_selected_id)})
        recorded = self._lookup(self.db.conn, experiment_id, key, "select_checkpoint", request_hash)
        if recorded is not None:
            return Experiment.model_validate_json(recorded)
        return self._guarded("experiment.select", experiment_id, request_hash, str(checkpoint_id),
                             lambda: self._select(experiment_id, checkpoint_id, expected_selected_id, key,
                                                  request_hash))

    def _select(self, experiment_id: str, checkpoint_id: Any, expected: Any, key: str, request_hash: str) -> Experiment:
        if not isinstance(expected, str):
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"expected_selected_id must be a checkpoint ID string, not {type(expected).__name__}; "
                "read the experiment and pass its selected_checkpoint_id",
            )
        self._owned_checkpoint(self.db.conn, experiment_id, checkpoint_id)
        self.store.load_manifest(checkpoint_id)  # hash-verified before it can be selected
        with self.db.transaction() as conn:
            recorded = self._lookup(conn, experiment_id, key, "select_checkpoint", request_hash)
            if recorded is not None:
                return Experiment.model_validate_json(recorded)
            row = self._experiment_row(experiment_id, conn)
            self._require_open(row)
            self._owned_checkpoint(conn, experiment_id, checkpoint_id)
            previous = row["selected_checkpoint_id"]
            updated = conn.execute(
                "UPDATE experiments SET selected_checkpoint_id = ? "
                "WHERE experiment_id = ? AND selected_checkpoint_id = ? AND state = 'OPEN'",
                (checkpoint_id, experiment_id, expected),
            ).rowcount
            if not updated:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"the selected checkpoint is {previous}, not {expected!r}; read the experiment and retry "
                    "with the current selected ID",
                    details={"expected_selected_id": str(expected), "selected_checkpoint_id": previous},
                )
            rollback = checkpoint_id in self._ancestors(conn, previous)
            record_event(conn, "experiment.rollback" if rollback else "experiment.selected", checkpoint_id,
                         {"from": previous, "to": checkpoint_id, "rollback": rollback},
                         experiment_id=experiment_id)
            result = self._experiment(self._experiment_row(experiment_id, conn), conn)
            self._remember(conn, experiment_id, key, "select_checkpoint", request_hash, result.canonical_json())
            return result

    # -- freezing -----------------------------------------------------------------

    def freeze(self, experiment_id: str, expected_selected_id: str) -> Experiment:
        """Freeze the selected checkpoint (operator only); a retry of a completed freeze returns it."""
        self._require_operator("freezing an experiment")
        self._experiment_row(experiment_id)
        request_hash = json_hash({"operation": "freeze", "expected_selected_id": _hashable(expected_selected_id)})
        return self._guarded("experiment.freeze", experiment_id, request_hash, str(expected_selected_id),
                             lambda: self._freeze(experiment_id, expected_selected_id))

    def _freeze(self, experiment_id: str, expected: Any) -> Experiment:
        if not isinstance(expected, str):
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           "expected_selected_id must be a checkpoint ID string; read the experiment and pass its "
                           "selected_checkpoint_id")
        self.engine.jobs.recover_lost_jobs()
        with self.db.transaction() as conn:
            row = self._experiment_row(experiment_id, conn)
            if row["selected_checkpoint_id"] != expected:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"the selected checkpoint is {row['selected_checkpoint_id']}, not {expected!r}; read the "
                    "experiment and retry with the current selected ID",
                    details={"expected_selected_id": expected,
                             "selected_checkpoint_id": row["selected_checkpoint_id"]},
                )
            if row["state"] != "OPEN":
                return self._experiment(row, conn)  # already frozen with this selection
            active = [r["job_id"] for r in conn.execute(
                "SELECT job_id FROM jobs WHERE experiment_id = ? AND state IN ('queued', 'running') "
                "ORDER BY created_at, rowid", (experiment_id,))]
            held = [r["reservation_id"] for r in conn.execute(
                "SELECT reservation_id FROM reservations WHERE experiment_id = ? AND state = 'HELD' "
                "ORDER BY created_at, rowid", (experiment_id,))]
            if active or held:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"experiment {experiment_id} has {len(active)} active jobs and {len(held)} unsettled "
                    "reservations; wait for or cancel them, then retry the freeze",
                    retryable=True,
                    details={"active_jobs": active, "held_reservations": held},
                )
            now = utc_now()
            conn.execute(
                "UPDATE experiments SET state = 'FROZEN', frozen_at = ? "
                "WHERE experiment_id = ? AND state = 'OPEN' AND selected_checkpoint_id = ?",
                (now, experiment_id, expected),
            )
            frozen = self._experiment(self._experiment_row(experiment_id, conn), conn)
            record_event(conn, "experiment.frozen", experiment_id,
                         {"selected_checkpoint_id": expected,
                          "consumed": {name: entry.consumed for name, entry in frozen.budget.items()}},
                         experiment_id=experiment_id)
            return frozen
