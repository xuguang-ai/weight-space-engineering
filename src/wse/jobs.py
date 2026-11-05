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
* Settled usage leaves no field silently empty: counters an evaluation job
  never uses are 0, and a field that cannot be measured is ``None`` with the
  reason in ``Usage.unavailable``.
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
    OBJECT_ID_PATTERNS,
    TERMINAL_JOB_STATES,
    DomainError,
    ErrorCode,
    EvalConfig,
    EvaluationReport,
    EvaluationSuite,
    Job,
    JobProgress,
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


def queued_cancel_usage() -> Usage:
    """Usage of a job cancelled before any worker started it."""
    return Usage(evaluation_batches=0, evaluated_examples=0, model_forward_calls=0, compute_seconds=0.0,
                 analysis_calls=0, tensor_read_calls=0, candidate_commits=0, patch_previews=0, artifact_bytes=0,
                 peak_memory_bytes=None,
                 unavailable={"peak_memory_bytes": "no worker executed this job, so no memory was measured"})


def lost_job_usage(requested: Mapping[str, Any]) -> Usage:
    """Usage charged for a lost job: its whole reservation, with unmeasured fields explained."""
    return Usage.model_validate({
        "evaluation_batches": 0, "evaluated_examples": 0, "model_forward_calls": 0, "compute_seconds": 0.0,
        **requested,
        "analysis_calls": 0, "tensor_read_calls": 0, "candidate_commits": 0, "patch_previews": 0, "artifact_bytes": 0,
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


class JobService:
    """Evaluation jobs of an ``Engine``, scoped like its experiment service."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.db = engine.store.db
        self.experiments = engine.experiments
        self.ledger = engine.experiments.ledger

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
        self.engine.registry.resolve_ref("evaluator", parsed.evaluator)
        if parsed.split == "final" and parsed.dataset is None:
            parsed = parsed.model_copy(update={"dataset": self.engine.datasets.put(parsed.data)})
        elif parsed.dataset is not None and parsed.data is not None:
            self.engine.datasets.put(parsed.data)
        if parsed.dataset is not None:
            try:
                parsed.resolved(self.engine.datasets.load(parsed.dataset))  # verifies data and preregistration
            except ValidationError as exc:
                raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid evaluation suite: {exc}") from exc
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

    def _development_suite(self, suite_id: Any, task_id: str) -> EvaluationSuite:
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
                f"suite {suite.suite_id} is not a development suite; agents may evaluate development suites only, "
                "and final evaluation is an operator action after freezing",
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
        per_batch = policy.budget.examples_per_batch
        if per_batch < 1:
            raise WSEError(ErrorCode.POLICY_DENIED, "the experiment's policy permits no evaluation batches")
        batches = -(-config.sample_count // per_batch)
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
                reservation = self.ledger.reserve(
                    experiment_id, job_id,
                    {"evaluation_batches": batches, "evaluated_examples": config.sample_count,
                     "model_forward_calls": batches, "compute_seconds": compute},
                    conn=conn,
                )
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
        return Job(
            job_id=row["job_id"],
            experiment_id=row["experiment_id"],
            kind=row["kind"],
            checkpoint_id=row["checkpoint_id"],
            suite_id=row["suite_id"],
            config=EvalConfig.model_validate_json(row["config_json"]),
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
