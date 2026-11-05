"""The evaluation worker: a separate process that claims, runs, meters, and settles jobs.

Run it as ``python -m wse.worker --workspace DIR`` (or ``wse worker``), or keep
one running with ``WorkerSupervisor``. It never executes agent-generated
code: it runs registered evaluators over registered adapters only.

* **Single worker.** A job is claimed only while no other job is running, in
  one ``BEGIN IMMEDIATE`` transaction, so at most one job executes at a time
  however many worker processes share the workspace.
* **Metering.** The evaluator receives a metered forward callable, never the
  model. It counts every forward call and example, refuses calls beyond the
  job's reservation, and in an experiment whose mode permits no inference
  (``weights_only``) the worker fails the job before loading the model, so
  the target network is never executed.
* **Leases and heartbeats.** A monitor thread renews the job's lease, records
  progress, and watches for cancellation and the time limit. A worker that
  dies stops renewing; its job is then settled as lost by crash recovery.
* **Supervision.** A worker started by ``WorkerSupervisor`` (the job
  scheduler of ``wse serve`` or of the runner) is given its supervisor's
  process ID and stops, after its current job, once that process is no longer
  its parent: a supervisor that dies without cleanup (a crash, SIGKILL)
  leaves no orphan worker behind to claim later runs' jobs.
* **Stopping.** Cancellation and the time limit (``seconds_per_operation``,
  capped by the remaining compute budget, as reserved) stop a job at its next
  forward call or ``forward.check()``. A job that does not yield within the
  grace period is hard-stopped: the monitor settles it with the work already
  started and terminates the process with ``EXIT_HARD_STOP``; a supervisor
  starts a fresh worker.
* **Settlement.** Started batches, examples, and forward calls are charged,
  with compute seconds measured from the start of execution (queue time is
  excluded) and capped at the reservation; unstarted work is released.
* **Reports.** The job body is the core ``evaluate_candidate`` with the
  metered forward callable. A successful result is persisted as an
  ``EvaluationReport`` bound to the checkpoint, task, suite, data, and
  execution configuration, in the transaction that settles the job; the job
  carries its summary inline. Stored examples are read from evaluator
  storage only here.
"""

from __future__ import annotations

import argparse
import importlib
import os
import resource
import secrets
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from wse.api import evaluate_candidate
from wse.audit import record_event
from wse.database import Database, utc_now
from wse.jobs import queued_cancel_usage
from wse.schemas import ErrorCode, EvalConfig, JobProgress, Usage, WSEError

EXIT_HARD_STOP = 70
DEFAULT_LEASE_SECONDS = 10.0
DEFAULT_POLL_SECONDS = 0.1
DEFAULT_CANCEL_GRACE_SECONDS = 2.0
DEFAULT_TIMEOUT_GRACE_SECONDS = 1.0


def peak_memory_bytes() -> int:
    """Peak resident memory of this worker process (process-wide, including imports)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class JobStopped(Exception):
    """Raised inside a job when it must stop (``cancelled``, ``timeout``, or ``lost``)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ForwardMeter:
    """The metered forward callable handed to evaluators."""

    def __init__(self, *, permitted: bool, batch_size: int, max_calls: int, max_examples: int,
                 deadline: float) -> None:
        self.permitted = permitted
        self.batch_size = batch_size
        self.max_calls = max_calls
        self.max_examples = max_examples
        self.deadline = deadline
        self.started_calls = 0
        self.completed_calls = 0
        self.started_examples = 0
        self.stop_reason: str | None = None
        self._forward: Callable[[torch.Tensor], torch.Tensor] | None = None

    def bind(self, forward: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._forward = forward

    def check(self) -> None:
        if self.stop_reason is None and time.monotonic() > self.deadline:
            self.stop_reason = "timeout"
        if self.stop_reason is not None:
            raise JobStopped(self.stop_reason)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.check()
        if not self.permitted or self._forward is None:
            raise WSEError(ErrorCode.POLICY_DENIED,
                           "this experiment permits no forward passes; the worker never runs the target network")
        if not isinstance(inputs, torch.Tensor) or inputs.dim() < 1 or not 1 <= inputs.shape[0] <= self.batch_size:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"a forward batch must have between 1 and {self.batch_size} rows")
        rows = int(inputs.shape[0])
        if self.started_calls + 1 > self.max_calls or self.started_examples + rows > self.max_examples:
            raise WSEError(
                ErrorCode.BUDGET_EXCEEDED,
                f"the evaluator requested more than the {self.max_calls} forward calls and {self.max_examples} "
                "examples reserved for this job",
                details={"reserved_forward_calls": self.max_calls, "reserved_examples": self.max_examples},
            )
        self.started_calls += 1
        self.started_examples += rows
        logits = self._forward(inputs)
        self.completed_calls += 1
        return logits

    def progress(self, batches_total: int) -> JobProgress:
        return JobProgress(batches_total=batches_total, batches_started=self.started_calls,
                           batches_completed=self.completed_calls, examples_started=self.started_examples)


def _stop_error(reason: str, limit: float) -> WSEError:
    if reason == "cancelled":
        return WSEError(ErrorCode.WORKER_FAILED,
                        "the job was cancelled while running; computation already performed remains charged",
                        details={"reason": "cancelled"})
    if reason == "timeout":
        return WSEError(ErrorCode.WORKER_FAILED,
                        f"the job exceeded its time limit of {limit:g} seconds; computation already performed "
                        "remains charged",
                        details={"reason": "timeout", "limit_seconds": limit})
    return WSEError(ErrorCode.WORKER_FAILED, "the worker lost the job's lease", details={"reason": reason})


class _Execution:
    """State shared by the main thread and the monitor of one running job."""

    def __init__(self, job_id: str, meter: ForwardMeter, batches_total: int, limit: float) -> None:
        self.job_id = job_id
        self.experiment_id: str | None = None
        self.checkpoint_id: str | None = None
        self.suite: Any = None
        self.meter = meter
        self.batches_total = batches_total
        self.limit = limit
        self.started = time.monotonic()
        self.done = threading.Event()
        self.cancel_seen: float | None = None

    def usage(self) -> Usage:
        elapsed = time.monotonic() - self.started
        return Usage(
            evaluation_batches=self.meter.started_calls,
            evaluated_examples=self.meter.started_examples,
            model_forward_calls=self.meter.started_calls,
            compute_seconds=round(min(elapsed, self.limit), 6),
            analysis_calls=0,
            tensor_read_calls=0,
            candidate_commits=0,
            patch_previews=0,
            artifact_bytes=0,
            peak_memory_bytes=peak_memory_bytes(),
        )


class Worker:
    """Claims and runs jobs of one workspace; see the module docstring."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        registry: Any = None,
        worker_id: str | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        heartbeat_seconds: float | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        cancel_grace_seconds: float = DEFAULT_CANCEL_GRACE_SECONDS,
        timeout_grace_seconds: float = DEFAULT_TIMEOUT_GRACE_SECONDS,
        hard_exit: Callable[[int], Any] = os._exit,
        parent_pid: int | None = None,
    ) -> None:
        from wse.api import Engine

        self.engine = Engine(workspace, registry=registry, recover=False)
        self.db = self.engine.store.db
        self.jobs = self.engine.jobs
        self.worker_id = worker_id or f"worker_{secrets.token_hex(8)}"
        self.lease_seconds = float(lease_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds or min(self.lease_seconds / 3, 0.5))
        self.poll_seconds = float(poll_seconds)
        self.cancel_grace_seconds = float(cancel_grace_seconds)
        self.timeout_grace_seconds = float(timeout_grace_seconds)
        self.hard_exit = hard_exit
        self.parent_pid = parent_pid
        self._stop = threading.Event()

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request_stop(self) -> None:
        """Stop after the current job (used for SIGTERM)."""
        self._stop.set()

    # -- loop -------------------------------------------------------------------

    def run(self, *, max_jobs: int | None = None, idle_exit_seconds: float | None = None) -> int:
        """Process jobs until stopped; return the number of jobs run."""
        count, idle_since = 0, time.monotonic()
        while not self._stop.is_set():
            if self.parent_pid is not None and os.getppid() != self.parent_pid:
                print(f"wse worker: supervisor process {self.parent_pid} is gone; stopping", file=sys.stderr,
                      flush=True)
                break
            if self.run_once() is not None:
                count += 1
                idle_since = time.monotonic()
                if max_jobs is not None and count >= max_jobs:
                    break
                continue
            if idle_exit_seconds is not None and time.monotonic() - idle_since >= idle_exit_seconds:
                break
            self._stop.wait(self.poll_seconds)
        return count

    def run_once(self) -> str | None:
        """Settle lost jobs, then claim and run the oldest queued job; return its ID."""
        self.jobs.recover_lost_jobs()
        row = self._claim()
        if row is None:
            return None
        self._execute(row)
        return row["job_id"]

    def _claim(self) -> Any:
        with self.db.transaction() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE state = 'running' LIMIT 1").fetchone():
                return None
            row = conn.execute(
                "SELECT j.job_id, j.experiment_id, r.state AS reservation_state FROM jobs j "
                "JOIN reservations r ON r.reservation_id = j.reservation_id "
                "WHERE j.state = 'queued' ORDER BY j.created_at, j.rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            if row["reservation_state"] != "HELD":
                # Allowance for this job is no longer held, so it must never run.
                self.jobs.finish(conn, row["job_id"], "failed", queued_cancel_usage(), from_state="queued",
                                 error=WSEError(ErrorCode.WORKER_FAILED,
                                                "the job's reservation was settled before it started, so it was "
                                                "not run", details={"reason": "reservation_settled"}))
                return None
            now = utc_now()
            conn.execute(
                "UPDATE jobs SET state = 'running', lease_owner = ?, lease_expires_at = ?, heartbeat_at = ?, "
                "started_at = ?, attempts = attempts + 1, lease_pid = ? WHERE job_id = ? AND state = 'queued'",
                (self.worker_id, time.time() + self.lease_seconds, now, now, os.getpid(), row["job_id"]),
            )
            record_event(conn, "job.started", row["job_id"], {"worker_id": self.worker_id},
                         experiment_id=row["experiment_id"])
            return conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()

    # -- execution --------------------------------------------------------------

    def _execute(self, row: Any) -> None:
        experiment = self.db.conn.execute("SELECT * FROM experiments WHERE experiment_id = ?",
                                          (row["experiment_id"],)).fetchone()
        reservation = self.jobs.ledger.get(row["reservation_id"])
        # Only allowance that is still held may be spent.
        requested = reservation.requested if reservation.state == "HELD" else {}
        progress = JobProgress.model_validate_json(row["progress_json"])
        limit = float(requested.get("compute_seconds", 0))
        permitted = experiment["mode"] == "bounded_eval" and requested.get("model_forward_calls", 0) > 0
        meter = ForwardMeter(permitted=permitted, batch_size=1, max_calls=int(requested.get("model_forward_calls", 0)),
                             max_examples=int(requested.get("evaluated_examples", 0)),
                             deadline=time.monotonic() + limit)
        execution = _Execution(row["job_id"], meter, progress.batches_total, limit)
        execution.experiment_id, execution.checkpoint_id = row["experiment_id"], row["checkpoint_id"]
        monitor = threading.Thread(target=self._monitor, args=(execution,), daemon=True,
                                   name=f"wse-monitor-{row['job_id']}")
        monitor.start()
        result: Any = None
        error: WSEError | None = None
        try:
            if experiment["mode"] != "bounded_eval":
                raise WSEError(
                    ErrorCode.POLICY_DENIED,
                    f"experiment mode {experiment['mode']} permits no forward passes; the worker never runs the "
                    "target network",
                    details={"mode": experiment["mode"]},
                )
            policy = self.engine.experiments.get_policy(experiment["policy_id"])
            if policy.policy_hash != experiment["policy_hash"]:
                raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"policy of experiment {experiment['experiment_id']} changed")
            meter.batch_size = policy.budget.examples_per_batch
            execution.suite = self.jobs.load_suite(row["suite_id"])
            result = self.evaluate(row["checkpoint_id"], row["suite_id"],
                                   EvalConfig.model_validate_json(row["config_json"]), meter)
            meter.check()  # a stop requested while the evaluator swallowed it still wins
            if time.monotonic() - execution.started > limit:
                raise JobStopped("timeout")
        except JobStopped as stop:
            error = _stop_error(stop.reason, limit)
        except WSEError as exc:
            error = exc
        except Exception as exc:  # noqa: BLE001 - any evaluator failure is a job failure
            error = WSEError(ErrorCode.WORKER_FAILED, f"the evaluation raised {type(exc).__name__}: {exc}")
        finally:
            execution.done.set()
            monitor.join()
        if error is not None and error.details.get("reason") == "lost":
            return  # recovery settled this job; this worker's result is discarded
        state = "succeeded" if error is None else ("cancelled" if error.details.get("reason") == "cancelled"
                                                   else "failed")
        self._finish(self.db, execution, state, result, error)

    def evaluate(self, checkpoint_id: str, suite_id: str, config: EvalConfig, meter: ForwardMeter) -> Any:
        """Evaluate a checkpoint on a suite through ``meter`` (worker-internal only).

        Stored examples are read from evaluator storage, the evaluator runs
        the core ``evaluate_candidate`` with the metered forward callable, and
        the returned ``EvaluationResult`` becomes the job's
        ``EvaluationReport`` when the job is settled.
        """
        suite = self.jobs.load_suite(suite_id)
        if suite.data is None:
            suite = suite.resolved(self.engine.datasets.load(suite.dataset))
        self.engine.registry.resolve_ref("evaluator", suite.evaluator)
        handle = self.engine.open_checkpoint(checkpoint_id)

        def run_forward(inputs: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                return handle.adapter.forward(handle._tensors, inputs)

        meter.bind(run_forward)
        return evaluate_candidate(handle, suite, config, registry=self.engine.registry, forward=meter,
                                  batch_size=meter.batch_size)

    def _finish(self, db: Database, execution: _Execution, state: str, result: Any,
                error: WSEError | None) -> bool:
        usage = execution.usage()
        report = None
        if result is not None and state == "succeeded":
            report = self.engine.evaluation.build_report(
                result, experiment_id=execution.experiment_id, job_id=execution.job_id,
                checkpoint_id=execution.checkpoint_id, suite=execution.suite, usage=usage, keep_scores=True)
        with db.transaction() as conn:
            return self.jobs.finish(conn, execution.job_id, state, usage,
                                    result=report.summary() if report is not None else None, error=error,
                                    progress=execution.meter.progress(execution.batches_total),
                                    lease_owner=self.worker_id, report=report)

    def _monitor(self, execution: _Execution) -> None:
        db = Database(self.db.path)
        try:
            while not execution.done.wait(self.heartbeat_seconds):
                try:
                    if not self._heartbeat(db, execution):
                        return
                except sqlite3.OperationalError as exc:  # a busy database: retry at the next heartbeat
                    print(f"wse worker: heartbeat of {execution.job_id} failed: {exc}", file=sys.stderr, flush=True)
        finally:
            db.close()

    def _heartbeat(self, db: Database, execution: _Execution) -> bool:
        """Renew the lease, record progress, and handle cancellation, timeouts, and hard stops.

        Returns whether monitoring should continue.
        """
        meter = execution.meter
        renewed = db.conn.execute(
            "UPDATE jobs SET lease_expires_at = ?, heartbeat_at = ?, progress_json = ? "
            "WHERE job_id = ? AND state = 'running' AND lease_owner = ?",
            (time.time() + self.lease_seconds, utc_now(), meter.progress(execution.batches_total).canonical_json(),
             execution.job_id, self.worker_id),
        ).rowcount
        if not renewed:
            meter.stop_reason = "lost"
            return False
        now = time.monotonic()
        cancelled = db.conn.execute("SELECT cancel_requested_at FROM jobs WHERE job_id = ?",
                                    (execution.job_id,)).fetchone()["cancel_requested_at"]
        if cancelled and execution.cancel_seen is None:
            execution.cancel_seen = now
            if meter.stop_reason is None:
                meter.stop_reason = "cancelled"
        if meter.stop_reason is None and now > meter.deadline:
            meter.stop_reason = "timeout"
        hard = (
            (meter.stop_reason == "cancelled" and execution.cancel_seen is not None
             and now - execution.cancel_seen > self.cancel_grace_seconds)
            or (meter.stop_reason == "timeout" and now > meter.deadline + self.timeout_grace_seconds)
        )
        if hard and not execution.done.is_set():
            state = "cancelled" if meter.stop_reason == "cancelled" else "failed"
            if self._finish(db, execution, state, None, _stop_error(meter.stop_reason, execution.limit)):
                print(f"wse worker: hard stop of {execution.job_id} ({meter.stop_reason})", file=sys.stderr,
                      flush=True)
                self.hard_exit(EXIT_HARD_STOP)
                return False
        return True


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


def spawn_worker(workspace: str | Path, *args: str, env: Mapping[str, str] | None = None,
                 stderr: Any = None) -> subprocess.Popen:
    """Start ``python -m wse.worker`` for ``workspace`` with extra command-line ``args``."""
    return subprocess.Popen(
        [sys.executable, "-m", "wse.worker", "--workspace", str(workspace), *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
        env=dict(env) if env is not None else None,
    )


class WorkerSupervisor:
    """Keeps exactly one worker process running for a workspace (the job-scheduler owner).

    A worker that exits (a hard stop or a crash) is replaced after
    ``restart_delay_seconds``. ``stop`` terminates the current worker, which
    finishes its current job first, and kills it after ``stop_timeout``.
    Each worker is told this process's ID and stops by itself if this process
    dies without calling ``stop``.
    """

    def __init__(self, workspace: str | Path, *, extra_args: Sequence[str] = (), env: Mapping[str, str] | None = None,
                 restart_delay_seconds: float = 0.5, poll_seconds: float = 0.1) -> None:
        self.workspace = Path(workspace)
        self.extra_args = tuple(extra_args)
        self.env = env
        self.restart_delay_seconds = restart_delay_seconds
        self.poll_seconds = poll_seconds
        self.process: subprocess.Popen | None = None
        self.restarts = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _spawn(self) -> subprocess.Popen:
        return spawn_worker(self.workspace, "--parent-pid", str(os.getpid()), *self.extra_args, env=self.env)

    def start(self) -> WorkerSupervisor:
        self._stop.clear()
        self.process = self._spawn()
        self._thread = threading.Thread(target=self._watch, daemon=True, name="wse-worker-supervisor")
        self._thread.start()
        return self

    def _watch(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            if self.process is not None and self.process.poll() is not None:
                if self._stop.wait(self.restart_delay_seconds):
                    return
                self.restarts += 1
                self.process = self._spawn()

    def stop(self, stop_timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=stop_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def __enter__(self) -> WorkerSupervisor:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _load_setup(spec: str) -> Callable[[Any], None]:
    module_name, _, function = spec.partition(":")
    if not module_name or not function:
        raise SystemExit(f"--setup must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module_name), function)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wse worker", description="Run the WSE evaluation worker")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--setup", action="append", default=[],
                        help="operator setup hook 'module:function' called with the registry (repeatable)")
    parser.add_argument("--lease-seconds", type=float, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--heartbeat-seconds", type=float, default=None)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--cancel-grace-seconds", type=float, default=DEFAULT_CANCEL_GRACE_SECONDS)
    parser.add_argument("--timeout-grace-seconds", type=float, default=DEFAULT_TIMEOUT_GRACE_SECONDS)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--idle-exit-seconds", type=float, default=None)
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="stop (after the current job) once this process is no longer the worker's parent")
    args = parser.parse_args(argv)

    from wse.registry import builtin_registry

    registry = builtin_registry()
    for spec in args.setup:
        _load_setup(spec)(registry)
    worker = Worker(args.workspace, registry=registry, lease_seconds=args.lease_seconds,
                    heartbeat_seconds=args.heartbeat_seconds, poll_seconds=args.poll_seconds,
                    cancel_grace_seconds=args.cancel_grace_seconds,
                    timeout_grace_seconds=args.timeout_grace_seconds, parent_pid=args.parent_pid)
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, lambda *_: worker.request_stop())
    try:
        worker.run(max_jobs=args.max_jobs, idle_exit_seconds=args.idle_exit_seconds)
    finally:
        worker.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
