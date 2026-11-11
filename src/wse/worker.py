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
* **Generation passes** (M2 plan Section 5.4). A job whose reservation holds
  ``generation_passes`` evaluates a generation suite: ``run_generation`` on
  the cached model of the checkpoint, at batch size 8, on the worker's device
  (``mps`` when available, else ``cpu``, or ``--device``). The
  ``GenerationMeter`` checks for cancellation and the time limit at every
  decoding step and counts the tokens generated: exactly for finished
  batches, and rows x steps for the batch in progress. Settlement charges
  one pass (once a batch started), those tokens, and the compute seconds,
  and releases the rest of the reservation. Each output is stored as a
  run-scoped ``GenerationOutput`` artifact (``wse://analyses/<sha256>``),
  which the run can read once the job has succeeded (ownership is recorded
  in the settling transaction); the report carries every sample's grade,
  length, limit hit, a tail excerpt of its text, and its artifact's URI,
  within the policy's inline-result limit (``report_with_tails``).
* **Model cache.** Generation passes load their checkpoint through
  ``ModelCache``, keyed by checkpoint ID, so consecutive jobs on a checkpoint
  load it once; see ``ModelCache`` for its bound and memory. M1 jobs load
  their (small) models per job, as before.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
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
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from wse.api import evaluate_candidate
from wse.audit import record_event
from wse.database import Database, utc_now
from wse.jobs import queued_cancel_usage, unused_counters
from wse.schemas import ErrorCode, EvalConfig, GenerationOutput, JobProgress, Usage, WSEError

if TYPE_CHECKING:
    from wse.api import EvaluationResult, ModelHandle
    from wse.evaluators.generation import SampleOutput, TextModel

EXIT_HARD_STOP = 70
DEFAULT_LEASE_SECONDS = 10.0
DEFAULT_POLL_SECONDS = 0.1
DEFAULT_CANCEL_GRACE_SECONDS = 2.0
DEFAULT_TIMEOUT_GRACE_SECONDS = 1.0
DEFAULT_MODEL_CACHE_SIZE = 2


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

    def counts(self) -> dict[str, int]:
        """The counters this job charges: the forward calls and examples it started."""
        return {"evaluation_batches": self.started_calls, "evaluated_examples": self.started_examples,
                "model_forward_calls": self.started_calls, "generation_passes": 0, "generated_tokens": 0}


class GenerationMeter:
    """The meter of a generation pass: its batches, its generated tokens, and its stops.

    The worker passes ``check``, ``step``, and ``batch_finished`` to
    ``run_generation``. ``step`` runs at every decoding step and checks for
    cancellation and the time limit, so a pass stops at its next step. A
    finished batch counts its outputs' tokens exactly (stop tokens
    included); the batch in progress counts rows x steps so far, an upper
    bound, because a row that stopped early still counts every step. The
    count never exceeds the reserved tokens, which cover every output at its
    token limit. The pass itself counts once its first batch started.
    The monitor thread reads the counts while the job runs, so they change
    under a lock.
    """

    def __init__(self, *, permitted: bool, max_tokens: int, deadline: float) -> None:
        self.permitted = permitted
        self.max_tokens = max_tokens
        self.deadline = deadline
        self.stop_reason: str | None = None
        self._lock = threading.Lock()
        self.batches_started = 0
        self.batches_completed = 0
        self.rows_started = 0
        self.completed_tokens = 0
        self._rows = 0  # rows of the batch in progress
        self._steps = 0  # its decoding steps so far

    def check(self) -> None:
        if self.stop_reason is None and time.monotonic() > self.deadline:
            self.stop_reason = "timeout"
        if self.stop_reason is not None:
            raise JobStopped(self.stop_reason)

    def step(self, rows: int, step: int) -> None:
        """``StepHook``: step 0 starts a batch of ``rows`` rows; step ``k`` is its ``k``-th decoding step."""
        with self._lock:
            if step == 0:
                self.batches_started += 1
                self.rows_started += rows
                self._rows, self._steps = rows, 0
            else:
                self._steps = step
        self.check()

    def batch_finished(self, outputs: Sequence[SampleOutput], seconds: float) -> None:
        with self._lock:
            self.completed_tokens += sum(len(output.tokens) for output in outputs)
            self.batches_completed += 1
            self._rows = self._steps = 0

    def progress(self, batches_total: int) -> JobProgress:
        with self._lock:
            return JobProgress(batches_total=batches_total, batches_started=self.batches_started,
                               batches_completed=self.batches_completed, examples_started=self.rows_started)

    def counts(self) -> dict[str, int]:
        """The counters this pass charges: the pass, once started, and the tokens generated so far."""
        with self._lock:
            tokens = min(self.max_tokens, self.completed_tokens + self._rows * self._steps)
            passes = 1 if self.batches_started else 0
        return {"evaluation_batches": 0, "evaluated_examples": 0, "model_forward_calls": 0,
                "generation_passes": passes, "generated_tokens": tokens}


@dataclass(frozen=True)
class LoadedModel:
    """A checkpoint loaded for generation: its hash-verified handle and its text model on the worker's device.

    The handle keeps the engine runtime it built (``ModelHandle.runtime``),
    so another job on the same device, a probe for example, reuses the module.
    """

    checkpoint_id: str
    handle: ModelHandle
    text_model: TextModel


def _release_device_memory(entry: LoadedModel) -> None:
    gc.collect()
    device = getattr(getattr(entry.text_model.runtime, "device", None), "type", None)
    if device == "mps":
        torch.mps.empty_cache()


class ModelCache:
    """Models the worker keeps loaded across jobs, keyed by checkpoint ID; the least recently used leaves first.

    **Memory.** An entry holds its checkpoint's tensors in CPU memory, their
    stored bytes (about 1.5 GB for Qwen3.5-0.8B's BF16 text model), and, on
    MPS, one device copy of every distinct tensor (about as much again), plus
    the tokenizer. At most ``capacity`` entries are held: the entry to evict
    is dropped before a new model loads, so a load never holds one model more.
    The default of 2 keeps a base and a candidate, which an experiment's loop
    alternates: about 6 GB for Qwen3.5-0.8B on MPS. ``capacity`` 0 disables
    the cache: each job loads its model and releases it when it ends.

    Checkpoints are immutable and verified against their hashes when loaded,
    so an entry never goes stale; the worker still refuses a checkpoint marked
    CORRUPT since it was loaded (``Worker.loaded_model``).
    """

    def __init__(self, load: Callable[[str], LoadedModel], capacity: int = DEFAULT_MODEL_CACHE_SIZE) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "the model cache size must be an integer >= 0")
        self._load = load
        self.capacity = capacity
        self._entries: OrderedDict[str, LoadedModel] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, checkpoint_id: object) -> bool:
        return checkpoint_id in self._entries

    def get(self, checkpoint_id: str) -> LoadedModel:
        """The cached model of ``checkpoint_id``, loading it (after evicting if full) when absent."""
        entry = self._entries.get(checkpoint_id)
        if entry is not None:
            self._entries.move_to_end(checkpoint_id)
            return entry
        while self._entries and len(self._entries) >= self.capacity:
            _, evicted = self._entries.popitem(last=False)
            _release_device_memory(evicted)
        loaded = self._load(checkpoint_id)
        if self.capacity > 0:
            self._entries[checkpoint_id] = loaded
        return loaded

    def discard(self, checkpoint_id: str) -> None:
        evicted = self._entries.pop(checkpoint_id, None)
        if evicted is not None:
            _release_device_memory(evicted)

    def clear(self) -> None:
        while self._entries:
            _, evicted = self._entries.popitem(last=False)
            _release_device_memory(evicted)


@dataclass(frozen=True)
class GenerationPass:
    """A finished generation pass: its result (output URIs attached) and each output's text by (problem, sample)."""

    result: EvaluationResult
    texts: Mapping[tuple[int, int], str]


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
    """State shared by the main thread and the monitor of one running job.

    ``output_uris`` and ``artifact_bytes`` are a generation pass's stored
    outputs, and ``report_limit`` the policy's inline-result limit for its
    report.
    """

    def __init__(self, job_id: str, meter: ForwardMeter | GenerationMeter, batches_total: int, limit: float) -> None:
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
        self.output_uris: list[str] = []
        self.artifact_bytes = 0
        self.report_limit: int | None = None

    def usage(self) -> Usage:
        """The usage to settle: the meter's counters, compute seconds capped at the reservation, and 0 elsewhere."""
        elapsed = time.monotonic() - self.started
        return Usage(
            **self.meter.counts(),
            compute_seconds=round(min(elapsed, self.limit), 6),
            **unused_counters(),
            artifact_bytes=self.artifact_bytes,
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
        device: str | None = None,
        model_cache_size: int = DEFAULT_MODEL_CACHE_SIZE,
    ) -> None:
        from wse.api import Engine

        if device is not None:
            from wse.engines.hf_torch import select_device

            device = select_device(device).type  # an unknown or unavailable device fails here, before any job
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
        self.device = device  # None: mps when available, else cpu
        self.models = ModelCache(self._load_model, model_cache_size)
        self._stop = threading.Event()

    def close(self) -> None:
        self.models.clear()
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
        # A generation pass is the job whose reservation holds a pass (jobs.JobService._generation_request).
        generation = "generation_passes" in requested
        meter: ForwardMeter | GenerationMeter
        if generation:
            meter = GenerationMeter(
                permitted=experiment["mode"] == "bounded_eval" and requested.get("generated_tokens", 0) > 0,
                max_tokens=int(requested.get("generated_tokens", 0)), deadline=time.monotonic() + limit)
        else:
            permitted = experiment["mode"] == "bounded_eval" and requested.get("model_forward_calls", 0) > 0
            meter = ForwardMeter(permitted=permitted, batch_size=1,
                                 max_calls=int(requested.get("model_forward_calls", 0)),
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
            execution.suite = self.jobs.load_suite(row["suite_id"])
            config = EvalConfig.model_validate_json(row["config_json"])
            if generation != self.jobs.generates_text(execution.suite):
                raise WSEError(ErrorCode.WORKER_FAILED,
                               f"the job's reservation does not match the evaluator of suite {row['suite_id']}, so it "
                               "was not run", details={"reason": "reservation_mismatch"})
            if isinstance(meter, GenerationMeter):
                execution.report_limit = policy.operations.limits.max_result_json_bytes
                result = self.generate(execution, config, meter, policy)
            else:
                meter.batch_size = policy.budget.examples_per_batch
                result = self.evaluate(row["checkpoint_id"], row["suite_id"], config, meter)
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
                return handle.forward(inputs)

        meter.bind(run_forward)
        return evaluate_candidate(handle, suite, config, registry=self.engine.registry, forward=meter,
                                  batch_size=meter.batch_size)

    # -- generation passes --------------------------------------------------------

    def _load_model(self, checkpoint_id: str) -> LoadedModel:
        from wse.evaluators import generation

        handle = self.engine.open_checkpoint(checkpoint_id)  # verified against the checkpoint's hashes
        text_model = generation.load_text_model(handle, self.engine.model_files(checkpoint_id), device=self.device)
        return LoadedModel(checkpoint_id=checkpoint_id, handle=handle, text_model=text_model)

    def loaded_model(self, checkpoint_id: str) -> LoadedModel:
        """The checkpoint's model, from the cache or loaded into it (worker-internal; jobs that run the engine).

        A checkpoint that is no longer READY (marked CORRUPT since it was
        loaded) is dropped from the cache and refused.
        """
        state = self.engine.store.checkpoint_state(checkpoint_id)
        if state != "READY":
            self.models.discard(checkpoint_id)
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"checkpoint {checkpoint_id} is {state} and cannot be used",
                           details={"checkpoint_id": checkpoint_id, "state": state})
        return self.models.get(checkpoint_id)

    def generate(self, execution: _Execution, config: EvalConfig, meter: GenerationMeter, policy: Any
                 ) -> GenerationPass:
        """Run the generation pass of a job through ``meter`` and store its outputs (worker-internal only).

        The suite's problems are read from evaluator storage, the model comes
        from the cache, and ``run_generation`` runs with the meter's hooks at
        the generation batch size. The pass is checked against the policy and
        the reservation again before it starts. Every output is then stored as
        a ``GenerationOutput`` artifact; the run may read them once the job
        has succeeded.
        """
        from wse.evaluators.generation import DEFAULT_BATCH_SIZE, plan_generation, run_generation

        if not meter.permitted:
            raise WSEError(ErrorCode.POLICY_DENIED,
                           "this experiment permits no generation; the worker never runs the target network")
        suite = execution.suite
        if suite.data is None:
            suite = suite.resolved(self.engine.datasets.load(suite.dataset))
        loaded = self.loaded_model(execution.checkpoint_id)
        plan = plan_generation(loaded.handle, suite, config, registry=self.engine.registry)
        budget, tokens = policy.budget, len(plan.rows) * plan.profile.max_new_tokens
        if len(plan.items) > budget.limit("problems_per_generation_pass") \
                or plan.profile.max_new_tokens > budget.limit("new_tokens_per_output"):
            raise WSEError(ErrorCode.POLICY_DENIED, "the generation pass exceeds the policy's per-pass limits",
                           details={"problems": len(plan.items), "max_new_tokens": plan.profile.max_new_tokens})
        if tokens > meter.max_tokens:
            raise WSEError(ErrorCode.BUDGET_EXCEEDED,
                           f"the generation pass may generate {tokens} tokens, more than the {meter.max_tokens} "
                           "reserved for this job", details={"reserved_generated_tokens": meter.max_tokens,
                                                             "generated_tokens": tokens})
        run = run_generation(loaded.handle, suite, config, text_model=loaded.text_model,
                             registry=self.engine.registry, batch_size=DEFAULT_BATCH_SIZE, check=meter.check,
                             on_batch=meter.batch_finished, on_step=meter.step)
        graded = {(problem.index, sample.sample): sample for problem in run.result.problems
                  for sample in problem.samples}
        uris: dict[tuple[int, int], str] = {}
        for output in run.outputs:
            key = (output.index, output.sample)
            sample = graded[key]
            record = GenerationOutput(
                experiment_id=execution.experiment_id, job_id=execution.job_id,
                checkpoint_id=execution.checkpoint_id, suite_id=suite.suite_id, index=output.index,
                item_id=output.item_id, sample=output.sample, seed=output.seed, prompt_sha256=output.prompt_sha256,
                output_sha256=sample.output_sha256, generated_tokens=sample.generated_tokens,
                finished=sample.finished, limit_hit=sample.limit_hit, tokens=list(output.tokens), text=output.text)
            data = record.canonical_json().encode("utf-8")
            uris[key] = self.engine.store.put_analysis_result(data)
            execution.artifact_bytes += len(data)
        execution.output_uris = list(uris.values())
        problems = tuple(problem.model_copy(update={"samples": [
            sample.model_copy(update={"output_uri": uris[(problem.index, sample.sample)]})
            for sample in problem.samples]}) for problem in run.result.problems)
        return GenerationPass(result=dataclasses.replace(run.result, problems=problems),
                              texts={(output.index, output.sample): output.text for output in run.outputs})

    def _finish(self, db: Database, execution: _Execution, state: str, result: Any,
                error: WSEError | None) -> bool:
        from wse.evaluators.generation import report_with_tails

        usage = execution.usage()
        report = None
        if result is not None and state == "succeeded":
            evaluated = result.result if isinstance(result, GenerationPass) else result
            report = self.engine.evaluation.build_report(
                evaluated, experiment_id=execution.experiment_id, job_id=execution.job_id,
                checkpoint_id=execution.checkpoint_id, suite=execution.suite, usage=usage, keep_scores=True)
            if isinstance(result, GenerationPass):
                report = report_with_tails(report, result.texts, limit=execution.report_limit)
        with db.transaction() as conn:
            finished = self.jobs.finish(conn, execution.job_id, state, usage,
                                        result=report.summary() if report is not None else None, error=error,
                                        progress=execution.meter.progress(execution.batches_total),
                                        lease_owner=self.worker_id, report=report)
            if finished and report is not None:
                # The run may read its pass's outputs from now on, as it reads its own analysis results.
                for uri in execution.output_uris:
                    conn.execute("INSERT OR IGNORE INTO analysis_results (uri, experiment_id, created_at) "
                                 "VALUES (?, ?, ?)", (uri, execution.experiment_id, utc_now()))
            return finished

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
    parser.add_argument("--device", choices=("mps", "cpu"), default=None,
                        help="the engine device of generation passes (default: mps when available, else cpu)")
    parser.add_argument("--model-cache-size", type=int, default=DEFAULT_MODEL_CACHE_SIZE,
                        help="models kept loaded across jobs (0 disables the cache; see wse.worker.ModelCache)")
    args = parser.parse_args(argv)

    from wse.registry import builtin_registry

    registry = builtin_registry()
    for spec in args.setup:
        _load_setup(spec)(registry)
    worker = Worker(args.workspace, registry=registry, lease_seconds=args.lease_seconds,
                    heartbeat_seconds=args.heartbeat_seconds, poll_seconds=args.poll_seconds,
                    cancel_grace_seconds=args.cancel_grace_seconds,
                    timeout_grace_seconds=args.timeout_grace_seconds, parent_pid=args.parent_pid,
                    device=args.device, model_cache_size=args.model_cache_size)
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
