"""Whole development evaluations of checkpoints, waited for by WSE, and baselines reused across experiments.

* **One evaluation.** ``evaluate_checkpoint`` (``Engine.evaluate_checkpoint``,
  the MCP tool ``wse_evaluate``) evaluates one checkpoint of an experiment on
  its task's development suite with the suite's full preregistered
  configuration (``development_target``): the task's
  ``development_examples`` as the sample count (the suite's largest sample
  count if the task's is not preregistered) and the suite's first seed. The
  controller names only the checkpoint; prompts, sampling, seeds, and
  grading are the suite's. The evaluation is a job that is checked,
  reserved, and settled exactly as one ``wse_submit_evaluation`` queues, so
  one evaluation is one generation pass of the whole suite (or its
  evaluation batches), and the policy's evaluation budget counts every one,
  the base checkpoint's included.
* **Waiting.** The call then waits until the job is terminal, at most
  ``evaluation_wait_cap`` (the policy's ``seconds_per_operation`` and the
  time a job stopped at that limit takes to settle), so a controller
  receives one result instead of polling ``wse_get_job``. The MCP server
  waits outside its lock, as ``wse_get_job`` does (``poll_job``).
* **Once per checkpoint.** A request for a checkpoint that already has a
  report on the suite at that configuration returns it
  (``existing_report``), however it was produced; a request for a checkpoint
  whose evaluation is queued or running waits for that job
  (``existing_job``). Neither creates a job or charges anything. Repeating a
  request with the same idempotency key returns its original outcome, a
  failed or cancelled job included; a new key after a failed or cancelled
  evaluation starts a new one, charged again.
* **Interrupted waits.** A wait that ends early, because the client's
  timeout cancelled the request, the server stopped, or ``stop()`` returned
  true, leaves the job to the worker, which finishes or stops it as any job;
  a later request for the checkpoint, or ``wse_get_job``, returns its result.
  The runner's finalization cancels a job that is still running.
* **Comparison with the base.** The report of a candidate is paired with the
  base checkpoint's first report on the same suite and configuration
  (``EvaluationService.compare``, which pairs generation passes with
  ``compare_generation``); ``comparison_unavailable`` says why there is none.
* **Baselines** (``ensure_baseline``, operator only). Before a controller
  starts, the execution layer evaluates the experiment's base checkpoint
  once. When another experiment of the workspace holds a report that a job
  produced of the same model (the base's whole identity: adapter,
  configuration, weights, and tokenizer files) on the same registered suite,
  at the same sample count and seed, with the batch size and device of this
  experiment's evaluations and without errors, that report is linked into
  this experiment instead: a new report ID, ``reused_from`` naming the
  report it repeats, and its generation outputs readable, and probe-able, by
  this run. Reuse is limited to the same registered suite (suites are
  immutable, so the same suite hash is the same content), because only such
  reports pair with the run's other reports in comparisons. The link is
  charged as the evaluation it stands for (the same passes or batches,
  examples, forward calls, and generated tokens), without compute seconds,
  and carries that charge as its usage. Otherwise the base is evaluated as
  above. The call returns once the experiment has the succeeded report and
  raises otherwise; repeating it returns the same report without charging
  again.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from wse.audit import record_event
from wse.database import utc_now
from wse.evaluation import _new_report_id
from wse.experiments import _check_key, _hashable, _new_id
from wse.jobs import JOB_COUNTERS, _wait_seconds, evaluation_wait_cap, job_counters, unused_counters
from wse.schemas import (
    EVALUATION_DEVICE,
    CheckpointEvaluation,
    ErrorCode,
    EvalConfig,
    EvaluationReport,
    EvaluationSuite,
    Job,
    ModelIdentity,
    ReportSource,
    ReportSummary,
    Usage,
    WSEError,
    canonical_json,
    json_hash,
)

if TYPE_CHECKING:
    import sqlite3

    from wse.api import Engine

OPERATION = "evaluate_checkpoint"
# The idempotency key of an experiment's baseline evaluation (``ensure_baseline``).
BASELINE_KEY = "wse:baseline"
FIRST_POLL_SECONDS = 0.05
MAX_POLL_SECONDS = 1.0
ACTIVE_STATES = ("queued", "running")


def development_target(engine: Engine, experiment_id: str) -> tuple[EvaluationSuite, EvalConfig]:
    """The experiment's development suite and the configuration of a whole evaluation on it.

    The suite is the task's ``development_suite_id``; the sample count is the
    task's ``development_examples`` when the suite preregisters it, else the
    suite's largest sample count; the seed is the suite's first.
    """
    row = engine.experiments._experiment_row(experiment_id)
    task = engine.evaluation._task_definition(row["task_id"])
    if task is None:
        raise WSEError(ErrorCode.NOT_FOUND, f"task {row['task_id']} is not registered, so it names no development "
                       "suite to evaluate on", details={"task_id": row["task_id"]})
    suite = engine.jobs._development_suite(task.development_suite_id, row["task_id"])
    count = task.development_examples if task.development_examples in suite.sample_counts else max(suite.sample_counts)
    return suite, EvalConfig(sample_count=count, seed=suite.seeds[0])


def poll_job(read: Callable[[], Job], seconds: float, stop: Callable[[], bool] | None = None) -> Job:
    """Read a job until it is terminal, ``seconds`` have passed, or ``stop()`` is true; return the last read.

    Reads start 50 ms apart and slow down to one a second, so a 37-minute
    generation pass costs about 2,200 reads of the job's row.
    """
    deadline = time.monotonic() + seconds
    pause = FIRST_POLL_SECONDS
    while True:
        job = read()
        left = deadline - time.monotonic()
        if job.terminal or left <= 0 or (stop is not None and stop()):
            return job
        time.sleep(min(pause, left))
        pause = min(2 * pause, MAX_POLL_SECONDS)


def _request_hash(checkpoint_id: Any) -> str:
    return json_hash({"operation": OPERATION, "checkpoint_id": _hashable(checkpoint_id)})


def _report_of(conn: sqlite3.Connection, experiment_id: str, checkpoint_id: str, suite: EvaluationSuite,
               config: EvalConfig) -> sqlite3.Row | None:
    """The experiment's first report of ``checkpoint_id`` on ``suite`` at ``config``, or ``None``."""
    return conn.execute(
        "SELECT report_id, job_id FROM reports WHERE experiment_id = ? AND checkpoint_id = ? AND suite_id = ? "
        "AND split = 'development' AND json_extract(report_json, '$.sample_count') = ? "
        "AND json_extract(report_json, '$.seed') = ? ORDER BY created_at, rowid LIMIT 1",
        (experiment_id, checkpoint_id, suite.suite_id, config.sample_count, config.seed)).fetchone()


def _existing(conn: sqlite3.Connection, experiment_id: str, checkpoint_id: str, suite: EvaluationSuite,
              config: EvalConfig) -> dict[str, Any] | None:
    """An earlier evaluation of the checkpoint: its report, else its queued or running job not asked to stop."""
    report = _report_of(conn, experiment_id, checkpoint_id, suite, config)
    if report is not None:
        return {"origin": "existing_report", "job_id": report["job_id"], "report_id": report["report_id"]}
    job = conn.execute(
        "SELECT job_id FROM jobs WHERE experiment_id = ? AND checkpoint_id = ? AND suite_id = ? "
        "AND kind = 'evaluation' AND state IN ('queued', 'running') AND cancel_requested_at IS NULL "
        "AND config_json = ? ORDER BY created_at, rowid LIMIT 1",
        (experiment_id, checkpoint_id, suite.suite_id, config.canonical_json())).fetchone()
    if job is not None:
        return {"origin": "existing_job", "job_id": job["job_id"], "report_id": None}
    return None


def _start(engine: Engine, experiment_id: str, checkpoint_id: Any, suite: EvaluationSuite, config: EvalConfig,
           key: str, request_hash: str) -> dict[str, Any]:
    """Queue the evaluation, or find the checkpoint's earlier one, and record the outcome under ``key``."""
    service, jobs = engine.experiments, engine.jobs
    # Every check of a submission runs first (state, mode, ownership, suite, configuration, policy limits); none
    # reserves anything, so a checkpoint with a report is answered even when no evaluation is left.
    request = jobs.evaluation_request(experiment_id, checkpoint_id, suite.suite_id, config.model_dump())
    with engine.store.db.transaction() as conn:
        again = service._lookup(conn, experiment_id, key, OPERATION, request_hash)
        if again is not None:
            return json.loads(again)
        service._require_open(service._experiment_row(experiment_id, conn))
        started = _existing(conn, experiment_id, checkpoint_id, suite, config)
        if started is None:
            job_id = _new_id("job")
            jobs.queue_evaluation(conn, experiment_id, job_id, request)
            started = {"origin": "new_job", "job_id": job_id, "report_id": None}
        service._remember(conn, experiment_id, key, OPERATION, request_hash, canonical_json(started))
    return started


def _begin(engine: Engine, experiment_id: str, checkpoint_id: Any, key: Any
           ) -> tuple[EvaluationSuite, EvalConfig, dict[str, Any]]:
    """The development target and the request's outcome: recorded under ``key``, or started now."""
    service = engine.experiments
    _check_key(key)
    service._experiment_row(experiment_id)
    suite, config = development_target(engine, experiment_id)
    request_hash = _request_hash(checkpoint_id)
    recorded = service._lookup(engine.store.db.conn, experiment_id, key, OPERATION, request_hash)
    if recorded is not None:
        return suite, config, json.loads(recorded)
    return suite, config, service._guarded(
        "evaluation.request", experiment_id, request_hash, str(checkpoint_id),
        lambda: _start(engine, experiment_id, checkpoint_id, suite, config, key, request_hash))


def _comparison_fields(engine: Engine, experiment_id: str, report: EvaluationReport, suite: EvaluationSuite,
                       config: EvalConfig) -> dict[str, Any]:
    """The paired comparison of ``report`` with the base checkpoint's report, or why there is none."""
    base = engine.experiments._experiment_row(experiment_id)["base_checkpoint_id"]
    if report.checkpoint_id == base:
        return {"comparison_unavailable": "this is the base checkpoint"}
    row = _report_of(engine.store.db.conn, experiment_id, base, suite, config)
    if row is None:
        return {"comparison_unavailable": f"the base checkpoint {base} has no report on suite {suite.suite_id} with "
                                          f"sample_count {config.sample_count} and seed {config.seed}; evaluate it "
                                          "to compare"}
    try:
        comparison = engine.evaluation.compare(experiment_id, row["report_id"], report.report_id)
    except WSEError as exc:
        if exc.code != ErrorCode.INCOMPATIBLE_REPORTS:
            raise
        return {"base_report_id": row["report_id"], "comparison_unavailable": exc.message}
    return {"base_report_id": row["report_id"], "comparison": comparison}


def _result(engine: Engine, experiment_id: str, checkpoint_id: str, suite: EvaluationSuite, config: EvalConfig,
            started: dict[str, Any]) -> CheckpointEvaluation:
    """The evaluation as it stands: the job's state, and the report and comparison once it succeeded."""
    job = engine.jobs.get_job(experiment_id, started["job_id"]) if started["job_id"] is not None else None
    report_id = job.report_id if job is not None else started["report_id"]
    report = engine.evaluation.get_report(experiment_id, report_id) if report_id is not None else None
    fields = _comparison_fields(engine, experiment_id, report, suite, config) if report is not None else {}
    return CheckpointEvaluation(
        experiment_id=experiment_id, checkpoint_id=checkpoint_id, suite_id=suite.suite_id, config=config,
        origin=started["origin"], state=job.state if job is not None else "succeeded", job_id=started["job_id"],
        report=ReportSummary.of(report) if report is not None else None,
        usage=job.usage if job is not None else report.usage, error=job.error if job is not None else None,
        **fields)


def evaluate_checkpoint(engine: Engine, experiment_id: str, checkpoint_id: str, key: str, *,
                        wait_seconds: float | None = None,
                        stop: Callable[[], bool] | None = None) -> CheckpointEvaluation:
    """Evaluate ``checkpoint_id`` on the whole development suite and wait for the result (module docstring).

    ``wait_seconds`` (default and at most ``evaluation_wait_cap`` of the
    policy; 0 does not wait) bounds the wait, and ``stop()``, when given,
    ends it early; either leaves the job running for a later request.
    """
    suite, config, started = _begin(engine, experiment_id, checkpoint_id, key)
    cap = evaluation_wait_cap(engine.experiments.experiment_policy(experiment_id))
    wait = cap if wait_seconds is None else _wait_seconds(wait_seconds, cap)
    if started["job_id"] is not None and wait > 0:
        poll_job(lambda: engine.jobs.get_job(experiment_id, started["job_id"]), wait, stop)
    return _result(engine, experiment_id, checkpoint_id, suite, config, started)


# -- baselines (operator) -----------------------------------------------------------------------------------------


def _reusable_report(conn: sqlite3.Connection, experiment_id: str, identity: ModelIdentity, suite: EvaluationSuite,
                     config: EvalConfig, batch_size: int, device: str) -> EvaluationReport | None:
    """The first report of another experiment that this experiment may link as its baseline, or ``None``."""
    rows = conn.execute(
        "SELECT report_hash, report_json FROM reports WHERE experiment_id != ? AND split = 'development' "
        "AND suite_id = ? AND job_id IS NOT NULL AND json_extract(report_json, '$.sample_count') = ? "
        "AND json_extract(report_json, '$.seed') = ? AND json_extract(report_json, '$.model.weights_hash') = ? "
        "ORDER BY created_at, rowid",
        (experiment_id, suite.suite_id, config.sample_count, config.seed, identity.weights_hash)).fetchall()
    for row in rows:
        report = EvaluationReport.model_validate_json(row["report_json"])
        if json_hash(report.model_dump(mode="json")) != row["report_hash"]:
            continue  # a report that does not match its hash is never repeated
        if (report.model == identity and report.suite_hash == suite.suite_hash and report.data_hash == suite.data_hash
                and report.evaluator == suite.evaluator and report.execution.batch_size == batch_size
                and report.execution.device == device and not report.errors and report.reused_from is None):
            return report
    return None


def _link_baseline(engine: Engine, experiment_id: str, base: str, suite: EvaluationSuite, config: EvalConfig,
                   request_hash: str, device: str | None) -> dict[str, Any] | None:
    """Link another experiment's identical report of the base as this experiment's baseline, or ``None``.

    ``None`` leaves the baseline to ``evaluate_checkpoint``: when no report
    may be reused, or this experiment already holds the base's report or job.
    """
    from wse.evaluators.generation import DEFAULT_BATCH_SIZE

    service, jobs = engine.experiments, engine.jobs
    jobs.evaluation_request(experiment_id, base, suite.suite_id, config.model_dump())  # every submission check
    if jobs.generates_text(suite):
        from wse.engines.hf_torch import select_device

        batch_size, shown_device = DEFAULT_BATCH_SIZE, select_device(device).type
    else:
        batch_size = service.experiment_policy(experiment_id).budget.examples_per_batch
        shown_device = EVALUATION_DEVICE
    identity = engine.store.load_manifest(base).identity
    with engine.store.db.transaction() as conn:
        again = service._lookup(conn, experiment_id, BASELINE_KEY, OPERATION, request_hash)
        if again is not None:
            return json.loads(again)
        service._require_open(service._experiment_row(experiment_id, conn))
        if _existing(conn, experiment_id, base, suite, config) is not None:
            return None
        source = _reusable_report(conn, experiment_id, identity, suite, config, batch_size, shown_device)
        if source is None:
            return None
        charge = {counter: amount for counter in JOB_COUNTERS if (amount := getattr(source.usage, counter) or 0)}
        for counter, amount in charge.items():
            service.ledger.charge(conn, experiment_id, counter, amount)
        origin = ReportSource(experiment_id=source.experiment_id, report_id=source.report_id)
        usage = Usage(**{**job_counters(), **charge}, compute_seconds=0.0, **unused_counters(), artifact_bytes=0,
                      peak_memory_bytes=None,
                      unavailable={"peak_memory_bytes": f"reused report {source.report_id} of experiment "
                                                        f"{source.experiment_id}; nothing was computed here"})
        linked = source.model_copy(update={"report_id": _new_report_id(), "experiment_id": experiment_id,
                                           "job_id": None, "checkpoint_id": base, "usage": usage,
                                           "created_at": utc_now(), "reused_from": origin})
        engine.evaluation.insert_report(conn, linked)
        outputs = [sample.output_uri for problem in linked.problems or [] for sample in problem.samples
                   if sample.output_uri]
        for uri in dict.fromkeys(outputs):  # this run reads and probes the outputs as its own pass's
            conn.execute("INSERT OR IGNORE INTO analysis_results (uri, experiment_id, created_at) VALUES (?, ?, ?)",
                         (uri, experiment_id, linked.created_at))
        record_event(conn, "report.reused", linked.report_id,
                     {"checkpoint_id": base, "suite_id": suite.suite_id, "reused_from": origin.model_dump(),
                      "charged": charge, "outputs": len(outputs)}, experiment_id=experiment_id)
        started = {"origin": "reused_report", "job_id": None, "report_id": linked.report_id}
        service._remember(conn, experiment_id, BASELINE_KEY, OPERATION, request_hash, canonical_json(started))
    return started


def ensure_baseline(engine: Engine, experiment_id: str, *, device: str | None = None,
                    wait_seconds: float | None = None,
                    stop: Callable[[], bool] | None = None) -> CheckpointEvaluation:
    """Give the experiment a succeeded whole development report of its base checkpoint (operator only).

    Links another experiment's identical report when the workspace has one,
    else evaluates the base (module docstring), and returns once the report
    exists. ``device`` is the engine device of the run's worker for
    generation suites (default: the worker's default, ``mps`` when available,
    else ``cpu``); a generation report of another device is not reused, since
    it would not pair with this run's reports. ``wait_seconds`` (default:
    until the job ends) and ``stop()`` bound the wait. A failed, cancelled,
    or still unfinished evaluation, and a report with errors, raise
    ``WORKER_FAILED``.
    """
    engine._require_operator("evaluating an experiment's baseline")
    service = engine.experiments
    base = service._experiment_row(experiment_id)["base_checkpoint_id"]
    suite, config = development_target(engine, experiment_id)
    wait = math.inf if wait_seconds is None else _wait_seconds(wait_seconds, math.inf)
    request_hash = _request_hash(base)
    if service._lookup(engine.store.db.conn, experiment_id, BASELINE_KEY, OPERATION, request_hash) is None:
        service._guarded("evaluation.baseline", experiment_id, request_hash, base,
                         lambda: _link_baseline(engine, experiment_id, base, suite, config, request_hash, device))
    evaluation = evaluate_checkpoint(engine, experiment_id, base, BASELINE_KEY, wait_seconds=0)
    if evaluation.job_id is not None and evaluation.state in ACTIVE_STATES and wait > 0:
        poll_job(lambda: engine.jobs.get_job(experiment_id, evaluation.job_id), wait, stop)
        evaluation = evaluate_checkpoint(engine, experiment_id, base, BASELINE_KEY, wait_seconds=0)
    if evaluation.state != "succeeded" or evaluation.report is None:
        ended = "is still " if evaluation.state in ACTIVE_STATES else "ended "
        reason = f": {evaluation.error.message}" if evaluation.error is not None else ""
        raise WSEError(ErrorCode.WORKER_FAILED, f"the baseline evaluation of {base} {ended}{evaluation.state}{reason}",
                       retryable=evaluation.state in ACTIVE_STATES,
                       details={"job_id": evaluation.job_id, "state": evaluation.state})
    if evaluation.report.errors:
        raise WSEError(ErrorCode.WORKER_FAILED, f"the baseline report {evaluation.report.report_id} has errors: "
                       f"{'; '.join(evaluation.report.errors)[:500]}",
                       details={"report_id": evaluation.report.report_id})
    return evaluation


__all__ = ["BASELINE_KEY", "OPERATION", "development_target", "ensure_baseline", "evaluate_checkpoint", "poll_job"]
