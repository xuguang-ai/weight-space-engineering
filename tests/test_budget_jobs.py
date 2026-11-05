"""Budget reservations, the job queue, the worker process, cancellation, and crash settlement."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
import torch

from wse.api import Engine
from wse.budget import BudgetLedger
from wse.database import utc_now
from wse.policy import SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID, BudgetLimits, ExperimentPolicy
from wse.registry import builtin_registry
from wse.schemas import ErrorCode, EvaluationSuite, WSEError
from wse.worker import Worker, WorkerSupervisor, spawn_worker

TESTS_DIR = Path(__file__).resolve().parent
TASK = "signed_sum_v1"
DEV = "signed_sum_dev_stub"
ROW1 = [{"start": 1, "stop": 2, "step": 1}, {"start": 0, "stop": 4, "step": 1}]
ORACLE = {"op": "scale_slice", "tensor": "layers.1.weight", "selection": ROW1, "factor": -1.0}
TINY = "tiny_eval_v1"


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _suite(signed_sum, suite_id=DEV, *, split="development", evaluator="forward_batch@1", parameters=None,
           sample_counts=(4, 8, 64), task_id=TASK):
    inputs, labels, _ = signed_sum.generate_examples(32, seed=7)
    return EvaluationSuite(suite_id=suite_id, task_id=task_id, split=split, evaluator=evaluator,
                           parameters=parameters or {}, sample_counts=list(sample_counts), seeds=[0, 1],
                           data={"inputs": inputs.tolist(), "labels": labels.tolist()})


def _tiny_policy(policy_id=TINY, **budget):
    limits = {"evaluation_batches": 3, "examples_per_batch": 4, "evaluated_examples": 12,
              "model_forward_calls": 3, "seconds_per_operation": 30.0, **budget}
    return ExperimentPolicy(policy_id=policy_id, mode="bounded_eval", budget=BudgetLimits(**limits))


@pytest.fixture
def base(engine, bundles, signed_sum):
    _plugins().setup(engine.registry)  # the operator registers the evaluators its suites name
    engine.register_suite(_suite(signed_sum))
    return engine.import_bundle(bundles["task"])


def _experiment(engine, base, policy_id=SMALL_MODEL_POLICY_ID, policy=None):
    if policy is not None:
        engine.register_policy(policy)
        policy_id = policy.policy_id
    grant = engine.grant_run(TASK, policy_id, base.checkpoint_id)
    return engine.create_experiment(grant.run_id, TASK, policy_id, "create")


@pytest.fixture
def experiment(engine, base):
    return _experiment(engine, base)


def _candidate(engine, base, experiment):
    eid = experiment.experiment_id
    spec = {"base_checkpoint_id": base.checkpoint_id, "hypothesis": "One output row has the wrong sign.",
            "operations": [ORACLE]}
    return engine.apply_patch(eid, engine.preview_patch(eid, spec, "p1").patch_id, "a1")


def _budget(engine, experiment_id):
    return {name: (entry.consumed, entry.reserved) for name, entry in engine.get_experiment(experiment_id).budget.items()}


def _in_threads(workspace, count, call):
    """Run ``call(engine, index)`` concurrently, each on its own Engine and SQLite connection."""
    barrier = threading.Barrier(count)
    outcomes: list[object] = [None] * count

    def run(index):
        with Engine(workspace, recover=False) as own:
            barrier.wait()
            try:
                outcomes[index] = call(own, index)
            except WSEError as exc:
                outcomes[index] = exc.code

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def _plugins():
    sys.path.insert(0, str(TESTS_DIR))
    try:
        import wse_test_plugins
    finally:
        sys.path.remove(str(TESTS_DIR))
    return wse_test_plugins


def _worker(workspace, **kwargs):
    registry = builtin_registry()
    _plugins().setup(registry)
    return Worker(workspace, registry=registry, **kwargs)


@pytest.fixture
def processes():
    """Worker processes started by a test; all are stopped afterwards, also on failure."""
    started: list[subprocess.Popen] = []
    yield started
    for process in started:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)


def _spawn(processes, workspace, *extra, setup="wse_test_plugins:setup", env=None):
    environment = {**os.environ, "PYTHONPATH": str(TESTS_DIR), **(env or {})}
    process = spawn_worker(workspace, "--setup", setup, "--poll-seconds", "0.02", *extra, env=environment)
    processes.append(process)
    return process


def _wait_for(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition not reached before the timeout")


# -- first: concurrency ---------------------------------------------------------------------


def test_one_unit_left_reserves_once_on_independent_connections(workspace, engine, base):
    experiment = _experiment(engine, base, policy=_tiny_policy(evaluation_batches=1))
    eid = experiment.experiment_id
    outcomes = _in_threads(workspace, 4, lambda own, i: own.budget.reserve(
        eid, f"job_{i:032x}", {"evaluation_batches": 1}))
    granted = [outcome for outcome in outcomes if not isinstance(outcome, ErrorCode)]
    assert len(granted) == 1 and outcomes.count(ErrorCode.BUDGET_EXCEEDED) == 3
    assert _budget(engine, eid)["evaluation_batches"] == (0, 1)


def test_one_unit_left_reserves_once_in_independent_processes(workspace, engine, base, tmp_path):
    experiment = _experiment(engine, base, policy=_tiny_policy(evaluation_batches=1))
    go = tmp_path / "go"
    code = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        from wse.api import Engine
        from wse.schemas import WSEError
        with Engine({str(workspace)!r}, recover=False) as engine:
            while not Path({str(go)!r}).exists():
                time.sleep(0.005)
            try:
                engine.budget.reserve({experiment.experiment_id!r}, "job_" + sys.argv[1] * 32, {{"evaluation_batches": 1}})
                print("RESERVED")
            except WSEError as exc:
                print(exc.code.value)
    """)
    children = [subprocess.Popen([sys.executable, "-c", code, str(i)], stdout=subprocess.PIPE, text=True)
                for i in range(3)]
    try:
        time.sleep(1.0)  # let every child open its connection
        go.write_text("go")
        outputs = sorted(child.communicate(timeout=60)[0].strip() for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
    assert outputs == ["BUDGET_EXCEEDED", "BUDGET_EXCEEDED", "RESERVED"]
    assert _budget(engine, experiment.experiment_id)["evaluation_batches"] == (0, 1)


def test_concurrent_submissions_with_one_batch_left_create_one_job(workspace, engine, base):
    experiment = _experiment(engine, base, policy=_tiny_policy(evaluation_batches=1))
    eid = experiment.experiment_id
    outcomes = _in_threads(workspace, 3, lambda own, i: own.submit_evaluation(
        eid, base.checkpoint_id, DEV, {"sample_count": 4, "seed": 0}, f"k{i}"))  # one batch of 4
    assert outcomes.count(ErrorCode.BUDGET_EXCEEDED) == 2
    assert len(engine.list_jobs(eid)) == 1


def test_repeated_idempotent_submissions_create_one_job(workspace, engine, base, experiment):
    eid = experiment.experiment_id
    config = {"sample_count": 64, "seed": 0}
    outcomes = _in_threads(workspace, 3, lambda own, i: own.submit_evaluation(
        eid, base.checkpoint_id, DEV, config, "same-key"))
    assert len({job.job_id for job in outcomes}) == 1
    again = engine.submit_evaluation(eid, base.checkpoint_id, DEV, config, "same-key")
    assert again.job_id == outcomes[0].job_id
    assert len(engine.list_jobs(eid)) == 1
    assert _budget(engine, eid)["evaluation_batches"] == (0, 1)  # one reservation only
    _expect(ErrorCode.IDEMPOTENCY_CONFLICT, engine.submit_evaluation, eid, base.checkpoint_id, DEV,
            {"sample_count": 8, "seed": 0}, "same-key")


# -- reservations and settlement --------------------------------------------------------------


def test_reserve_and_settle_charge_work_done_and_release_the_rest(engine, base, experiment):
    eid = experiment.experiment_id
    ledger = engine.budget
    reservation = ledger.reserve(eid, "job_" + "a" * 32, {"evaluation_batches": 2, "evaluated_examples": 64,
                                                           "compute_seconds": 30})
    assert reservation.state == "HELD" and reservation.requested["evaluation_batches"] == 2
    entry = engine.get_experiment(eid).budget["evaluation_batches"]
    assert (entry.consumed, entry.reserved, entry.remaining) == (0, 2, 8)
    # Retrying the same job's reservation does not reserve again.
    assert ledger.reserve(eid, "job_" + "a" * 32, {"evaluation_batches": 2, "evaluated_examples": 64,
                                                    "compute_seconds": 30}) == reservation
    _expect(ErrorCode.VERSION_CONFLICT, ledger.reserve, eid, "job_" + "a" * 32, {"evaluation_batches": 1})
    settled = ledger.settle(reservation.reservation_id, {"evaluation_batches": 1, "evaluated_examples": 99,
                                                         "compute_seconds": 1.5}, "cancelled")
    assert settled.state == "SETTLED" and settled.terminal_state == "cancelled"
    # Never more than was reserved; unstarted work is released.
    assert settled.charged == {"evaluation_batches": 1, "evaluated_examples": 64, "compute_seconds": 1.5}
    budget = _budget(engine, eid)
    assert budget["evaluation_batches"] == (1, 0) and budget["evaluated_examples"] == (64, 0)
    assert budget["compute_seconds"] == (1.5, 0)
    # Settling again is a no-op that returns the recorded settlement.
    assert ledger.settle(reservation.reservation_id, {"evaluation_batches": 2}, "succeeded") == settled
    assert _budget(engine, eid)["evaluation_batches"] == (1, 0)
    _expect(ErrorCode.INVALID_ARGUMENT, ledger.reserve, eid, "job_" + "b" * 32, {"free_lunch": 1})
    _expect(ErrorCode.INVALID_ARGUMENT, ledger.reserve, eid, "job_" + "b" * 32, {"evaluation_batches": -1})
    _expect(ErrorCode.BUDGET_EXCEEDED, ledger.reserve, eid, "job_" + "b" * 32, {"evaluation_batches": 10})
    _expect(ErrorCode.NOT_FOUND, ledger.settle, "resv_" + "0" * 32, {}, "failed")
    _expect(ErrorCode.INVALID_ARGUMENT, ledger.settle, reservation.reservation_id, {}, "exploded")


def test_reservations_require_an_open_experiment(engine, base, experiment):
    eid = experiment.experiment_id
    with engine.store.db.transaction() as conn:
        conn.execute("UPDATE experiments SET state = 'FROZEN' WHERE experiment_id = ?", (eid,))
    _expect(ErrorCode.VERSION_CONFLICT, engine.budget.reserve, eid, "job_" + "c" * 32, {"evaluation_batches": 1})
    _expect(ErrorCode.VERSION_CONFLICT, engine.submit_evaluation, eid, base.checkpoint_id, DEV,
            {"sample_count": 8, "seed": 0}, "k1")
    assert engine.list_jobs(eid) == []


# -- submission rules -------------------------------------------------------------------------


def test_submission_is_validated_without_partial_mutation(engine, bundles, base, signed_sum, experiment):
    eid = experiment.experiment_id
    engine.register_suite(_suite(signed_sum, "signed_sum_final_stub", split="final"))
    engine.register_suite(_suite(signed_sum, "other_task_dev", task_id="other_task"))
    reference = engine.import_bundle(bundles["reference"])
    good = {"sample_count": 8, "seed": 0}
    cases = [
        (ErrorCode.POLICY_DENIED, base.checkpoint_id, "signed_sum_final_stub", good),  # development suites only
        (ErrorCode.NOT_FOUND, base.checkpoint_id, "no_such_suite", good),
        (ErrorCode.NOT_FOUND, base.checkpoint_id, "other_task_dev", good),
        (ErrorCode.NOT_FOUND, reference.checkpoint_id, DEV, good),  # not in this experiment
        (ErrorCode.POLICY_DENIED, base.checkpoint_id, DEV, {"sample_count": 9, "seed": 0}),  # not preregistered
        (ErrorCode.POLICY_DENIED, base.checkpoint_id, DEV, {"sample_count": 8, "seed": 5}),
        (ErrorCode.INVALID_ARGUMENT, base.checkpoint_id, DEV, {"sample_count": 8}),
        (ErrorCode.INVALID_ARGUMENT, base.checkpoint_id, DEV, {"sample_count": 8, "seed": 0, "device": "cuda"}),
        (ErrorCode.INVALID_ARGUMENT, base.checkpoint_id, DEV, "all"),
    ]
    for index, (code, checkpoint_id, suite_id, config) in enumerate(cases):
        _expect(code, engine.submit_evaluation, eid, checkpoint_id, suite_id, config, f"bad{index}")
    assert engine.list_jobs(eid) == []
    assert all(reserved == 0 for _, reserved in _budget(engine, eid).values())
    failures = [e for e in engine.audit_log(eid) if e["event_type"] == "job.submit.failed"]
    assert len(failures) == len(cases)


def test_evaluation_is_unavailable_in_weights_only_mode(engine, base):
    experiment = _experiment(engine, base, WEIGHTS_ONLY_POLICY_ID)
    error = _expect(ErrorCode.POLICY_DENIED, engine.submit_evaluation, experiment.experiment_id, base.checkpoint_id,
                    DEV, {"sample_count": 8, "seed": 0}, "k1")
    assert "weights_only" in error.message
    assert engine.list_jobs(experiment.experiment_id) == []


def test_submission_reserves_the_planned_usage(engine, base):
    experiment = _experiment(engine, base, policy=_tiny_policy())
    job = engine.submit_evaluation(experiment.experiment_id, base.checkpoint_id, DEV,
                                   {"sample_count": 8, "seed": 1}, "k1")
    assert job.state == "queued" and job.kind == "evaluation" and job.progress.batches_total == 2
    assert job.reservation.requested == {"evaluation_batches": 2, "evaluated_examples": 8,
                                         "model_forward_calls": 2, "compute_seconds": 30.0}
    budget = _budget(engine, experiment.experiment_id)
    assert budget["evaluation_batches"] == (0, 2) and budget["evaluated_examples"] == (0, 8)
    # 64 examples would need 16 batches of 4, more than the 3-batch allowance.
    _expect(ErrorCode.BUDGET_EXCEEDED, engine.submit_evaluation, experiment.experiment_id, base.checkpoint_id,
            DEV, {"sample_count": 64, "seed": 0}, "k2")


# -- the worker -------------------------------------------------------------------------------


def test_worker_runs_an_evaluation_and_settles_actual_usage(workspace, engine, base, experiment):
    eid = experiment.experiment_id
    candidate = _candidate(engine, base, experiment)
    first = engine.submit_evaluation(eid, base.checkpoint_id, DEV, {"sample_count": 64, "seed": 0}, "e1")
    second = engine.submit_evaluation(eid, candidate.checkpoint_id, DEV, {"sample_count": 64, "seed": 0}, "e2")
    with _worker(workspace) as worker:
        assert worker.run_once() == first.job_id
        assert worker.run_once() == second.job_id
        assert worker.run_once() is None
    done = engine.get_job(eid, first.job_id)
    assert done.state == "succeeded" and done.result["metrics"] == {"accuracy": 0.5}
    assert done.result["sample_count"] == 64 and done.error is None
    assert engine.get_job(eid, second.job_id).result["metrics"] == {"accuracy": 1.0}
    assert done.usage.evaluation_batches == 1 and done.usage.evaluated_examples == 64
    assert done.usage.model_forward_calls == 1 and 0 < done.usage.compute_seconds < 30
    assert done.usage.peak_memory_bytes > 0
    assert done.progress.batches_completed == done.progress.batches_started == 1
    assert done.reservation.state == "SETTLED" and done.reservation.terminal_state == "succeeded"
    assert done.started_at and done.finished_at and done.attempts == 1
    budget = engine.get_experiment(eid).budget
    assert (budget["evaluation_batches"].consumed, budget["evaluation_batches"].reserved) == (2, 0)
    assert budget["model_forward_calls"].consumed == 2 and budget["evaluated_examples"].consumed == 128
    assert budget["compute_seconds"].reserved == 0 and budget["compute_seconds"].consumed > 0
    events = [e["event_type"] for e in engine.audit_log(eid) if e["event_type"].startswith("job.")]
    assert events == ["job.submitted", "job.submitted", "job.started", "job.succeeded", "job.started",
                      "job.succeeded"]
    # Budgets and results survive a restart.
    engine.close()
    with Engine(workspace) as restarted:
        assert restarted.get_job(eid, first.job_id) == done
        assert restarted.get_experiment(eid).budget == budget


def test_the_worker_never_executes_the_target_network_in_weights_only_mode(workspace, engine, base, monkeypatch):
    experiment = _experiment(engine, base, WEIGHTS_ONLY_POLICY_ID)
    job_id = _enqueue_directly(engine, experiment.experiment_id, base.checkpoint_id)
    from wse.adapters.mlp import MLPAdapter

    calls = []
    monkeypatch.setattr(MLPAdapter, "forward", lambda self, tensors, inputs: calls.append(1))
    with _worker(workspace) as worker:
        assert worker.run_once() == job_id
    job = engine.get_job(experiment.experiment_id, job_id)
    assert calls == []
    assert job.state == "failed" and job.error.code == ErrorCode.POLICY_DENIED
    assert job.usage.model_forward_calls == 0 and job.usage.evaluation_batches == 0
    assert all(consumed == 0 and reserved == 0 for name, (consumed, reserved) in
               _budget(engine, experiment.experiment_id).items() if name != "compute_seconds")


def test_the_forward_meter_refuses_work_beyond_the_reservation(workspace, engine, base, monkeypatch, signed_sum):
    # An evaluator that tries one more forward call than was reserved is stopped by the worker.
    from wse.evaluators.forward_batch import ForwardBatchEvaluator

    def greedy(self, forward, suite, config, *, batch_size):
        inputs = torch.tensor(suite.data.inputs[:4], dtype=torch.float32)
        for _ in range(3):
            forward(inputs)
        return {"metrics": {}, "sample_count": 4}

    monkeypatch.setattr(ForwardBatchEvaluator, "evaluate", greedy)
    experiment = _experiment(engine, base, policy=_tiny_policy())
    job = engine.submit_evaluation(experiment.experiment_id, base.checkpoint_id, DEV, {"sample_count": 8, "seed": 0},
                                   "k1")
    with _worker(workspace) as worker:
        worker.run_once()
    done = engine.get_job(experiment.experiment_id, job.job_id)
    assert done.state == "failed" and done.error.code == ErrorCode.BUDGET_EXCEEDED
    assert done.usage.model_forward_calls == 2  # the refused third call never ran


def test_only_one_worker_executes_at_a_time(workspace, engine, base, signed_sum):
    engine.register_suite(_suite(signed_sum, "slow_dev", evaluator="slow_batches@1",
                                 parameters={"sleep_seconds": 0.6, "cooperative": True}))
    experiment = _experiment(engine, base, policy=_tiny_policy())
    eid = experiment.experiment_id
    jobs = [engine.submit_evaluation(eid, base.checkpoint_id, "slow_dev", {"sample_count": 4, "seed": i},
                                     f"k{i}") for i in range(2)]
    with _worker(workspace, lease_seconds=2.0) as first, _worker(workspace) as second:
        runner = threading.Thread(target=first.run_once)
        runner.start()
        try:
            _wait_for(lambda: engine.get_job(eid, jobs[0].job_id).state == "running")
            assert second.run_once() is None  # a job is running, so nothing else starts
            assert engine.get_job(eid, jobs[1].job_id).state == "queued"
        finally:
            runner.join()
        assert second.run_once() == jobs[1].job_id


def test_heartbeats_renew_the_lease_of_a_long_job(workspace, engine, base, signed_sum):
    engine.register_suite(_suite(signed_sum, "slow_dev", evaluator="slow_batches@1",
                                 parameters={"sleep_seconds": 0.5, "cooperative": True}))
    experiment = _experiment(engine, base, policy=_tiny_policy())
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, "slow_dev", {"sample_count": 8, "seed": 0}, "k1")
    with _worker(workspace, lease_seconds=0.3) as worker:
        runner = threading.Thread(target=worker.run_once)
        runner.start()
        try:
            _wait_for(lambda: engine.get_job(eid, job.job_id).state == "running")
            expiries = set()
            for _ in range(5):
                expiries.add(engine.store.db.conn.execute(
                    "SELECT lease_expires_at FROM jobs WHERE job_id = ?", (job.job_id,)).fetchone()[0])
                assert engine.recover_jobs() == []  # the lease is live, so nothing is declared lost
                time.sleep(0.12)
        finally:
            runner.join()
    assert len(expiries) > 1
    assert engine.get_job(eid, job.job_id).state == "succeeded"


def test_get_job_waits_for_a_terminal_state(workspace, engine, base, experiment):
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, DEV, {"sample_count": 8, "seed": 0}, "k1")
    for bad in (-1, 10.5, 11, "5", True, None):
        _expect(ErrorCode.INVALID_ARGUMENT, engine.get_job, eid, job.job_id, wait_seconds=bad)
    start = time.monotonic()
    assert engine.get_job(eid, job.job_id).state == "queued"
    assert engine.get_job(eid, job.job_id, wait_seconds=0.2).state == "queued"
    assert time.monotonic() - start >= 0.2
    with _worker(workspace) as worker:
        timer = threading.Timer(0.3, worker.run_once)
        timer.start()
        try:
            done = engine.get_job(eid, job.job_id, wait_seconds=10)
        finally:
            timer.join()
    assert done.state == "succeeded"
    _expect(ErrorCode.NOT_FOUND, engine.get_job, eid, "job_" + "0" * 32)
    _expect(ErrorCode.NOT_FOUND, engine.get_job, eid, {"job": 1})


# -- cancellation -----------------------------------------------------------------------------


def test_cancelling_a_queued_job_releases_its_whole_reservation(workspace, engine, base, experiment):
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, DEV, {"sample_count": 64, "seed": 0}, "k1")
    cancelled = engine.cancel_job(eid, job.job_id, "c1")
    assert cancelled.state == "cancelled" and cancelled.reservation.state == "SETTLED"
    assert cancelled.usage.evaluation_batches == 0 and cancelled.usage.model_forward_calls == 0
    assert all(value == (0, 0) for name, value in _budget(engine, eid).items() if name != "compute_seconds")
    assert engine.cancel_job(eid, job.job_id, "c1") == cancelled  # idempotent
    _expect(ErrorCode.IDEMPOTENCY_CONFLICT, engine.cancel_job, eid, "job_" + "0" * 32, "c1")
    with _worker(workspace) as worker:
        assert worker.run_once() is None  # a cancelled job never starts


def test_cancelling_after_success_refunds_nothing(workspace, engine, base, experiment):
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, DEV, {"sample_count": 64, "seed": 0}, "k1")
    with _worker(workspace) as worker:
        worker.run_once()
    before = _budget(engine, eid)
    after_cancel = engine.cancel_job(eid, job.job_id, "c1")
    assert after_cancel.state == "succeeded" and not after_cancel.cancel_requested
    assert _budget(engine, eid) == before
    assert before["evaluation_batches"] == (1, 0) and before["model_forward_calls"] == (1, 0)


def test_cancelling_a_running_job_stops_it_between_batches(workspace, engine, base, signed_sum, tmp_path):
    marker = tmp_path / "first-batch"
    engine.register_suite(_suite(signed_sum, "slow_dev", evaluator="slow_batches@1",
                                 parameters={"sleep_seconds": 5.0, "cooperative": True, "marker": str(marker)}))
    experiment = _experiment(engine, base, policy=_tiny_policy())
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, "slow_dev", {"sample_count": 8, "seed": 0}, "k1")
    with _worker(workspace) as worker:
        runner = threading.Thread(target=worker.run_once)
        runner.start()
        try:
            _wait_for(marker.exists)
            requested = engine.cancel_job(eid, job.job_id, "c1")
            # Cancelling a request does not stop work by itself: the job is still running.
            assert requested.state == "running" and requested.cancel_requested
            done = engine.get_job(eid, job.job_id, wait_seconds=10)
        finally:
            runner.join()
    assert done.state == "cancelled" and done.error.code == ErrorCode.WORKER_FAILED
    assert done.error.details["reason"] == "cancelled"
    # The first batch ran and stays charged; the second never started and was released.
    assert (done.usage.evaluation_batches, done.usage.evaluated_examples, done.usage.model_forward_calls) == (1, 4, 1)
    budget = _budget(engine, eid)
    assert budget["evaluation_batches"] == (1, 0) and budget["evaluated_examples"] == (4, 0)
    assert budget["compute_seconds"][1] == 0 and budget["compute_seconds"][0] >= done.usage.compute_seconds


def test_cancellation_hard_stops_a_job_that_does_not_yield(workspace, engine, base, signed_sum, processes, tmp_path):
    marker = tmp_path / "first-batch"
    engine.register_suite(_suite(signed_sum, "blocking_dev", evaluator="slow_batches@1",
                                 parameters={"sleep_seconds": 60.0, "marker": str(marker)}))
    experiment = _experiment(engine, base, policy=_tiny_policy())
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, "blocking_dev", {"sample_count": 8, "seed": 0}, "k1")
    worker = _spawn(processes, workspace, "--cancel-grace-seconds", "0.3", "--lease-seconds", "1")
    _wait_for(marker.exists)
    engine.cancel_job(eid, job.job_id, "c1")
    done = engine.get_job(eid, job.job_id, wait_seconds=10)
    assert done.state == "cancelled" and done.error.details["reason"] == "cancelled"
    assert done.usage.evaluation_batches == 1 and done.usage.model_forward_calls == 1
    from wse.worker import EXIT_HARD_STOP

    assert worker.wait(timeout=30) == EXIT_HARD_STOP
    assert _budget(engine, eid)["evaluation_batches"] == (1, 0)


def test_a_job_that_exceeds_its_time_limit_fails_and_stays_charged(workspace, engine, base, signed_sum, processes):
    engine.register_suite(_suite(signed_sum, "blocking_dev", evaluator="slow_batches@1",
                                 parameters={"sleep_seconds": 60.0}))
    experiment = _experiment(engine, base, policy=_tiny_policy(seconds_per_operation=1.0))
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, "blocking_dev", {"sample_count": 8, "seed": 0}, "k1")
    assert job.reservation.requested["compute_seconds"] == 1.0
    worker = _spawn(processes, workspace, "--timeout-grace-seconds", "0.3")
    done = engine.get_job(eid, job.job_id, wait_seconds=10)
    if not done.terminal:
        done = engine.get_job(eid, job.job_id, wait_seconds=10)
    assert done.state == "failed" and done.error.code == ErrorCode.WORKER_FAILED
    assert done.error.details["reason"] == "timeout"
    assert done.usage.compute_seconds == 1.0 and done.usage.evaluation_batches == 1
    from wse.worker import EXIT_HARD_STOP

    assert worker.wait(timeout=30) == EXIT_HARD_STOP
    assert _budget(engine, eid)["compute_seconds"][0] >= 1.0


def test_a_cooperative_job_stops_at_its_time_limit(workspace, engine, base, signed_sum):
    engine.register_suite(_suite(signed_sum, "slow_dev", evaluator="slow_batches@1",
                                 parameters={"sleep_seconds": 5.0, "cooperative": True}))
    experiment = _experiment(engine, base, policy=_tiny_policy(seconds_per_operation=0.4))
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, "slow_dev", {"sample_count": 8, "seed": 0}, "k1")
    with _worker(workspace) as worker:
        worker.run_once()
    done = engine.get_job(eid, job.job_id)
    assert done.state == "failed" and done.error.details["reason"] == "timeout"
    assert done.usage.compute_seconds == pytest.approx(0.4)
    assert done.usage.evaluation_batches == 1


# -- crashes and restart ------------------------------------------------------------------------


def test_jobs_lost_in_a_crash_are_charged_in_full_and_never_rerun(workspace, engine, base, signed_sum, processes,
                                                                    tmp_path):
    marker = tmp_path / "first-batch"
    engine.register_suite(_suite(signed_sum, "blocking_dev", evaluator="slow_batches@1",
                                 parameters={"sleep_seconds": 60.0, "marker": str(marker)}))
    experiment = _experiment(engine, base, policy=_tiny_policy())
    eid = experiment.experiment_id
    lost = engine.submit_evaluation(eid, base.checkpoint_id, "blocking_dev", {"sample_count": 8, "seed": 0}, "k1")
    queued = engine.submit_evaluation(eid, base.checkpoint_id, DEV, {"sample_count": 4, "seed": 1}, "k2")
    worker = _spawn(processes, workspace, "--lease-seconds", "0.5")
    _wait_for(marker.exists)
    worker.send_signal(signal.SIGKILL)
    worker.wait(timeout=30)
    engine.close()
    time.sleep(0.6)  # let the dead worker's lease expire
    with Engine(workspace) as restarted:  # startup recovery settles lost jobs
        failed = restarted.get_job(eid, lost.job_id)
        assert failed.state == "failed" and failed.error.code == ErrorCode.WORKER_FAILED
        assert failed.error.details["reason"] == "lost"
        # The usage could not be verified, so the whole reservation is charged.
        assert failed.reservation.charged == failed.reservation.requested
        assert failed.usage.evaluation_batches == 2 and failed.usage.compute_seconds == 30.0
        assert restarted.get_job(eid, queued.job_id).state == "queued"
        budget = _budget(restarted, eid)
        assert budget["evaluation_batches"] == (2, 1)
        replacement = _spawn(processes, workspace)
        assert restarted.get_job(eid, queued.job_id, wait_seconds=10).state == "succeeded"
        assert restarted.get_job(eid, lost.job_id).attempts == 1  # not rerun
        replacement.terminate()
        assert replacement.wait(timeout=30) == 0


def test_the_supervisor_keeps_one_worker_running(workspace, engine, base, experiment, signed_sum):
    eid = experiment.experiment_id
    job = engine.submit_evaluation(eid, base.checkpoint_id, DEV, {"sample_count": 64, "seed": 0}, "k1")
    with WorkerSupervisor(workspace, extra_args=("--poll-seconds", "0.02"), restart_delay_seconds=0.05) as supervisor:
        assert engine.get_job(eid, job.job_id, wait_seconds=10).state == "succeeded"
        first = supervisor.process.pid
        supervisor.process.kill()
        _wait_for(lambda: supervisor.process is not None and supervisor.process.pid != first
                  and supervisor.process.poll() is None)
        again = engine.submit_evaluation(eid, base.checkpoint_id, DEV, {"sample_count": 64, "seed": 1}, "k2")
        assert engine.get_job(eid, again.job_id, wait_seconds=10).state == "succeeded"
        assert supervisor.restarts == 1
    assert supervisor.process is None


def test_a_worker_process_never_runs_forward_in_weights_only_mode(workspace, engine, base, processes, tmp_path):
    log = tmp_path / "forward.log"
    experiment = _experiment(engine, base, WEIGHTS_ONLY_POLICY_ID)
    job_id = _enqueue_directly(engine, experiment.experiment_id, base.checkpoint_id)
    bounded = _experiment(engine, base)
    control = engine.submit_evaluation(bounded.experiment_id, base.checkpoint_id, DEV,
                                       {"sample_count": 8, "seed": 0}, "k1")
    _spawn(processes, workspace, setup="wse_test_plugins:setup_forward_marker",
           env={"WSE_TEST_FORWARD_LOG": str(log)})
    assert engine.get_job(bounded.experiment_id, control.job_id, wait_seconds=10).state == "succeeded"
    job = engine.get_job(experiment.experiment_id, job_id)
    assert job.state == "failed" and job.error.code == ErrorCode.POLICY_DENIED
    # The control job proves the marker works: exactly its one forward call was logged.
    assert log.read_text().splitlines() == ["forward"]


# -- run scoping and experiment-scoped reads ---------------------------------------------------


def test_a_bound_engine_sees_only_its_own_jobs_and_cannot_touch_the_ledger(workspace, engine, base, signed_sum):
    mine, other = _experiment(engine, base), _experiment(engine, base)
    foreign = engine.submit_evaluation(other.experiment_id, base.checkpoint_id, DEV, {"sample_count": 8, "seed": 0},
                                       "k1")
    with Engine(workspace, run_id=mine.run_id) as bound:
        own = bound.submit_evaluation(mine.experiment_id, base.checkpoint_id, DEV, {"sample_count": 8, "seed": 0},
                                      "k1")
        assert [job.job_id for job in bound.list_jobs(mine.experiment_id)] == [own.job_id]
        _expect(ErrorCode.NOT_FOUND, bound.get_job, mine.experiment_id, foreign.job_id)
        _expect(ErrorCode.NOT_FOUND, bound.cancel_job, mine.experiment_id, foreign.job_id, "c1")
        _expect(ErrorCode.NOT_FOUND, bound.get_job, other.experiment_id, foreign.job_id)
        _expect(ErrorCode.POLICY_DENIED, bound.reserve, mine.experiment_id, "job_" + "d" * 32,
                {"evaluation_batches": 1})
        _expect(ErrorCode.POLICY_DENIED, bound.settle, own.reservation.reservation_id, {}, "failed")
        _expect(ErrorCode.POLICY_DENIED, bound.register_suite, _suite(signed_sum, "bound_suite"))
    assert engine.get_job(other.experiment_id, foreign.job_id).state == "queued"


def test_experiment_scoped_reads_and_analyses_are_metered(workspace, engine, bundles, base):
    experiment = _experiment(engine, base, policy=_tiny_policy(tensor_read_calls=2, analysis_calls=1))
    eid = experiment.experiment_id
    reference = engine.import_bundle(bundles["reference"])
    assert engine.inspect_model(eid, base.checkpoint_id).checkpoint_id == base.checkpoint_id
    _expect(ErrorCode.NOT_FOUND, engine.inspect_model, eid, reference.checkpoint_id)
    _expect(ErrorCode.NOT_FOUND, engine.read_model_tensor, eid, reference.checkpoint_id, "layers.1.weight", ROW1)
    _expect(ErrorCode.NOT_FOUND, engine.analyze_model, eid, reference.checkpoint_id, "stats", "layers.1.weight", ROW1)
    # An invalid request is rejected before anything is charged.
    _expect(ErrorCode.NOT_FOUND, engine.read_model_tensor, eid, base.checkpoint_id, "nope", ROW1)
    read = engine.read_model_tensor(eid, base.checkpoint_id, "layers.1.weight", ROW1)
    assert read.values == [-1.0, 1.0, -1.0, 1.0] and read.usage.tensor_read_calls == 1
    stats = engine.analyze_model(eid, base.checkpoint_id, "stats", "layers.1.weight", ROW1)
    assert stats.result["count"] == 4 and stats.usage.analysis_calls == 1
    _expect(ErrorCode.BUDGET_EXCEEDED, engine.analyze_model, eid, base.checkpoint_id, "norm", "layers.1.weight", ROW1)
    budget = engine.get_experiment(eid).budget
    assert (budget["tensor_read_calls"].consumed, budget["analysis_calls"].consumed) == (1, 1)
    assert budget["compute_seconds"].consumed > 0
    # One read left: concurrent readers on independent connections get exactly one.
    outcomes = _in_threads(workspace, 3, lambda own, i: own.read_model_tensor(
        eid, base.checkpoint_id, "layers.1.weight", ROW1))
    assert outcomes.count(ErrorCode.BUDGET_EXCEEDED) == 2
    assert engine.get_experiment(eid).budget["tensor_read_calls"].consumed == 2


# -- helpers ----------------------------------------------------------------------------------


def _enqueue_directly(engine, experiment_id, checkpoint_id):
    """Insert a queued evaluation job without the submission checks (simulates a bypass)."""
    job_id, reservation_id, now = "job_" + os.urandom(16).hex(), "resv_" + os.urandom(16).hex(), utc_now()
    with engine.store.db.transaction() as conn:
        conn.execute(
            "INSERT INTO reservations (reservation_id, experiment_id, job_id, state, requested_json, created_at) "
            "VALUES (?, ?, ?, 'HELD', ?, ?)", (reservation_id, experiment_id, job_id,
                                              json.dumps({"compute_seconds": 1.0}), now))
        conn.execute("UPDATE budget_ledger SET reserved = reserved + 1.0 WHERE experiment_id = ? "
                     "AND counter = 'compute_seconds'", (experiment_id,))
        conn.execute(
            "INSERT INTO jobs (job_id, experiment_id, kind, checkpoint_id, suite_id, config_json, state, "
            "reservation_id, progress_json, created_at) VALUES (?, ?, 'evaluation', ?, ?, ?, 'queued', ?, ?, ?)",
            (job_id, experiment_id, checkpoint_id, DEV, json.dumps({"sample_count": 8, "seed": 0}), reservation_id,
             json.dumps({"batches_total": 1}), now))
    return job_id


def test_job_modules_import_no_mcp_or_agent_sdk():
    code = ("import sys, wse.budget, wse.jobs, wse.worker, wse.evaluators.forward_batch; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('mcp', 'agents', 'openai', 'claude_agent_sdk')))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_the_cli_runs_the_worker(workspace, engine, base, experiment):
    from wse.cli import main

    job = engine.submit_evaluation(experiment.experiment_id, base.checkpoint_id, DEV,
                                   {"sample_count": 64, "seed": 0}, "k1")
    assert main(["--workspace", str(workspace), "worker", "--max-jobs", "1", "--poll-seconds", "0.01"]) == 0
    assert engine.get_job(experiment.experiment_id, job.job_id).state == "succeeded"
