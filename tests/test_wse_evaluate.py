"""Whole development evaluations waited for by WSE (``wse_evaluate``) and baselines reused across experiments.

``Engine.evaluate_checkpoint`` and the MCP tool ``wse_evaluate`` evaluate one
checkpoint on its task's whole development suite and wait for the result: the
report summary and, for a candidate, the paired comparison with the base
checkpoint's report. A checkpoint is evaluated once: repeated requests return
its report or wait for its job without charging again. A wait that ends early
(the client's timeout, a killed server, ``stop()``) leaves the job to the
worker, and a later request returns its result; the runner's finalization
cancels a job that is still running. ``Engine.ensure_baseline`` gives an
experiment its base checkpoint's report before the controller starts, linking
another experiment's identical report when the workspace has one.

The signed-sum fixture covers the rules quickly, a slow test evaluator
(``slow_batches@1``) the waits, and the tiny language-model fixture (``lm``)
generation passes. The MCP tests run a real ``wse serve`` with its worker.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import jsonschema
import pytest
from mcp.shared.exceptions import MCPError
from test_llm_policy_mcp import TASK, _open_run, _samples, lm  # noqa: F401 - lm is a fixture
from test_mcp_e2e import (  # noqa: F401 - trial and plugins are fixtures
    DEV,
    ORACLE,
    ROW1,
    TESTS_DIR,
    _create,
    _spec,
    connect,
    plugins,
    run,
    trial,
)

from wse.api import Engine
from wse.jobs import evaluation_wait_cap, process_exists
from wse.policy import SMALL_MODEL_POLICY_ID, builtin_experiment_policies
from wse.registry import builtin_registry
from wse.runner import EVALUATION_TOOLS, ExperimentExecutor, ReferenceRunner, StopMonitor, allowed_tools, \
    render_instructions
from wse.schemas import (
    ComparisonReport,
    ErrorCode,
    EvalConfig,
    EvaluationReport,
    GenerationComparison,
    GenerationOutput,
    ReportSource,
    TaskDefinition,
    WSEError,
)
from wse.strategies.base import STRATEGY_TOOLS, Action

SLOW_TASK, SLOW_SUITE = "signed_sum_slow_v1", "signed_sum_dev_slow"
COUNTERS = ("evaluation_batches", "evaluated_examples", "model_forward_calls")
LOOP_TASK = {"task_id": TASK, "title": "Finish answers", "description": "Make the model finish its answers.",
             "objective": {"task_id": TASK, "description": "Fewer outputs at the token limit.",
                           "gates": [{"metric": "accuracy", "min_delta": 0.0}]},
             "development_suite_id": "practice", "development_examples": 4}
CANCELLED_WAIT = "the request was cancelled while waiting for job"


def _expect(code: ErrorCode, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@contextlib.contextmanager
def _worker(workspace: Path, registry: Any = None) -> Iterator[Any]:
    """A worker thread that runs queued jobs, as the run's worker process does, until the block ends."""
    from wse.worker import Worker

    worker = Worker(workspace, registry=registry, device="cpu", poll_seconds=0.02, hard_exit=lambda code: None)
    thread = threading.Thread(target=worker.run, name="test-worker", daemon=True)
    thread.start()
    try:
        yield worker
    finally:
        worker.request_stop()
        thread.join(timeout=120)
        worker.close()


def _plugin_registry(plugins: Any) -> Any:
    registry = builtin_registry()
    plugins.setup(registry)
    return registry


def _experiment(workspace: Path, run_id: str, task_id: str, policy_id: str = SMALL_MODEL_POLICY_ID) -> str:
    with Engine(workspace, run_id=run_id) as bound:
        return bound.create_experiment(run_id, task_id, policy_id, "create").experiment_id


def _candidate(engine: Engine, eid: str, base: str, *, factor: float = -1.0, key: str = "flip") -> str:
    """A candidate that scales row 1 of ``layers.1.weight`` by ``factor`` (-1 repairs the fixture)."""
    spec = {"base_checkpoint_id": base, "hypothesis": "One output row has the wrong sign.",
            "operations": [{"op": "scale_slice", "tensor": "layers.1.weight", "selection": ROW1, "factor": factor}]}
    preview = engine.preview_patch(eid, spec, f"preview-{key}")
    return engine.apply_patch(eid, preview.patch_id, f"apply-{key}").checkpoint_id


def _ledger(engine: Engine, eid: str, *names: str) -> dict[str, tuple[Any, Any]]:
    """(consumed, reserved) of ledger counters (all of them when none are named)."""
    budget = engine.get_experiment(eid).budget
    return {name: (entry.consumed, entry.reserved) for name, entry in budget.items() if not names or name in names}


def _slow_task(engine: Engine, signed_sum: Any, marker: Path, sleep_seconds: float) -> None:
    """A task whose development suite runs one batch, writes ``marker``, then waits ``sleep_seconds``."""
    engine.register_suite({"suite_id": SLOW_SUITE, "task_id": SLOW_TASK, "split": "development",
                           "evaluator": "slow_batches@1", "sample_counts": [64], "seeds": [0],
                           "parameters": {"sleep_seconds": sleep_seconds, "cooperative": True,
                                          "marker": str(marker)},
                           "data": signed_sum.suite_data(32, signed_sum.DEV_SEED)})
    task = signed_sum.load_task()
    engine.register_task({**task, "task_id": SLOW_TASK, "development_suite_id": SLOW_SUITE, "target_accuracy": None,
                          "objective": {**task["objective"], "task_id": SLOW_TASK,
                                        "gates": [{"metric": "batches", "min_value": 1}]}})


def _settled(engine: Engine, eid: str, job_id: str, seconds: float = 60) -> Any:
    deadline = time.monotonic() + seconds
    job = engine.get_job(eid, job_id)
    while not job.terminal and time.monotonic() < deadline:
        job = engine.get_job(eid, job_id, 10)
    return job


# -- one call per evaluation ---------------------------------------------------------------------------------------


def test_one_call_returns_the_report_and_the_comparison_with_the_base(trial, workspace):
    base = trial["task_checkpoint_id"]
    eid = _experiment(workspace, trial["run_id"], trial["task_id"])
    with Engine(workspace, run_id=trial["run_id"]) as bound, _worker(workspace):
        first = bound.evaluate_checkpoint(eid, base, "base")
        assert (first.origin, first.state, first.suite_id) == ("new_job", "succeeded", "signed_sum_dev_v1")
        assert first.config == EvalConfig(sample_count=64, seed=0)  # the whole suite at its first seed
        report = first.report
        assert (report.metrics, report.sample_count, report.target_met) == ({"accuracy": 0.5}, 64, False)
        assert set(report.intervals) == {"accuracy", "fully_correct_clusters"} and report.reused_from is None
        assert first.comparison is None and first.comparison_unavailable == "this is the base checkpoint"

        candidate = _candidate(bound, eid, base)
        second = bound.evaluate_checkpoint(eid, candidate, "candidate")
        assert second.report.metrics == {"accuracy": 1.0} and second.report.target_met is True
        comparison = second.comparison
        assert isinstance(comparison, ComparisonReport) and second.comparison_unavailable is None
        assert (comparison.difference, comparison.clusters_improved) == (0.5, 32)
        assert second.base_report_id == comparison.base_report_id == report.report_id
        assert comparison.candidate_report_id == second.report.report_id
        # Each result carries its job's settled usage; the two calls charged two whole evaluations, nothing held.
        assert second.usage == bound.get_job(eid, second.job_id).usage
        assert _ledger(bound, eid, *COUNTERS) == {"evaluation_batches": (2, 0), "evaluated_examples": (128, 0),
                                                  "model_forward_calls": (2, 0)}
        assert bound.get_experiment(eid).budget["compute_seconds"].reserved == 0
        # A strategy's executor returns the same result; the suite, sample count, and seed are not arguments.
        executor = ExperimentExecutor(bound, constraint_status="verified")
        observation = executor.execute(Action.call("wse_evaluate", checkpoint_id=candidate,
                                                   idempotency_key="candidate"))
        assert observation.ok and observation.result == second.model_dump(mode="json")
        refused = executor.execute(Action.call("wse_evaluate", checkpoint_id=candidate, idempotency_key="custom",
                                               eval_config=DEV))
        assert refused.error_code == "INVALID_ARGUMENT"


def test_a_checkpoint_is_evaluated_once_and_repeated_requests_charge_nothing(trial, workspace):
    base = trial["task_checkpoint_id"]
    eid = _experiment(workspace, trial["run_id"], trial["task_id"])
    with Engine(workspace, run_id=trial["run_id"]) as bound:
        policy = bound.experiment_policy(eid)
        _expect(ErrorCode.INVALID_ARGUMENT, bound.evaluate_checkpoint, eid, base, "too-long",
                wait_seconds=evaluation_wait_cap(policy) + 1)
        assert bound.list_jobs(eid) == []  # a refused wait queues nothing
        # With no worker running, the job waits in the queue; a second request waits for the same job.
        queued = bound.evaluate_checkpoint(eid, base, "first", wait_seconds=0)
        assert (queued.origin, queued.state, queued.report, queued.usage) == ("new_job", "queued", None, None)
        held = _ledger(bound, eid)
        assert held["evaluation_batches"] == (0, 1)
        attached = bound.evaluate_checkpoint(eid, base, "second", wait_seconds=0)
        assert (attached.origin, attached.job_id, attached.state) == ("existing_job", queued.job_id, "queued")
        assert _ledger(bound, eid) == held and len(bound.list_jobs(eid)) == 1
        with _worker(workspace):
            waited = bound.evaluate_checkpoint(eid, base, "second")  # the same key waits for the same job
        assert (waited.origin, waited.job_id, waited.state) == ("existing_job", queued.job_id, "succeeded")
        # A new key returns the earlier report; a key repeats its original outcome; nothing is charged again.
        later = bound.evaluate_checkpoint(eid, base, "third")
        assert (later.origin, later.job_id, later.report) == ("existing_report", queued.job_id, waited.report)
        repeated = bound.evaluate_checkpoint(eid, base, "first")
        assert (repeated.origin, repeated.state, repeated.report) == ("new_job", "succeeded", waited.report)
        assert _ledger(bound, eid, *COUNTERS) == {"evaluation_batches": (1, 0), "evaluated_examples": (64, 0),
                                                  "model_forward_calls": (1, 0)}
        candidate = _candidate(bound, eid, base)
        _expect(ErrorCode.IDEMPOTENCY_CONFLICT, bound.evaluate_checkpoint, eid, candidate, "first")
        # A report of wse_submit_evaluation at the whole configuration is the checkpoint's evaluation as well.
        submitted = bound.submit_evaluation(eid, candidate, "signed_sum_dev_v1", DEV, "submitted")
        with _worker(workspace):
            assert _settled(bound, eid, submitted.job_id).state == "succeeded"
        found = bound.evaluate_checkpoint(eid, candidate, "after-submission", wait_seconds=0)
        assert (found.origin, found.job_id, found.comparison.difference) == ("existing_report", submitted.job_id, 0.5)
        # A cancelled evaluation is reported as cancelled under its key; a new key evaluates again, charged again.
        other = _candidate(bound, eid, base, factor=0.5, key="half")
        pending = bound.evaluate_checkpoint(eid, other, "half-1", wait_seconds=0)
        bound.cancel_job(eid, pending.job_id, "cancel-half")
        cancelled = bound.evaluate_checkpoint(eid, other, "half-1")
        assert (cancelled.state, cancelled.report, cancelled.error.code) == ("cancelled", None, ErrorCode.WORKER_FAILED)
        assert cancelled.usage.evaluation_batches == 0  # cancelled while queued: its reservation was released
        retried = bound.evaluate_checkpoint(eid, other, "half-2", wait_seconds=0)
        assert retried.origin == "new_job" and retried.job_id != pending.job_id
        assert _ledger(bound, eid, "evaluation_batches") == {"evaluation_batches": (2, 1)}


def test_an_interrupted_wait_leaves_the_job_running_and_a_later_request_returns_it(engine, trial, workspace,
                                                                                    signed_sum, plugins, tmp_path):
    base, marker = trial["task_checkpoint_id"], tmp_path / "started"
    _slow_task(engine, signed_sum, marker, sleep_seconds=2.0)
    run_id = engine.grant_run(SLOW_TASK, trial["policy_id"], base).run_id
    eid = _experiment(workspace, run_id, SLOW_TASK)
    registry = _plugin_registry(plugins)  # as the server and the worker of a run load the operator's setup
    with Engine(workspace, registry=registry, run_id=run_id) as bound, _worker(workspace, registry):
        # The wait ends when stop() says so (the job's first batch is done), not with the job.
        interrupted = bound.evaluate_checkpoint(eid, base, "first", stop=marker.exists)
        assert (interrupted.origin, interrupted.state, interrupted.report) == ("new_job", "running", None)
        later = bound.evaluate_checkpoint(eid, base, "later")
        assert (later.origin, later.job_id, later.state) == ("existing_job", interrupted.job_id, "succeeded")
        assert later.report.metrics == {"batches": 1.0}
        assert bound.get_job(eid, interrupted.job_id).report_id == later.report.report_id
        assert len(bound.list_jobs(eid)) == 1 and _ledger(bound, eid, "evaluation_batches") == {
            "evaluation_batches": (1, 0)}


# -- baselines -----------------------------------------------------------------------------------------------------


def test_baselines_are_evaluated_once_and_reused_by_later_experiments(engine, trial, workspace):
    base, task_id, policy_id = trial["task_checkpoint_id"], trial["task_id"], trial["policy_id"]
    eid_a = _experiment(workspace, trial["run_id"], task_id)
    with Engine(workspace, run_id=trial["run_id"]) as bound:
        _expect(ErrorCode.POLICY_DENIED, bound.ensure_baseline, eid_a)  # an operator action, never an agent's
    with _worker(workspace):
        first = engine.ensure_baseline(eid_a)
    assert (first.origin, first.state, first.report.metrics) == ("new_job", "succeeded", {"accuracy": 0.5})
    charged_a = _ledger(engine, eid_a)
    assert charged_a["evaluation_batches"] == (1, 0) and charged_a["compute_seconds"][0] > 0
    assert engine.ensure_baseline(eid_a) == first and _ledger(engine, eid_a) == charged_a  # no worker, no charge
    # Another model gets a baseline of its own.
    reference = trial["reference_checkpoint_id"]
    eid_c = _experiment(workspace, engine.grant_run(task_id, policy_id, reference).run_id, task_id)
    with _worker(workspace):
        other_model = engine.ensure_baseline(eid_c)
    assert (other_model.origin, other_model.report.metrics) == ("new_job", {"accuracy": 1.0})

    # A later experiment links the report instead: no job runs (no worker here), one evaluation is charged.
    run_b = engine.grant_run(task_id, policy_id, base).run_id
    eid_b = _experiment(workspace, run_b, task_id)
    reused = engine.ensure_baseline(eid_b)
    assert (reused.origin, reused.state, reused.job_id) == ("reused_report", "succeeded", None)
    assert reused.report.reused_from == ReportSource(experiment_id=eid_a, report_id=first.report.report_id)
    assert reused.report.report_id != first.report.report_id and engine.list_jobs(eid_b) == []
    assert (reused.report.metrics, reused.report.intervals) == (first.report.metrics, first.report.intervals)
    assert _ledger(engine, eid_b, *COUNTERS, "compute_seconds") == {
        "evaluation_batches": (1, 0), "evaluated_examples": (64, 0), "model_forward_calls": (1, 0),
        "compute_seconds": (0, 0)}
    assert reused.usage.compute_seconds == 0 and reused.usage.evaluation_batches == 1
    linked, source = engine.get_report(eid_b, reused.report.report_id), engine.get_report(eid_a, first.report.report_id)
    differing = {name for name in EvaluationReport.model_fields if getattr(linked, name) != getattr(source, name)}
    assert differing == {"report_id", "experiment_id", "job_id", "usage", "created_at", "reused_from"}
    events = [event for event in engine.audit_log(eid_b) if event["event_type"] == "report.reused"]
    assert [event["payload"]["charged"] for event in events] == [
        {"evaluation_batches": 1, "evaluated_examples": 64, "model_forward_calls": 1}]
    # The runner finds it as the experiment's baseline, and repeating the call charges nothing.
    task = engine.get_task(task_id)
    assert [r.report_id for r in StopMonitor(engine, eid_b, task)._full_development_reports()] == [linked.report_id]
    before = _ledger(engine, eid_b)
    assert engine.ensure_baseline(eid_b) == reused and _ledger(engine, eid_b) == before

    # The run's own requests return the baseline, and its candidates are compared with it.
    with Engine(workspace, run_id=run_b) as bound:
        again = bound.evaluate_checkpoint(eid_b, base, "base")
        assert (again.origin, again.report) == ("existing_report", reused.report)
        candidate = _candidate(bound, eid_b, base)
        with _worker(workspace):
            evaluated = bound.evaluate_checkpoint(eid_b, candidate, "candidate")
        assert evaluated.base_report_id == linked.report_id and evaluated.comparison.difference == 0.5
    assert _ledger(engine, eid_b, "evaluation_batches") == {"evaluation_batches": (2, 0)}

    # Evaluations of another batch size get a baseline of their own: their reports would not pair.
    small = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID]
    engine.register_policy(small.model_copy(update={
        "policy_id": "small_model_batches_of_32", "budget": small.budget.model_copy(update={"examples_per_batch": 32})}))
    eid_d = _experiment(workspace, engine.grant_run(task_id, "small_model_batches_of_32", base).run_id, task_id,
                        "small_model_batches_of_32")
    with _worker(workspace):
        other_batches = engine.ensure_baseline(eid_d)
    assert other_batches.origin == "new_job" and other_batches.usage.evaluation_batches == 2
    # Reuse follows the model's identity, not its checkpoint ID: B's repaired candidate has the reference's weights,
    # and a run on the reference reuses the earliest report of that model, the one of experiment C.
    eid_e = _experiment(workspace, engine.grant_run(task_id, policy_id, reference).run_id, task_id)
    assert engine.load_manifest(candidate).weights_hash == engine.load_manifest(reference).weights_hash
    same_model = engine.ensure_baseline(eid_e)
    assert same_model.origin == "reused_report" and same_model.report.checkpoint_id == reference
    assert same_model.report.reused_from == ReportSource(experiment_id=eid_c, report_id=other_model.report.report_id)


def test_a_reused_generation_baseline_pairs_with_the_runs_passes_and_its_outputs_stay_readable(lm):
    """A run with its own suite and task of the same problems reuses another run's pass of the same model."""
    with Engine(lm.workspace) as engine:
        engine.register_task(LOOP_TASK)
        suite = engine.get_suite("practice")
        engine.register_suite({**suite.model_dump(mode="json"), "suite_id": "practice_again", "task_id": "loops_v2"})
        engine.register_task({**LOOP_TASK, "task_id": "loops_v2", "development_suite_id": "practice_again",
                              "objective": {**LOOP_TASK["objective"], "task_id": "loops_v2"}})
        run_b = engine.grant_run("loops_v2", "qwen_loop_v1", lm.base).run_id
    run_a, eid_a = _open_run(lm.workspace, lm.base)
    eid_b = _experiment(lm.workspace, run_b, "loops_v2", "qwen_loop_v1")
    with Engine(lm.workspace) as engine:
        with _worker(lm.workspace):
            first = engine.ensure_baseline(eid_a, device="cpu")
        assert (first.origin, first.config) == ("new_job", EvalConfig(sample_count=4, seed=0))
        source = engine.get_report(eid_a, first.report.report_id)
        tokens = source.usage.generated_tokens
        assert tokens == sum(sample.generated_tokens for sample in _samples(source)) > 0
        reused = engine.ensure_baseline(eid_b, device="cpu")  # no worker runs: the pass is linked
        assert (reused.origin, reused.job_id, reused.suite_id) == ("reused_report", None, "practice_again")
        assert (reused.usage.generation_passes, reused.usage.generated_tokens, reused.usage.compute_seconds) == (
            1, tokens, 0)
        assert _ledger(engine, eid_b, "generation_passes", "generated_tokens", "compute_seconds") == {
            "generation_passes": (1, 0), "generated_tokens": (tokens, 0), "compute_seconds": (0, 0)}
        linked = engine.get_report(eid_b, reused.report.report_id)
        assert (linked.suite_id, linked.task_id, linked.suite_hash) == ("practice_again", "loops_v2",
                                                                        engine.get_suite("practice_again").suite_hash)
    uris = [sample.output_uri for sample in _samples(linked)]
    assert uris == [sample.output_uri for sample in _samples(source)]
    with Engine(lm.workspace, run_id=run_b) as bound:
        # The run reads and probes the reused outputs as its own pass's.
        output = GenerationOutput.model_validate(bound.read_analysis_result(uris[0]))
        assert output.experiment_id == eid_a
        probe = bound.submit_probe(eid_b, lm.base, [{"output_uri": uris[0]}, {"output_uri": uris[1]}],
                                   ["residual:1"], "probe")
        spec = {"base_checkpoint_id": lm.base, "hypothesis": "Scaling one row changes some samples.",
                "operations": [{"op": "scale_slice", "tensor": "model.layers.0.mlp.down_proj.weight", "factor": -3.0,
                                "selection": [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 32,
                                                                                   "step": 1}]}]}
        candidate = bound.apply_patch(eid_b, bound.preview_patch(eid_b, spec, "p").patch_id, "a").checkpoint_id
        with _worker(lm.workspace):
            probed = _settled(bound, eid_b, probe.job_id)
            evaluated = bound.evaluate_checkpoint(eid_b, candidate, "candidate")
        assert probed.state == "succeeded", probed.error
        assert evaluated.state == "succeeded", evaluated.error
        assert isinstance(evaluated.comparison, GenerationComparison) and evaluated.base_report_id == linked.report_id
        assert set(evaluated.comparison.differences) == {"accuracy", "limit_hit_rate", "mean_generated_tokens"}
    # A run that reused nothing cannot read them.
    run_c, _ = _open_run(lm.workspace, lm.base)
    with Engine(lm.workspace, run_id=run_c) as other:
        _expect(ErrorCode.NOT_FOUND, other.read_analysis_result, uris[0])


def test_a_blocking_evaluation_refreshes_the_agents_inputs_under_a_sandbox_policy(lm):
    """As a succeeded job polled with wse_get_job does, wse_evaluate exports its pass to the run's sandbox inputs."""
    from wse.mcp_server import WSEMCPServer
    from wse.sandbox_runs import layout_for

    with Engine(lm.workspace) as engine:
        engine.register_task(LOOP_TASK)
    run_id, eid = _open_run(lm.workspace, lm.base, policy_id="qwen_loop_sandbox_v1")
    with Engine(lm.workspace, run_id=run_id) as bound, _worker(lm.workspace):
        response = WSEMCPServer(bound).wse_evaluate(experiment_id=eid, checkpoint_id=lm.base, idempotency_key="base")
        result = response.structured_content["result"]
        assert (result["origin"], result["state"]) == ("new_job", "succeeded")
        inputs = layout_for(bound.experiments, run_id).inputs
    assert (inputs / "passes" / f"{result['report']['report_id']}.jsonl").is_file()
    assert (inputs / "weights" / "base.safetensors").is_file()


# -- through MCP ---------------------------------------------------------------------------------------------------


def test_wse_evaluate_through_a_real_wse_serve(trial, tmp_path):
    workspace, base = tmp_path / "workspace", trial["task_checkpoint_id"]

    async def body():
        async with connect(workspace, trial["run_id"], trial["log"], worker=True) as session:
            tools = {tool.name: tool for tool in (await session.client.list_tools()).tools}
            tool = tools["wse_evaluate"]
            assert set(tool.input_schema["properties"]) == {"experiment_id", "checkpoint_id", "idempotency_key"}
            assert set(tool.input_schema["required"]) == {"checkpoint_id", "idempotency_key"}
            assert tool.annotations.read_only_hint is False and "evaluated once" in tool.description
            await _create(session, trial)
            first = await session.call("wse_evaluate", checkpoint_id=base, idempotency_key="base")
            jsonschema.validate(first, tool.output_schema)
            result = first["result"]
            assert (result["origin"], result["state"], result["report"]["metrics"]) == (
                "new_job", "succeeded", {"accuracy": 0.5})
            assert first["uris"] == [result["report"]["report_uri"]] and "a new evaluation" in first["summary"]
            assert first["usage"]["call"] == result["usage"]
            assert first["usage"]["cumulative"]["evaluation_batches"]["consumed"] == 1
            preview = (await session.call("wse_preview_patch", patch_spec=_spec(trial, [ORACLE]),
                                          idempotency_key="p"))["result"]
            candidate = (await session.call("wse_apply_patch", patch_id=preview["patch_id"],
                                            idempotency_key="a"))["result"]["checkpoint_id"]
            second = await session.call("wse_evaluate", checkpoint_id=candidate, idempotency_key="candidate")
            jsonschema.validate(second, tool.output_schema)
            assert second["result"]["comparison"]["difference"] == 0.5
            assert second["result"]["base_report_id"] == result["report"]["report_id"]
            assert second["uris"] == [second["result"]["report"]["report_uri"], result["report"]["report_uri"]]
            assert "target met; difference from the base accuracy +0.5" in second["summary"]
            again = await session.call("wse_evaluate", checkpoint_id=candidate, idempotency_key="again")
            assert again["result"]["origin"] == "existing_report"
            assert again["result"]["report"] == second["result"]["report"] and "not charged again" in again["summary"]
            assert again["usage"]["cumulative"]["evaluation_batches"]["consumed"] == 2

    run(body)


def test_a_client_timeout_ends_the_wait_and_a_later_call_returns_the_job(engine, trial, tmp_path, signed_sum):
    workspace, base, marker = tmp_path / "workspace", trial["task_checkpoint_id"], tmp_path / "started"
    _slow_task(engine, signed_sum, marker, sleep_seconds=10.0)
    slow = {**trial, "task_id": SLOW_TASK, "run_id": engine.grant_run(SLOW_TASK, trial["policy_id"], base).run_id}

    async def body():
        async with connect(workspace, slow["run_id"], trial["log"], worker=True) as session:
            eid = (await _create(session, slow))["experiment"]["experiment_id"]
            started = time.monotonic()
            with pytest.raises(MCPError) as excinfo:
                await session.client.call_tool("wse_evaluate", {"checkpoint_id": base, "idempotency_key": "first"},
                                               read_timeout_seconds=1.0)
            assert "timed out" in str(excinfo.value) and time.monotonic() - started < 5
            # The server answers while the job runs on in the worker (a blocked server would take about 9 s).
            answered = time.monotonic()
            await session.call("wse_get_experiment")
            assert time.monotonic() - answered < 4
            with Engine(workspace) as operator:
                (job,) = operator.list_jobs(eid)
            assert job.state in ("queued", "running")
            later = await session.call("wse_evaluate", checkpoint_id=base, idempotency_key="later")
            assert (later["result"]["origin"], later["result"]["job_id"], later["result"]["state"]) == (
                "existing_job", job.job_id, "succeeded")
            assert later["usage"]["cumulative"]["evaluation_batches"]["consumed"] == 1
            first = await session.call("wse_evaluate", checkpoint_id=base, idempotency_key="first")
            assert (first["result"]["origin"], first["result"]["state"]) == ("new_job", "succeeded")
            polled = await session.call("wse_get_job", job_id=job.job_id)
            assert polled["result"]["report_id"] == later["result"]["report"]["report_id"]

    run(body)
    assert CANCELLED_WAIT in trial["log"].read_text(encoding="utf-8")


def test_a_killed_server_leaves_the_job_to_its_worker_and_a_new_server_returns_it(engine, trial, tmp_path,
                                                                                   signed_sum):
    workspace, base, marker = tmp_path / "workspace", trial["task_checkpoint_id"], tmp_path / "started"
    _slow_task(engine, signed_sum, marker, sleep_seconds=4.0)
    slow = {**trial, "task_id": SLOW_TASK, "run_id": engine.grant_run(SLOW_TASK, trial["policy_id"], base).run_id}
    state: dict[str, Any] = {}

    async def first_connection():
        async with connect(workspace, slow["run_id"], trial["log"], worker=True) as session:
            state["eid"] = (await _create(session, slow))["experiment"]["experiment_id"]

            async def evaluate():
                with contextlib.suppress(Exception):  # the server dies during this call
                    await session.client.call_tool("wse_evaluate", {"checkpoint_id": base, "idempotency_key": "first"})

            async with anyio.create_task_group() as group:
                group.start_soon(evaluate)
                with anyio.fail_after(60):
                    while not marker.exists():
                        await anyio.sleep(0.02)
                with Engine(workspace) as operator:
                    row = operator.store.db.conn.execute("SELECT job_id, lease_pid FROM jobs WHERE experiment_id = ?",
                                                         (state["eid"],)).fetchone()
                state["job_id"], state["worker"] = row["job_id"], row["lease_pid"]
                server = subprocess.run(["ps", "-o", "ppid=", "-p", str(row["lease_pid"])], capture_output=True,
                                        text=True, check=True).stdout
                os.kill(int(server), signal.SIGKILL)  # the server waiting in wse_evaluate, not its worker

    with contextlib.suppress(Exception):  # the client's connection to the killed server breaks
        run(first_connection)
    # The orphaned worker finishes the job it runs, then stops; the job is never lost or run twice.
    job = _settled(engine, state["eid"], state["job_id"])
    assert job.state == "succeeded", job.error
    deadline = time.monotonic() + 30
    while process_exists(state["worker"]) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not process_exists(state["worker"])

    async def second_connection():
        async with connect(workspace, slow["run_id"], trial["log"], worker=True) as session:
            same = await session.call("wse_evaluate", checkpoint_id=base, idempotency_key="first")
            assert (same["result"]["origin"], same["result"]["state"], same["result"]["job_id"]) == (
                "new_job", "succeeded", state["job_id"])
            other = await session.call("wse_evaluate", checkpoint_id=base, idempotency_key="second")
            assert other["result"]["origin"] == "existing_report" and other["result"]["report"] == same["result"]["report"]
            assert other["usage"]["cumulative"]["evaluation_batches"]["consumed"] == 1

    run(second_connection)
    assert [job.job_id for job in engine.list_jobs(state["eid"])] == [state["job_id"]]


def test_the_runner_cancels_a_blocking_evaluation_and_freezes_when_the_agent_stops(engine, trial, workspace,
                                                                                    signed_sum, tmp_path):
    pytest.importorskip("agents", reason="the optional agents-openai extra is not installed")
    from test_agent_adapter import ToolModel

    from wse.agent_adapters.openai_agents import OpenAIAgentsAdapter

    base = trial["task_checkpoint_id"]
    _slow_task(engine, signed_sum, tmp_path / "started", sleep_seconds=60.0)
    run_id = engine.grant_run(SLOW_TASK, trial["policy_id"], base).run_id

    def factory(config, eid):
        return OpenAIAgentsAdapter(model=ToolModel([
            ("call", "wse_get_experiment", {}),
            ("call", "wse_evaluate", {"checkpoint_id": base, "idempotency_key": "slow"}),
            ("say", "done")]))

    runner = ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], worker_args=["--poll-seconds", "0.02"],
                             serve_env={"PYTHONPATH": str(TESTS_DIR)}, adapter_factory=factory,
                             settle_timeout_seconds=60)
    started = time.monotonic()
    eid = runner.run_experiment(run_id, SLOW_TASK, trial["policy_id"],
                                {"kind": "agent", "model": "test-double", "wall_clock_seconds": 4})
    elapsed = time.monotonic() - started
    result = runner.last_result["result"]
    assert result["status"] == "timeout", result
    (job,) = engine.list_jobs(eid)
    assert result["finalization"]["cancelled_jobs"] == [job.job_id]
    assert (job.state, job.usage.evaluation_batches) == ("cancelled", 1)  # the batch it started stays charged
    assert job.error.details["reason"] == "cancelled" and engine.get_experiment(eid).state == "FROZEN"
    # The SDK abandons the call without cancelling the request, so the server's wait ended when the finalization
    # cancelled the job, and the server then closed at once: the job alone would have run for 30 s.
    assert elapsed < 30
    assert "Traceback" not in Path(runner.serve_log(run_id)).read_text(encoding="utf-8")


# -- grants and instructions ---------------------------------------------------------------------------------------


def test_the_tool_is_offered_with_development_evaluation_and_named_in_the_instructions():
    policies = builtin_experiment_policies()
    assert "wse_evaluate" in STRATEGY_TOOLS and "wse_evaluate" in EVALUATION_TOOLS
    for policy_id in (SMALL_MODEL_POLICY_ID, "qwen_loop_v1", "qwen_loop_sandbox_v1"):
        assert "wse_evaluate" in allowed_tools(policies[policy_id]), policy_id
    weights_only = policies["small_model_weights_only_v1"]
    assert "wse_evaluate" not in allowed_tools(weights_only)
    task = TaskDefinition.model_validate(LOOP_TASK)
    qwen = policies["qwen_loop_v1"]
    text = render_instructions(task, qwen, experiment_id="exp_0123", tools=allowed_tools(qwen))
    assert "wse_evaluate evaluates a checkpoint on all of development suite practice" in text
    assert "wse_evaluate" not in render_instructions(task, weights_only, experiment_id="exp_0123",
                                                     tools=allowed_tools(weights_only))
