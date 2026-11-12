"""The agent-adapter contract through the reference runner, with the Codex adapter and its offline stand-in.

Every test runs a real ``wse serve`` subprocess bound to a run grant, as
Codex's MCP server, through ``CodexCLIAdapter`` and its process handling;
only Codex itself is replaced by ``wse.agent_adapters.codex_test_double``,
which plays a registered strategy or a fixed script of tool calls,
messages, and sleeps (``scripted``). Nothing reaches a model or the network:
outbound IP connections are blocked in this process and no API key is set.
Each test stops its server and worker processes when the run ends,
including on failure.
"""

from __future__ import annotations

import asyncio
import json
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from test_mcp_e2e import FAULT_DESCRIPTIONS, FINAL_SEED, _final_secrets, _is_oracle, _walk

from wse.agent_adapters.base import AgentConfig
from wse.agent_adapters.codex_cli import TEST_DOUBLE_MODULE, CodexCLIAdapter
from wse.api import Engine
from wse.policy import SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID, ExperimentPolicy
from wse.runner import ReferenceRunner, allowed_tools
from wse.strategies.scripted import ScriptedSearchStrategy

TESTS_DIR = Path(__file__).resolve().parent
FAST_WORKER = ["--poll-seconds", "0.02"]
W1_ROW = [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 2, "step": 1}]
DOUBLE = "wse-test-double:scripted_search@1"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Block outbound IP connections in this process and remove credentials; record every attempt."""
    attempts: list[Any] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guard(method):
        def blocked(self, address, *args, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                attempts.append(address)
                raise OSError(f"network access is blocked in this test: {address!r}")
            return method(self, address, *args, **kwargs)
        return blocked

    monkeypatch.setattr(socket.socket, "connect", guard(original_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(original_connect_ex))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield attempts
    assert attempts == [], f"network access was attempted: {attempts}"


def scripted(script: list[tuple], *, latency: float = 0.0, **options: Any) -> CodexCLIAdapter:
    """The Codex adapter with its stand-in playing ``script`` (steps as in ``CodexCLIAdapter.scripted``).

    ``latency`` delays each model request, so a stop that the runner requests
    after a tool result lands before the next request.
    """
    behavior = {"request_seconds": latency} if latency else {}
    return CodexCLIAdapter.scripted(script, extra_env={"WSE_CODEX_TEST_DOUBLE": json.dumps(behavior)},
                                    interrupt_grace_seconds=5.0, terminate_grace_seconds=2.0,
                                    orphan_grace_seconds=3.0, **options)


def codex_dir(workspace: Path, run_id: str) -> Path:
    return workspace / "logs" / f"codex-{run_id}"


def shown_to_model(workspace: Path, run_id: str) -> list[dict[str, Any]]:
    """What the stand-in recorded as shown to the model, one entry per segment: the prompt and the offered tools."""
    records = sorted((codex_dir(workspace, run_id) / "home" / "test-double").glob("*.requests.jsonl"))
    return [json.loads(line) for path in records for line in path.read_text().splitlines()]


def segment_items(result: dict[str, Any], item_type: str) -> list[dict[str, Any]]:
    """The completed items of one type in the raw Codex event streams of a run, in order."""
    items = []
    for segment in result["provider_metadata"]["segments"]:
        for line in Path(segment["log"]).read_text().splitlines():
            event = json.loads(line)
            if event.get("type") == "item.completed" and event["item"].get("type") == item_type:
                items.append(event["item"])
    return items


@pytest.fixture
def prepared(engine, signed_sum, tmp_path, plugins):
    plugins.setup(engine.registry)
    return signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED, with_reference=True)


@pytest.fixture
def plugins():
    import wse_test_plugins

    return wse_test_plugins


def _runner(workspace: Path, factory) -> ReferenceRunner:
    return ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], worker_args=FAST_WORKER,
                           serve_env={"PYTHONPATH": str(TESTS_DIR)}, adapter_factory=factory,
                           settle_timeout_seconds=60)


def _agent_run(engine, workspace, prepared, adapter_for, policy_id=SMALL_MODEL_POLICY_ID, model="test-double",
               **config):
    """Run one agent trial; ``adapter_for(experiment_id)`` builds the adapter."""
    run_id = engine.grant_run(prepared["task_id"], policy_id, prepared["task_checkpoint_id"]).run_id
    made: dict[str, Any] = {}

    def factory(agent_config: AgentConfig, eid: str) -> CodexCLIAdapter:
        made["adapter"] = adapter_for(eid)
        return made["adapter"]

    runner = _runner(workspace, factory)
    eid = runner.run_experiment(run_id, prepared["task_id"], policy_id, {"kind": "agent", "model": model, **config})
    return eid, runner.last_result, run_id


def _events(engine, eid) -> list[dict[str, Any]]:
    rows = engine.store.db.conn.execute("SELECT kind, payload FROM controller_events WHERE experiment_id = ? "
                                        "ORDER BY seq", (eid,)).fetchall()
    return [{"kind": row["kind"], **json.loads(row["payload"])} for row in rows]


# -- the reference loop -------------------------------------------------------------------------------------------


def test_the_agent_loop_repairs_through_wse_serve_and_the_runner_settles_and_freezes(engine, workspace, prepared):
    eid, record, run_id = _agent_run(engine, workspace, prepared,
                                     lambda eid: CodexCLIAdapter.test_double(ScriptedSearchStrategy()), model=DOUBLE)
    result = record["result"]
    assert result["status"] == "stopped" and result["stop_condition"].startswith("target_met")
    # Completion of the agent's loop closes nothing: the runner settled and froze; final evaluation is separate.
    experiment = engine.get_experiment(eid)
    assert experiment.state == "FROZEN" and result["finalization"]["state"] == "FROZEN"
    assert not [job for job in engine.list_jobs(eid) if not job.terminal]
    # Normalized usage: tokens per model request; what Codex cannot report is null with a reason.
    usage = result["usage"]
    responses = [event for event in _events(engine, eid) if event["kind"] == "model_response"]
    assert usage["requests"] == len(responses) == result["turns"] > 0
    assert usage["input_tokens"] > 0 and usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["input_tokens"] == sum(event["usage"]["input_tokens"] for event in responses)
    assert usage["provider"] == "wse-test-double" and usage["model"] == DOUBLE
    assert usage["cost_usd"] is None and usage["unavailable"]["cost_usd"]
    assert usage["model_version"] is None and usage["unavailable"]["model_version"]
    metadata = result["provider_metadata"]
    assert metadata["max_turns"] == 100 and metadata["wall_clock_seconds"] == 900
    assert metadata["allowed_tools"] == allowed_tools(engine.get_policy(SMALL_MODEL_POLICY_ID))
    assert record["kind"] == "agent" and record["controller"] == {"extension_id": "codex_cli", "version": "1"}
    assert record["config"]["test_double"] == {"strategy": "scripted_search@1",
                                               "resolved_config": ScriptedSearchStrategy().config()}
    # Normalized events: the run's start, tool calls paired with results, the stop, and the end.
    events = _events(engine, eid)
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished" and "cancel_requested" in kinds
    calls = [event for event in events if event["kind"] == "tool_call"]
    results = [event for event in events if event["kind"] == "tool_result"]
    assert results and all(event["ok"] and event["output_hash"].startswith("sha256:") for event in results)
    assert [(e["tool"], e["call_id"]) for e in calls] == [(e["tool"], e["call_id"]) for e in results]
    assert calls[0]["tool"] == "wse_get_experiment"
    # The model was shown exactly the allow-listed WSE tools and instructions generated from task.json and the
    # policy, with the run's and the experiment's IDs.
    policy = engine.get_policy(SMALL_MODEL_POLICY_ID)
    task = engine.get_task(prepared["task_id"])
    shown = shown_to_model(workspace, run_id)
    assert shown and all(sorted(tool["name"] for tool in entry["tools"]) == sorted(allowed_tools(policy))
                         for entry in shown)
    prompt = shown[0]["prompt"]
    assert f"Experiment ID: {eid}" in prompt and task.description in prompt and run_id in prompt
    assert "Available tools: " + ", ".join(allowed_tools(policy)) + "." in prompt
    # The leak test of M1 Section 7.1 over everything the model was shown before the answer was found.
    reference = engine.load_manifest(prepared["reference_checkpoint_id"])
    text = "\n".join([prompt, json.dumps(shown[0]["tools"])])
    for secret in [reference.checkpoint_id, reference.weights_hash, *_final_secrets(engine, prepared)]:
        assert secret not in text
    assert sorted({m.group(0) for m in FAULT_DESCRIPTIONS.finditer(text)}) == []
    assert not any(_is_oracle(node) for node in _walk(shown[0]["tools"]))
    assert engine.final_evaluate(eid).comparison.candidate_value == 1.0


# -- limits and cancellation ----------------------------------------------------------------------------------


def test_the_turn_limit_ends_the_agent_loop(engine, workspace, prepared):
    looping = lambda eid: scripted([("call", "wse_get_experiment", {"experiment_id": eid})],  # noqa: E731
                                   latency=1.0)
    eid, record, _ = _agent_run(engine, workspace, prepared, looping, WEIGHTS_ONLY_POLICY_ID, max_turns=3)
    assert record["result"]["status"] == "max_turns" and record["result"]["turns"] == 3
    assert record["result"]["provider_metadata"]["max_turns"] == 3
    assert engine.get_experiment(eid).state == "FROZEN"


def test_the_wall_clock_limit_and_external_cancellation_stop_the_agent_loop(engine, workspace, prepared):
    sleeper = lambda eid: scripted([("call", "wse_get_experiment", {"experiment_id": eid}),  # noqa: E731
                                    ("sleep", 60)])
    started = time.monotonic()
    eid, record, _ = _agent_run(engine, workspace, prepared, sleeper, WEIGHTS_ONLY_POLICY_ID, wall_clock_seconds=1.5)
    assert record["result"]["status"] == "timeout" and "wall-clock limit of 1.5 s" in record["result"]["reason"]
    assert time.monotonic() - started < 30 and engine.get_experiment(eid).state == "FROZEN"

    # cancel() from another thread stops a running loop with the given reason.
    box: dict[str, Any] = {}
    run_id = engine.grant_run(prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, prepared["task_checkpoint_id"]).run_id

    def factory(config, eid):
        box["adapter"] = scripted([("call", "wse_get_experiment", {"experiment_id": eid}), ("sleep", 60)])
        box["timer"] = threading.Timer(3.0, lambda: box["adapter"].cancel(eid, "operator request"))
        box["timer"].start()
        return box["adapter"]

    runner = _runner(workspace, factory)
    eid = runner.run_experiment(run_id, prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, {"model": "test-double"})
    box["timer"].join()
    result = runner.last_result["result"]
    assert result["status"] == "stopped" and result["reason"] == "operator request"
    assert engine.get_experiment(eid).state == "FROZEN"


def test_a_budget_error_terminates_the_controller(engine, workspace, prepared):
    tight = ExperimentPolicy.model_validate({"policy_id": "small_model_reads_v1", "mode": "bounded_eval",
                                             "budget": {"tensor_read_calls": 1}})
    engine.register_policy(tight)

    def reader(eid):
        read = {"experiment_id": eid, "checkpoint_id": prepared["task_checkpoint_id"], "tensor": "layers.0.weight",
                "selection": W1_ROW}
        return scripted([("call", "wse_get_experiment", {"experiment_id": eid}), ("call", "wse_read_tensor", read)],
                        latency=0.5)

    eid, record, _ = _agent_run(engine, workspace, prepared, reader, tight.policy_id)
    result = record["result"]
    assert result["status"] == "stopped" and result["stop_condition"].startswith("insufficient_budget")
    # The controller did not keep requesting denied work: one successful read, one denied, then the stop.
    assert result["tool_calls"] == 3 and result["turns"] == 3
    errors = [e["error_code"] for e in _events(engine, eid) if e["kind"] == "tool_result" and not e["ok"]]
    assert errors == ["BUDGET_EXCEEDED"]
    assert engine.get_experiment(eid).budget["tensor_read_calls"].consumed == 1


def test_jobs_unfinished_when_the_agent_stops_are_settled_before_freezing(engine, workspace, prepared, signed_sum):
    engine.register_suite({"suite_id": "signed_sum_dev_slow", "task_id": prepared["task_id"], "split": "development",
                           "evaluator": "slow_batches@1", "sample_counts": [64], "seeds": [0],
                           "parameters": {"sleep_seconds": 30, "cooperative": True},
                           "data": signed_sum.suite_data(32, signed_sum.DEV_SEED)})

    def submit_and_leave(eid):
        job = {"experiment_id": eid, "checkpoint_id": prepared["task_checkpoint_id"], "suite_id": "signed_sum_dev_slow",
               "eval_config": {"sample_count": 64, "seed": 0}, "idempotency_key": "slow"}
        return scripted([("call", "wse_submit_evaluation", job), ("say", "Leaving the evaluation running.")])

    eid, record, _ = _agent_run(engine, workspace, prepared, submit_and_leave)
    result = record["result"]
    assert result["status"] == "completed" and result["final_message"] == "Leaving the evaluation running."
    (job,) = engine.list_jobs(eid)
    assert result["finalization"]["cancelled_jobs"] == [job.job_id]
    assert job.state == "cancelled" and job.reservation.state == "SETTLED"
    experiment = engine.get_experiment(eid)
    assert experiment.state == "FROZEN" and all(entry.reserved == 0 for entry in experiment.budget.values())


def test_a_connection_failure_is_reported_and_the_experiment_is_still_finalized(engine, workspace, prepared,
                                                                                monkeypatch):
    original = ReferenceRunner.connection

    def broken(self, *args, **kwargs):
        connection = original(self, *args, **kwargs)
        return connection.model_copy(update={"serve_command": [sys.executable, "-c", "import sys; sys.exit(3)"]})

    monkeypatch.setattr(ReferenceRunner, "connection", broken)
    never = lambda eid: scripted([("say", "unreachable")])  # noqa: E731
    eid, record, _ = _agent_run(engine, workspace, prepared, never, connect_timeout_seconds=20)
    result = record["result"]
    assert result["status"] == "connection_failed" and result["turns"] == 0
    assert result["usage"]["input_tokens"] is None and "no model request" in result["usage"]["unavailable"][
        "input_tokens"]
    experiment = engine.get_experiment(eid)
    assert experiment.state == "FROZEN" and experiment.selected_checkpoint_id == experiment.base_checkpoint_id


def test_a_connection_failure_names_its_cause_and_points_to_the_server_log(engine, workspace, prepared, monkeypatch):
    original = ReferenceRunner.connection

    def unloadable(self, *args, **kwargs):  # the server exits at startup: its setup hook cannot be imported
        connection = original(self, *args, **kwargs)
        return connection.model_copy(update={"serve_command": [*connection.serve_command, "--setup",
                                                               "wse_no_such_module:setup"]})

    monkeypatch.setattr(ReferenceRunner, "connection", unloadable)
    never = lambda eid: scripted([("say", "unreachable")])  # noqa: E731
    eid, record, _ = _agent_run(engine, workspace, prepared, never, connect_timeout_seconds=20)
    result = record["result"]
    assert result["status"] == "connection_failed" and result["turns"] == 0
    log = Path(result["serve_log"])
    assert log.name == f"serve-{engine.get_experiment(eid).run_id}.log"
    assert "ModuleNotFoundError: No module named 'wse_no_such_module'" in log.read_text()
    # The reason names the underlying failure (the server did not start in Codex) and where the server's log is.
    assert "MCP servers failed to initialize" in result["reason"]
    assert f"see the server log {log}" in result["reason"]
    assert engine.get_experiment(eid).state == "FROZEN"


def test_a_model_name_run_records_its_provider_but_not_the_endpoint(engine, workspace, prepared, monkeypatch):
    original = ReferenceRunner.connection

    def broken(self, *args, **kwargs):  # no model request is ever made
        connection = original(self, *args, **kwargs)
        return connection.model_copy(update={"serve_command": [sys.executable, "-c", "import sys; sys.exit(3)"]})

    monkeypatch.setattr(ReferenceRunner, "connection", broken)
    endpoint = "https://example-resource.services.ai.azure.com/openai/v1"
    provider = {"model_provider": "wse_proxy", "model_providers": {"wse_proxy": {
        "name": "WSE model proxy", "base_url": endpoint, "env_key": "WSE_MODEL_PROXY_TOKEN", "wire_api": "responses"}}}
    adapter_for = lambda eid: CodexCLIAdapter(  # noqa: E731 - the model name goes to Codex as in a live trial
        codex_command=(sys.executable, "-m", TEST_DOUBLE_MODULE), provider_config=provider, proxy_token="t" * 32)
    eid, record, _ = _agent_run(engine, workspace, prepared, adapter_for, model="gpt-5.6-luna",
                                connect_timeout_seconds=20)
    usage = record["result"]["usage"]
    assert record["result"]["status"] == "connection_failed"
    assert usage["provider"] == "wse_proxy" and usage["model"] == "gpt-5.6-luna"
    assert "example-resource" not in json.dumps(record) + json.dumps(_events(engine, eid))


def test_an_operator_interrupt_cancels_the_agent_loop_and_the_trial_is_still_frozen(engine, workspace, prepared):
    run_id = engine.grant_run(prepared["task_id"], SMALL_MODEL_POLICY_ID, prepared["task_checkpoint_id"]).run_id

    def factory(config, eid):  # the stand-in sends SIGINT to this process, as the operator's Ctrl-C would
        return scripted([("call", "wse_get_experiment", {"experiment_id": eid}), ("interrupt", 60)])

    runner = _runner(workspace, factory)
    handler = signal.getsignal(signal.SIGINT)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):  # the interrupt reaches the caller after finalization
        runner.run_experiment(run_id, prepared["task_id"], SMALL_MODEL_POLICY_ID, {"model": "test-double"})
    assert time.monotonic() - started < 45  # the model's 60 s sleep was cancelled
    assert signal.getsignal(signal.SIGINT) is handler
    row = engine.store.db.conn.execute("SELECT experiment_id FROM experiments WHERE run_id = ?", (run_id,)).fetchone()
    experiment = engine.get_experiment(row["experiment_id"])
    assert experiment.state == "FROZEN" and not [job for job in engine.list_jobs(experiment.experiment_id)
                                                 if not job.terminal]
    result = runner.last_result["result"]
    assert result["status"] == "cancelled" and result["stop_condition"].startswith("operator_interrupt")
    assert result["finalization"]["state"] == "FROZEN" and result["turns"] == 2


def test_completion_of_the_agent_loop_alone_does_not_freeze_or_close_an_experiment(engine, workspace, prepared):
    run_id = engine.grant_run(prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    with Engine(workspace, run_id=run_id) as bound:
        eid = bound.create_experiment(run_id, prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, "c").experiment_id
    runner = _runner(workspace, None)
    connection = runner.connection(run_id, eid, engine.get_task(prepared["task_id"]),
                                   engine.get_policy(WEIGHTS_ONLY_POLICY_ID), "verified")
    adapter = scripted([("call", "wse_get_experiment", {"experiment_id": eid}), ("say", "Finished.")])
    config = AgentConfig(model="test-double")
    events = []

    async def body():
        await adapter.start(connection, config)
        try:
            return await adapter.run(eid, config, events.append)
        finally:
            await adapter.close(eid)

    result = asyncio.run(body())
    assert result.status == "completed" and result.final_message == "Finished." and result.tool_calls == 1
    kinds = [event.kind for event in events]
    assert [kind for kind in kinds if kind != "model_response"] == ["run_started", "tool_call", "tool_result",
                                                                    "message", "run_finished"]
    assert kinds.count("model_response") == 2 == result.turns
    assert engine.get_experiment(eid).state == "OPEN"  # only the runner freezes, and only the operator closes


def test_the_network_guard_blocks_outbound_connections(offline):
    with pytest.raises(OSError, match="network access is blocked"):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    assert offline == [("127.0.0.1", 9)]
    offline.clear()

