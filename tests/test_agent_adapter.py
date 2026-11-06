"""The OpenAI Agents SDK adapter, driven by SDK test doubles (Task 7).

Every test runs the SDK's real ``Runner`` and ``Agent`` with a real
``MCPServerStdio`` connection to a real ``wse serve`` subprocess bound to a
run grant; only the model is replaced, by an implementation of
``agents.models.interface.Model`` that returns scripted tool calls. Nothing
reaches the network: outbound IP connections are blocked in this process,
no API key is set, and SDK tracing is disabled (stdio MCP uses pipes).
Each test stops its server and worker processes when the run ends,
including on failure.
"""

from __future__ import annotations

import _socket
import asyncio
import json
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

agents = pytest.importorskip("agents", reason="the optional agents-openai extra is not installed")

from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from agents.usage import Usage  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from test_mcp_e2e import FAULT_DESCRIPTIONS, FINAL_SEED, _final_secrets, _is_oracle, _walk  # noqa: E402

from wse.agent_adapters.base import AgentConfig  # noqa: E402
from wse.agent_adapters.openai_agents import OpenAIAgentsAdapter, parse_tool_output, strategy_model  # noqa: E402
from wse.api import Engine  # noqa: E402
from wse.policy import SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID, ExperimentPolicy  # noqa: E402
from wse.runner import ReferenceRunner, allowed_tools, render_instructions  # noqa: E402
from wse.strategies.scripted import ScriptedSearchStrategy  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
FAST_WORKER = ["--poll-seconds", "0.02"]
W1_ROW = [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 2, "step": 1}]


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
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
    yield attempts
    assert attempts == [], f"network access was attempted: {attempts}"


class ToolModel(Model):
    """A fake SDK model that plays a script of tool calls, messages, and sleeps.

    Each step is ``("call", tool, arguments)`` (``arguments`` may be a
    function of the previous tool output), ``("say", text)``,
    ``("sleep", seconds)``, or ``("interrupt", seconds)`` (SIGINT to this
    process, as from the operator's Ctrl-C, then a sleep). After the script,
    it keeps repeating its last step. It records the instructions and tools
    of every request.
    """

    provider_name = "test-double"

    def __init__(self, script: list[tuple]) -> None:
        self.script = script
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.outputs: list[Any] = []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,  # noqa: A002
                           handoffs, tracing, *, previous_response_id=None, conversation_id=None, prompt=None):
        self.requests.append({"system_instructions": system_instructions,
                              "tools": [{"name": t.name, "description": t.description,
                                         "parameters": t.params_json_schema} for t in tools]})
        outputs = [item for item in (input if isinstance(input, list) else [])
                   if isinstance(item, dict) and item.get("type") == "function_call_output"]
        last = parse_tool_output(outputs[-1]["output"])[3] if outputs else None
        self.outputs.append(last)
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if step[0] in ("sleep", "interrupt"):
            if step[0] == "interrupt":  # stands in for the operator's Ctrl-C while the model is working
                os.kill(os.getpid(), signal.SIGINT)
            await asyncio.sleep(step[1])
            step = ("say", "done")
        if step[0] == "say":
            output = [ResponseOutputMessage(id=f"m{self.calls}", type="message", role="assistant", status="completed",
                                            content=[ResponseOutputText(type="output_text", text=step[1],
                                                                        annotations=[])])]
        else:
            arguments = step[2](last) if callable(step[2]) else step[2]
            output = [ResponseFunctionToolCall(id=f"fc{self.calls}", call_id=f"c{self.calls}", name=step[1],
                                               arguments=json.dumps(arguments), type="function_call",
                                               status="completed")]
        return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=100 + self.calls, output_tokens=7,
                                                        total_tokens=107 + self.calls), response_id=None)

    async def stream_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError
        yield  # pragma: no cover


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


def _agent_run(engine, workspace, prepared, model_for, policy_id=SMALL_MODEL_POLICY_ID, **config):
    """Run one SDK-backed trial; ``model_for(experiment_id)`` builds the fake model."""
    run_id = engine.grant_run(prepared["task_id"], policy_id, prepared["task_checkpoint_id"]).run_id
    made: dict[str, Any] = {}

    def factory(agent_config: AgentConfig, eid: str) -> OpenAIAgentsAdapter:
        made["model"] = model_for(eid)
        made["adapter"] = OpenAIAgentsAdapter(model=made["model"])
        return made["adapter"]

    runner = _runner(workspace, factory)
    eid = runner.run_experiment(run_id, prepared["task_id"], policy_id, {"kind": "agent", "model": "test-double",
                                                                         **config})
    return eid, runner.last_result, made


def _events(engine, eid) -> list[dict[str, Any]]:
    rows = engine.store.db.conn.execute("SELECT kind, payload FROM controller_events WHERE experiment_id = ? "
                                        "ORDER BY seq", (eid,)).fetchall()
    return [{"kind": row["kind"], **json.loads(row["payload"])} for row in rows]


# -- the reference loop through the real SDK runner ------------------------------------------------------------


def test_the_sdk_loop_repairs_through_wse_serve_and_the_runner_settles_and_freezes(engine, workspace, prepared,
                                                                                   monkeypatch):
    from agents import Runner

    run_calls: list[dict[str, Any]] = []
    original = Runner.run

    async def spy(*args, **kwargs):
        run_calls.append(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(Runner, "run", spy)
    eid, record, made = _agent_run(engine, workspace, prepared,
                                   lambda eid: strategy_model(ScriptedSearchStrategy(), experiment_id=eid))
    model = made["model"]
    # max_turns is passed explicitly (the SDK default is 10) and tracing is disabled for the benchmark run.
    assert len(run_calls) == 1 and run_calls[0]["max_turns"] == 100
    assert run_calls[0]["run_config"].tracing_disabled is True
    assert run_calls[0]["run_config"].trace_include_sensitive_data is False
    result = record["result"]
    assert result["status"] == "stopped" and result["stop_condition"].startswith("target_met")
    # SDK completion alone closes nothing: the runner settled and froze; final evaluation is separate.
    experiment = engine.get_experiment(eid)
    assert experiment.state == "FROZEN" and result["finalization"]["state"] == "FROZEN"
    assert not [job for job in engine.list_jobs(eid) if not job.terminal]
    # Normalized usage: token fields come from the fake responses; unreportable fields are null with a reason.
    usage = result["usage"]
    assert usage["requests"] == model.calls == result["turns"]
    assert usage["input_tokens"] > 0 and usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["provider"] == "wse-test-double" and usage["model"] == "test-double"
    assert usage["cost_usd"] is None and usage["unavailable"]["cost_usd"]
    assert usage["model_version"] is None and usage["unavailable"]["model_version"]
    metadata = result["provider_metadata"]
    assert metadata["tracing_disabled"] is True and metadata["max_turns"] == 100
    assert metadata["wall_clock_seconds"] == 900 and metadata["mcp_protocol_version"] == "2026-07-28"
    assert metadata["tools_seen"] == sorted(allowed_tools(engine.get_policy(SMALL_MODEL_POLICY_ID)))
    assert record["tracing"]["sdk_tracing_disabled"] is True and record["constraint_status"] == "verified"
    assert record["kind"] == "agent" and record["controller"] == {"extension_id": "openai_agents", "version": "1"}
    # Normalized events: model responses with usage, tool calls and results, the stop, and the end.
    events = _events(engine, eid)
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished" and "cancel_requested" in kinds
    responses = [event for event in events if event["kind"] == "model_response"]
    assert sum(event["usage"]["input_tokens"] for event in responses) == usage["input_tokens"]
    # A test double reports no provider IDs; the latency of each model call is still measured.
    assert all(e["details"]["request_id"] is None and e["details"]["latency_seconds"] >= 0 for e in responses)
    results = [event for event in events if event["kind"] == "tool_result"]
    assert results and all(event["ok"] and event["output_hash"].startswith("sha256:") for event in results)
    assert [e["tool"] for e in events if e["kind"] == "tool_call"] == [e["tool"] for e in results]
    # The model saw exactly the allow-listed WSE tools and instructions generated from task.json and the policy.
    policy = engine.get_policy(SMALL_MODEL_POLICY_ID)
    task = engine.get_task(prepared["task_id"])
    for request in model.requests:
        assert sorted(tool["name"] for tool in request["tools"]) == sorted(allowed_tools(policy))
        assert request["system_instructions"] == render_instructions(task, policy, experiment_id=eid,
                                                                     tools=allowed_tools(policy))
    # The section 7.1 leak test over everything the model was shown before the answer was found.
    reference = engine.load_manifest(prepared["reference_checkpoint_id"])
    surfaces = [model.requests[0]["system_instructions"], json.dumps(model.requests[0]["tools"])]
    text = "\n".join(surfaces)
    for secret in [reference.checkpoint_id, reference.weights_hash, *_final_secrets(engine, prepared)]:
        assert secret not in text
    assert sorted({m.group(0) for m in FAULT_DESCRIPTIONS.finditer(text)}) == []
    assert not any(_is_oracle(node) for node in _walk(model.requests[0]["tools"]))
    assert engine.final_evaluate(eid).comparison.candidate_value == 1.0


# -- limits and cancellation ----------------------------------------------------------------------------------


def test_the_turn_limit_ends_the_sdk_loop(engine, workspace, prepared):
    looping = lambda eid: ToolModel([("call", "wse_get_experiment", {"experiment_id": eid})])  # noqa: E731
    eid, record, made = _agent_run(engine, workspace, prepared, looping, WEIGHTS_ONLY_POLICY_ID, max_turns=3)
    assert record["result"]["status"] == "max_turns" and made["model"].calls == 3
    assert record["result"]["provider_metadata"]["max_turns"] == 3
    assert engine.get_experiment(eid).state == "FROZEN"


def test_the_wall_clock_limit_and_external_cancellation_stop_the_sdk_loop(engine, workspace, prepared):
    sleeper = lambda eid: ToolModel([("call", "wse_get_experiment", {"experiment_id": eid}),  # noqa: E731
                                     ("sleep", 60)])
    eid, record, _ = _agent_run(engine, workspace, prepared, sleeper, WEIGHTS_ONLY_POLICY_ID, wall_clock_seconds=1.5)
    assert record["result"]["status"] == "timeout" and "wall-clock limit of 1.5 s" in record["result"]["reason"]
    assert record["result"]["elapsed_seconds"] < 30 and engine.get_experiment(eid).state == "FROZEN"

    # cancel() from another thread stops a running loop with the given reason.
    adapter_box: dict[str, Any] = {}

    def model_for(eid):
        model = ToolModel([("call", "wse_get_experiment", {"experiment_id": eid}), ("sleep", 60)])
        timer = threading.Timer(1.0, lambda: adapter_box["adapter"].cancel(eid, "operator request"))
        adapter_box["timer"] = timer
        timer.start()
        return model

    run_id = engine.grant_run(prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, prepared["task_checkpoint_id"]).run_id

    def factory(config, eid):
        adapter_box["adapter"] = OpenAIAgentsAdapter(model=model_for(eid))
        return adapter_box["adapter"]

    runner = _runner(workspace, factory)
    eid = runner.run_experiment(run_id, prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, {"model": "test-double"})
    adapter_box["timer"].join()
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
        return ToolModel([("call", "wse_get_experiment", {"experiment_id": eid}), ("call", "wse_read_tensor", read)])

    eid, record, made = _agent_run(engine, workspace, prepared, reader, tight.policy_id)
    result = record["result"]
    assert result["status"] == "stopped" and result["stop_condition"].startswith("insufficient_budget")
    # The controller did not keep requesting denied work: one successful read, one denied, then the stop.
    assert result["tool_calls"] == 3 and made["model"].calls == 3
    errors = [e["error_code"] for e in _events(engine, eid) if e["kind"] == "tool_result" and not e["ok"]]
    assert errors == ["BUDGET_EXCEEDED"]
    assert engine.get_experiment(eid).budget["tensor_read_calls"].consumed == 1


def test_jobs_unfinished_at_sdk_exit_are_settled_before_freezing(engine, workspace, prepared, signed_sum, tmp_path):
    engine.register_suite({"suite_id": "signed_sum_dev_slow", "task_id": prepared["task_id"], "split": "development",
                           "evaluator": "slow_batches@1", "sample_counts": [64], "seeds": [0],
                           "parameters": {"sleep_seconds": 30, "cooperative": True},
                           "data": signed_sum.suite_data(32, signed_sum.DEV_SEED)})

    def submit_and_leave(eid):
        job = {"experiment_id": eid, "checkpoint_id": prepared["task_checkpoint_id"], "suite_id": "signed_sum_dev_slow",
               "eval_config": {"sample_count": 64, "seed": 0}, "idempotency_key": "slow"}
        return ToolModel([("call", "wse_submit_evaluation", job), ("say", "Leaving the evaluation running.")])

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
    never = lambda eid: ToolModel([("say", "unreachable")])  # noqa: E731
    eid, record, made = _agent_run(engine, workspace, prepared, never, connect_timeout_seconds=20)
    result = record["result"]
    assert result["status"] == "connection_failed" and made["model"].calls == 0
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
    never = lambda eid: ToolModel([("say", "unreachable")])  # noqa: E731
    eid, record, made = _agent_run(engine, workspace, prepared, never, connect_timeout_seconds=20)
    result = record["result"]
    assert result["status"] == "connection_failed" and made["model"].calls == 0
    log = Path(result["serve_log"])
    assert log.name == f"serve-{engine.get_experiment(eid).run_id}.log"
    assert "ModuleNotFoundError: No module named 'wse_no_such_module'" in log.read_text()
    # The reason names the underlying failure, not only an exception group, and where the server's log is.
    assert f"see the server log {log}" in result["reason"]
    assert "unhandled errors in a TaskGroup" not in result["reason"].split(";")[0]
    assert engine.get_experiment(eid).state == "FROZEN"


def test_the_provider_is_the_class_of_the_endpoint_the_sdk_calls():
    from wse.agent_adapters.openai_agents import provider_for_base_url

    # Unset (None) is the SDK client's default, https://api.openai.com/v1; an empty value is used as given.
    for url in (None, "https://api.openai.com/v1", "https://eu.api.openai.com/v1/"):
        assert provider_for_base_url(url) == "openai", url
    for url in ("https://r.openai.azure.com/openai/v1", "https://r.services.ai.azure.com/openai/v1/",
                "https://r.cognitiveservices.azure.com/openai/v1", "HTTPS://R.Services.AI.Azure.com/openai/v1"):
        assert provider_for_base_url(url) == "azure_openai", url
    for url in ("http://127.0.0.1:8000/v1", "https://openrouter.ai/api/v1", "https://r.openai.azure.com.example.org/v1",
                "https://openai.azure.com/openai/v1", "https://api.openai.com.example.org/v1", "not a url", ""):
        assert provider_for_base_url(url) == "openai_compatible", url


def test_a_model_name_run_records_its_endpoint_class_but_not_the_endpoint(engine, workspace, prepared, monkeypatch):
    original = ReferenceRunner.connection

    def broken(self, *args, **kwargs):  # no SDK request is ever made
        connection = original(self, *args, **kwargs)
        return connection.model_copy(update={"serve_command": [sys.executable, "-c", "import sys; sys.exit(3)"]})

    monkeypatch.setattr(ReferenceRunner, "connection", broken)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example-resource.services.ai.azure.com/openai/v1")
    # model=None: the adapter passes the configured model name to the SDK, as in a live trial
    eid, record, _ = _agent_run(engine, workspace, prepared, lambda eid: None, connect_timeout_seconds=20)
    usage = record["result"]["usage"]
    assert record["result"]["status"] == "connection_failed"
    assert usage["provider"] == "azure_openai" and usage["model"] == "test-double"
    assert "example-resource" not in json.dumps(record) + json.dumps(_events(engine, eid))


class _ResponsesStub(BaseHTTPRequestHandler):
    """A loopback stand-in for the Responses API: every request gets a final assistant message.

    HTTP/1.1 keeps each connection open after its response, as real endpoints
    do, so the client's connection pool can reuse it. Each answer comes after
    ``server.delay_seconds`` and carries an ``x-request-id`` header, as OpenAI
    and Azure OpenAI responses do.
    """

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        self.server.requests.append((self.path, request.get("model"), self.headers.get("Authorization")))
        number = len(self.server.requests)
        body = json.dumps({
            "id": f"resp_{number}", "object": "response", "created_at": 0, "status": "completed",
            "model": "stub", "error": None, "incomplete_details": None, "parallel_tool_calls": True,
            "tool_choice": "auto", "tools": [],
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "Finished.", "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        }).encode()
        time.sleep(self.server.delay_seconds)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("x-request-id", f"req_{number}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:  # noqa: A002 - keep the test output quiet
        pass


@pytest.fixture
def responses_endpoint(offline, monkeypatch):
    """Serve ``_ResponsesStub`` on loopback, the one address the offline guard lets the test reach."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ResponsesStub)
    server.block_on_close = False  # a connection the client keeps open must not hold up shutdown
    server.requests = []
    server.delay_seconds = 0.0
    address = server.server_address[:2]
    guarded = socket.socket.connect

    def connect(self, target, *args, **kwargs):
        if self.family == socket.AF_INET and tuple(target[:2]) == address:
            return _socket.socket.connect(self, target, *args, **kwargs)
        return guarded(self, target, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://{address[0]}:{address[1]}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_consecutive_model_name_trials_in_one_process_each_reach_the_endpoint(engine, workspace, prepared,
                                                                               responses_endpoint):
    # run_trials.py runs every trial in its own event loop, all in one process. A model client shared between
    # trials would reuse a pooled connection of an earlier, closed loop ("RuntimeError: Event loop is closed").
    results = [_agent_run(engine, workspace, prepared, lambda eid: None)[1]["result"] for _ in range(3)]
    assert [result["status"] for result in results] == ["completed"] * 3, [result["reason"] for result in results]
    assert [result["usage"]["input_tokens"] for result in results] == [10, 10, 10]
    # Each trial's own client calls the endpoint as the SDK's default client does: OPENAI_BASE_URL, the
    # configured model name, and OPENAI_API_KEY as a bearer token.
    assert responses_endpoint.requests == [("/v1/responses", "test-double", "Bearer test-only-key")] * 3


def test_each_model_response_records_its_latency_and_the_provider_ids(engine, workspace, prepared,
                                                                      responses_endpoint):
    # A request's latency and the provider's response and request IDs locate a slow or failed request in the
    # provider's own logs; the latency spans the SDK's whole model call, including the client's retries.
    responses_endpoint.delay_seconds = 0.2
    eid, last, _ = _agent_run(engine, workspace, prepared, lambda eid: None)
    assert last["result"]["status"] == "completed", last["result"]["reason"]
    responses = [event for event in _events(engine, eid) if event["kind"] == "model_response"]
    assert [(e["details"]["response_id"], e["details"]["request_id"]) for e in responses] == [("resp_1", "req_1")]
    assert all(0.2 <= e["details"]["latency_seconds"] < 30 for e in responses)


def test_an_operator_interrupt_cancels_the_sdk_loop_and_the_trial_is_still_frozen(engine, workspace, prepared):
    run_id = engine.grant_run(prepared["task_id"], SMALL_MODEL_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    made: dict[str, Any] = {}

    def factory(config, eid):
        made["model"] = ToolModel([("call", "wse_get_experiment", {"experiment_id": eid}), ("interrupt", 60)])
        made["adapter"] = OpenAIAgentsAdapter(model=made["model"])
        return made["adapter"]

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
    assert result["finalization"]["state"] == "FROZEN" and made["model"].calls == 2


def test_sdk_completion_alone_does_not_freeze_or_close_an_experiment(engine, workspace, prepared):
    run_id = engine.grant_run(prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    with Engine(workspace, run_id=run_id) as bound:
        eid = bound.create_experiment(run_id, prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, "c").experiment_id
    runner = _runner(workspace, None)
    connection = runner.connection(run_id, eid, engine.get_task(prepared["task_id"]),
                                   engine.get_policy(WEIGHTS_ONLY_POLICY_ID), "verified")
    model = ToolModel([("call", "wse_get_experiment", {"experiment_id": eid}), ("say", "Finished.")])
    adapter = OpenAIAgentsAdapter(model=model)
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
    assert [event.kind for event in events] == ["run_started", "model_response", "tool_call", "tool_result",
                                                "model_response", "message", "run_finished"]
    assert engine.get_experiment(eid).state == "OPEN"  # only the runner freezes, and only the operator closes


def test_the_network_guard_blocks_outbound_connections(offline):
    with pytest.raises(OSError, match="network access is blocked"):
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    assert offline == [("127.0.0.1", 9)]
    offline.clear()
