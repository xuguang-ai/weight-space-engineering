"""The Codex CLI adapter (``codex_cli@1``) offline: configuration, skill, events, resume, limits, and full runs.

Runs here use the Codex stand-in (``wse.agent_adapters.codex_test_double``)
with a real ``wse serve`` subprocess bound to a run grant: a registered
strategy chooses every tool call, so these are toolchain checks, not agent
runs. Nothing reaches a model or the network. Tests that need the real
Codex binary are in ``test_codex_isolation.py``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

from wse.agent_adapters import codex_cli
from wse.agent_adapters.base import AgentConfig, AgentEvent, RunConnection
from wse.agent_adapters.codex_cli import (
    MCP_NAMESPACE,
    PROXY_TOKEN_ENV,
    CodexCLIAdapter,
    _Recorder,
    _RolloutTail,
    _Totals,
    _Translator,
    codex_config,
    find_codex,
    parse_mcp_result,
    python_access,
    skill_text,
    toml_dumps,
)
from wse.api import Engine
from wse.policy import SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID
from wse.runner import AGENT_ADAPTERS, ReferenceRunner, agent_session_timeout, allowed_tools
from wse.strategies.scripted import ScriptedSearchStrategy

TESTS_DIR = Path(__file__).resolve().parent
FAST_WORKER = ["--poll-seconds", "0.02"]
FINAL_SEED = 5_550_199_223_310_017
TOKEN = "wse-proxy-token-for-tests-7f3a9c2e51d84b60"
DOUBLE = "wse-test-double:scripted_search@1"


@pytest.fixture
def plugins():
    import wse_test_plugins

    return wse_test_plugins


@pytest.fixture
def prepared(engine, signed_sum, tmp_path, plugins):
    plugins.setup(engine.registry)
    return signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED, with_reference=True)


def _runner(workspace: Path, factory, **options: Any) -> ReferenceRunner:
    return ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], worker_args=FAST_WORKER,
                           serve_env={"PYTHONPATH": str(TESTS_DIR)}, adapter_factory=factory,
                           settle_timeout_seconds=60, **options)


def _codex_run(engine, workspace, prepared, *, behavior: dict[str, Any] | None = None,
               policy_id: str = SMALL_MODEL_POLICY_ID, adapter_options: dict[str, Any] | None = None,
               **config: Any) -> tuple[str, dict[str, Any], CodexCLIAdapter, str]:
    """One run of the Codex stand-in for ``scripted_search@1`` through the reference runner."""
    run_id = engine.grant_run(prepared["task_id"], policy_id, prepared["task_checkpoint_id"]).run_id
    made: dict[str, Any] = {}
    options = {"interrupt_grace_seconds": 5.0, "terminate_grace_seconds": 2.0, "orphan_grace_seconds": 3.0,
               **(adapter_options or {})}
    if behavior:
        options["extra_env"] = {"WSE_CODEX_TEST_DOUBLE": json.dumps(behavior)}

    def factory(agent_config: AgentConfig, eid: str) -> CodexCLIAdapter:
        made["adapter"] = CodexCLIAdapter.test_double(ScriptedSearchStrategy(), **options)
        return made["adapter"]

    runner = _runner(workspace, factory)
    eid = runner.run_experiment(run_id, prepared["task_id"], policy_id,
                                {"kind": "agent", "adapter": "codex_cli@1", "model": DOUBLE, **config})
    return eid, runner.last_result, made["adapter"], run_id


def _events(engine, eid) -> list[dict[str, Any]]:
    rows = engine.store.db.conn.execute("SELECT kind, payload FROM controller_events WHERE experiment_id = ? "
                                        "ORDER BY seq", (eid,)).fetchall()
    return [{"kind": row["kind"], **json.loads(row["payload"])} for row in rows]


def _run_processes(run_id: str) -> list[str]:
    """Live processes whose command line names this run (its server, its worker, or a stand-in)."""
    table = subprocess.run(["/bin/ps", "-A", "-o", "pid=,command="], capture_output=True, text=True).stdout
    return [line for line in table.splitlines() if run_id in line and "/bin/ps" not in line]


def _codex_dir(workspace: Path, run_id: str) -> Path:
    return workspace / "logs" / f"codex-{run_id}"


# -- a full offline run -------------------------------------------------------------------------------------------


def test_a_full_offline_run_through_wse_serve_freezes_the_expected_candidate(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared)
    result = record["result"]
    # The runner's stop monitor found the target and stopped Codex; the runner settled and froze.
    assert result["status"] == "stopped" and result["stop_condition"].startswith("target_met")
    experiment = engine.get_experiment(eid)
    assert experiment.state == "FROZEN" and result["finalization"]["state"] == "FROZEN"
    assert experiment.selected_checkpoint_id != experiment.base_checkpoint_id
    assert engine.final_evaluate(eid).comparison.candidate_value == 1.0
    assert record["kind"] == "agent" and record["controller"] == {"extension_id": "codex_cli", "version": "1"}
    assert record["config"]["test_double"] == {"strategy": "scripted_search@1",
                                               "resolved_config": ScriptedSearchStrategy().config()}
    # Normalized events: the run's start, tool calls paired with results, per-request usage, the stop, the end.
    events = _events(engine, eid)
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished" and "cancel_requested" in kinds
    calls = [event for event in events if event["kind"] == "tool_call"]
    results = [event for event in events if event["kind"] == "tool_result"]
    assert calls[0]["tool"] == "wse_get_experiment" and [e["call_id"] for e in calls] == [e["call_id"] for e in results]
    assert all(event["ok"] and event["output_hash"].startswith("sha256:") for event in results)
    assert {event["tool"] for event in calls} <= set(allowed_tools(engine.get_policy(SMALL_MODEL_POLICY_ID)))
    responses = [event for event in events if event["kind"] == "model_response"]
    usage = result["usage"]
    assert usage["requests"] == len(responses) == result["turns"] > 0
    assert usage["input_tokens"] == sum(event["usage"]["input_tokens"] for event in responses) > 0
    assert usage["cached_input_tokens"] == 0 and usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["provider"] == "wse-test-double" and usage["model"] == DOUBLE
    assert usage["cost_usd"] is None and usage["unavailable"]["cost_usd"]
    # One segment, stopped with SIGINT; the record names the session, the logs, and the configuration.
    metadata = result["provider_metadata"]
    (segment,) = metadata["segments"]
    assert segment["kind"] == "initial" and segment["signals"] == ["SIGINT"] and segment["session_id"]
    assert result["conversation_ref"] == segment["session_id"] and metadata["test_double"] is True
    assert metadata["tool_call_counts"]["wse"] == len(results) and metadata["resumes"] == 0
    assert metadata["permission_profile"] == "wse"
    # Codex lets one WSE call run for the run's wall-clock limit, at least the policy's longest call.
    longest = agent_session_timeout(engine.get_policy(SMALL_MODEL_POLICY_ID), 30.0)
    assert metadata["tool_timeout_sec"] >= max(900, longest)
    written = tomllib.loads((_codex_dir(workspace, run_id) / "home" / "config.toml").read_text())
    assert written["mcp_servers"]["wse"]["tool_timeout_sec"] == metadata["tool_timeout_sec"]
    assert written["mcp_servers"]["wse"]["args"][-1] == "--no-worker" and metadata["worker"] == "adapter"
    assert not [line for line in subprocess.run(["/bin/ps", "-A", "-o", "command="], capture_output=True,
                                                text=True).stdout.splitlines()
                if "wse.worker" in line and str(workspace) in line]  # the adapter stopped its worker
    log = Path(segment["log"])
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines[0]["type"] == "thread.started" and any(line["type"] == "item.completed" for line in lines)
    # Nothing outlives the run: no process names the run any more.
    assert _run_processes(run_id) == []


def test_the_runner_selects_the_stand_in_by_model_name(engine, workspace, prepared):
    """``AgentConfig(adapter="codex_cli@1", model="wse-test-double:<strategy>")`` needs no adapter factory."""
    assert AGENT_ADAPTERS["codex_cli@1"] == ("wse.agent_adapters.codex_cli", "CodexCLIAdapter")
    run_id = engine.grant_run(prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    runner = ReferenceRunner(workspace, setup=["wse_test_plugins:setup"], worker_args=FAST_WORKER,
                             serve_env={"PYTHONPATH": str(TESTS_DIR)}, settle_timeout_seconds=60)
    eid = runner.run_experiment(run_id, prepared["task_id"], WEIGHTS_ONLY_POLICY_ID,
                                {"kind": "agent", "adapter": "codex_cli@1", "model": DOUBLE})
    record = runner.last_result
    assert record["controller"] == {"extension_id": "codex_cli", "version": "1"}
    assert record["config"]["test_double"]["strategy"] == "scripted_search@1"
    assert record["result"]["status"] in ("stopped", "completed")
    assert engine.get_experiment(eid).state == "FROZEN"


def test_a_deliberate_stop_ends_the_run_without_a_resume(engine, workspace, prepared):
    run_id = engine.grant_run(prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    runner = ReferenceRunner(workspace, setup=["codex_test_plugins:setup"], worker_args=FAST_WORKER,
                             serve_env={"PYTHONPATH": str(TESTS_DIR)}, settle_timeout_seconds=60,
                             adapter_factory=lambda config, eid: CodexCLIAdapter.test_double(None))
    eid = runner.run_experiment(run_id, prepared["task_id"], WEIGHTS_ONLY_POLICY_ID,
                                {"kind": "agent", "adapter": "codex_cli@1", "model": "wse-test-double:stop_at_once@1"})
    result = runner.last_result["result"]
    assert result["status"] == "completed" and result["reason"] == ("the agent stopped: WSE-STOP: nothing is worth "
                                                                    "the budget")
    assert result["final_message"].startswith("WSE-STOP:") and len(result["provider_metadata"]["segments"]) == 1
    assert engine.get_experiment(eid).state == "FROZEN"


# -- configuration and the skill ------------------------------------------------------------------------------------


def _connection(workspace: Path, tools: list[str] | None = None) -> RunConnection:
    return RunConnection(workspace=str(workspace), run_id="run_cfg", experiment_id="exp_cfg", task_id="task",
                         policy_id="policy", serve_command=[sys.executable, "-m", "wse.cli", "--workspace",
                                                            str(workspace), "serve", "--run-id", "run_cfg"],
                         serve_env={"PATH": "/usr/bin:/bin", "HOME": "/home/x", "OPENAI_AGENTS_DISABLE_TRACING": "1"},
                         cwd=str(workspace), allowed_tools=tools or ["wse_get_experiment", "wse_preview_patch"],
                         instructions="Instructions.", initial_message="Begin work on experiment exp_cfg.",
                         constraint_status="unverified")


def test_the_configuration_isolates_commands_and_round_trips_as_toml(tmp_path):
    sandbox = tmp_path / "workspace" / "sandboxes" / "run_cfg"
    python = python_access(tmp_path / "pythonpath")
    provider = {"model_provider": "wse_proxy", "model_providers": {"wse_proxy": {
        "name": "WSE model proxy", "base_url": "http://127.0.0.1:9/v1", "env_key": PROXY_TOKEN_ENV,
        "wire_api": "responses"}}}
    config = codex_config(model="gpt-5.6-luna", connection=_connection(tmp_path / "workspace"), sandbox_root=sandbox,
                          skill_dir=tmp_path / "home" / "skills" / "wse", python=python, provider_config=provider,
                          startup_timeout_seconds=60, tool_timeout_seconds=7200.5,
                          model_settings={"reasoning_effort": "high"})
    text = toml_dumps(config)
    assert tomllib.loads(text) == config and TOKEN not in text
    assert config["model"] == "gpt-5.6-luna" and config["model_reasoning_effort"] == "high"
    assert config["approval_policy"] == "never" and config["default_permissions"] == "wse"
    assert config["allow_login_shell"] is False
    assert config["project_doc_max_bytes"] == 0 and config["project_root_markers"] == []
    assert config["model_provider"] == "wse_proxy"
    assert config["model_providers"]["wse_proxy"]["env_key"] == PROXY_TOKEN_ENV
    # Commands: an explicit environment (nothing inherited) and a filesystem allow-list without network.
    policy = config["shell_environment_policy"]
    assert policy["inherit"] == "none" and PROXY_TOKEN_ENV not in policy["set"]
    assert policy["set"]["PATH"] == f"{python.bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    assert policy["set"]["HOME"] == str(sandbox / "work") and policy["set"]["PYTHONPATH"] == python.pythonpath
    profile = config["permissions"]["wse"]
    assert profile["network"] == {"enabled": False}
    rules = profile["filesystem"]
    assert rules[":minimal"] == "read" and rules["/tmp"] == "deny" and rules[str(sandbox)] == "read"
    assert {rules[str(sandbox / name)] for name in ("work", "outputs", "tmp")} == {"write"}
    assert str(sandbox / "inputs") not in rules  # read only, as part of the sandbox directory
    assert python.denied and all(rules[path] == "deny" for path in python.denied)
    assert all(rules[path] == "read" for path in python.readable)
    readable = [Path(path) for path, rule in rules.items() if rule == "read" and not path.startswith(":")]
    assert Path.home() not in readable and tmp_path / "workspace" not in readable
    # The WSE server: the runner's command and allow-listed environment, the tool allow-list, long tool calls.
    server = config["mcp_servers"]["wse"]
    assert [server["command"], *server["args"]] == _connection(tmp_path / "workspace").serve_command
    assert server["env"] == _connection(tmp_path / "workspace").serve_env and server["required"] is True
    assert server["enabled_tools"] == ["wse_get_experiment", "wse_preview_patch"]
    assert server["tool_timeout_sec"] == 7201 and server["startup_timeout_sec"] == 60
    assert server["default_tools_approval_mode"] == "approve" and server["startup_readiness"] == "catalog"
    assert config["features"]["code_mode"] == {"direct_only_tool_namespaces": [MCP_NAMESPACE],
                                               "excluded_tool_namespaces": ["multi_agent_v1"]}
    assert config["features"]["multi_agent"] is False and config["skills"] == {"bundled": {"enabled": False}}
    with pytest.raises(codex_cli.WSEError, match="model setting"):
        codex_config(model="m", connection=_connection(tmp_path), sandbox_root=sandbox, skill_dir=tmp_path,
                     python=python, provider_config=None, startup_timeout_seconds=1, tool_timeout_seconds=1,
                     model_settings={"temperature": 0.2})


def test_the_tool_timeout_covers_the_policys_longest_call_even_under_a_short_wall_clock(tmp_path):
    """The runner raises client_session_timeout_seconds to agent_session_timeout(policy); Codex gets at least that."""
    from wse.policy import ExperimentPolicy

    policy = ExperimentPolicy.model_validate({"policy_id": "long_waits_v1", "mode": "bounded_eval",
                                              "max_job_wait_seconds": 600.0,
                                              "budget": {"evaluation_batches": 2, "seconds_per_operation": 3600.0}})
    longest = agent_session_timeout(policy, 30.0)
    assert longest > 3600
    config = AgentConfig(adapter="codex_cli@1", model="gpt-5.6-luna", wall_clock_seconds=60,
                         client_session_timeout_seconds=longest)
    adapter = CodexCLIAdapter(codex_command=["true"], provider_config={})
    asyncio.run(adapter.start(_connection(tmp_path / "workspace"), config))
    session = adapter._sessions["exp_cfg"]
    written = tomllib.loads((session.paths.home / "config.toml").read_text())
    asyncio.run(adapter.close("exp_cfg"))
    assert written["mcp_servers"]["wse"]["tool_timeout_sec"] == math.ceil(longest)


def test_toml_dumps_quotes_keys_and_strings_that_need_it():
    data = {"a": "x\"y\\z\n", "b": [1, 2.5, True], "t": {"/path/with spaces": "read", ":minimal": "read",
                                                         "nested": {"k": "é"}}}
    assert tomllib.loads(toml_dumps(data)) == data


def test_the_analysis_python_reaches_the_libraries_through_links_and_never_wse(tmp_path):
    python = python_access(tmp_path / "links")
    linked = set(python.linked)
    assert {"numpy", "torch", "safetensors"} <= linked and not any(name.startswith("wse") for name in linked)
    assert all((tmp_path / "links" / name).is_symlink() for name in linked)
    assert not any(Path(path).name.startswith(("wse", "_editable_impl_wse")) for path in python.readable)


def test_the_skill_is_installed_where_codex_lists_it_and_holds_no_run_details(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared)
    installed = _codex_dir(workspace, run_id) / "home" / "skills" / "wse" / "SKILL.md"
    text = installed.read_text()
    assert text == skill_text()
    front = text.split("---")[1]
    assert "name: wse" in front and "description:" in front
    for word in ("wse_get_experiment", "wse_preview_patch", "wse_apply_patch", "wse_submit_candidate",
                 "wse_evaluate", "wse_get_job", "WSE-STOP:", "inputs/weights/base.safetensors"):
        assert word in text
    # Task, budget, and run IDs arrive in the prompt, never in the skill.
    for detail in (run_id, eid, prepared["task_id"], SMALL_MODEL_POLICY_ID, "candidate commits: 20"):
        assert detail not in text
    prompt = (_codex_dir(workspace, run_id) / "prompt.txt").read_text()
    assert run_id in prompt and eid in prompt and str(installed) in prompt and "WSE-STOP:" in prompt
    assert "candidate commits: 20" in prompt


# -- events and usage ---------------------------------------------------------------------------------------------

# A real ``codex exec --json`` stream (Codex 0.167, a scripted local model, a stdio MCP server), commands shortened.
REAL_EVENTS = [
    {"type": "thread.started", "thread_id": "01a1ce50-8fac-7912-a2ef-915fea656446"},
    {"type": "turn.started"},
    {"type": "item.started", "item": {"id": "item_0", "type": "mcp_tool_call", "server": "wse", "tool": "wse_echo",
                                      "arguments": {"text": "hi"}, "result": None, "error": None,
                                      "status": "in_progress"}},
    {"type": "item.completed", "item": {"id": "item_0", "type": "mcp_tool_call", "server": "wse", "tool": "wse_echo",
                                        "arguments": {"text": "hi"}, "result": {"content": [{"type": "text", "text":
                                        "{\"summary\": \"echo\", \"result\": {\"text\": \"hi\"}}"}],
                                        "structured_content": None}, "error": None, "status": "completed"}},
    {"type": "item.started", "item": {"id": "item_1", "type": "command_execution", "command": "/bin/bash -c \"cat x\"",
                                      "aggregated_output": "", "exit_code": None, "status": "in_progress"}},
    {"type": "item.completed", "item": {"id": "item_1", "type": "command_execution",
                                        "command": "/bin/bash -c \"cat x\"", "aggregated_output":
                                        "wrote-work\n/bin/bash: inputs/b.txt: Operation not permitted\n",
                                        "exit_code": 0, "status": "completed"}},
    {"type": "item.started", "item": {"id": "item_2", "type": "mcp_tool_call", "server": "wse", "tool": "wse_fail",
                                      "arguments": {}, "result": None, "error": None, "status": "in_progress"}},
    {"type": "item.completed", "item": {"id": "item_2", "type": "mcp_tool_call", "server": "wse", "tool": "wse_fail",
                                        "arguments": {}, "result": {"content": [{"type": "text", "text":
                                        "Error executing tool wse_fail: {\"error\": {\"code\": \"POLICY_DENIED\", "
                                        "\"message\": \"denied\", \"retryable\": false, \"details\": {}}}"}],
                                        "structured_content": None}, "error": None, "status": "failed"}},
    {"type": "item.completed", "item": {"id": "item_3", "type": "agent_message", "text": "All done."}},
    {"type": "turn.completed", "usage": {"input_tokens": 4010, "cached_input_tokens": 1000,
                                         "cache_write_input_tokens": 0, "output_tokens": 210,
                                         "reasoning_output_tokens": 28}},
]


def _translate(events: list[dict[str, Any]], segment: int = 1) -> tuple[list[AgentEvent], _Translator, _Totals]:
    seen: list[AgentEvent] = []
    totals = _Totals()
    translator = _Translator(_Recorder(seen.append), totals, segment)
    for event in events:
        translator.handle(event)
    return seen, translator, totals


def test_codex_events_become_normalized_events():
    seen, translator, totals = _translate(REAL_EVENTS)
    assert translator.session_id == "01a1ce50-8fac-7912-a2ef-915fea656446" and translator.ending == "turn_completed"
    assert [(e.kind, e.tool) for e in seen] == [("tool_call", "wse_echo"), ("tool_result", "wse_echo"),
                                                ("tool_call", "shell"), ("tool_result", "shell"),
                                                ("tool_call", "wse_fail"), ("tool_result", "wse_fail"),
                                                ("message", None)]
    echo, shell_result, failure = (e for e in seen if e.kind == "tool_result")
    assert echo.ok and echo.summary == "echo" and echo.call_id == "s1:item_0"
    assert echo.output_hash.startswith("sha256:")
    assert shell_result.ok and shell_result.details["exit_code"] == 0
    assert "Operation not permitted" in shell_result.output
    assert not failure.ok and failure.error_code == "POLICY_DENIED" and failure.summary == "denied"
    assert seen[0].arguments == {"text": "hi"} and seen[2].arguments == {"command": "/bin/bash -c \"cat x\""}
    assert seen[-1].output == "All done." and translator.final_message == "All done."
    assert totals.tool_calls == {"wse": 2, "shell": 1, "apply_patch": 0, "other": 0}
    assert totals.turn_usage == {"input_tokens": 4010, "cached_input_tokens": 1000, "output_tokens": 210,
                                 "reasoning_output_tokens": 28, "total_tokens": 4220}


def test_mcp_results_report_domain_errors_and_codex_timeouts():
    ok = {"status": "completed", "result": {"content": [{"type": "text", "text": "{\"summary\": \"s\"}"}]}}
    assert parse_mcp_result(ok)[1:3] == (True, None)
    structured = {"status": "completed", "result": {"content": [], "structured_content": {"summary": "s"}}}
    assert parse_mcp_result(structured)[3] == {"summary": "s"}
    timeout = {"status": "failed", "result": None, "error": {"message": "tool call timed out after 2s"}}
    assert parse_mcp_result(timeout)[1:3] == (False, "TOOL_TIMEOUT")
    transport = {"status": "failed", "result": None, "error": {"message": "connection closed"}}
    assert parse_mcp_result(transport)[1:3] == (False, "TOOL_ERROR")
    garbled = {"status": "failed", "result": {"content": [{"type": "text", "text": "Error executing tool x: nope"}]}}
    assert parse_mcp_result(garbled)[1:3] == (False, "TOOL_ERROR")


def test_usage_comes_from_the_session_record_one_event_per_request(tmp_path):
    home = tmp_path / "home"
    record = home / "sessions" / "2026" / "11" / "12" / "rollout-2026-11-12T00-00-00-thread-1.jsonl"
    record.parent.mkdir(parents=True)
    lines = [{"type": "session_meta", "payload": {}},
             {"type": "token_usage_record", "payload": {"response_id": "resp_1", "turn_id": "t", "usage": {
                 "input_tokens": 1001, "cached_input_tokens": 100, "output_tokens": 51,
                 "reasoning_output_tokens": 7, "total_tokens": 1052}}},
             {"type": "token_usage_record", "payload": {"response_id": "resp_2", "turn_id": "t", "usage": {
                 "input_tokens": 1002, "cached_input_tokens": 200, "output_tokens": 52,
                 "reasoning_output_tokens": 7, "total_tokens": 1054}}}]
    record.write_text("\n".join(json.dumps(line) for line in lines) + "\n" + '{"type": "token_usage_rec')
    seen: list[AgentEvent] = []
    totals = _Totals()
    tail = _RolloutTail(home, _Recorder(seen.append), totals)
    assert tail.poll(None, 1) == 0 and tail.poll("thread-1", 1) == 2 and tail.poll("thread-1", 1) == 0
    with record.open("a") as handle:  # the partial line completes, and a repeated record is not counted twice
        handle.write('ord", "payload": {"response_id": "resp_3", "usage": {"input_tokens": 5, "output_tokens": 1}}}\n')
        handle.write(json.dumps(lines[1]) + "\n")
    assert tail.poll("thread-1", 2) == 1
    assert [e.details["response_id"] for e in seen] == ["resp_1", "resp_2", "resp_3"]
    assert all(e.kind == "model_response" and e.usage["requests"] == 1 for e in seen)
    from wse.agent_adapters.codex_cli import _Session

    adapter = CodexCLIAdapter(codex_command=["true"], provider_config={})
    usage = adapter._usage(_Session(connection=_connection(tmp_path), config=AgentConfig(model="gpt-5.6-luna")),
                           totals)
    assert (usage.requests, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens,
            usage.reasoning_output_tokens, usage.total_tokens) == (3, 2008, 300, 104, 14, 2112)
    assert usage.cost_usd is None and "cost_usd" in usage.unavailable


# -- resume -------------------------------------------------------------------------------------------------------


def test_an_early_end_of_turn_is_resumed_in_the_same_session(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared, behavior={"end_turn_after": 2})
    result = record["result"]
    assert result["status"] == "stopped" and result["stop_condition"].startswith("target_met")
    first, second = result["provider_metadata"]["segments"]
    assert (first["kind"], first["ending"], second["kind"]) == ("initial", "turn_completed", "resume")
    assert first["session_id"] == second["session_id"] == result["conversation_ref"]
    assert "without a stop condition" in second["reason"] and result["provider_metadata"]["resumes"] == 1
    events = _events(engine, eid)
    resume = [e for e in events if e["kind"] == "message" and e["details"].get("role") == "user"]
    assert len(resume) == 1 and "WSE has not ended this run" in resume[0]["output"]
    log = Path(second["log"]).read_text().splitlines()
    assert json.loads(log[0]) == {"type": "thread.started", "thread_id": first["session_id"]}
    assert engine.get_experiment(eid).state == "FROZEN" and _run_processes(run_id) == []


def test_a_recoverable_failure_is_resumed(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared, behavior={"fail_first": True})
    result = record["result"]
    assert result["status"] == "stopped" and result["stop_condition"].startswith("target_met")
    first, second = result["provider_metadata"]["segments"]
    assert first["ending"] == "turn_failed" and "stream disconnected" in first["errors"][0]
    assert second["kind"] == "resume" and second["reason"].startswith("recoverable failure")


def test_resumes_are_bounded(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared, behavior={"end_turn_after": 1},
                                              adapter_options={"max_resumes": 0})
    result = record["result"]
    assert result["status"] == "completed" and result["reason"] == "the agent ended its turn; 0 of 0 resumes used"
    assert len(result["provider_metadata"]["segments"]) == 1 and engine.get_experiment(eid).state == "FROZEN"


# -- limits, cancellation, and processes --------------------------------------------------------------------------


def test_the_wall_clock_limit_stops_codex_and_leaves_no_process(engine, workspace, prepared):
    started = time.monotonic()
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared, behavior={"hang": "sleep"},
                                              policy_id=WEIGHTS_ONLY_POLICY_ID, wall_clock_seconds=4)
    result = record["result"]
    assert result["status"] == "timeout" and "wall-clock limit of 4 s" in result["reason"]
    (segment,) = result["provider_metadata"]["segments"]
    assert segment["signals"] == ["SIGINT"] and time.monotonic() - started < 60
    assert engine.get_experiment(eid).state == "FROZEN" and _run_processes(run_id) == []


def test_a_codex_that_ignores_interrupts_is_terminated_then_killed(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared, behavior={"hang": "ignore-signals"},
                                              policy_id=WEIGHTS_ONLY_POLICY_ID, wall_clock_seconds=3,
                                              adapter_options={"interrupt_grace_seconds": 1.0,
                                                               "terminate_grace_seconds": 1.0})
    (segment,) = record["result"]["provider_metadata"]["segments"]
    assert segment["signals"] == ["SIGINT", "SIGTERM", "SIGKILL"] and segment["exit_code"] == -signal.SIGKILL
    assert record["result"]["status"] == "timeout" and _run_processes(run_id) == []
    assert engine.get_experiment(eid).state == "FROZEN"


def test_a_process_that_left_codex_s_process_group_is_reaped(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared,
                                              behavior={"hang": "sleep", "grandchild": True},
                                              policy_id=WEIGHTS_ONLY_POLICY_ID, wall_clock_seconds=4)
    (segment,) = record["result"]["provider_metadata"]["segments"]
    reaped = [entry for entry in segment["reaped_processes"] if "time.sleep(3600)" in entry["command"]]
    assert reaped and reaped[0]["signal"] == "SIGTERM"
    table = subprocess.run(["/bin/ps", "-A", "-o", "pid=,command="], capture_output=True, text=True).stdout
    assert str(reaped[0]["pid"]) not in [line.split()[0] for line in table.splitlines() if "time.sleep(3600)" in line]


def test_cancel_from_another_thread_stops_the_run(engine, workspace, prepared):
    import threading

    run_id = engine.grant_run(prepared["task_id"], WEIGHTS_ONLY_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    box: dict[str, Any] = {}

    def factory(agent_config, eid):
        box["adapter"] = CodexCLIAdapter.test_double(ScriptedSearchStrategy(), interrupt_grace_seconds=5.0,
                                                     extra_env={"WSE_CODEX_TEST_DOUBLE": json.dumps({"hang": "sleep"})})
        box["timer"] = threading.Timer(3.0, lambda: box["adapter"].cancel(eid, "operator request"))
        box["timer"].start()
        return box["adapter"]

    runner = _runner(workspace, factory)
    eid = runner.run_experiment(run_id, prepared["task_id"], WEIGHTS_ONLY_POLICY_ID,
                                {"kind": "agent", "adapter": "codex_cli@1", "model": DOUBLE})
    box["timer"].join()
    result = runner.last_result["result"]
    assert result["status"] == "stopped" and result["reason"] == "operator request"
    assert engine.get_experiment(eid).state == "FROZEN" and _run_processes(run_id) == []


def test_the_turn_limit_counts_model_requests(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared, behavior={"request_seconds": 1.5},
                                              max_turns=3)
    result = record["result"]
    assert result["status"] == "max_turns" and "turn limit of 3 model requests" in result["reason"]
    assert result["turns"] == 3 and engine.get_experiment(eid).state == "FROZEN"


def test_a_server_that_does_not_start_is_a_connection_failure(engine, workspace, prepared, monkeypatch):
    original = ReferenceRunner.connection

    def broken(self, *args, **kwargs):
        connection = original(self, *args, **kwargs)
        return connection.model_copy(update={"serve_command": [sys.executable, "-c", "import sys; sys.exit(3)"]})

    monkeypatch.setattr(ReferenceRunner, "connection", broken)
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared, policy_id=WEIGHTS_ONLY_POLICY_ID)
    result = record["result"]
    assert result["status"] == "connection_failed" and "MCP servers failed to initialize" in result["reason"]
    assert engine.get_experiment(eid).state == "FROZEN"


# -- secrets and the binary ---------------------------------------------------------------------------------------


def test_no_log_or_record_holds_the_proxy_token(engine, workspace, prepared):
    eid, record, adapter, run_id = _codex_run(engine, workspace, prepared, adapter_options={"proxy_token": TOKEN})
    assert record["result"]["status"] == "stopped"
    files = [p for p in workspace.rglob("*") if p.is_file() and not p.is_symlink()]
    assert files and not [str(p) for p in files if TOKEN.encode() in p.read_bytes()]
    assert TOKEN not in json.dumps(record) and "token_found_in" not in record["result"]["provider_metadata"]
    config = tomllib.loads((_codex_dir(workspace, run_id) / "home" / "config.toml").read_text())
    assert TOKEN not in json.dumps(config) and PROXY_TOKEN_ENV not in json.dumps(config["mcp_servers"])


def test_without_a_provider_the_adapter_runs_a_model_proxy_for_the_operators_endpoint(tmp_path, monkeypatch):
    """The default model access: ``wse.model_proxy`` keeps the operator's key; Codex gets the proxy and its token."""
    import types

    calls: list[tuple[Any, ...]] = []

    class FakeProxy:
        def __init__(self, upstream, key, *, log_path=None):
            calls.append(("init", upstream, key == "sk-operator-key", log_path is not None))
            self.token, self.url = "proxy-token-for-this-run", None

        def start(self):
            self.url = "http://127.0.0.1:5555/v1"
            calls.append(("start",))
            return self

        def stop(self):
            calls.append(("stop",))

    def provider(proxy):
        return {"model_providers": {"wse_proxy": {"name": "WSE model proxy", "base_url": proxy.url,
                                                  "env_key": PROXY_TOKEN_ENV, "wire_api": "responses"}},
                "model_provider": "wse_proxy"}

    module = types.ModuleType("wse.model_proxy")
    module.ModelProxy, module.codex_provider_config = FakeProxy, provider
    monkeypatch.setitem(sys.modules, "wse.model_proxy", module)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/openai/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-operator-key")
    connection = _connection(tmp_path / "workspace")
    config = AgentConfig(adapter="codex_cli@1", model="gpt-5.6-luna")
    adapter = CodexCLIAdapter(codex_command=["true"])
    asyncio.run(adapter.start(connection, config))
    session = adapter._sessions["exp_cfg"]
    assert session.connect_error is None
    assert session.env[PROXY_TOKEN_ENV] == "proxy-token-for-this-run"
    assert "sk-operator-key" not in json.dumps(session.env) and "OPENAI_API_KEY" not in session.env
    written = (session.paths.home / "config.toml").read_text()
    assert tomllib.loads(written)["model_providers"]["wse_proxy"]["base_url"] == "http://127.0.0.1:5555/v1"
    assert "sk-operator-key" not in written and "proxy-token-for-this-run" not in written
    asyncio.run(adapter.close("exp_cfg"))
    assert calls == [("init", "https://example.invalid/openai/v1", True, True), ("start",), ("stop",)]
    monkeypatch.delenv("OPENAI_API_KEY")
    missing = CodexCLIAdapter(codex_command=["true"])
    asyncio.run(missing.start(connection, config))
    assert "set OPENAI_BASE_URL and OPENAI_API_KEY" in missing._sessions["exp_cfg"].connect_error


def test_worker_arguments_come_from_the_serve_command():
    command = ["python", "-m", "wse.cli", "--workspace", "w", "serve", "--run-id", "r", "--setup", "a:b",
               "--setup", "c:d", "--worker-args", "--poll-seconds 0.02 --device cpu"]
    assert codex_cli.worker_arguments(command) == ["--setup", "a:b", "--setup", "c:d", "--poll-seconds", "0.02",
                                                   "--device", "cpu"]


def test_credentials_are_refused_in_codex_s_extra_environment():
    with pytest.raises(codex_cli.WSEError, match="looks like a credential"):
        CodexCLIAdapter(extra_env={"OPENAI_API_KEY": "x"})


def test_an_unsigned_binary_is_refused_on_macos(tmp_path, monkeypatch):
    fake = tmp_path / "codex"
    fake.write_text("#!/bin/sh\necho 'codex-cli 9.9.9'\n")
    fake.chmod(0o755)
    if sys.platform == "darwin":
        with pytest.raises(codex_cli.WSEError, match="Team ID|signature"):
            find_codex(fake)
        monkeypatch.setenv(codex_cli.CODEX_PATH_ENV, str(fake))
        with pytest.raises(codex_cli.WSEError, match="WSE_CODEX_PATH"):
            find_codex()
    binary = find_codex(fake, verify=False)
    assert binary.command == (str(fake.resolve()),) and binary.version == "codex-cli 9.9.9"
    with pytest.raises(codex_cli.WSEError, match="not an executable file"):
        find_codex(tmp_path / "missing", verify=False)


def test_the_search_prefers_path_then_the_newest_extension_bundle(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for version in ("26.1007.21434", "26.1106.51428", "26.1021.22027"):
        binary = home / ".vscode" / "extensions" / f"openai.chatgpt-{version}-darwin-arm64" / "bin" / "x" / "codex"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
    on_path = tmp_path / "bin" / "codex"
    on_path.parent.mkdir()
    on_path.write_text("#!/bin/sh\n")
    on_path.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{on_path.parent}:/usr/bin:/bin")
    found = codex_cli.codex_candidates()
    assert [source for source, _ in found] == ["PATH", "vscode-extension", "vscode-extension", "vscode-extension"]
    assert [path.parts[-4] for _, path in found[1:]] == ["openai.chatgpt-26.1106.51428-darwin-arm64",
                                                         "openai.chatgpt-26.1021.22027-darwin-arm64",
                                                         "openai.chatgpt-26.1007.21434-darwin-arm64"]
