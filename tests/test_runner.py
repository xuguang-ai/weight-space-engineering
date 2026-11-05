"""The reference runner with stepwise strategies, agent instructions, exports, and replay (Task 7).

Every run goes through ``ReferenceRunner.run_experiment`` on an
operator-prepared signed-sum workspace: the runner persists the experiment,
executes the strategy's actions through a run-bound Engine (the same service
the MCP tools call), owns the evaluation worker, settles, and freezes. The
scripted strategy never sees the fixture builder, the clean reference model,
or the oracle patch: it is constructed from its registered factory and
reaches the experiment only through the run-bound executor.
"""

from __future__ import annotations

import json
import re

import pytest
from test_mcp_e2e import FAULT_DESCRIPTIONS, FINAL_SEED, _final_secrets

from wse.api import Engine
from wse.cli import main as cli_main
from wse.policy import SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID, builtin_experiment_policies
from wse.runner import (
    EVALUATION_TOOLS,
    ExperimentExecutor,
    ReferenceRunner,
    StopMonitor,
    allowed_tools,
    render_instructions,
    serve_environment,
    summarize_trial,
)
from wse.schemas import ErrorCode, TaskDefinition, WSEError
from wse.strategies.base import Action

FAST_WORKER = ["--poll-seconds", "0.02"]


@pytest.fixture
def prepared(engine, signed_sum, tmp_path):
    ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED, with_reference=True)
    return ids


def _grant(engine, ids, policy_id=SMALL_MODEL_POLICY_ID):
    return engine.grant_run(ids["task_id"], policy_id, ids["task_checkpoint_id"]).run_id


def _run(workspace, run_id, ids, policy_id=SMALL_MODEL_POLICY_ID, **config):
    runner = ReferenceRunner(workspace, worker_args=FAST_WORKER)
    eid = runner.run_experiment(run_id, ids["task_id"], policy_id, {"kind": "strategy", **config})
    return eid, runner.last_result


# -- the scripted strategy -----------------------------------------------------------------------------------


def test_the_scripted_strategy_repairs_the_fixture_within_small_model_v1(engine, workspace, prepared):
    run_id = _grant(engine, prepared)
    eid, record = _run(workspace, run_id, prepared)
    experiment = engine.get_experiment(eid)
    # Persisted, settled, and frozen by the runner; never closed by it.
    assert experiment.state == "FROZEN" and experiment.selected_checkpoint_id != experiment.base_checkpoint_id
    assert not [job for job in engine.list_jobs(eid) if not job.terminal]
    result = record["result"]
    assert result["status"] == "stopped" and result["stop_condition"].startswith("target_met")
    assert result["finalization"]["state"] == "FROZEN"
    # The strategy's identity, version, and configuration are recorded with the experiment.
    assert record["kind"] == "strategy" and record["controller"] == {"extension_id": "scripted_search", "version": "1"}
    assert record["config"]["resolved_config"]["factors"] == [-1.0] and record["config"]["strategy"] == \
        "scripted_search@1"
    assert record["constraint_status"] == "verified"
    for name, reason in result["usage"]["unavailable"].items():
        assert result["usage"][name] is None and reason.startswith("not applicable")
    # Within the small_model_v1 budget.
    limits = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID].budget
    for counter, entry in experiment.budget.items():
        assert entry.consumed <= getattr(limits, counter), counter
    assert experiment.budget["evaluated_examples"].consumed == 3 * 64  # base and two candidates
    # The trace records every operation; a run-bound executor was the only channel.
    trace = engine.store.db.conn.execute(
        "SELECT kind, payload FROM controller_events WHERE experiment_id = ? ORDER BY seq", (eid,)).fetchall()
    tools = [json.loads(row["payload"]).get("tool") for row in trace if row["kind"] == "tool_call"]
    assert tools[:3] == ["wse_get_experiment", "wse_inspect_model", "wse_submit_evaluation"]
    assert "wse_select_checkpoint" in tools and set(tools) <= set(allowed_tools(engine.get_policy(
        SMALL_MODEL_POLICY_ID)))
    events = [event["event_type"] for event in engine.audit_log(eid)]
    assert events[0] == "experiment.created" and "controller.started" in events and "experiment.frozen" in events
    assert not [e for e in events if e.endswith(".failed")]
    # The final evaluation is a separate operator step.
    final = engine.final_evaluate(eid)
    assert final.comparison.candidate_value == 1.0 and final.comparison.base_value == 0.5


def test_the_weights_only_control_stops_after_its_single_candidate_commit(engine, workspace, prepared):
    run_id = _grant(engine, prepared, WEIGHTS_ONLY_POLICY_ID)
    eid, record = _run(workspace, run_id, prepared, WEIGHTS_ONLY_POLICY_ID)
    experiment = engine.get_experiment(eid)
    assert experiment.state == "FROZEN" and experiment.mode == "weights_only"
    assert experiment.budget["candidate_commits"].consumed == 1
    assert experiment.budget["evaluation_batches"].consumed == 0 and engine.list_jobs(eid) == []
    assert record["result"]["status"] == "stopped"
    assert record["result"]["stop_condition"].startswith("candidate_committed")
    assert set(record["config"]["allowed_tools"]).isdisjoint(EVALUATION_TOOLS)
    final = engine.final_evaluate(eid)  # no development score guided this candidate
    assert final.candidate_checkpoint_id == experiment.selected_checkpoint_id


def test_trial_summaries_report_costs_changed_fraction_and_failure_causes(engine, workspace, prepared, capsys):
    run_id = _grant(engine, prepared, WEIGHTS_ONLY_POLICY_ID)
    eid, _ = _run(workspace, run_id, prepared, WEIGHTS_ONLY_POLICY_ID)
    pending = summarize_trial(engine, eid)
    assert pending["success"] is None and pending["final"] is None and pending["state"] == "FROZEN"
    engine.final_evaluate(eid)
    capsys.readouterr()
    assert cli_main(["--workspace", str(workspace), "trials", eid, "--format", "markdown"]) == 0
    text = capsys.readouterr().out
    assert "0/1 trials found a candidate with final accuracy >= 99%" in text and eid in text
    row = summarize_trial(engine, eid)
    assert row["candidates"] == 1 and row["evaluated_examples"] == 0 and row["changed_elements"] > 0
    assert row["changed_fraction"] == row["changed_elements"] / row["parameter_count"]
    assert row["tokens"]["input_tokens"] is None and row["tokens_unavailable"]["input_tokens"]
    assert row["final"]["clusters"] == 512 and row["controller"] == "scripted_search@1"
    assert row["success"] is (row["final"]["candidate_accuracy"] >= 0.99)
    if not row["success"]:
        assert row["failure_cause"].startswith("final accuracy")


def test_three_evaluated_candidates_without_improvement_stop_the_run(engine, workspace, prepared):
    run_id = _grant(engine, prepared)
    eid, record = _run(workspace, run_id, prepared,
                       parameters={"tensors": ["layers.0.weight"], "patience": 10})
    result = record["result"]
    assert result["status"] == "stopped" and result["stop_condition"].startswith("no_improvement: 3")
    experiment = engine.get_experiment(eid)
    assert experiment.selected_checkpoint_id == experiment.base_checkpoint_id
    assert experiment.budget["candidate_commits"].consumed == 3 and experiment.state == "FROZEN"


def test_the_stop_monitor_counts_consecutive_candidates_without_improvement():
    class Report:
        def __init__(self, checkpoint_id, accuracy):
            self.checkpoint_id, self.metrics = checkpoint_id, {"accuracy": accuracy}

    streak = StopMonitor.no_improvement_streak
    assert streak([Report("b", 0.5), Report("c1", 0.5), Report("c2", 0.4), Report("c3", 0.5)], "b") == 3
    assert streak([Report("b", 0.5), Report("c1", 0.6), Report("c2", 0.6)], "b") == 1
    assert streak([Report("c1", 0.1), Report("c2", 0.1), Report("b", 0.9), Report("c3", 0.2)], "b") == 2
    assert streak([Report("b", 0.5), Report("c1", 0.4), Report("c1", 0.9)], "b") == 1  # first report per checkpoint


def test_a_strategy_reaches_only_the_run_bound_operations(engine, workspace, prepared):
    run_id = _grant(engine, prepared)
    with Engine(workspace, run_id=run_id) as bound:
        eid = bound.create_experiment(run_id, prepared["task_id"], SMALL_MODEL_POLICY_ID, "c").experiment_id
        executor = ExperimentExecutor(bound, constraint_status="verified")
        denied = executor.execute(Action.call("wse_create_experiment", run_id=run_id))
        assert denied.error_code == "POLICY_DENIED"
        hidden = executor.execute(Action.call("wse_inspect_model", experiment_id=eid,
                                              checkpoint_id=prepared["reference_checkpoint_id"]))
        assert hidden.error_code == "NOT_FOUND"
        bad = executor.execute(Action.call("wse_read_tensor", experiment_id=eid))
        assert bad.error_code == "INVALID_ARGUMENT"
    with pytest.raises(WSEError) as excinfo:
        ExperimentExecutor(engine, constraint_status="verified")
    assert excinfo.value.code == ErrorCode.POLICY_DENIED


# -- agent instructions ------------------------------------------------------------------------------------------


def test_generated_instructions_come_from_the_task_and_policy_and_reveal_no_answer(engine, prepared, signed_sum):
    task = TaskDefinition(**signed_sum.load_task())
    reference = engine.load_manifest(prepared["reference_checkpoint_id"])
    secrets = [reference.checkpoint_id, reference.weights_hash, *_final_secrets(engine, prepared)]
    texts = []
    for policy_id in (SMALL_MODEL_POLICY_ID, WEIGHTS_ONLY_POLICY_ID):
        policy = engine.get_policy(policy_id)
        tools = allowed_tools(policy)
        text = render_instructions(task, policy, experiment_id="exp_0123", tools=tools)
        assert text == render_instructions(task, policy, experiment_id="exp_0123", tools=tools)  # deterministic
        assert task.description in text and task.objective.description in text and "exp_0123" in text
        assert ", ".join(tools) in text
        texts.append(text)
    assert "wse_submit_evaluation" in texts[0] and "wse_submit_evaluation" not in texts[1]
    combined = "\n".join(texts)
    for secret in secrets:
        assert secret not in combined
    assert sorted({m.group(0) for m in FAULT_DESCRIPTIONS.finditer(combined)}) == []
    assert not re.search(r"plan|build_fixture|\.py\b|docs/", combined)


def test_the_server_environment_is_allow_listed_and_never_carries_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("WSE_UNRELATED", "1")
    env = serve_environment({"PYTHONPATH": "/x"})
    assert "OPENAI_API_KEY" not in env and "WSE_UNRELATED" not in env
    assert env["OPENAI_AGENTS_DISABLE_TRACING"] == "1" and env["PYTHONPATH"] == "/x"
    with pytest.raises(WSEError) as excinfo:
        serve_environment({"OPENAI_API_KEY": "x"})
    assert excinfo.value.code == ErrorCode.POLICY_DENIED


# -- export and replay ----------------------------------------------------------------------------------------------


def test_exports_carry_hashes_traces_policies_and_costs_and_replay_to_identical_weights(
        engine, workspace, prepared, tmp_path, capsys):
    run_id = _grant(engine, prepared)
    eid, _ = _run(workspace, run_id, prepared)
    engine.final_evaluate(eid)
    out = tmp_path / "export"
    capsys.readouterr()
    assert cli_main(["--workspace", str(workspace), "export", eid, "--output", str(out)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest == json.loads((out / "export_manifest.json").read_text())
    experiment = engine.get_experiment(eid)
    assert manifest["state"] == "CLOSED" and manifest["management"] == "managed"
    assert manifest["base"]["weights_hash"] == engine.load_manifest(experiment.base_checkpoint_id).weights_hash
    assert manifest["policy_hash"] == experiment.policy_hash and (out / "policy.json").exists()
    assert {p["applied_checkpoint_id"] for p in manifest["patches"]} == \
        {c.checkpoint_id for c in engine.list_experiment_checkpoints(eid) if c.role == "candidate"}
    assert all(p["exact_patch_hash"].startswith("sha256:") for p in manifest["patches"])
    assert len(manifest["reports"]) == 3 + 2 and manifest["final"]["candidate_accuracy"] == 1.0
    assert manifest["constraint_status"] == "verified" and manifest["controller"]["controller"]["extension_id"] == \
        "scripted_search"
    assert manifest["tracing"] == {"sdk_tracing": "not applicable: no agent SDK runs"}
    assert manifest["costs"]["candidates"] == 2 and manifest["costs"]["budget"]["evaluated_examples"]["consumed"] == 192
    assert manifest["costs"]["selected_changed_elements"] == 4 and manifest["costs"]["parameter_count"] == 22
    trace = json.loads((out / "tool_trace.json").read_text())
    assert [e for e in trace if e["kind"] == "tool_call"] and trace[-1]["kind"] == "run_finished"
    audit = json.loads((out / "audit_events.json").read_text())
    assert audit[0]["event_type"] == "experiment.created" and audit[-1]["event_type"] == "experiment.closed"
    assert manifest["excluded"] == ["final-set inputs", "final-set labels", "final-set generation seed"]
    exported = "\n".join(path.read_text(errors="replace") for path in out.rglob("*") if path.suffix == ".json")
    for secret in _final_secrets(engine, prepared)[1:]:  # the seed and an input value; the data hash is an identity
        assert secret not in exported
    assert "\"labels\"" not in exported and "\"final_data/" not in exported

    # Replay in a clean workspace recovers byte-identical weights without running the editing operator.
    clean = tmp_path / "clean"
    capsys.readouterr()
    assert cli_main(["--workspace", str(clean), "replay", str(out)]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["all_identical"] and len(replay["candidates"]) == 2
    with Engine(clean) as fresh:
        for candidate in replay["candidates"]:
            replayed = fresh.store.weights_path(candidate["replayed_checkpoint_id"])
            tensors = fresh.store.load_tensors(candidate["replayed_checkpoint_id"])
            original = engine.store.load_tensors(candidate["checkpoint_id"])
            assert replayed.exists() and all(tensors[name].numpy().tobytes() == original[name].numpy().tobytes()
                                             for name in original)


def test_exports_include_final_data_only_on_request_and_detect_tampering(engine, workspace, prepared, tmp_path):
    run_id = _grant(engine, prepared, WEIGHTS_ONLY_POLICY_ID)
    eid, _ = _run(workspace, run_id, prepared, WEIGHTS_ONLY_POLICY_ID)
    with_data = engine.export_experiment(eid, tmp_path / "with-data", include_final_data=True)
    assert with_data.excluded == [] and with_data.included_final_data
    stored = json.loads((tmp_path / "with-data" / "final_data" / "final_set.json").read_text())
    assert stored["generation"]["seed"] == FINAL_SEED
    plain = engine.export_experiment(eid, tmp_path / "plain", include_base_weights=False)
    assert plain.replay["base_bundle"] is None and not (tmp_path / "plain" / "base_model").exists()
    with pytest.raises(WSEError):
        engine.export_experiment(eid, tmp_path / "plain")  # not empty
    patch = next((tmp_path / "with-data" / "patches").iterdir())
    patch.write_text(patch.read_text().replace('"hypothesis"', '"hypothesis" ', 1))
    with Engine(tmp_path / "clean") as fresh, pytest.raises(WSEError) as excinfo:
        fresh.replay_export(tmp_path / "with-data")
    assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT
    with Engine(workspace, run_id=run_id) as bound, pytest.raises(WSEError) as excinfo:
        bound.export_experiment(eid, tmp_path / "agent-export")
    assert excinfo.value.code == ErrorCode.POLICY_DENIED


def test_the_cli_runs_a_scripted_controller(engine, workspace, prepared, capsys):
    run_id = _grant(engine, prepared, WEIGHTS_ONLY_POLICY_ID)
    capsys.readouterr()
    assert cli_main(["--workspace", str(workspace), "run", "--run-id", run_id, "--controller", "scripted",
                     "--param", "factors=[-1.0]"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["result"]["finalization"]["state"] == "FROZEN"
    assert cli_main(["--workspace", str(workspace), "run", "--run-id", run_id]) == 2  # one controller per grant



# -- the optional SDK -----------------------------------------------------------------------------------------------

NO_SDK = """
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('agents', 'openai', 'claude_agent_sdk'):
            raise ImportError('blocked optional SDK: ' + name)
sys.meta_path.insert(0, Block())
import asyncio, json
import wse, wse.cli, wse.runner, wse.agent_adapters.base
from wse.agent_adapters.openai_agents import OpenAIAgentsAdapter
from wse.api import Engine
from wse.schemas import WSEError
spec = importlib.util.spec_from_file_location('fixture', sys.argv[1])
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
with Engine(sys.argv[2]) as engine:
    ids = fixture.setup_workspace(engine, sys.argv[3], final_seed=1)
    run_id = engine.grant_run(ids['task_id'], 'small_model_weights_only_v1', ids['task_checkpoint_id']).run_id
runner = wse.runner.ReferenceRunner(sys.argv[2])
eid = runner.run_experiment(run_id, ids['task_id'], 'small_model_weights_only_v1', {'kind': 'strategy'})
try:
    asyncio.run(OpenAIAgentsAdapter('a-model').start(None, None))
except WSEError as exc:
    code = exc.code.value
print(json.dumps({'state': runner.last_result['result']['finalization']['state'], 'code': code,
                  'loaded': sorted(m for m in sys.modules if m.split('.')[0] in ('agents', 'openai'))}))
"""


def test_the_base_package_and_scripted_runs_work_without_the_optional_sdk(tmp_path, signed_sum):
    import subprocess
    import sys

    probe = subprocess.run([sys.executable, "-c", NO_SDK, str(signed_sum.__file__), str(tmp_path / "ws"),
                            str(tmp_path / "bundles")], capture_output=True, text=True, timeout=120,
                           env={**__import__("os").environ, "OPENAI_AGENTS_DISABLE_TRACING": "1"})
    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout.strip().splitlines()[-1]) == {"state": "FROZEN", "code": "UNSUPPORTED_CAPABILITY",
                                                                 "loaded": []}
