"""One agent experiment in one call: ``wse.agent_runs.run_agent_experiment`` on the signed-sum fixture.

Every run is offline: the scripted strategy, or the ``openai_agents@1``
adapter driven by the SDK test double (``wse-test-double:STRATEGY``) through
a real ``wse serve`` process. No live model is started.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
from test_mcp_e2e import FINAL_SEED

from wse.agent_runs import AgentRun, run_agent_experiment
from wse.api import Engine
from wse.audit import EXPORT_MANIFEST_NAME
from wse.policy import SMALL_MODEL_POLICY_ID
from wse.runner import TEST_DOUBLE_PREFIX, ReferenceRunner, summarize_trial
from wse.schemas import ErrorCode, WSEError

SCRIPTED = "scripted_search@1"


@pytest.fixture
def prepared(engine, signed_sum, tmp_path):
    return signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)


def _run(workspace, prepared, **options) -> AgentRun:
    return run_agent_experiment(workspace, task=prepared["task_id"], base=prepared["task_checkpoint_id"],
                                policy=SMALL_MODEL_POLICY_ID, **options)


def _grant_count(engine) -> int:
    return engine.store.db.conn.execute("SELECT COUNT(*) AS n FROM run_grants").fetchone()["n"]


def test_a_scripted_run_closes_succeeds_and_replays_to_identical_weights(engine, workspace, prepared):
    run = _run(workspace, prepared, strategy=SCRIPTED, final_evaluation=True)
    assert isinstance(run, AgentRun) and dataclasses.is_dataclass(run)
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.state = "OPEN"  # type: ignore[misc]
    experiment = engine.get_experiment(run.experiment_id)
    assert run.run_id == experiment.run_id and run.state == experiment.state == "CLOSED"
    assert run.summary == summarize_trial(engine, run.experiment_id)
    assert run.success is True and run.final == run.summary["final"] and run.final["candidate_accuracy"] >= 0.99
    assert run.summary["controller"] == SCRIPTED
    # The default output directory lies inside the workspace, one per run grant.
    assert run.out_dir == workspace.resolve() / "agent_runs" / run.run_id
    assert run.export_dir == run.out_dir / "export" and (run.export_dir / EXPORT_MANIFEST_NAME).is_file()
    assert run.replay_identical is True
    assert (run.out_dir / "replay" / "metadata.sqlite").is_file()
    # The engine ignores that subdirectory: reopening the workspace recovers nothing and exports still work.
    with Engine(workspace) as reopened:
        recovery = reopened.last_recovery
        assert not (recovery.removed_temp_files or recovery.removed_orphans or recovery.removed_blobs
                    or recovery.corrupted_checkpoints)
        assert reopened.get_experiment(run.experiment_id).state == "CLOSED"


def test_without_replay_verification_nothing_is_exported(workspace, prepared, tmp_path):
    out = tmp_path / "out"
    run = _run(workspace, prepared, strategy=SCRIPTED, out_dir=out, final_evaluation=True, verify_replay=False)
    assert run.state == "CLOSED" and run.success is True
    assert run.out_dir == out and run.export_dir is None and run.replay_identical is None
    assert not (out / "export").exists() and not (out / "replay").exists()


def test_by_default_the_frozen_experiment_is_returned_before_its_final_evaluation(engine, workspace, prepared,
                                                                                   tmp_path):
    # A series of trials runs its final evaluations only after every trial has frozen (M1 plan Section 7.3).
    controller = {"kind": "strategy", "strategy": SCRIPTED, "wall_clock_seconds": 300.0}
    run = _run(workspace, prepared, controller=controller, out_dir=tmp_path / "out")
    assert run.state == engine.get_experiment(run.experiment_id).state == "FROZEN"
    assert run.success is None and run.final is None
    assert run.export_dir is None and run.replay_identical is None


@pytest.mark.parametrize("options", [
    {},
    {"model": "some-model", "strategy": SCRIPTED},
    {"strategy": SCRIPTED, "controller": {"kind": "strategy"}},
    {"model": "some-model", "strategy": SCRIPTED, "controller": {"kind": "strategy"}},
])
def test_exactly_one_controller_option_is_required(engine, workspace, prepared, options):
    with pytest.raises(WSEError) as caught:
        _run(workspace, prepared, **options)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT
    assert all(name in caught.value.message for name in ("model", "strategy", "controller"))
    assert _grant_count(engine) == 0


@pytest.mark.parametrize("options", [
    {"strategy": SCRIPTED, "max_turns": 5},  # a strategy has no SDK turns
    {"controller": {"kind": "strategy"}, "wall_clock_seconds": 10.0},  # limits belong in the mapping
    {"controller": {"kind": "agent"}},  # an agent controller needs a model
])
def test_an_invalid_controller_is_rejected_before_any_run_grant(engine, workspace, prepared, options):
    with pytest.raises(WSEError) as caught:
        _run(workspace, prepared, **options)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT
    assert _grant_count(engine) == 0


def test_a_used_output_directory_is_refused_before_any_run_grant(engine, workspace, prepared, tmp_path):
    out = tmp_path / "out"
    (out / "replay").mkdir(parents=True)
    (out / "replay" / "leftover.txt").write_text("x", encoding="utf-8")
    with pytest.raises(WSEError) as caught:
        _run(workspace, prepared, strategy=SCRIPTED, out_dir=out)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT and "not empty" in caught.value.message
    assert _grant_count(engine) == 0


@pytest.mark.parametrize("field", ["task", "base"])
def test_an_unknown_task_or_checkpoint_propagates_the_engines_error(workspace, prepared, field):
    options = {"task": prepared["task_id"], "base": prepared["task_checkpoint_id"], field: "unknown_v9"}
    with pytest.raises(WSEError) as caught:
        run_agent_experiment(workspace, policy=SMALL_MODEL_POLICY_ID, strategy=SCRIPTED, **options)
    assert caught.value.code == ErrorCode.NOT_FOUND


def test_a_runner_error_propagates_unchanged(workspace, prepared, monkeypatch):
    def fail(self, *args, **kwargs):
        raise WSEError(ErrorCode.WORKER_FAILED, "1 jobs did not settle within 120 s; the experiment stays open")

    monkeypatch.setattr(ReferenceRunner, "_run_strategy", fail)
    with pytest.raises(WSEError) as caught:
        _run(workspace, prepared, strategy=SCRIPTED)
    assert caught.value.code == ErrorCode.WORKER_FAILED


def test_an_agent_run_through_the_sdk_test_double_closes_and_replays(workspace, prepared, tmp_path, monkeypatch):
    pytest.importorskip("agents", reason="the optional agents-openai extra is not installed")
    monkeypatch.delenv("OPENAI_AGENTS_DISABLE_TRACING", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = f"{TEST_DOUBLE_PREFIX}{SCRIPTED}"
    run = _run(workspace, prepared, model=model, max_turns=60, out_dir=tmp_path / "out", final_evaluation=True)
    assert os.environ["OPENAI_AGENTS_DISABLE_TRACING"] == "1"
    assert run.summary["controller"] == "openai_agents@1" and run.summary["model"] == model
    assert run.state == "CLOSED" and run.success is True and run.replay_identical is True
    assert (tmp_path / "out" / "logs" / f"serve-{run.run_id}.log").is_file()
