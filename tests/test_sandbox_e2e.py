"""Offline end-to-end check of code execution through MCP (M3 Task 8, the exit of stage S1).

``sandbox_search@1`` drives the ``openai_agents@1`` adapter through the SDK
test double and a real ``wse serve`` process, under a policy that grants the
sandbox. Its code runs in the sandbox and reads the base weights from the
sandbox's inputs; it finds the pair of identical rows of ``layers.1.weight``
and computes the values for the second row; the strategy previews,
applies, evaluates, and selects a ``set_slice`` with those values. The run
closes with a final accuracy of 1.0, is unverified with its reason, and its
export (with the sandbox call) replays to identical weights. No live model
is started.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_mcp_e2e import FINAL_SEED
from test_sandbox_policy import SANDBOX_POLICY_ID, _small_sandbox_policy

from wse import sandbox
from wse.agent_runs import run_agent_experiment
from wse.runner import SANDBOX_CONSTRAINT_REASON, SANDBOX_TIME_LABEL, TEST_DOUBLE_PREFIX

AVAILABLE, REASON = sandbox.available()
pytestmark = pytest.mark.skipif(not AVAILABLE, reason=REASON)


def test_a_scripted_sandbox_strategy_repairs_signed_sum_through_mcp(engine, workspace, signed_sum, tmp_path,
                                                                    monkeypatch) -> None:
    pytest.importorskip("agents", reason="the optional agents-openai extra is not installed")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
    engine.register_policy(_small_sandbox_policy())
    model = f"{TEST_DOUBLE_PREFIX}sandbox_search@1"
    run = run_agent_experiment(workspace, task=ids["task_id"], base=ids["task_checkpoint_id"],
                               policy=SANDBOX_POLICY_ID, model=model, max_turns=60, out_dir=tmp_path / "out",
                               final_evaluation=True)
    assert run.state == "CLOSED" and run.success is True and run.replay_identical is True
    assert run.final["candidate_accuracy"] == 1.0
    summary = run.summary
    assert (summary["controller"], summary["model"]) == ("openai_agents@1", model)
    assert summary["constraint_status"] == "unverified"
    assert summary["constraint_reason"] == SANDBOX_CONSTRAINT_REASON
    assert summary["sandbox"]["calls"] == 1 and summary["sandbox"]["label"] == SANDBOX_TIME_LABEL

    # The sandbox call: its code read the sandbox's copy of the base weights and printed the pair it found.
    export = run.export_dir
    manifest = json.loads((export / "export_manifest.json").read_text(encoding="utf-8"))
    [call] = manifest["sandbox"]["calls"]
    assert "inputs/weights/base.safetensors" in (export / call["code.py"]).read_text(encoding="utf-8")
    [pair] = json.loads((export / call["stdout.txt"]).read_text(encoding="utf-8"))
    assert {key: pair[key] for key in ("tensor", "first", "row", "columns")} == {
        "tensor": "layers.1.weight", "first": 0, "row": 1, "columns": 4}
    saved = np.load(export / call["outputs"]["values_0.npy"], allow_pickle=False)
    assert saved.tolist() == pair["values"]
    [array] = manifest["sandbox"]["arrays"]
    assert json.loads((export / array["file"]).read_text(encoding="utf-8"))["values"] == pair["values"]

    # The selected candidate is exactly that set_slice, with the values the sandbox computed.
    selected = next(p for p in manifest["patches"] if p["applied_checkpoint_id"] == manifest["selected_checkpoint_id"])
    stored = json.loads((export / selected["path"]).read_text(encoding="utf-8"))
    [operation] = stored["artifact"]["spec"]["operations"]
    assert (operation["op"], operation["tensor"]) == ("set_slice", "layers.1.weight")
    assert operation["parameters"]["values"] == [pair["values"]]
