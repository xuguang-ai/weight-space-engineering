"""Long waits for jobs and the agent's instructions under a sandbox policy (M3 plan Sections 2.4 and 2.7, Tasks 6-7).

A policy's ``max_job_wait_seconds`` caps how long one ``wse_get_job`` call
may wait; unset, the cap stays 10 seconds. The run-bound MCP server
publishes the cap in the tool's schema, the runner raises the agent's MCP
session timeout above the longest call a policy permits, and the
instructions state the cap and, under a sandbox policy, how to use the
sandbox.
"""

from __future__ import annotations

import anyio
import pytest
from test_mcp_e2e import FINAL_SEED

from wse.api import Engine
from wse.mcp_server import WSEMCPServer
from wse.policy import QWEN_LOOP_SANDBOX_POLICY_ID, SMALL_MODEL_POLICY_ID, builtin_experiment_policies
from wse.runner import agent_session_timeout, allowed_tools, render_instructions
from wse.schemas import ErrorCode, TaskDefinition, WSEError

CAPPED = "small_model_capped_waits"


def _capped_policy():
    small = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID]
    return small.model_copy(update={"policy_id": CAPPED, "max_job_wait_seconds": 30.0})


@pytest.fixture
def runs(workspace, signed_sum, tmp_path):
    """One run under the capped policy and one under small_model_v1, each with a finished evaluation job."""
    from wse.worker import Worker

    found = {}
    with Engine(workspace) as engine:
        ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
        engine.register_policy(_capped_policy())
    for policy_id in (CAPPED, SMALL_MODEL_POLICY_ID):
        with Engine(workspace) as engine:
            grant = engine.grant_run("signed_sum_v1", policy_id, ids["task_checkpoint_id"])
        with Engine(workspace, run_id=grant.run_id) as bound:
            eid = bound.create_experiment(grant.run_id, "signed_sum_v1", policy_id, "create").experiment_id
            job = bound.submit_evaluation(eid, ids["task_checkpoint_id"], "signed_sum_dev_v1",
                                          {"sample_count": 64, "seed": 0}, "evaluate")
        assert Worker(workspace, device="cpu").run_once() == job.job_id
        found[policy_id] = (grant.run_id, eid, job.job_id)
    return found


def test_get_job_waits_up_to_the_policy_cap(workspace, runs) -> None:
    run_id, eid, job_id = runs[CAPPED]
    with Engine(workspace, run_id=run_id) as bound:
        assert bound.get_job(eid, job_id, 25).terminal  # a finished job returns at once
        with pytest.raises(WSEError) as caught:
            bound.get_job(eid, job_id, 31)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT and "30" in caught.value.message


def test_policies_without_a_cap_keep_ten_seconds(workspace, runs) -> None:
    run_id, eid, job_id = runs[SMALL_MODEL_POLICY_ID]
    with Engine(workspace, run_id=run_id) as bound:
        assert bound.get_job(eid, job_id, 10).terminal
        with pytest.raises(WSEError) as caught:
            bound.get_job(eid, job_id, 11)
    assert caught.value.code == ErrorCode.INVALID_ARGUMENT


def test_the_published_schema_states_the_policy_cap(workspace, runs) -> None:
    for policy_id, cap in ((CAPPED, 30), (SMALL_MODEL_POLICY_ID, 10)):
        run_id, eid, job_id = runs[policy_id]
        with Engine(workspace, run_id=run_id) as bound:
            server = WSEMCPServer(bound)
            tools = {tool.name: tool for tool in anyio.run(server.server.list_tools)}
            wait = tools["wse_get_job"].input_schema["properties"]["wait_seconds"]
            assert wait["maximum"] == cap and f"0-{cap}" in wait["description"]
            assert f"up to {cap} seconds" in tools["wse_get_job"].description
            assert server.wse_get_job(experiment_id=eid, job_id=job_id, wait_seconds=cap).structured_content[
                "result"]["state"] == "succeeded"


def test_the_agent_session_timeout_exceeds_the_longest_call() -> None:
    from wse.jobs import evaluation_wait_cap

    policies = builtin_experiment_policies()
    # wse_evaluate waits for its job up to the policy's seconds_per_operation and a settling margin.
    small = policies[SMALL_MODEL_POLICY_ID]
    assert agent_session_timeout(small, 30.0) > evaluation_wait_cap(small) > small.budget.seconds_per_operation == 30
    sandboxed = agent_session_timeout(policies[QWEN_LOOP_SANDBOX_POLICY_ID], 30.0)
    assert sandboxed > 600 and sandboxed > 300 + 60 and sandboxed > 2 * 3600
    assert agent_session_timeout(policies[QWEN_LOOP_SANDBOX_POLICY_ID], 9000.0) == 9000.0
    # Without development evaluations (weights_only), the longest call is still a wse_get_job wait.
    assert agent_session_timeout(policies["small_model_weights_only_v1"], 30.0) == 30.0


def test_instructions_describe_the_sandbox_and_the_wait_cap() -> None:
    policies = builtin_experiment_policies()
    task = TaskDefinition.model_validate({
        "task_id": "loops_v1", "title": "Finish answers", "description": "Make the model finish its answers.",
        "objective": {"task_id": "loops_v1", "description": "Fewer outputs at the token limit.",
                      "gates": [{"metric": "accuracy", "min_delta": 0.0}]},
        "development_suite_id": "practice", "development_examples": 4})
    sandboxed = policies[QWEN_LOOP_SANDBOX_POLICY_ID]
    text = render_instructions(task, sandboxed, experiment_id="exp_0123", tools=allowed_tools(sandboxed))
    for fragment in ("wse_run_python", "inputs/weights/base.safetensors", "safetensors.torch", "passes/",
                     "probes/", "outputs/", "#/values", "sandbox calls: 100", "300 s", "not metered",
                     "wait_seconds up to 600"):
        assert fragment in text, fragment
    assert "you cannot run code" not in text
    tools_only = policies["qwen_loop_v1"]
    plain = render_instructions(task, tools_only, experiment_id="exp_0123", tools=allowed_tools(tools_only))
    assert "you cannot run code, read files, or use the network" in plain
    assert "wse_run_python" not in plain and "wait_seconds up to 10" in plain
