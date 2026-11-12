"""Policy, ledger, and labels of the sandbox mode (M3 plan Sections 2.4 and 2.5, Task 2)."""

from __future__ import annotations

import sqlite3

import pytest
from test_mcp_e2e import FINAL_SEED

from wse.api import Engine
from wse.policy import (
    QWEN_LOOP_SANDBOX_POLICY_ID,
    SMALL_MODEL_POLICY_ID,
    BudgetLimits,
    ExperimentPolicy,
    SandboxPolicy,
    builtin_experiment_policies,
    sandbox_available,
)
from wse.runner import StrategyConfig, ReferenceRunner, render_trials_markdown, summarize_trial
from wse.schemas import ErrorCode, Usage, WSEError

# The hashes of the built-in policies before the sandbox fields existed: they must never change.
FROZEN_HASHES = {
    "small_model_v1": "sha256:d8761471bfb88e1d322da1ca06003729886daafa028963f1dcf7ef807cc218b5",
    "small_model_weights_only_v1": "sha256:88d3c1e4f73e0eb8204c30b65aceb753608114c4ed05e074302db630e665fa75",
    "qwen_loop_v1": "sha256:cbf2ccceaca494aa0e4c96af88201a7aa28d0c77cdcb407aaa59978ee3a333f4",
}
SANDBOX_POLICY_ID = "small_model_sandbox_test"


def _small_sandbox_policy() -> ExperimentPolicy:
    small = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID]
    budget = small.budget.model_copy(update={"sandbox_calls": 5, "sandbox_seconds": 60.0})
    return small.model_copy(update={"policy_id": SANDBOX_POLICY_ID, "budget": budget,
                                    "sandbox": SandboxPolicy(seconds_per_call=20.0)})


@pytest.fixture
def prepared(engine, signed_sum, tmp_path):
    ids = signed_sum.setup_workspace(engine, tmp_path / "bundles", final_seed=FINAL_SEED)
    engine.register_policy(_small_sandbox_policy())
    return ids


def test_existing_policies_and_their_hashes_are_unchanged() -> None:
    policies = builtin_experiment_policies()
    assert {policy_id: policies[policy_id].policy_hash for policy_id in FROZEN_HASHES} == FROZEN_HASHES
    for policy_id in FROZEN_HASHES:
        dumped = policies[policy_id].model_dump(mode="json")
        assert "sandbox" not in dumped and "max_job_wait_seconds" not in dumped
        assert not sandbox_available(policies[policy_id])


def test_qwen_loop_sandbox_v1_limits() -> None:
    policies = builtin_experiment_policies()
    policy, tools_only = policies[QWEN_LOOP_SANDBOX_POLICY_ID], policies["qwen_loop_v1"]
    assert sandbox_available(policy)
    assert policy.budget.limit("sandbox_calls") == 100
    assert policy.budget.limit("sandbox_seconds") == 7200
    assert policy.sandbox.seconds_per_call == 300 and policy.sandbox.max_output_bytes == 16384
    assert policy.max_job_wait_seconds == 600
    shared = {name: value for name, value in policy.budget.model_dump().items() if not name.startswith("sandbox")}
    assert shared == tools_only.budget.model_dump()
    assert policy.operations == tools_only.operations


def test_a_sandbox_and_its_counters_come_together() -> None:
    small = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID]
    with pytest.raises(ValueError, match="sandbox_calls"):
        ExperimentPolicy.model_validate({**small.model_dump(), "policy_id": "x", "sandbox": {"seconds_per_call": 1.0}})
    with pytest.raises(ValueError, match="sandbox"):
        ExperimentPolicy.model_validate({**small.model_dump(), "policy_id": "x",
                                         "budget": {**small.budget.model_dump(), "sandbox_calls": 3,
                                                    "sandbox_seconds": 10.0}})
    assert BudgetLimits().limit("sandbox_calls") == 0


def test_sandbox_usage_is_charged_to_its_own_counters(engine, prepared) -> None:
    grant = engine.grant_run("signed_sum_v1", SANDBOX_POLICY_ID, prepared["task_checkpoint_id"])
    eid = engine.create_experiment(grant.run_id, "signed_sum_v1", SANDBOX_POLICY_ID, "create").experiment_id
    reservation = engine.reserve(eid, "job_" + "1" * 32, Usage(sandbox_calls=1, sandbox_seconds=20.0))
    engine.settle(reservation.reservation_id, Usage(sandbox_calls=1, sandbox_seconds=2.5), "succeeded")
    budget = engine.get_experiment(eid).budget
    assert (budget["sandbox_calls"].consumed, budget["sandbox_seconds"].consumed) == (1, 2.5)
    assert budget["compute_seconds"].consumed == 0
    with pytest.raises(WSEError) as caught:
        engine.reserve(eid, "job_" + "2" * 32, Usage(sandbox_calls=1, sandbox_seconds=60.0))
    assert caught.value.code == ErrorCode.BUDGET_EXCEEDED


def test_a_sandbox_policy_marks_every_run_unverified_with_its_reason(workspace, prepared) -> None:
    rows = {}
    for policy_id in (SMALL_MODEL_POLICY_ID, SANDBOX_POLICY_ID):
        with Engine(workspace) as engine:
            run_id = engine.grant_run("signed_sum_v1", policy_id, prepared["task_checkpoint_id"]).run_id
        experiment_id = ReferenceRunner(workspace).run_experiment(
            run_id, "signed_sum_v1", policy_id, StrategyConfig(strategy="scripted_search@1"))
        with Engine(workspace) as engine:
            rows[policy_id] = summarize_trial(engine, experiment_id)
    assert (rows[SMALL_MODEL_POLICY_ID]["constraint_status"], rows[SMALL_MODEL_POLICY_ID]["constraint_reason"]) == (
        "verified", None)
    assert rows[SANDBOX_POLICY_ID]["constraint_status"] == "unverified"
    assert rows[SANDBOX_POLICY_ID]["constraint_reason"] == "sandbox code execution: WSE meters only its own calls"


def test_reports_separate_wse_metered_calls_from_sandbox_time(workspace, prepared) -> None:
    with Engine(workspace) as engine:
        run_id = engine.grant_run("signed_sum_v1", SANDBOX_POLICY_ID, prepared["task_checkpoint_id"]).run_id
    experiment_id = ReferenceRunner(workspace).run_experiment(
        run_id, "signed_sum_v1", SANDBOX_POLICY_ID, StrategyConfig(strategy="scripted_search@1"))
    with Engine(workspace) as engine:
        row = summarize_trial(engine, experiment_id)
    assert row["sandbox"] == {"calls": 0, "wall_seconds": 0.0,
                              "label": "sandbox wall-clock time (not metered computation)"}
    report = render_trials_markdown([row])
    assert "WSE-metered calls" in report
    assert "sandbox wall-clock time (not metered computation)" in report


def test_migration_12_backfills_sandbox_counters_and_the_reason_table(workspace, prepared) -> None:
    from wse.database import LATEST_SCHEMA_VERSION

    with Engine(workspace) as engine:
        grant = engine.grant_run("signed_sum_v1", SMALL_MODEL_POLICY_ID, prepared["task_checkpoint_id"])
        eid = engine.create_experiment(grant.run_id, "signed_sum_v1", SMALL_MODEL_POLICY_ID, "create").experiment_id
    with sqlite3.connect(workspace / "metadata.sqlite") as conn:
        conn.execute("DELETE FROM budget_ledger WHERE counter IN ('sandbox_calls', 'sandbox_seconds')")
        conn.execute("DROP TABLE run_constraints")
        conn.execute("PRAGMA user_version = 11")
    with Engine(workspace) as engine:
        assert engine.store.db.schema_version() == LATEST_SCHEMA_VERSION == 12
        budget = engine.get_experiment(eid).budget
        assert (budget["sandbox_calls"].limit, budget["sandbox_seconds"].limit) == (0, 0)
        tables = {row[0] for row in engine.store.db.conn.execute("SELECT name FROM sqlite_master")}
        assert "run_constraints" in tables
