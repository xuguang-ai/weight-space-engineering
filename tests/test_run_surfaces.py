"""What a run-bound Engine may compute and see (plan Sections 4.1, 4.2, 6.2).

A run-bound Engine is the agent-facing side of the service. It reaches
weights only through experiment-scoped, metered operations; it cannot open a
checkpoint (whose handle reads every weight and runs forward unmetered), and
its views of a patch omit the materialized after-values, which would reveal
base weights. Previews are counted because their deltas reveal the edited
region's base values, and inspection statistics are charged as compute time.
"""

from __future__ import annotations

import json

import pytest
from test_analysis import WIDE_CONFIG, DumpOperator, _extended, _full, _mlp_tensors, _normal

from wse.api import Engine
from wse.policy import (
    SMALL_MODEL_POLICY_ID,
    BudgetLimits,
    ExperimentPolicy,
    builtin_experiment_policies,
)
from wse.schemas import ErrorCode, WSEError

TASK = "signed_sum_v1"
W1 = "layers.0.weight"
FULL_W1 = [{"start": 0, "stop": 4, "step": 1}, {"start": 0, "stop": 2, "step": 1}]
CELL = [{"start": 0, "stop": 1, "step": 1}, {"start": 0, "stop": 1, "step": 1}]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _scale(factor: float, tensor: str = W1, selection=FULL_W1) -> dict:
    return {"op": "scale_slice", "tensor": tensor, "selection": selection, "factor": factor}


def _spec(base_id: str, *operations) -> dict:
    return {"base_checkpoint_id": base_id, "hypothesis": "h", "expected_behavior": "b",
            "operations": list(operations or [_scale(1.0)])}


def _consumed(engine: Engine, experiment_id: str, counter: str):
    return engine.get_experiment(experiment_id).budget[counter].consumed


@pytest.fixture
def base(engine, bundles):
    return engine.import_bundle(bundles["task"])


@pytest.fixture
def grant(engine, base):
    return engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base.checkpoint_id)


@pytest.fixture
def bound(workspace, engine, grant):
    with Engine(workspace, run_id=grant.run_id) as run_engine:
        yield run_engine


@pytest.fixture
def eid(bound, grant):
    return bound.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "create").experiment_id


# -- no unmetered weight access or forward on a run-bound engine ------------------------------------


def test_a_run_bound_engine_cannot_open_checkpoints_or_read_unmetered(bound, eid, base, engine):
    candidate = bound.apply_patch(eid, bound.preview_patch(eid, _spec(base.checkpoint_id, _scale(2.0)), "p").patch_id,
                                  "a")
    missing = "ckpt_" + "0" * 32
    calls = {
        "open_checkpoint": lambda c: bound.open_checkpoint(c),
        "read_tensor_slice": lambda c: bound.read_tensor_slice(c, W1, FULL_W1),
        "read_tensor": lambda c: bound.read_tensor(c, W1, FULL_W1),
        "analyze": lambda c: bound.analyze(c, "stats", W1, FULL_W1),
    }
    for name, call in calls.items():
        messages = {_expect(ErrorCode.POLICY_DENIED, call, c).message
                    for c in (base.checkpoint_id, candidate.checkpoint_id, missing)}
        # The same denial for every ID, so the call confirms nothing about existence.
        assert len(messages) == 1, (name, messages)
    # The metered, experiment-scoped operations remain available and are charged.
    before = _consumed(bound, eid, "tensor_read_calls")
    assert bound.read_model_tensor(eid, base.checkpoint_id, W1, FULL_W1).element_count == 8
    assert _consumed(bound, eid, "tensor_read_calls") == before + 1
    assert bound.analyze_model(eid, base.checkpoint_id, "stats", W1, FULL_W1).result["count"] == 8
    # The operator keeps direct access.
    assert engine.open_checkpoint(base.checkpoint_id).weights_hash == base.weights_hash


# -- patch views omit materialized values -----------------------------------------------------------


def test_run_views_of_a_patch_omit_materialized_values(bound, eid, base, engine, signed_sum):
    preview = bound.preview_patch(eid, _spec(base.checkpoint_id, _scale(1.0)), "identity")
    document = bound.find_patch(preview.patch_id).model_dump(mode="json")
    assert document["preview"] == preview.model_dump(mode="json")
    assert document["spec"]["operations"][0]["parameters"] == {"factor": 1.0}
    step = document["steps"][0]
    assert step["tensor"] == W1 and step["changed_elements"] == 0 and "values" not in step
    assert "values" not in json.dumps(document)
    _expect(ErrorCode.POLICY_DENIED, bound.read_patch, eid, preview.patch_id)
    # The operator view keeps the exact patch for replay and export.
    exact = engine.read_patch(eid, preview.patch_id)
    assert exact.patch.steps[0].values == signed_sum.task_tensors()[W1].flatten().tolist()


def test_a_patch_view_keeps_the_values_the_run_supplied(bound, eid, base):
    supplied = {"op": "set_slice", "tensor": W1, "selection": CELL, "values": [[0.25]]}
    preview = bound.preview_patch(eid, _spec(base.checkpoint_id, supplied), "set")
    document = bound.find_patch(preview.patch_id).model_dump(mode="json")
    assert document["spec"]["operations"][0]["parameters"]["values"] == [[0.25]]
    assert all("values" not in step for step in document["steps"])


# -- previews are counted ----------------------------------------------------------------------------


def test_builtin_and_example_policies_limit_previews(signed_sum):
    for policy in builtin_experiment_policies().values():
        assert policy.budget.patch_previews == 50
    assert signed_sum.load_policy()["budget"]["patch_previews"] == 50
    assert ExperimentPolicy.model_validate(signed_sum.load_policy()).budget.patch_previews == 50


def test_previews_are_counted_once_per_key_and_limited(workspace, engine, base):
    engine.register_policy(ExperimentPolicy(policy_id="two_previews_v1", mode="bounded_eval",
                                            budget=BudgetLimits(patch_previews=2)))
    grant = engine.grant_run(TASK, "two_previews_v1", base.checkpoint_id)
    with Engine(workspace, run_id=grant.run_id) as run_engine:
        eid = run_engine.create_experiment(grant.run_id, TASK, "two_previews_v1", "c").experiment_id
        first = run_engine.preview_patch(eid, _spec(base.checkpoint_id, _scale(2.0)), "p1")
        assert first.usage.patch_previews == 1
        assert run_engine.preview_patch(eid, _spec(base.checkpoint_id, _scale(2.0)), "p1") == first
        assert _consumed(run_engine, eid, "patch_previews") == 1
        run_engine.preview_patch(eid, _spec(base.checkpoint_id, _scale(3.0)), "p2")
        assert _consumed(run_engine, eid, "patch_previews") == 2
        error = _expect(ErrorCode.BUDGET_EXCEEDED, run_engine.preview_patch, eid,
                        _spec(base.checkpoint_id, _scale(4.0)), "p3")
        assert error.details["counter"] == "patch_previews"
        assert _consumed(run_engine, eid, "patch_previews") == 2
    stored = engine.experiments.db.conn.execute("SELECT COUNT(*) FROM patches WHERE experiment_id = ?", (eid,)).fetchone()[0]
    assert stored == 2


# -- inspection statistics are charged as compute time ---------------------------------------------


def test_inspection_statistics_are_charged_as_compute_time(bound, eid, base, engine):
    before = bound.get_experiment(eid).budget
    inspection = bound.inspect_model_details(eid, base.checkpoint_id)
    after = bound.get_experiment(eid).budget
    assert all(entry.statistics is not None for entry in inspection.tensor_statistics)
    assert inspection.usage.compute_seconds > 0
    assert after["compute_seconds"].consumed > before["compute_seconds"].consumed
    assert after["analysis_calls"].consumed == before["analysis_calls"].consumed
    engine.freeze(eid, base.checkpoint_id)
    frozen = bound.inspect_model_details(eid, base.checkpoint_id)
    assert all(entry.statistics is None and "FROZEN" in entry.unavailable for entry in frozen.tensor_statistics)
    assert bound.get_experiment(eid).budget["compute_seconds"].consumed == after["compute_seconds"].consumed


def test_inspection_stops_computing_at_the_per_operation_time_limit(workspace, engine, base):
    engine.register_policy(ExperimentPolicy(policy_id="instant_v1", mode="bounded_eval",
                                            budget=BudgetLimits(seconds_per_operation=1e-9)))
    grant = engine.grant_run(TASK, "instant_v1", base.checkpoint_id)
    with Engine(workspace, run_id=grant.run_id) as run_engine:
        eid = run_engine.create_experiment(grant.run_id, TASK, "instant_v1", "c").experiment_id
        inspection = run_engine.inspect_model_details(eid, base.checkpoint_id)
    reasons = [entry.unavailable or "" for entry in inspection.tensor_statistics]
    assert any("time limit" in reason for reason in reasons), reasons


# -- analysis result URIs are scoped to the run that produced them -------------------------------------


def test_analysis_result_uris_are_readable_only_by_their_run(workspace):
    registry, operations = _extended(("dump", DumpOperator()))
    tensors = _mlp_tensors(WIDE_CONFIG, _normal(5))
    policy = ExperimentPolicy(policy_id="dump_v1", mode="bounded_eval", operations=operations)
    with Engine(workspace, registry=registry) as operator:
        checkpoint = operator.import_model({"adapter": "mlp@1", "config": WIDE_CONFIG}, tensors).checkpoint_id
        operator.register_policy(policy)
        grants = [operator.grant_run(TASK, "dump_v1", checkpoint) for _ in range(2)]
    with Engine(workspace, registry=registry, run_id=grants[0].run_id) as owner:
        eid = owner.create_experiment(grants[0].run_id, TASK, "dump_v1", "c").experiment_id
        uri = owner.analyze_model(eid, checkpoint, "dump", W1, _full([64, 64])).result_uri
        assert uri is not None and uri.startswith("wse://analyses/")
        assert len(owner.read_analysis_result(uri)["values"]) == 64 * 64
    with Engine(workspace, registry=registry, run_id=grants[1].run_id) as other:
        other.create_experiment(grants[1].run_id, TASK, "dump_v1", "c")
        missing = "wse://analyses/" + "0" * 64
        hidden = _expect(ErrorCode.NOT_FOUND, other.read_analysis_result, uri).message.replace(uri, "<uri>")
        absent = _expect(ErrorCode.NOT_FOUND, other.read_analysis_result, missing).message.replace(missing, "<uri>")
        assert hidden == absent
    with Engine(workspace, registry=registry) as operator:
        assert len(operator.read_analysis_result(uri)["values"]) == 64 * 64
