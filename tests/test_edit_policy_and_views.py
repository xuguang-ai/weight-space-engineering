"""Managed edit policy, agent-facing patch views, and whole-tensor scaling (M2 plan Sections 5.1 and 5.5, Task 9).

* A managed preview runs only the edit operators that its experiment policy
  lists, whether an operation names a public alias or an exact reference.
  Direct use accepts the exact reference of any registered edit operator.
* The agent-facing views of a patch echo each inline vector of the M2 edit
  operators as ``{length, sha256}``, so a 2,048-value direction stays within
  the inline-result limit; the stored patch keeps the values.
* An operation policy may refuse edits that scale a whole tensor by one
  constant (``refuse_whole_tensor_scaling``); the option is off by default.

``project_out`` and ``add_low_rank`` need ``weights.direction_edit``, which
M1's MLP adapter lacks, so the signed-sum weights go through the test adapter
of ``tests/test_direction_ops.py`` that declares it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from mcp.server.mcpserver.exceptions import ToolError

from test_direction_ops import TEST_ADAPTER, _registry

from wse.api import Engine, apply_operator
from wse.mcp_server import WSEMCPServer
from wse.patches import materialize_patch
from wse.policy import (
    SMALL_MODEL_POLICY_ID,
    WEIGHTS_ONLY_POLICY_ID,
    ExperimentPolicy,
    OperationPolicy,
    builtin_experiment_policies,
    default_operation_policy,
    json_size,
)
from wse.schemas import ErrorCode, WSEError, json_hash

LIMIT = 31 * 1024  # the default inline-result limit: 32 KiB less the 1 KiB response envelope
TASK = "signed_sum_v1"
W0 = "layers.0.weight"  # [4, 2] in the signed-sum model
W1 = "layers.1.weight"  # [2, 4]
NO_INFERENCE = {"evaluation_batches": 0, "examples_per_batch": 0, "evaluated_examples": 0, "model_forward_calls": 0}
EDITS = (default_operation_policy().with_edit("project_out", "project_out@1").with_edit("fill_slice", "fill_slice@1")
         .with_edit("add_low_rank", "add_low_rank@1"))
REFUSING = OperationPolicy.model_validate({**EDITS.model_dump(mode="json"), "refuse_whole_tensor_scaling": True})


def _sel(*ranges):
    return [{"start": start, "stop": stop, "step": 1} for start, stop in ranges]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _consumed(engine, experiment_id, counter):
    return engine.get_experiment(experiment_id).budget[counter].consumed


@pytest.fixture
def task_model(signed_sum):
    from wse.api import load_model

    return load_model(signed_sum.task_tensors(), TEST_ADAPTER, signed_sum.CONFIG, registry=_registry())


# -- exact references in managed previews ---------------------------------------------------------------------------

REQUESTS = {
    "project_out": {"tensor": W1, "selection": _sel((0, 2), (0, 4)), "direction": [1.0, 1.0], "columns": [1, 3]},
    "fill_slice": {"tensor": W1, "selection": _sel((0, 1), (0, 4)), "value": 0.0},
    "add_low_rank": {"tensor": W0, "selection": _sel((0, 4), (0, 2)), "u": [[1.0, 0.5, -0.5, 0.25]],
                     "v": [[0.1, -0.2]]},
}


def test_managed_previews_run_only_the_edit_operators_their_policy_lists(workspace, signed_sum, task_model):
    registry = _registry()
    with Engine(workspace, registry=registry) as operator:
        base = operator.import_model({"adapter": TEST_ADAPTER, "config": signed_sum.CONFIG},
                                     signed_sum.task_tensors()).checkpoint_id
        operator.register_policy(ExperimentPolicy(policy_id="m2_edits_v1", mode="weights_only", operations=EDITS,
                                                  budget={"candidate_commits": 1, **NO_INFERENCE}))
        grants = {policy: operator.grant_run(TASK, policy, base) for policy in (SMALL_MODEL_POLICY_ID, "m2_edits_v1")}

    def spec(op, request):
        return {"base_checkpoint_id": base, "hypothesis": "This edit repairs the output sign.",
                "operations": [{"op": op, **request}]}

    # small_model_v1 lists set_slice and scale_slice. An exact reference to any other edit operator is refused like
    # an alias the policy does not list, through the Python SDK and through MCP, and nothing is created or charged.
    grant = grants[SMALL_MODEL_POLICY_ID]
    with Engine(workspace, run_id=grant.run_id, registry=registry) as bound:
        eid = bound.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "c").experiment_id
        server = WSEMCPServer(bound)
        for name, request in REQUESTS.items():
            alias = _expect(ErrorCode.POLICY_DENIED, bound.preview_patch, eid, spec(name, request), f"alias-{name}")
            exact = _expect(ErrorCode.POLICY_DENIED, bound.preview_patch, eid, spec(f"{name}@1", request),
                            f"exact-{name}")
            assert exact.message == alias.message.replace(name, f"{name}@1")
            assert exact.details["permitted_operations"] == ["scale_slice", "set_slice"]
            assert exact.details["operation_index"] == 0
            with pytest.raises(ToolError) as excinfo:
                server.wse_preview_patch(experiment_id=eid, patch_spec=spec(f"{name}@1", request),
                                         idempotency_key=f"mcp-{name}")
            error = json.loads(str(excinfo.value))["error"]
            assert (error["code"], error["message"]) == ("POLICY_DENIED", exact.message)
        # A version the policy does not list is refused too, whether or not it is registered.
        _expect(ErrorCode.POLICY_DENIED, bound.preview_patch, eid,
                spec("scale_slice@2", {"tensor": W1, "selection": _sel((1, 2), (0, 4)), "factor": -1.0}), "v2")
        assert _consumed(bound, eid, "patch_previews") == 0
        assert bound.store.db.conn.execute("SELECT COUNT(*) FROM patches WHERE experiment_id = ?",
                                           (eid,)).fetchone()[0] == 0
        failures = [event for event in bound.audit_log(eid) if event["event_type"] == "patch.preview.failed"]
        assert len(failures) == 3 * len(REQUESTS) + 1
        assert {event["payload"]["code"] for event in failures} == {"POLICY_DENIED"}
        # The exact reference of a listed operator is accepted.
        listed = bound.preview_patch(eid, spec("scale_slice@1", {"tensor": W1, "selection": _sel((1, 2), (0, 4)),
                                                                 "factor": -1.0}), "listed")
        assert str(listed.operations[0].operator) == "scale_slice@1"
    # A policy that grants the operators accepts their aliases and their exact references alike.
    grant = grants["m2_edits_v1"]
    with Engine(workspace, run_id=grant.run_id, registry=registry) as bound:
        eid = bound.create_experiment(grant.run_id, TASK, "m2_edits_v1", "c").experiment_id
        for name, request in REQUESTS.items():
            by_alias = bound.preview_patch(eid, spec(name, request), f"alias-{name}")
            by_reference = bound.preview_patch(eid, spec(f"{name}@1", request), f"exact-{name}")
            assert by_alias.result_weights_hash == by_reference.result_weights_hash
            assert [str(operation.operator) for operation in by_reference.operations] == [f"{name}@1"]
            # Direct use accepts the exact reference of any registered edit operator, whatever the policy.
            direct = apply_operator(task_model, f"{name}@1", request)
            assert direct.patch.result_weights_hash == by_reference.result_weights_hash
        candidate = bound.apply_patch(eid, by_reference.patch_id, "apply")
        assert candidate.weights_hash == by_reference.result_weights_hash


def test_invalid_patch_specs_give_bounded_errors_and_audit_records(engine, bundles):
    base = engine.import_bundle(bundles["task"]).checkpoint_id
    grant = engine.grant_run(TASK, SMALL_MODEL_POLICY_ID, base)
    eid = engine.create_experiment(grant.run_id, TASK, SMALL_MODEL_POLICY_ID, "c").experiment_id
    spec = {"base_checkpoint_id": base, "hypothesis": "Many operations.",
            "operations": [{"op": "scale_slice", "selection": [], "factor": 2.0}] * 2000}  # each lacks a tensor
    error = _expect(ErrorCode.INVALID_PATCH, engine.preview_patch, eid, spec, "p1")
    assert len(error.message) < 2048 and "1995 more problems" in error.message
    assert error.details["error_count"] == 2000 and len(error.details["errors"]) == 5
    assert error.details["errors"][0]["loc"] == ["operations", 0, "tensor"]
    (failure,) = [event for event in engine.audit_log(eid) if event["event_type"] == "patch.preview.failed"]
    assert json_size(failure["payload"]) < 4096


# -- agent-facing patch views ---------------------------------------------------------------------------------------


def test_patch_views_echo_inline_vectors_by_length_and_hash_within_the_inline_limit(workspace):
    rows = 2048
    direction = [-1.2345679e-05] * rows  # every value needs 14 characters in its shortest digits, the worst case
    column = [-1.2345678e-05] * rows
    assert len(json.dumps(direction[0])) == len(json.dumps(column[0])) == 14
    weights = torch.from_numpy(np.random.default_rng(0).normal(0.0, 0.02, size=(rows, 2)).astype(np.float32))
    registry = _registry()
    with Engine(workspace, registry=registry) as operator:
        base = operator.import_model({"adapter": TEST_ADAPTER, "config": {"layer_sizes": [2, rows],
                                                                          "activation": "relu", "bias": False}},
                                     {W0: weights}).checkpoint_id
        operator.register_policy(ExperimentPolicy(
            policy_id="views_v1", mode="weights_only", operations=EDITS.with_analysis("svd_top_k", "svd_top_k@1"),
            budget={"candidate_commits": 1, **NO_INFERENCE}))
        grant = operator.grant_run("views_task", "views_v1", base)
    full = _sel((0, rows), (0, 2))
    views = {}
    with Engine(workspace, run_id=grant.run_id, registry=registry) as bound:
        eid = bound.create_experiment(grant.run_id, "views_task", "views_v1", "c").experiment_id
        svd = bound.analyze_model(eid, base, "svd_top_k", W0, full, parameters={"k": 2})
        stored = bound.read_analysis_result(svd.result_uri)
        left, right = stored["left_vectors"][0], stored["right_vectors"][0]
        patches = {
            "project_out": {"op": "project_out", "tensor": W0, "selection": full, "direction": direction,
                            "columns": [0, 1]},
            "add_low_rank": {"op": "add_low_rank", "tensor": W0, "selection": full, "alpha": 0.5,
                             "u": [column, f"{svd.result_uri}#/left_vectors/0"],
                             "v": [[0.5, -0.25], f"{svd.result_uri}#/right_vectors/0"]},
        }
        for name, operation in patches.items():
            preview = bound.preview_patch(eid, {"base_checkpoint_id": base, "operations": [operation],
                                                "hypothesis": "Removing this direction repairs the output."},
                                          f"p-{name}")
            document = bound.find_patch(preview.patch_id)  # the wse://patches/{patch_id} resource
            for view in (preview, bound.get_patch(eid, preview.patch_id), document):
                assert json_size(view.model_dump(mode="json")) <= LIMIT, (name, type(view).__name__)
            views[name] = preview.patch_id, document.spec.operations[0].parameters
        assert json_size({"events": bound.audit_log(eid)}) <= LIMIT

    def summary(values):
        return {"length": len(values), "sha256": json_hash(values)}

    def record(pointer, values):
        return {"reference": f"{svd.result_uri}#{pointer}", **summary(values)}

    assert views["project_out"][1] == {"direction": summary(direction), "columns": [0, 1], "alpha": 1.0}
    assert views["add_low_rank"][1] == {"u": [summary(column), record("/left_vectors/0", left)],
                                        "v": [summary([0.5, -0.25]), record("/right_vectors/0", right)],
                                        "alpha": 0.5}
    # The stored patch, which only the operator reads whole, keeps every value in its frozen spec and its steps.
    with Engine(workspace, registry=registry) as operator:
        artifact = operator.read_patch(eid, views["project_out"][0])
        assert artifact.spec.operations[0].parameters["direction"] == direction
        assert artifact.patch.steps[0].parameters["direction"] == direction
        low_rank = operator.read_patch(eid, views["add_low_rank"][0]).patch.steps[0].parameters
        assert low_rank["u"] == [column, record("/left_vectors/0", left)]
        assert low_rank["v"] == [[0.5, -0.25], record("/right_vectors/0", right)]


# -- whole-tensor scaling -------------------------------------------------------------------------------------------


def test_whole_tensor_scaling_is_refused_only_under_the_policy_option(task_model):
    whole = {"tensor": W0, "selection": _sel((0, 4), (0, 2)), "factor": 2.0}
    # Off by default: direct use and the built-in policies scale whole tensors.
    assert apply_operator(task_model, "scale_slice", whole).patch.steps[0].changed_elements > 0
    assert default_operation_policy().refuse_whole_tensor_scaling is None
    for op in ("scale_slice", "scale_slice@1"):
        for factor in (2.0, -1.0, 1.0, 0.0):
            error = _expect(ErrorCode.POLICY_DENIED, apply_operator, task_model, op, {**whole, "factor": factor},
                            policy=REFUSING)
            # What is refused, why, and what to do instead.
            for fragment in (repr(W0), "whole", "refuse_whole_tensor_scaling", "rows", "project_out"):
                assert fragment in error.message, (fragment, error.message)
            assert error.details == {"tensor": W0, "shape": [4, 2], "operator": "scale_slice@1",
                                     "policy_option": "refuse_whole_tensor_scaling",
                                     "permitted_operations": sorted(REFUSING.edits)}
    bias = {"tensor": "layers.0.bias", "selection": _sel((0, 4)), "factor": 0.5}
    _expect(ErrorCode.POLICY_DENIED, apply_operator, task_model, "scale_slice", bias, policy=REFUSING)
    # A proper part of a tensor may still be scaled.
    for request in ({**whole, "selection": _sel((0, 3), (0, 2))}, {**whole, "selection": _sel((0, 4), (1, 2))},
                    {**bias, "selection": _sel((1, 4))}):
        apply_operator(task_model, "scale_slice", request, policy=REFUSING)
    # The other edit operators may cover a whole tensor: none of them scales it by one constant.
    for op, parameters in (("set_slice", {"values": [[0.5, 0.5]] * 4}), ("fill_slice", {"value": 0.5}),
                           ("add_low_rank", {"u": [[1.0] * 4], "v": [[0.5, 0.5]]}),
                           ("project_out", {"direction": [1.0] * 4, "columns": [0, 1]})):
        apply_operator(task_model, op, {"tensor": W0, "selection": _sel((0, 4), (0, 2)), **parameters},
                       policy=REFUSING)
    # In a patch the refusal names its operation.
    error = _expect(ErrorCode.POLICY_DENIED, materialize_patch, task_model,
                    [{"op": "scale_slice", **whole, "selection": _sel((0, 3), (0, 2))}, {"op": "scale_slice", **whole}],
                    policy=REFUSING)
    assert error.details["operation_index"] == 1


def test_the_whole_tensor_scaling_option_is_stored_with_its_policy_and_off_by_default(workspace, bundles):
    # Unset, the option is absent from a policy's JSON, so built-in and stored policies keep their hashes.
    assert "refuse_whole_tensor_scaling" not in default_operation_policy().model_dump(mode="json")
    policies = builtin_experiment_policies()
    assert policies[SMALL_MODEL_POLICY_ID].policy_hash == (
        "sha256:d8761471bfb88e1d322da1ca06003729886daafa028963f1dcf7ef807cc218b5")
    assert policies[WEIGHTS_ONLY_POLICY_ID].policy_hash == (
        "sha256:88d3c1e4f73e0eb8204c30b65aceb753608114c4ed05e074302db630e665fa75")
    stored = ExperimentPolicy.model_validate_json(policies[SMALL_MODEL_POLICY_ID].canonical_json())
    assert stored.policy_hash == policies[SMALL_MODEL_POLICY_ID].policy_hash
    assert stored.operations.refuse_whole_tensor_scaling is None
    refusing = ExperimentPolicy(policy_id="no_whole_scaling_v1", mode="weights_only", operations=REFUSING,
                                budget={"candidate_commits": 1, **NO_INFERENCE})
    assert refusing.operations.model_dump(mode="json")["refuse_whole_tensor_scaling"] is True
    unset = refusing.model_copy(update={"operations": EDITS})
    assert refusing.policy_hash != unset.policy_hash
    with Engine(workspace) as engine:
        engine.register_policy(refusing)
        assert engine.get_policy("no_whole_scaling_v1") == refusing
        base = engine.import_bundle(bundles["task"]).checkpoint_id
        grant = engine.grant_run(TASK, "no_whole_scaling_v1", base)
        eid = engine.create_experiment(grant.run_id, TASK, "no_whole_scaling_v1", "c").experiment_id

        def spec(selection):
            return {"base_checkpoint_id": base, "hypothesis": "A smaller first layer helps.",
                    "operations": [{"op": "scale_slice", "tensor": W0, "selection": selection, "factor": 0.5}]}

        error = _expect(ErrorCode.POLICY_DENIED, engine.preview_patch, eid, spec(_sel((0, 4), (0, 2))), "whole")
        assert error.details["operation_index"] == 0 and "refuse_whole_tensor_scaling" in error.message
        assert _consumed(engine, eid, "patch_previews") == 0
        assert engine.preview_patch(eid, spec(_sel((0, 2), (0, 2))), "part").changed_elements > 0
