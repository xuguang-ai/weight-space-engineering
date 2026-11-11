"""Edit operators, materialized patches, exact replay, and managed previews/applies."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wse.adapters.base import CAP_FORWARD_LOGITS, CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT, AdapterCapabilities
from wse.adapters.mlp import MLPAdapter
from wse.api import Engine, apply_operator, load_model
from wse.operators.base import BaseEditOperator, OperatorParameters, make_spec
from wse.operators.tensor import ScaleSliceOperator, SetSliceOperator
from wse.patches import materialize_patch, replay_patch
from wse.policy import ExperimentPolicy, default_operation_policy
from wse.registry import Registry, builtin_registry
from wse.schemas import ErrorCode, PatchStep, UsageContract, WSEError
from wse.tensors import tensor_hash, weights_hash


def _sel(*ranges):
    return [{"start": start, "stop": stop, "step": 1} for start, stop in ranges]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


@pytest.fixture
def task_model(signed_sum):
    return load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)


ROW1 = _sel((1, 2), (0, 4))
ORACLE_SCALE = {"op": "scale_slice", "tensor": "layers.1.weight", "selection": ROW1, "factor": -1.0}


# -- known values: set_slice and scale_slice ------------------------------------------


def test_scale_slice_known_values_and_changed_element_count(task_model):
    result = apply_operator(task_model, "scale_slice", {"tensor": "layers.0.weight", "selection": _sel((0, 2), (0, 2)),
                                                        "factor": 0.5})
    after = result.candidate.tensor("layers.0.weight")
    assert after.tolist() == [[0.5, 0.0], [-0.5, 0.0], [0.0, 1.0], [0.0, -1.0]]
    # Zeros scaled by 0.5 keep their bytes and are not counted.
    (step,) = result.patch.steps
    assert step.changed_elements == 2 and step.values == [0.5, 0.0, -0.5, 0.0]
    assert str(step.operator) == "scale_slice@1" and step.op == "scale_slice"
    assert step.parameters == {"factor": 0.5}
    diff = {d.tensor: d for d in result.diff}
    assert diff["layers.0.weight"].changed_elements == 2
    assert diff["layers.0.weight"].l2_delta == pytest.approx(np.sqrt(0.5))
    assert diff["layers.0.weight"].max_abs_delta == 0.5
    assert diff["layers.0.weight"].relative_l2_delta == pytest.approx(np.sqrt(0.5) / 2.0)
    assert diff["layers.1.weight"].changed_elements == 0 and diff["layers.1.weight"].relative_l2_delta == 0.0
    assert diff["layers.0.bias"].relative_l2_delta is None  # zero base norm
    assert result.management == "unmanaged"
    # The input model is untouched.
    assert task_model.tensor("layers.0.weight").tolist() == [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]


def test_scale_slice_is_computed_in_float32(task_model):
    result = apply_operator(task_model, "scale_slice@1", {"tensor": "layers.0.weight", "selection": _sel((0, 1), (0, 1)),
                                                          "factor": 0.1})
    value = result.candidate.tensor("layers.0.weight")[0, 0]
    assert value.item() == float(np.float32(1.0) * np.float32(0.1))
    assert result.patch.steps[0].values == [0.1]  # shortest float32 digits


def test_oracle_scale_restores_the_reference_weights(task_model, signed_sum):
    reference = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
    request = {k: v for k, v in ORACLE_SCALE.items() if k != "op"}
    result = apply_operator(task_model, "scale_slice", request)
    assert result.candidate.weights_hash == reference.weights_hash
    assert result.patch.result_weights_hash == reference.weights_hash
    assert result.patch.base == task_model.identity
    assert sum(d.changed_elements for d in result.diff) == 4


def test_set_slice_known_values(task_model, signed_sum):
    result = apply_operator(task_model, "set_slice", {"tensor": "layers.1.weight", "selection": ROW1,
                                                      "values": [[1, -1, 1, -1]]})
    reference = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
    assert result.candidate.weights_hash == reference.weights_hash
    (step,) = result.patch.steps
    assert step.values == [1.0, -1.0, 1.0, -1.0] and step.changed_elements == 4
    bias = apply_operator(task_model, "set_slice", {"tensor": "layers.1.bias", "selection": _sel((0, 2)),
                                                    "values": [0.25, -3]})
    assert bias.candidate.tensor("layers.1.bias").tolist() == [0.25, -3.0]


def test_no_op_edits_are_recorded_but_change_nothing(task_model):
    result = apply_operator(task_model, "scale_slice", {"tensor": "layers.1.weight", "selection": ROW1, "factor": 1})
    assert result.candidate.weights_hash == task_model.weights_hash
    (step,) = result.patch.steps
    assert step.changed_elements == 0 and step.before_hash == step.after_hash
    assert all(d.changed_elements == 0 and d.l2_delta == 0.0 for d in result.diff)


def test_changed_elements_compare_stored_bytes(task_model):
    # Writing -0.0 over 0.0 changes the stored bytes but not the value.
    result = apply_operator(task_model, "set_slice", {"tensor": "layers.0.bias", "selection": _sel((0, 2)),
                                                      "values": [-0.0, 0.0]})
    assert result.patch.steps[0].changed_elements == 1
    diff = {d.tensor: d for d in result.diff}["layers.0.bias"]
    assert diff.changed_elements == 1 and diff.l2_delta == 0.0 and diff.before_hash != diff.after_hash


def test_edits_write_only_the_selected_region(signed_sum):
    rng = np.random.default_rng(1)
    config = {"layer_sizes": [8, 6, 3], "activation": "relu", "bias": True}
    tensors = {
        "layers.0.weight": rng.standard_normal((6, 8)).astype(np.float32),
        "layers.0.bias": rng.standard_normal(6).astype(np.float32),
        "layers.1.weight": rng.standard_normal((3, 6)).astype(np.float32),
        "layers.1.bias": rng.standard_normal(3).astype(np.float32),
    }
    model = load_model(tensors, "mlp@1", config)
    result = apply_operator(model, "scale_slice", {"tensor": "layers.0.weight", "selection": _sel((2, 4), (1, 5)),
                                                   "factor": 3.0})
    before = tensors["layers.0.weight"]
    after = result.candidate.tensor("layers.0.weight").numpy()
    mask = np.zeros_like(before, dtype=bool)
    mask[2:4, 1:5] = True
    assert after[~mask].tobytes() == before[~mask].tobytes()
    assert np.array_equal(after[mask], (before[2:4, 1:5] * np.float32(3.0)).reshape(-1))
    for name in ("layers.0.bias", "layers.1.weight", "layers.1.bias"):
        assert result.candidate.tensor(name).numpy().tobytes() == tensors[name].tobytes()


def test_multi_operation_patch_applies_in_order_to_one_snapshot(task_model):
    result = materialize_patch(task_model, [
        {"op": "set_slice", "tensor": "layers.1.weight", "selection": _sel((0, 1), (0, 2)), "values": [[2, 3]]},
        {"op": "scale_slice", "tensor": "layers.1.weight", "selection": _sel((0, 1), (0, 4)), "factor": 2},
        {"op": "scale_slice", "tensor": "layers.0.weight", "selection": _sel((3, 4), (0, 2)), "factor": -1},
    ])
    w2 = result.candidate.tensor("layers.1.weight").tolist()
    assert w2 == [[4.0, 6.0, -2.0, 2.0], [-1.0, 1.0, -1.0, 1.0]]
    assert result.candidate.tensor("layers.0.weight").tolist()[3] == [0.0, 1.0]
    assert [step.op for step in result.patch.steps] == ["set_slice", "scale_slice", "scale_slice"]
    # Negating row 3 turns its 0.0 into -0.0: different bytes, so both elements count.
    assert [step.changed_elements for step in result.patch.steps] == [2, 4, 2]
    # Step hashes chain: each step starts from the previous step's tensor.
    assert result.patch.steps[1].before_hash == result.patch.steps[0].after_hash
    assert replay_patch(task_model, result.patch).weights_hash == result.candidate.weights_hash


def test_later_failure_in_a_multi_operation_patch_produces_no_candidate(task_model):
    error = _expect(ErrorCode.INVALID_PATCH, materialize_patch, task_model, [
        ORACLE_SCALE,
        {"op": "scale_slice", "tensor": "layers.0.weight", "selection": _sel((0, 1), (0, 2)), "factor": 3e38},
        {"op": "scale_slice", "tensor": "layers.0.weight", "selection": _sel((0, 1), (0, 2)), "factor": 10.0},
    ])
    assert error.details["operation_index"] == 2 and error.details["nonfinite_count"] == 1
    assert "operation 2" in error.message


# -- invalid edits --------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [-1, 1, -1, 1, -1],  # a flat list for a [1, 4] region must hold its 4 values (M2 accepts exactly 4)
        [[-1, 1, -1]],  # too few
        [[1.0]],  # no broadcasting
        [[-1, 1], [-1, 1]],  # wrong shape
        [[-1, 1, -1, [1]]],  # ragged
        [[True, 1, -1, 1]],  # booleans are not numbers
        [["1", 1, -1, 1]],
        [[float("nan"), 1, -1, 1]],
        [[float("inf"), 1, -1, 1]],
        [[1e39, 1, -1, 1]],  # outside the float32 range
    ],
)
def test_set_slice_rejects_mismatched_or_nonfinite_values(task_model, values):
    error = _expect(ErrorCode.INVALID_PATCH, materialize_patch, task_model,
                    [{"op": "set_slice", "tensor": "layers.1.weight", "selection": ROW1, "values": values}])
    assert error.details["operation_index"] == 0


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), 1e39, True, "2", None])
def test_scale_slice_rejects_nonfinite_or_non_numeric_factors(task_model, factor):
    _expect(ErrorCode.INVALID_ARGUMENT, apply_operator, task_model, "scale_slice",
            {"tensor": "layers.1.weight", "selection": ROW1, "factor": factor})


@pytest.mark.parametrize(
    "operation, code",
    [
        ({"tensor": "layers.9.weight", "selection": ROW1, "factor": 2.0}, ErrorCode.INVALID_PATCH),
        ({"tensor": "layers.1.weight ", "selection": ROW1, "factor": 2.0}, ErrorCode.INVALID_PATCH),
        ({"tensor": "layers.1.weight", "selection": _sel((1, 3), (0, 4)), "factor": 2.0}, ErrorCode.INVALID_PATCH),
        ({"tensor": "layers.1.weight", "selection": _sel((1, 1), (0, 4)), "factor": 2.0}, ErrorCode.INVALID_PATCH),
        ({"tensor": "layers.1.weight", "selection": "W[1]", "factor": 2.0}, ErrorCode.INVALID_PATCH),
        ({"tensor": "layers.1.weight", "selection": ROW1, "factor": 2.0, "extra": 1}, ErrorCode.INVALID_PATCH),
        ({"tensor": "layers.1.weight", "selection": ROW1}, ErrorCode.INVALID_PATCH),
    ],
)
def test_invalid_operations_are_actionable(task_model, operation, code):
    error = _expect(code, materialize_patch, task_model, [{"op": "scale_slice", **operation}])
    assert error.details["operation_index"] == 0 and error.message.startswith("operation 0 (scale_slice)")


def test_operations_must_be_permitted_and_registered(task_model):
    _expect(ErrorCode.POLICY_DENIED, materialize_patch, task_model,
            [{"op": "add_low_rank", "tensor": "layers.1.weight", "selection": ROW1}])
    _expect(ErrorCode.POLICY_DENIED, materialize_patch, task_model,
            [{"op": "tensor_stats@1", "tensor": "layers.1.weight", "selection": ROW1}])
    policy = default_operation_policy().with_edit("scale_v2", "scale_slice@2")
    _expect(ErrorCode.NOT_FOUND, materialize_patch, task_model,
            [{"op": "scale_v2", "tensor": "layers.1.weight", "selection": ROW1, "factor": 2.0}], policy=policy)
    _expect(ErrorCode.INVALID_PATCH, materialize_patch, task_model, [])


def test_patch_size_limits(task_model):
    policy = default_operation_policy().with_limits(max_patch_operations=2, max_patch_elements=6)
    _expect(ErrorCode.INVALID_PATCH, materialize_patch, task_model, [ORACLE_SCALE] * 3, policy=policy)
    error = _expect(ErrorCode.INVALID_PATCH, materialize_patch, task_model, [ORACLE_SCALE] * 2, policy=policy)
    assert error.details["max_patch_elements"] == 6


# -- extension contract and capabilities ---------------------------------------------


class NegateOperator(BaseEditOperator):
    """A separately supplied edit operator."""

    spec = make_spec(
        operator_id="negate_slice", version="1", effect="edit", description="Negate a selection.",
        required_capabilities=[CAP_WEIGHTS_SLICE_EDIT], parameters_model=OperatorParameters,
        result_model=PatchStep, usage=UsageContract(counter="candidate_commits"),
    )
    calls = 0

    def apply(self, region, parameters):
        type(self).calls += 1
        assert not region.values.flags.writeable
        return -region.values.astype(np.float32)


def test_separately_supplied_edit_operator_materializes_standard_patches(task_model, signed_sum):
    registry = builtin_registry()
    registry.register("operator", "negate_slice", "1", NegateOperator())
    result = apply_operator(task_model, "negate_slice@1", {"tensor": "layers.1.weight", "selection": ROW1},
                            registry=registry)
    reference = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
    assert result.candidate.weights_hash == reference.weights_hash
    assert str(result.patch.steps[0].operator) == "negate_slice@1"
    # Replay needs no registry: it writes stored values only.
    NegateOperator.calls = 0
    assert replay_patch(task_model, result.patch).weights_hash == reference.weights_hash
    assert NegateOperator.calls == 0


class ReadOnlyAdapter(MLPAdapter):
    adapter_id = "readonly_mlp"
    version = "1"

    def capabilities(self):
        return AdapterCapabilities(capabilities=frozenset({CAP_WEIGHTS_READ, CAP_FORWARD_LOGITS}))


def test_unsupported_adapter_capabilities_fail_before_candidate_creation(signed_sum, monkeypatch):
    registry = builtin_registry()
    registry.register("adapter", "readonly_mlp", "1", ReadOnlyAdapter())
    model = load_model(signed_sum.task_tensors(), "readonly_mlp@1", signed_sum.CONFIG, registry=registry)

    def never(*args, **kwargs):
        raise AssertionError("no candidate may be computed")

    monkeypatch.setattr(ScaleSliceOperator, "apply", never)
    error = _expect(ErrorCode.UNSUPPORTED_CAPABILITY, materialize_patch, model, [ORACLE_SCALE], registry=registry)
    assert error.details["missing_capabilities"] == [CAP_WEIGHTS_SLICE_EDIT]


# -- replay -----------------------------------------------------------------------------


def test_replay_does_not_execute_the_editing_operator(task_model, monkeypatch):
    result = materialize_patch(task_model, [ORACLE_SCALE, {"op": "set_slice", "tensor": "layers.0.bias",
                                                           "selection": _sel((0, 4)), "values": [1, 2, 3, 4]}])

    def never(*args, **kwargs):
        raise AssertionError("replay must not run an editing operator")

    monkeypatch.setattr(ScaleSliceOperator, "apply", never)
    monkeypatch.setattr(SetSliceOperator, "apply", never)
    replayed = replay_patch(task_model, result.patch)
    assert replayed.weights_hash == result.candidate.weights_hash
    for name in task_model.tensor_names():
        assert replayed.tensor(name).numpy().tobytes() == result.candidate.tensor(name).numpy().tobytes()


def test_replay_rejects_another_base_or_a_tampered_patch(task_model, signed_sum):
    result = apply_operator(task_model, "scale_slice", {k: v for k, v in ORACLE_SCALE.items() if k != "op"})
    other = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
    _expect(ErrorCode.VERSION_CONFLICT, replay_patch, other, result.patch)
    step = result.patch.steps[0].model_copy(update={"values": [-1.0, 1.0, -1.0, 2.0]})
    tampered = result.patch.model_copy(update={"steps": [step]})
    _expect(ErrorCode.INVALID_PATCH, replay_patch, task_model, tampered)


def test_patch_round_trips_through_json_exactly(task_model):
    rng = np.random.default_rng(5)
    values = rng.normal(0, 0.02, size=(1, 4)).astype(np.float32)
    result = apply_operator(task_model, "set_slice", {"tensor": "layers.1.weight", "selection": ROW1,
                                                      "values": values.tolist()})
    restored = type(result.patch).model_validate_json(result.patch.canonical_json())
    assert replay_patch(task_model, restored).tensor("layers.1.weight")[1].numpy().tobytes() == values.tobytes()


# -- managed path ---------------------------------------------------------------------


def _experiment(engine, checkpoint_id, policy_id="small_model_v1"):
    grant = engine.grant_run("signed_sum_v1", policy_id, checkpoint_id)
    return engine.create_experiment(grant.run_id, "signed_sum_v1", policy_id, "create-1")


def _spec(base, *operations, **extra):
    return {"base_checkpoint_id": base, "hypothesis": "Row 1 of the output layer has the wrong sign.",
            "expected_behavior": "Accuracy rises.", "development_suite_id": "example_dev_v1",
            "operations": list(operations), **extra}


def test_direct_and_managed_edits_yield_identical_tensor_bytes(engine, bundles, task_model):
    base = engine.import_bundle(bundles["task"])
    experiment = _experiment(engine, base.checkpoint_id)
    operations = [ORACLE_SCALE, {"op": "set_slice", "tensor": "layers.0.bias", "selection": _sel((1, 3)),
                                 "values": [0.5, -0.25]}]
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, *operations), "p1")
    candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
    direct = materialize_patch(task_model, operations)
    assert candidate.weights_hash == direct.candidate.weights_hash == preview.result_weights_hash
    stored = engine.store.load_tensors(candidate.checkpoint_id)
    for name in task_model.tensor_names():
        assert stored[name].numpy().tobytes() == direct.candidate.tensor(name).numpy().tobytes()
    artifact = engine.read_patch(experiment.experiment_id, preview.patch_id)
    assert artifact.patch == direct.patch
    assert [str(op.operator) for op in artifact.spec.operations] == ["scale_slice@1", "set_slice@1"]


def test_managed_apply_replays_the_stored_patch_without_operators(engine, bundles, monkeypatch):
    base = engine.import_bundle(bundles["task"])
    experiment = _experiment(engine, base.checkpoint_id)
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, ORACLE_SCALE), "p1")

    def never(*args, **kwargs):
        raise AssertionError("apply must replay the materialized patch")

    monkeypatch.setattr(ScaleSliceOperator, "apply", never)
    candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
    assert candidate.weights_hash == preview.result_weights_hash


def test_managed_multi_operation_failure_leaves_no_patch_and_no_candidate(engine, bundles):
    base = engine.import_bundle(bundles["task"])
    experiment = _experiment(engine, base.checkpoint_id)
    spec = _spec(base.checkpoint_id, ORACLE_SCALE,
                 {"op": "set_slice", "tensor": "layers.1.weight", "selection": ROW1, "values": [[1, 2]]})
    error = _expect(ErrorCode.INVALID_PATCH, engine.preview_patch, experiment.experiment_id, spec, "p1")
    assert error.details["operation_index"] == 1
    conn = engine.store.db.conn
    assert conn.execute("SELECT COUNT(*) FROM patches").fetchone()[0] == 0
    assert engine.store.list_ready() == [base.checkpoint_id]
    assert engine.list_experiment_checkpoints(experiment.experiment_id)[0].role == "base"
    assert list((engine.store.artifacts_dir / "patches").iterdir()) == []
    after = engine.get_experiment(experiment.experiment_id)
    assert after.selected_checkpoint_id == base.checkpoint_id
    assert after.budget["candidate_commits"].consumed == 0
    events = [event["event_type"] for event in engine.audit_log(experiment.experiment_id)]
    assert events[-1] == "patch.preview.failed"


def test_registered_edit_operator_runs_through_a_managed_policy(workspace, bundles, signed_sum):
    registry = builtin_registry()
    registry.register("operator", "negate_slice", "1", NegateOperator())
    with Engine(workspace, registry=registry) as engine:
        policy = ExperimentPolicy(policy_id="negate_policy_v1", mode="weights_only",
                                  budget={"candidate_commits": 1, "evaluation_batches": 0, "examples_per_batch": 0,
                                          "evaluated_examples": 0, "model_forward_calls": 0},
                                  operations=default_operation_policy().with_edit("negate_slice", "negate_slice@1"))
        engine.register_policy(policy)
        base = engine.import_bundle(bundles["task"])
        experiment = _experiment(engine, base.checkpoint_id, "negate_policy_v1")
        spec = _spec(base.checkpoint_id, {"op": "negate_slice", "tensor": "layers.1.weight", "selection": ROW1})
        preview = engine.preview_patch(experiment.experiment_id, spec, "p1")
        assert str(preview.operations[0].operator) == "negate_slice@1"
        candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
        reference = load_model(signed_sum.reference_tensors(), "mlp@1", signed_sum.CONFIG)
        assert candidate.weights_hash == reference.weights_hash
        # The default policy does not permit the extension.
        other = _experiment(engine, base.checkpoint_id)
        _expect(ErrorCode.POLICY_DENIED, engine.preview_patch, other.experiment_id,
                _spec(base.checkpoint_id, {"op": "negate_slice", "tensor": "layers.1.weight", "selection": ROW1}),
                "p1")


def test_original_checkpoint_bytes_never_change(engine, bundles, signed_sum):
    base = engine.import_bundle(bundles["task"])
    weights_file = engine.store.weights_path(base.checkpoint_id)
    before = weights_file.read_bytes()
    experiment = _experiment(engine, base.checkpoint_id)
    preview = engine.preview_patch(experiment.experiment_id, _spec(base.checkpoint_id, ORACLE_SCALE), "p1")
    engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
    assert weights_file.read_bytes() == before
    assert weights_hash(engine.store.load_tensors(base.checkpoint_id)) == base.weights_hash
    assert tensor_hash("layers.1.weight", engine.store.load_tensors(base.checkpoint_id)["layers.1.weight"]) == \
        tensor_hash("layers.1.weight", signed_sum.task_tensors()["layers.1.weight"])


def test_core_patch_modules_import_no_mcp_or_agent_sdk():
    import subprocess
    import sys

    code = ("import sys, wse.patches, wse.experiments, wse.audit; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('mcp', 'agents', 'openai', 'claude_agent_sdk')))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_replay_resolves_nothing_from_any_registry(task_model, monkeypatch):
    result = apply_operator(task_model, "scale_slice", {k: v for k, v in ORACLE_SCALE.items() if k != "op"})

    def refuse(*args, **kwargs):
        raise AssertionError("replay must not resolve any registered extension")

    monkeypatch.setattr(Registry, "resolve", refuse)
    monkeypatch.setattr(Registry, "resolve_ref", refuse)
    monkeypatch.setattr("wse.registry.builtin_registry", refuse)
    assert replay_patch(task_model, result.patch).weights_hash == result.candidate.weights_hash
    assert isinstance(result.candidate.tensor("layers.1.weight"), torch.Tensor)
