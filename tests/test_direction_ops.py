"""The M2 edit operators ``project_out@1``, ``fill_slice@1``, and ``add_low_rank@1`` (M2 plan Section 5.1, Task 9).

* ``project_out``: ``W'[:, C] = W[:, C] - alpha d (d^T W[:, C]) / (d^T d)`` for a
  two-dimensional ``[out_features, in_features]`` tensor, with ``d`` on the
  output axis and ``C`` a strictly increasing list of input-axis indices.
* ``fill_slice``: every element of a selection becomes one finite scalar,
  recorded as the engine rounds it to the stored dtype.
* ``add_low_rank``: ``W'[R, C] = W[R, C] + alpha U V^T`` on a selection, with
  ``U`` given as ``r <= 16`` columns of one value per selected row and ``V`` as
  ``r`` columns of one value per selected column.

Directions and factor columns may be inline or vector references to stored
analysis results of the run. The in-memory model is the signed-sum fixture;
BF16, F16, and F32 open-weight tensors come from the tiny offline fixtures of
``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
import torch

from wse.adapters.families import model_family, role_tensor
from wse.api import Engine, apply_operator, load_model
from wse.experiments import ExperimentService
from wse.operators.base import TensorRegion
from wse.operators.directions import (
    MAX_DIRECTION_LENGTH,
    MAX_INLINE_FACTOR_VALUES,
    MAX_LOW_RANK,
    MIN_DIRECTION_NORM,
    AddLowRankOperator,
    FillSliceOperator,
    ProjectOutOperator,
)
from wse.patches import _derive, materialize_patch, replay_patch
from wse.policy import ExperimentPolicy, builtin_experiment_policies, default_operation_policy
from wse.registry import builtin_registry
from wse.schemas import AxisSlice, ErrorCode, PatchStep, UsageContract, WSEError, json_hash

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DOWN = "model.layers.0.mlp.down_proj.weight"  # [16, 32] in the Llama (BF16), Qwen3 (F16), and Gemma 3 (F32) fixtures
W1 = "layers.0.weight"  # [4, 2] in the signed-sum model
POLICY = default_operation_policy().with_edit("project_out", "project_out@1")
EDITS = POLICY.with_edit("fill_slice", "fill_slice@1").with_edit("add_low_rank", "add_low_rank@1")
TASK = "tiny_lm_v1"


def _sel(*ranges):
    return [{"start": start, "stop": stop, "step": 1} for start, stop in ranges]


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _bits(tensor: torch.Tensor) -> list[int]:
    """Stored bit patterns of a 16-bit tensor, row-major."""
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1).tolist()


def _bf16(bits: list[int]) -> torch.Tensor:
    return torch.tensor(np.array(bits, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


def _project(model, tensor, selection, **parameters):
    return apply_operator(model, "project_out", {"tensor": tensor, "selection": selection, **parameters},
                          policy=POLICY)


def _fill(model, tensor, selection, **parameters):
    return apply_operator(model, "fill_slice", {"tensor": tensor, "selection": selection, **parameters},
                          policy=EDITS)


def _low_rank(model, tensor, selection, **parameters):
    return apply_operator(model, "add_low_rank", {"tensor": tensor, "selection": selection, **parameters},
                          policy=EDITS)


def _consumed(engine, experiment_id, counter):
    return engine.get_experiment(experiment_id).budget[counter].consumed


@pytest.fixture
def task_model(signed_sum):
    return load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)


@pytest.fixture
def counting_model(task_model):
    """The signed-sum model with ``layers.0.weight = [[1, 2], [3, 4], [5, 6], [7, 8]]``."""
    return _derive(task_model, {**task_model._tensors, W1: torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                                                                         [7.0, 8.0]])})


@pytest.fixture(scope="module")
def hf_workspace(hf_snapshots, tmp_path_factory):
    workspace = tmp_path_factory.mktemp("direction_ops") / "workspace"
    with Engine(workspace) as engine:
        ids = {name: engine.import_hf_snapshot(hf_snapshots[name], repo=REPO, revision=REVISION).checkpoint_id
               for name in ("llama_untied", "llama_tied", "gemma3", "qwen3")}
    return workspace, ids


@pytest.fixture(scope="module")
def handles(hf_workspace):
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        return {name: engine.open_checkpoint(checkpoint_id) for name, checkpoint_id in ids.items()}


@pytest.fixture(scope="module")
def bf16_model(hf_workspace):
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        return engine.open_checkpoint(ids["llama_untied"])


@pytest.fixture(scope="module")
def f32_model(hf_workspace):
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        return engine.open_checkpoint(ids["gemma3"])


def _direction(rows: int, seed: int) -> list[float]:
    return np.random.default_rng(seed).normal(0.0, 1.0, size=rows).astype(np.float32).tolist()


def _reference(weights: np.ndarray, direction: list[float], columns: list[int], alpha: float) -> np.ndarray:
    """The operator's rule written out with scalar Python floats (IEEE binary64)."""
    out = weights.astype(np.float32).copy()
    d = [float(np.float32(value)) for value in direction]
    norm_squared = math.fsum(value * value for value in d)
    for c in columns:
        column = [float(value) for value in weights[:, c]]
        coefficient = alpha * (math.fsum(di * wi for di, wi in zip(d, column)) / norm_squared)
        for i, (di, wi) in enumerate(zip(d, column)):
            term = di * coefficient
            out[i, c] = np.float32(wi if term == 0.0 else wi - term)
    return out


# -- registration and policies --------------------------------------------------------------------


def test_project_out_is_a_registered_builtin_edit_operator():
    spec = builtin_registry().resolve_ref("operator", "project_out@1").spec
    assert spec.effect == "edit"
    assert spec.required_capabilities == sorted(["weights.read", "weights.slice_edit"])
    assert spec.usage == UsageContract(counter="candidate_commits", units_per_call=1, model_forward_calls=0)
    assert spec.output_schema == PatchStep.model_json_schema()
    assert sorted(spec.input_schema["properties"]) == ["alpha", "columns", "direction"]
    assert MAX_DIRECTION_LENGTH == 2048 and MIN_DIRECTION_NORM > 0


def test_fill_slice_and_add_low_rank_are_registered_builtin_edit_operators():
    registry = builtin_registry()
    expected = {"fill_slice@1": ["value"], "add_low_rank@1": ["alpha", "u", "v"]}
    for ref, properties in expected.items():
        operator = registry.resolve_ref("operator", ref)
        spec = operator.spec
        assert spec.effect == "edit"
        assert spec.required_capabilities == sorted(["weights.read", "weights.slice_edit"])
        assert spec.usage == UsageContract(counter="candidate_commits", units_per_call=1, model_forward_calls=0)
        assert spec.output_schema == PatchStep.model_json_schema()
        assert sorted(spec.input_schema["properties"]) == properties
    assert registry.resolve_ref("operator", "add_low_rank@1").reference_parameters == ("u", "v")
    assert registry.resolve_ref("operator", "project_out@1").reference_parameters == ("direction",)
    assert registry.resolve_ref("operator", "fill_slice@1").reference_parameters == ()
    assert MAX_LOW_RANK == 16 and MAX_INLINE_FACTOR_VALUES == 16_384


@pytest.mark.parametrize("declaration", ["reference_parameters", "inline_parameters"])
def test_edit_operator_vector_declarations_are_checked_at_registration(declaration):
    from wse.registry import Registry

    class Custom(ProjectOutOperator):
        pass

    setattr(Custom, declaration, ("missing",))
    error = _expect(ErrorCode.INVALID_ARGUMENT, Registry().register, "operator", "project_out", "1", Custom())
    assert "missing" in error.message and "edit operator project_out@1" in error.message


def test_builtin_policies_do_not_permit_project_out_and_keep_their_hashes(task_model):
    policies = builtin_experiment_policies()
    assert {name: sorted(policy.operations.edits) for name, policy in policies.items()} == {
        "small_model_v1": ["scale_slice", "set_slice"],
        "small_model_weights_only_v1": ["scale_slice", "set_slice"],
    }
    assert policies["small_model_v1"].policy_hash == (
        "sha256:d8761471bfb88e1d322da1ca06003729886daafa028963f1dcf7ef807cc218b5")
    assert policies["small_model_weights_only_v1"].policy_hash == (
        "sha256:88d3c1e4f73e0eb8204c30b65aceb753608114c4ed05e074302db630e665fa75")
    request = {"tensor": W1, "selection": _sel((0, 4), (0, 2)), "direction": [1.0, 0.0, 0.0, 0.0], "columns": [0]}
    _expect(ErrorCode.POLICY_DENIED, apply_operator, task_model, "project_out", request)
    # The exact reference needs no policy alias.
    assert apply_operator(task_model, "project_out@1", request).patch.steps[0].operator.extension_id == "project_out"
    for op, parameters in (("fill_slice", {"value": 0.0}), ("add_low_rank", {"u": [[1.0] * 4], "v": [[1.0] * 2]})):
        request = {"tensor": W1, "selection": _sel((0, 4), (0, 2)), **parameters}
        _expect(ErrorCode.POLICY_DENIED, apply_operator, task_model, op, request)
        assert apply_operator(task_model, f"{op}@1", request).patch.steps[0].operator.extension_id == op


# -- known values ---------------------------------------------------------------------------------


def test_project_out_known_values_in_fp32(counting_model):
    result = _project(counting_model, W1, _sel((0, 4), (0, 2)), direction=[1, 1, 1, 1], columns=[1], alpha=0.5)
    # d^T W[:, 1] = 20 and d^T d = 4, so half the projection subtracts 2.5 from every entry of column 1.
    assert result.candidate.tensor(W1).tolist() == [[1.0, -0.5], [3.0, 1.5], [5.0, 3.5], [7.0, 5.5]]
    (step,) = result.patch.steps
    assert step.op == "project_out" and str(step.operator) == "project_out@1"
    assert step.parameters == {"direction": [1.0, 1.0, 1.0, 1.0], "columns": [1], "alpha": 0.5}
    assert step.changed_elements == 4
    assert step.values == [1.0, -0.5, 3.0, 1.5, 5.0, 3.5, 7.0, 5.5]
    full = _project(counting_model, W1, _sel((0, 4), (0, 2)), direction=[1.0, 1.0, 1.0, 1.0], columns=[1])
    assert full.patch.steps[0].parameters["alpha"] == 1.0
    assert full.candidate.tensor(W1).tolist() == [[1.0, -3.0], [3.0, -1.0], [5.0, 1.0], [7.0, 3.0]]
    # The input model is untouched.
    assert counting_model.tensor(W1).tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]


def test_project_out_matches_the_scalar_reference_bytes_in_fp32(f32_model):
    weights = f32_model.tensor(DOWN).numpy()
    direction, columns, alpha = _direction(16, 3), [0, 5, 6, 31], 0.75
    result = _project(f32_model, DOWN, _sel((0, 16), (0, 32)), direction=direction, columns=columns, alpha=alpha)
    after = result.candidate.tensor(DOWN).numpy()
    assert after.tobytes() == _reference(weights, direction, columns, alpha).tobytes()


def test_project_out_known_values_in_bf16(bf16_model):
    edited = bf16_model._tensors[DOWN].clone()
    edited[:, 3] = _bf16([0x3F80, 0x3F80] + [0x3F00] * 14)  # 1.0, 1.0, then 0.5
    edited[0, 4] = _bf16([0x3F80])[0]
    model = _derive(bf16_model, {**bf16_model._tensors, DOWN: edited})
    direction = [1.0, 2.0] + [0.0] * 14
    result = _project(model, DOWN, _sel((0, 16), (3, 5)), direction=direction, columns=[3])
    after = result.candidate._tensors[DOWN]
    # d^T W[:, 3] = 3 and d^T d = 5: rows 0 and 1 become 1 - 0.6 = 0.4 and 1 - 1.2 = -0.2, computed in
    # float64, cast to float32, and rounded to nearest even in BF16 (0x3ECD and 0xBE4D); rows with d_i = 0
    # and column 4 keep their bytes.
    assert _bits(after[:, 3]) == [0x3ECD, 0xBE4D] + [0x3F00] * 14
    expected = torch.tensor([0.4, -0.2], dtype=torch.float64).to(torch.float32).to(torch.bfloat16)
    assert _bits(after[:2, 3]) == _bits(expected)
    assert _bits(after[:, 4]) == _bits(edited[:, 4])
    assert result.patch.steps[0].changed_elements == 2
    assert result.patch.steps[0].block is not None and result.patch.steps[0].values is None


# -- the projection's properties ------------------------------------------------------------------


@pytest.mark.parametrize("which", ["bf16", "f32"])
def test_project_out_leaves_columns_outside_c_byte_identical(which, bf16_model, f32_model):
    model = bf16_model if which == "bf16" else f32_model
    before = model._tensors[DOWN]
    columns = [2, 3, 17]
    result = _project(model, DOWN, _sel((0, 16), (0, 32)), direction=_direction(16, 7), columns=columns)
    after = result.candidate._tensors[DOWN]
    outside = [c for c in range(32) if c not in columns]
    width = torch.int16 if which == "bf16" else torch.int32

    def raw(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(width)

    assert torch.equal(raw(after[:, outside]), raw(before[:, outside]))
    step = result.patch.steps[0]
    assert 0 < step.changed_elements <= 16 * len(columns)
    assert int((raw(after) != raw(before)).sum()) == step.changed_elements
    diff = {d.tensor: d for d in result.diff}
    assert diff[DOWN].changed_elements == step.changed_elements
    assert all(d.changed_elements == 0 for name, d in diff.items() if name != DOWN)


def test_project_out_removes_d_from_the_selected_columns_in_fp32(f32_model):
    weights = f32_model.tensor(DOWN).numpy().astype(np.float64)
    direction = np.array(_direction(16, 11), dtype=np.float64)
    columns = [1, 4, 9, 30]
    result = _project(f32_model, DOWN, _sel((0, 16), (0, 32)), direction=direction.tolist(), columns=columns)
    after = result.candidate.tensor(DOWN).numpy().astype(np.float64)
    for c in columns:
        scale = np.linalg.norm(direction) * np.linalg.norm(weights[:, c])
        assert abs(direction @ weights[:, c]) > 1e-3 * scale  # there was something to remove
        assert abs(direction @ after[:, c]) <= 1e-6 * scale


def test_project_out_with_alpha_zero_changes_nothing(task_model):
    signed = torch.tensor([[-0.0, 2.0], [3.0, -0.0], [-5.0, 0.0], [-0.0, -8.0]])
    model = _derive(task_model, {**task_model._tensors, W1: signed})
    for direction in ([1.0, -2.0, 0.5, 3.0], [-1.0, 0.0, 0.0, 0.25]):
        result = _project(model, W1, _sel((0, 4), (0, 2)), direction=direction, columns=[0, 1], alpha=0.0)
        assert result.patch.steps[0].changed_elements == 0
        assert result.candidate.weights_hash == model.weights_hash
        assert result.candidate.tensor(W1).numpy().tobytes() == signed.numpy().tobytes()


def test_rows_with_a_zero_direction_entry_keep_their_bytes_including_negative_zero(task_model):
    signed = torch.tensor([[-0.0, 2.0], [3.0, -0.0], [-5.0, 1.0], [-0.0, -8.0]])
    model = _derive(task_model, {**task_model._tensors, W1: signed})
    # d^T W[:, 1] = -1 * 2 + 1 * 1 = -1 and d^T d = 2, so the coefficient is -0.5 and the update term of
    # row 1 is 0 * -0.5 = -0.0; subtracting it would turn the stored -0.0 into +0.0.
    result = _project(model, W1, _sel((0, 4), (1, 2)), direction=[-1.0, 0.0, 1.0, 0.0], columns=[1])
    after = result.candidate.tensor(W1).numpy()
    assert after[:, 1].tolist() == [1.5, 0.0, 1.5, -8.0]
    assert after[1, 1].tobytes() == signed.numpy()[1, 1].tobytes()  # still -0.0
    assert result.patch.steps[0].changed_elements == 2


def test_dot_products_are_correctly_rounded_sums(task_model):
    big = 2.0 ** 60
    model = _derive(task_model, {**task_model._tensors, W1: torch.tensor([[big, 0.0], [1.0, 0.0], [-big / 2, 0.0],
                                                                          [-big / 2, 0.0]])})
    # d^T W[:, 0] is exactly 1. A float64 sum in any order of pairwise additions loses the 1 against 2^60
    # and returns 0, which would leave the column unchanged.
    result = _project(model, W1, FULL, direction=[1.0, 1.0, 1.0, 1.0], columns=[0])
    assert result.candidate.tensor(W1)[:, 0].tolist() == [big, 0.75, -big / 2, -big / 2]


def test_direction_values_are_recorded_as_float32(counting_model):
    result = _project(counting_model, W1, _sel((0, 4), (0, 2)), direction=[0.1, 1.0, 1.0, 1.0], columns=[0])
    assert result.patch.steps[0].parameters["direction"] == [0.1, 1.0, 1.0, 1.0]  # shortest float32 digits
    again = _project(counting_model, W1, _sel((0, 4), (0, 2)), direction=[float(np.float32(0.1)), 1.0, 1.0, 1.0],
                     columns=[0])
    assert again.patch == result.patch


# -- refusals -------------------------------------------------------------------------------------

VALID = {"direction": [1.0, 0.0, 0.0, 0.0], "columns": [0]}
FULL = _sel((0, 4), (0, 2))


@pytest.mark.parametrize(
    "tensor,selection,overrides,fragment",
    [
        (W1, FULL, {"direction": [1.0, 0.0, 0.0]}, "one value per row"),
        (W1, FULL, {"direction": [1.0] * (MAX_DIRECTION_LENGTH + 1)}, "direction"),
        (W1, FULL, {"direction": []}, "direction"),
        (W1, FULL, {"direction": [float("nan"), 1.0, 0.0, 0.0]}, "finite"),
        (W1, FULL, {"direction": [float("inf"), 1.0, 0.0, 0.0]}, "finite"),
        (W1, FULL, {"direction": [1e39, 1.0, 0.0, 0.0]}, "finite"),
        (W1, FULL, {"direction": ["1.0", 1.0, 0.0, 0.0]}, "direction"),
        (W1, FULL, {"direction": [0.0, -0.0, 0.0, 0.0]}, "norm"),
        (W1, FULL, {"direction": [1e-46, 0.0, 0.0, 0.0]}, "norm"),  # zero in float32
        (W1, FULL, {"direction": [MIN_DIRECTION_NORM / 2, 0.0, 0.0, 0.0]}, "norm"),
        (W1, FULL, {"columns": [2]}, "columns"),
        (W1, FULL, {"columns": [-1]}, "columns"),
        (W1, _sel((0, 4), (1, 2)), {"columns": [0]}, "columns"),
        (W1, FULL, {"columns": [0, 0]}, "strictly increasing"),
        (W1, FULL, {"columns": [1, 0]}, "strictly increasing"),
        (W1, FULL, {"columns": []}, "columns"),
        (W1, FULL, {"columns": [True]}, "columns"),
        (W1, FULL, {"alpha": float("nan")}, "alpha"),
        (W1, FULL, {"alpha": float("inf")}, "alpha"),
        (W1, FULL, {"alpha": 1e39}, "alpha"),
        (W1, FULL, {"factor": 2.0}, "Accepted parameters"),
        (W1, _sel((0, 3), (0, 2)), {}, "every row"),
        (W1, _sel((1, 4), (0, 2)), {}, "every row"),
        ("layers.0.bias", _sel((0, 4)), {}, "two-dimensional"),
    ],
)
def test_invalid_project_out_requests_are_refused(task_model, tensor, selection, overrides, fragment):
    error = _expect(ErrorCode.INVALID_ARGUMENT, _project, task_model, tensor, selection, **{**VALID, **overrides})
    assert fragment in error.message


def test_project_out_requires_an_output_by_input_layout(bf16_model):
    embed = "model.embed_tokens.weight"  # [64, 16] with axes (num_embeddings, embedding_dim)
    # llama@1 protects the embedding (Task 8); the same weights at Level 0 have no protected role.
    level0 = load_model(bf16_model.tensors(), "hf_causal_lm@1", {**bf16_model.config, "family": None})
    error = _expect(ErrorCode.INVALID_ARGUMENT, _project, level0, embed, _sel((0, 64), (0, 16)),
                    direction=[1.0] * 64, columns=[0])
    assert "out_features" in error.message


VALID_REQUESTS = {
    "project_out": lambda shape: {"direction": [1.0] * shape[0], "columns": [0]},
    "fill_slice": lambda shape: {"value": 0.0},
    "add_low_rank": lambda shape: {"u": [[1.0] * shape[0]], "v": [[1.0] * shape[-1]]},
}


@pytest.mark.parametrize("op", sorted(VALID_REQUESTS))
def test_alias_groups_and_protected_roles_are_refused(op, handles):
    # Alias groups: Gemma 3 (Level 0, no roles) ties its embeddings and output head.
    gemma = handles["gemma3"]
    assert gemma.architecture.alias_groups == [["lm_head.weight", "model.embed_tokens.weight"]]
    assert model_family(gemma) is None
    for tensor in gemma.architecture.alias_groups[0]:
        request = {"tensor": tensor, "selection": _sel((0, 64), (0, 16)), **VALID_REQUESTS[op]([64, 16])}
        error = _expect(ErrorCode.UNSUPPORTED_CAPABILITY, apply_operator, gemma, op, request, policy=EDITS)
        assert "alias group" in error.message
    # Protected roles: llama@1 protects the embeddings, the output head, and the final norm. They are refused
    # before the operator sees its parameters (here none), whether or not the tensor is also in an alias group.
    for name in ("llama_untied", "llama_tied"):
        handle = handles[name]
        for role in ("embed", "lm_head", "final_norm"):
            target = role_tensor(handle, role)
            for parameters in ({}, VALID_REQUESTS[op](target.shape)):
                error = _expect(ErrorCode.POLICY_DENIED, apply_operator, handle, op,
                                {"tensor": target.tensor, "selection": target.selection(), **parameters}, policy=EDITS)
                assert error.details["role"] == role and "llama@1" in error.message and "mlp.down" in error.message
    # In a patch the refused step keeps its code and reports its position.
    handle = handles["llama_untied"]
    down = role_tensor(handle, "mlp.down", layer=0)
    embed = role_tensor(handle, "embed")
    error = _expect(ErrorCode.POLICY_DENIED, materialize_patch, handle, [
        {"op": op, "tensor": down.tensor, "selection": down.selection(), **VALID_REQUESTS[op](down.shape)},
        {"op": op, "tensor": embed.tensor, "selection": embed.selection(), **VALID_REQUESTS[op](embed.shape)},
    ], policy=EDITS)
    assert error.details["operation_index"] == 1 and error.details["role"] == "embed"
    # Unprotected roles stay editable.
    edit = apply_operator(handle, op, {"tensor": down.tensor, "selection": down.selection(),
                                       **VALID_REQUESTS[op](down.shape)}, policy=EDITS)
    assert edit.patch.steps[0].changed_elements > 0


# -- replay, re-execution, and the managed path ---------------------------------------------------


def test_project_out_patches_replay_hash_exact_without_the_operator(counting_model, bf16_model, monkeypatch):
    results = [
        _project(counting_model, W1, FULL, direction=[0.3, -1.0, 2.0, 0.5], columns=[0, 1], alpha=0.9),
        _project(bf16_model, DOWN, _sel((0, 16), (4, 20)), direction=_direction(16, 5), columns=[4, 8, 19]),
    ]

    def never(*args, **kwargs):
        raise AssertionError("replay must not run an editing operator")

    monkeypatch.setattr(ProjectOutOperator, "apply", never)
    for base, result in zip((counting_model, bf16_model), results):
        replayed = replay_patch(base, result.patch, blocks=result.blocks)
        assert replayed.weights_hash == result.candidate.weights_hash == result.patch.result_weights_hash
    assert results[1].blocks and not results[0].blocks


def test_re_executing_the_same_call_gives_the_same_patch(bf16_model):
    request = dict(direction=_direction(16, 9), columns=[0, 1, 2, 30, 31], alpha=1.25)
    first = _project(bf16_model, DOWN, _sel((0, 16), (0, 32)), **request)
    second = _project(bf16_model, DOWN, _sel((0, 16), (0, 32)), **request)
    assert first.patch.canonical_json() == second.patch.canonical_json()
    assert dict(first.blocks) == dict(second.blocks)


def test_project_out_runs_through_a_managed_policy(workspace, bundles, task_model):
    with Engine(workspace) as engine:
        policy = ExperimentPolicy(policy_id="project_out_policy_v1", mode="weights_only",
                                  budget={"candidate_commits": 1, "evaluation_batches": 0, "examples_per_batch": 0,
                                          "evaluated_examples": 0, "model_forward_calls": 0},
                                  operations=POLICY)
        engine.register_policy(policy)
        base = engine.import_bundle(bundles["task"])
        grant = engine.grant_run("signed_sum_v1", "project_out_policy_v1", base.checkpoint_id)
        experiment = engine.create_experiment(grant.run_id, "signed_sum_v1", "project_out_policy_v1", "create-1")
        operation = {"op": "project_out", "tensor": "layers.1.weight", "selection": _sel((0, 2), (0, 4)),
                     "direction": [1.0, 1.0], "columns": [1, 3], "alpha": 0.5}
        spec = {"base_checkpoint_id": base.checkpoint_id, "hypothesis": "Removing the shared direction helps.",
                "operations": [operation]}
        preview = engine.preview_patch(experiment.experiment_id, spec, "p1")
        assert str(preview.operations[0].operator) == "project_out@1"
        candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
        direct = _project(task_model, "layers.1.weight", _sel((0, 2), (0, 4)), direction=[1.0, 1.0],
                          columns=[1, 3], alpha=0.5)
        assert candidate.weights_hash == direct.candidate.weights_hash == preview.result_weights_hash
        assert engine.read_patch(experiment.experiment_id, preview.patch_id).patch == direct.patch


# -- fill_slice -----------------------------------------------------------------------------------


def test_fill_slice_known_values_in_fp32(counting_model):
    result = _fill(counting_model, W1, _sel((1, 3), (0, 2)), value=0.1)
    expected = np.array([[1.0, 2.0], [0.1, 0.1], [0.1, 0.1], [7.0, 8.0]], dtype=np.float32)
    assert result.candidate.tensor(W1).numpy().tobytes() == expected.tobytes()
    (step,) = result.patch.steps
    assert step.op == "fill_slice" and str(step.operator) == "fill_slice@1"
    assert step.parameters == {"value": 0.1}  # the shortest digits of the stored float32 value
    assert step.changed_elements == 4 and step.values == [0.1, 0.1, 0.1, 0.1]
    # Elements that already hold the value are not counted as changed; an integer value is recorded as a float.
    again = _fill(counting_model, W1, _sel((1, 2), (0, 2)), value=3)
    assert again.candidate.tensor(W1)[1].tolist() == [3.0, 3.0]
    assert again.patch.steps[0].parameters == {"value": 3.0} and again.patch.steps[0].changed_elements == 1
    # The input model is untouched.
    assert counting_model.tensor(W1).tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]


def test_fill_slice_known_values_in_bf16_and_f16(handles):
    # BF16 rounds 0.1 to 0x3DCD (0.10009765625) and F16 to 0x2E66 (0.0999755859375). 1000.3 becomes 1000.0 in BF16
    # (spacing 4) and 1000.5 in F16 (spacing 0.5). The step records the value as the engine stores it, in the digits
    # of reads: the shortest digits of its float32 widening (BF16; 0.10009766 would round to another float32) or of
    # the float16 value (F16).
    cases = {
        "llama_untied": (torch.bfloat16, {0.1: (0x3DCD, 0.100097656), 1000.3: (0x447A, 1000.0),
                                          -0.0: (0x8000, -0.0)}),
        "qwen3": (torch.float16, {0.1: (0x2E66, 0.1), 1000.3: (0x63D1, 1000.5), -0.0: (0x8000, -0.0)}),
    }
    for name, (dtype, values) in cases.items():
        model = handles[name]
        before = model._tensors[DOWN]
        assert before.dtype == dtype
        outside = torch.ones(16, 32, dtype=torch.bool)
        outside[2:5, 3:7] = False
        for value, (bits, recorded) in values.items():
            result = _fill(model, DOWN, _sel((2, 5), (3, 7)), value=value)
            after = result.candidate._tensors[DOWN]
            assert set(_bits(after[2:5, 3:7])) == {bits}, (name, value)
            assert _bits(after[outside]) == _bits(before[outside])
            step = result.patch.steps[0]
            assert step.parameters == {"value": recorded}
            assert math.copysign(1.0, step.parameters["value"]) == math.copysign(1.0, recorded)
            assert step.changed_elements == sum(1 for old in _bits(before[2:5, 3:7]) if old != bits)
            assert step.block is not None and step.values is None
            # Filling with the recorded value writes the same bytes.
            same = _fill(model, DOWN, _sel((2, 5), (3, 7)), value=step.parameters["value"])
            assert same.candidate.weights_hash == result.candidate.weights_hash


def test_invalid_fill_slice_requests_are_refused(task_model, handles):
    full = _sel((0, 4), (0, 2))
    cases = [
        (W1, full, {}, "value"),
        (W1, full, {"value": float("nan")}, "finite"),
        (W1, full, {"value": float("inf")}, "finite"),
        (W1, full, {"value": 1e39}, "finite"),
        (W1, full, {"value": "0.5"}, "value"),
        (W1, full, {"value": True}, "value"),
        (W1, full, {"value": [0.5]}, "value"),
        (W1, full, {"value": 0.5, "factor": 2.0}, "Accepted parameters"),
        ("layers.0.bias", _sel((0, 4)), {"value": 0.5}, "two-dimensional"),
    ]
    for tensor, selection, parameters, fragment in cases:
        error = _expect(ErrorCode.INVALID_ARGUMENT, _fill, task_model, tensor, selection, **parameters)
        assert fragment in error.message, (parameters, error.message)
    # A value that is finite in float32 but rounds to infinity in the stored dtype is refused, with the largest
    # finite value of that dtype; that value itself is accepted.
    for name, value, largest, text in (("llama_untied", 3.4e38, 3.3895313892515355e38, "3.3895314e+38"),
                                       ("qwen3", 65520.0, 65504.0, "65504")):
        error = _expect(ErrorCode.INVALID_ARGUMENT, _fill, handles[name], DOWN, _sel((0, 1), (0, 1)), value=value)
        assert "infinity" in error.message and text in error.message, error.message
        accepted = _fill(handles[name], DOWN, _sel((0, 1), (0, 1)), value=largest)
        assert accepted.candidate._tensors[DOWN][0, 0].item() == largest


def test_fill_slice_and_add_low_rank_require_two_known_axes():
    values = np.zeros((2, 3), dtype=np.float32)
    values.flags.writeable = False
    region = TensorRegion(tensor="w", dtype="float32", axes=("unknown", "unknown"), tensor_shape=(2, 3),
                          selection=(AxisSlice(start=0, stop=2, step=1), AxisSlice(start=0, stop=3, step=1)),
                          values=values)
    requests = ((FillSliceOperator(), {"value": 1.0}), (AddLowRankOperator(), {"u": [[1.0, 1.0]], "v": [[1.0] * 3]}))
    for operator, parameters in requests:
        error = _expect(ErrorCode.INVALID_ARGUMENT, operator.resolve_parameters, parameters, region)
        assert "known axes" in error.message and "unknown" in error.message
    # Any two known axes will do, for example those of an embedding.
    embedding = dataclasses.replace(region, axes=("num_embeddings", "embedding_dim"))
    for operator, parameters in requests:
        operator.resolve_parameters(parameters, embedding)


# -- add_low_rank ---------------------------------------------------------------------------------


def _low_rank_reference(weights: np.ndarray, u: list[list[float]], v: list[list[float]], alpha: float,
                        row_start: int = 0, column_start: int = 0) -> np.ndarray:
    """The operator's rule written out with scalar Python floats (IEEE binary64) and ``math.fsum``."""
    out = weights.astype(np.float32).copy()
    us = [[float(np.float32(value)) for value in column] for column in u]
    vs = [[float(np.float32(value)) for value in column] for column in v]
    for i in range(len(us[0])):
        for j in range(len(vs[0])):
            term = alpha * math.fsum(a[i] * b[j] for a, b in zip(us, vs))
            w = float(weights[row_start + i, column_start + j])
            out[row_start + i, column_start + j] = np.float32(w if term == 0.0 else w + term)
    return out


def test_add_low_rank_known_values_in_fp32(counting_model):
    result = _low_rank(counting_model, W1, _sel((0, 4), (0, 2)), u=[[1, 0, 0, 1]], v=[[0.5, -1]], alpha=2)
    # alpha u v^T adds [1, -2] to rows 0 and 3; rows with u_i = 0 keep their bytes.
    assert result.candidate.tensor(W1).tolist() == [[2.0, 0.0], [3.0, 4.0], [5.0, 6.0], [8.0, 6.0]]
    (step,) = result.patch.steps
    assert step.op == "add_low_rank" and str(step.operator) == "add_low_rank@1"
    assert step.parameters == {"u": [[1.0, 0.0, 0.0, 1.0]], "v": [[0.5, -1.0]], "alpha": 2.0}
    assert step.changed_elements == 4
    # Rank 2 on a sub-selection: each column of U has one value per selected row, each column of V one per selected
    # column, and alpha defaults to 1.
    rank2 = _low_rank(counting_model, W1, _sel((1, 3), (0, 2)), u=[[1.0, 1.0], [1.0, -1.0]], v=[[1.0, 0.0], [0.0, 1.0]])
    assert rank2.patch.steps[0].parameters["alpha"] == 1.0
    assert rank2.candidate.tensor(W1).tolist() == [[1.0, 2.0], [4.0, 5.0], [6.0, 5.0], [7.0, 8.0]]
    assert counting_model.tensor(W1).tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]


def test_add_low_rank_known_values_in_bf16(handles):
    model = handles["llama_untied"]
    edited = model._tensors[DOWN].clone()
    edited[0:2, 0:2] = 1.0
    model = _derive(model, {**model._tensors, DOWN: edited})
    result = _low_rank(model, DOWN, _sel((0, 2), (0, 3)), u=[[1.0, 2.0]], v=[[0.1, -0.1, 0.0]])
    after = result.candidate._tensors[DOWN]
    # 1 + 0.1, 1 - 0.1, 1 + 0.2, and 1 - 0.2, from float32(0.1) in float64, rounded to float32 and then to nearest
    # even in BF16: 0x3F8D (1.1015625), 0x3F66 (0.8984375), 0x3F9A (1.203125), and 0x3F4D (0.80078125).
    assert _bits(after[0:2, 0:2]) == [0x3F8D, 0x3F66, 0x3F9A, 0x3F4D]
    outer = torch.tensor([[1.0], [2.0]], dtype=torch.float64) * torch.tensor([0.1, -0.1]).to(torch.float64)
    assert _bits(after[0:2, 0:2]) == _bits((1.0 + outer).to(torch.float32).to(torch.bfloat16))
    # Column 2 has v_j = 0 and keeps its bytes, as does everything outside the selection.
    assert _bits(after[0:2, 2]) == _bits(edited[0:2, 2])
    assert _bits(after[2:]) == _bits(edited[2:]) and _bits(after[:, 3:]) == _bits(edited[:, 3:])
    step = result.patch.steps[0]
    assert step.changed_elements == 4 and step.block is not None and step.values is None
    assert step.parameters == {"u": [[1.0, 2.0]], "v": [[0.1, -0.1, 0.0]], "alpha": 1.0}


@pytest.mark.parametrize("rank", [1, 2, 3, MAX_LOW_RANK])
def test_add_low_rank_matches_the_scalar_reference_bytes_in_fp32(rank, handles):
    model = handles["gemma3"]
    weights = model.tensor(DOWN).numpy()
    rng = np.random.default_rng(rank)
    u = rng.normal(0.0, 1.0, size=(rank, 12)).astype(np.float32).tolist()
    v = rng.normal(0.0, 0.01, size=(rank, 20)).astype(np.float32).tolist()
    result = _low_rank(model, DOWN, _sel((2, 14), (5, 25)), u=u, v=v, alpha=-0.75)
    assert result.candidate.tensor(DOWN).numpy().tobytes() == _low_rank_reference(weights, u, v, -0.75, 2, 5).tobytes()
    assert 0 < result.patch.steps[0].changed_elements <= 12 * 20


def test_low_rank_sums_are_correctly_rounded_whatever_the_column_order(task_model):
    model = _derive(task_model, {**task_model._tensors, W1: torch.tensor([[0.5, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                                          [0.0, 0.0]])})
    big = 2.0 ** 30
    # The products are 2^60, 1, and -2^60, whose sum is exactly 1. Float64 additions in column order lose the 1
    # against 2^60 and return 0 in the first two orders below.
    columns = [([big], [big]), ([1.0], [1.0]), ([big], [-big])]
    hashes = set()
    for order in ((0, 1, 2), (1, 0, 2), (0, 2, 1)):
        result = _low_rank(model, W1, _sel((0, 1), (0, 1)), u=[columns[k][0] for k in order],
                           v=[columns[k][1] for k in order])
        assert result.candidate.tensor(W1)[0, 0].item() == 1.5
        hashes.add(result.candidate.weights_hash)
    assert len(hashes) == 1


def test_low_rank_entries_with_a_zero_update_keep_their_bytes_including_negative_zero(task_model):
    signed = torch.tensor([[-0.0, 2.0], [3.0, -0.0], [-5.0, 1.0], [-0.0, -8.0]])
    model = _derive(task_model, {**task_model._tensors, W1: signed})
    # Row 1 has u_i = 0 and column 1 has v_j = 0, so their update terms are zero (0.0 or -0.0); adding them would
    # turn the stored -0.0 at [1, 1] into +0.0.
    result = _low_rank(model, W1, _sel((0, 4), (0, 2)), u=[[1.0, 0.0, -1.0, 2.0]], v=[[0.5, 0.0]])
    after = result.candidate.tensor(W1).numpy()
    assert after.tolist() == [[0.5, 2.0], [3.0, -0.0], [-5.5, 1.0], [1.0, -8.0]]
    assert after[1, 1].tobytes() == signed.numpy()[1, 1].tobytes()  # still -0.0
    assert result.patch.steps[0].changed_elements == 3
    # alpha = 0 changes nothing at all.
    zero = _low_rank(model, W1, _sel((0, 4), (0, 2)), u=[[1.0, 2.0, 3.0, 4.0]], v=[[1.0, -1.0]], alpha=0.0)
    assert zero.patch.steps[0].changed_elements == 0 and zero.candidate.weights_hash == model.weights_hash


def test_low_rank_rank_and_shape_limits(task_model, handles):
    full = _sel((0, 4), (0, 2))
    one_u, one_v = [1.0, 0.0, 0.0, 0.0], [1.0, 0.0]
    cases = [
        ({"u": [], "v": []}, "at least 1"),
        ({"u": [one_u] * (MAX_LOW_RANK + 1), "v": [one_v] * (MAX_LOW_RANK + 1)}, f"at most {MAX_LOW_RANK}"),
        ({"u": [one_u, one_u], "v": [one_v]}, "same number of columns"),
        ({"u": [one_u[:3]], "v": [one_v]}, "u/0 must hold one value per selected row"),
        ({"u": [one_u], "v": [one_v + [0.0]]}, "v/0 must hold one value per selected column"),
        ({"u": [one_u, one_u], "v": [one_v, [1.0]]}, "v/1"),
        ({"u": [[]], "v": [one_v]}, "u"),
        ({"u": one_u, "v": one_v}, "u"),  # flat lists, not lists of columns
        ({"u": [[float("nan"), 0.0, 0.0, 0.0]], "v": [one_v]}, "finite"),
        ({"u": [one_u], "v": [[1e39, 0.0]]}, "finite"),
        ({"u": [["1.0", 0.0, 0.0, 0.0]], "v": [one_v]}, "u"),
        ({"u": [one_u], "v": [one_v], "alpha": float("inf")}, "alpha"),
        ({"u": [one_u], "v": [one_v], "factor": 2.0}, "Accepted parameters"),
        ({"u": [one_u]}, "v"),
    ]
    for parameters, fragment in cases:
        error = _expect(ErrorCode.INVALID_ARGUMENT, _low_rank, task_model, W1, full, **parameters)
        assert fragment in error.message, (parameters, error.message)
    # The rank may reach the limit.
    top = _low_rank(task_model, W1, full, u=[one_u] * MAX_LOW_RANK, v=[one_v] * MAX_LOW_RANK)
    assert top.candidate.tensor(W1)[0, 0].item() == task_model.tensor(W1)[0, 0].item() + MAX_LOW_RANK
    # A one-dimensional tensor has no low-rank form.
    error = _expect(ErrorCode.INVALID_ARGUMENT, _low_rank, task_model, "layers.0.bias", _sel((0, 4)), u=[one_u],
                    v=[[1.0]])
    assert "two-dimensional" in error.message
    # Inline factor columns hold at most MAX_INLINE_FACTOR_VALUES values in all; longer ones go by reference.
    wide = load_model({W1: torch.zeros(2, 4096)}, "mlp@1", {"layer_sizes": [4096, 2], "activation": "relu",
                                                           "bias": False})
    at_bound = _low_rank(wide, W1, _sel((0, 2), (0, 4094)), u=[[1.0, 1.0]] * 4, v=[[0.5] * 4094] * 4)
    assert at_bound.patch.steps[0].changed_elements == 2 * 4094  # 8 + 16,376 = 16,384 inline values
    error = _expect(ErrorCode.INVALID_ARGUMENT, _low_rank, wide, W1, _sel((0, 2), (0, 4096)), u=[[1.0, 1.0]] * 4,
                    v=[[0.5] * 4096] * 4)
    assert f"at most {MAX_INLINE_FACTOR_VALUES}" in error.message and "reference" in error.message


def test_inline_vectors_and_columns_count_toward_the_patch_element_limit(handles):
    model = handles["llama_untied"]  # blocks: the operator results themselves add nothing
    policy = EDITS.with_limits(max_patch_elements=35)
    low_rank = {"op": "add_low_rank", "tensor": DOWN, "selection": _sel((0, 2), (0, 4)), "u": [[1.0, 1.0]],
                "v": [[0.5] * 4]}  # 6 inline values
    project = {"op": "project_out", "tensor": DOWN, "selection": _sel((0, 16), (0, 32)), "direction": [1.0] * 16,
               "columns": list(range(8))}  # 16 + 8 inline values
    fill = {"op": "fill_slice", "tensor": DOWN, "selection": _sel((0, 16), (0, 32)), "value": 0.0}  # none
    assert materialize_patch(model, [low_rank, project, fill], policy=policy).patch.steps[2].changed_elements > 35
    error = _expect(ErrorCode.INVALID_PATCH, materialize_patch, model, [low_rank, project, low_rank, fill],
                    policy=policy)  # 6 + 24 + 6 = 36
    assert error.details["operation_index"] == 2 and error.details["max_patch_elements"] == 35
    error = _expect(ErrorCode.INVALID_PATCH, materialize_patch, model, [dict(project, columns=list(range(32)))],
                    policy=EDITS.with_limits(max_patch_elements=47))
    assert "48" in error.message and error.details["operation_index"] == 0
    # The columns are bounded by the selection's columns, and so by the tensor's input axis.
    error = _expect(ErrorCode.INVALID_ARGUMENT, _project, model, DOWN, _sel((0, 16), (4, 8)), direction=[1.0] * 16,
                    columns=list(range(4, 9)))
    assert "at most 4" in error.message


def test_float32_maximum_is_accepted_wherever_a_float32_value_is(handles, counting_model):
    largest = float(np.finfo(np.float32).max)
    shortest = float(str(np.float32(largest)))  # 3.4028235e+38, above the maximum as a float64
    assert shortest > largest and np.float32(shortest) == np.float32(largest)
    for value in (largest, shortest, -shortest):
        direction = [value] + [0.0] * 15
        result = _project(handles["gemma3"], DOWN, _sel((0, 16), (0, 32)), direction=direction, columns=[0])
        assert result.patch.steps[0].parameters["direction"][0] == math.copysign(shortest, value)
        assert _fill(counting_model, W1, _sel((0, 1), (0, 1)), value=value).patch.steps[0].parameters == {
            "value": math.copysign(shortest, value)}
        recorded = _low_rank(counting_model, W1, _sel((0, 1), (0, 1)), u=[[value]], v=[[0.0]],
                             alpha=value).patch.steps[0].parameters
        assert recorded["u"] == [[math.copysign(shortest, value)]] and recorded["alpha"] == value
    # Beyond the float32 rounding range a value is refused as a domain error, never a raw validation error.
    beyond = 2.0 ** 128 - 2.0 ** 103  # rounds to infinity in float32
    for parameters in ({"direction": [beyond] + [0.0] * 15, "columns": [0]}, {"direction": [1.0] * 16, "columns": [0],
                                                                                "alpha": beyond}):
        error = _expect(ErrorCode.INVALID_ARGUMENT, _project, handles["gemma3"], DOWN, _sel((0, 16), (0, 32)),
                        **parameters)
        assert "float32" in error.message


def test_parameter_errors_are_bounded(task_model):
    error = _expect(ErrorCode.INVALID_ARGUMENT, _project, task_model, W1, FULL, direction=["x"] * 2048, columns=[0])
    assert len(error.message) < 2048 and "more" in error.message
    assert len(json.dumps(error.details)) < 1024 and error.details["error_count"] > 5
    factors = _expect(ErrorCode.INVALID_ARGUMENT, _low_rank, task_model, W1, FULL, u=[["x"] * 4] * 16,
                      v=[["y"] * 2] * 16)
    assert len(factors.message) < 2048


# -- replay and the managed path ------------------------------------------------------------------


def test_fill_and_low_rank_patches_replay_hash_exact_without_the_operators(counting_model, handles, monkeypatch):
    bf16 = handles["llama_untied"]
    rng = np.random.default_rng(4)
    results = [
        (counting_model, _fill(counting_model, W1, _sel((0, 2), (0, 2)), value=-2.5)),
        (bf16, _fill(bf16, DOWN, _sel((0, 16), (4, 8)), value=0.0)),
        (counting_model, _low_rank(counting_model, W1, FULL, u=[[0.3, -1.0, 2.0, 0.5]], v=[[0.25, -0.75]],
                                   alpha=0.9)),
        (bf16, _low_rank(bf16, DOWN, _sel((0, 16), (0, 32)), u=rng.normal(size=(3, 16)).tolist(),
                         v=rng.normal(size=(3, 32)).tolist(), alpha=0.01)),
    ]

    def never(*args, **kwargs):
        raise AssertionError("replay must not run an editing operator")

    monkeypatch.setattr(FillSliceOperator, "apply", never)
    monkeypatch.setattr(AddLowRankOperator, "apply", never)
    for base, result in results:
        replayed = replay_patch(base, result.patch, blocks=result.blocks)
        assert replayed.weights_hash == result.candidate.weights_hash == result.patch.result_weights_hash


def test_fill_slice_and_add_low_rank_run_through_a_managed_policy(workspace, bundles, task_model):
    with Engine(workspace) as engine:
        engine.register_policy(ExperimentPolicy(
            policy_id="new_edits_v1", mode="weights_only", operations=EDITS,
            budget={"candidate_commits": 1, "evaluation_batches": 0, "examples_per_batch": 0, "evaluated_examples": 0,
                    "model_forward_calls": 0}))
        base = engine.import_bundle(bundles["task"])
        grant = engine.grant_run("signed_sum_v1", "new_edits_v1", base.checkpoint_id)
        experiment = engine.create_experiment(grant.run_id, "signed_sum_v1", "new_edits_v1", "create-1")
        operations = [
            {"op": "fill_slice", "tensor": "layers.1.weight", "selection": _sel((0, 1), (0, 4)), "value": 0.0},
            {"op": "add_low_rank", "tensor": "layers.0.weight", "selection": _sel((0, 4), (0, 2)),
             "u": [[1.0, 0.5, -0.5, 0.25]], "v": [[0.1, -0.2]], "alpha": 0.5},
        ]
        spec = {"base_checkpoint_id": base.checkpoint_id, "hypothesis": "Silencing one output unit helps.",
                "operations": operations}
        preview = engine.preview_patch(experiment.experiment_id, spec, "p1")
        assert [str(operation.operator) for operation in preview.operations] == ["fill_slice@1", "add_low_rank@1"]
        candidate = engine.apply_patch(experiment.experiment_id, preview.patch_id, "a1")
        direct = materialize_patch(task_model, operations, policy=EDITS)
        assert candidate.weights_hash == direct.candidate.weights_hash == preview.result_weights_hash
        assert engine.read_patch(experiment.experiment_id, preview.patch_id).patch == direct.patch


# -- vector references ----------------------------------------------------------------------------

REFERENCE_POLICY_ID = "edit_references_v1"


@pytest.fixture
def run_setup(workspace, hf_snapshots):
    """A Llama (BF16) base checkpoint, a policy granting svd_top_k and the three edits, and two run grants."""
    with Engine(workspace) as operator:
        base = operator.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION).checkpoint_id
        operator.register_policy(ExperimentPolicy(
            policy_id=REFERENCE_POLICY_ID, mode="weights_only",
            operations=EDITS.with_analysis("svd_top_k", "svd_top_k@1"),
            budget={"candidate_commits": 2, "evaluation_batches": 0, "examples_per_batch": 0, "evaluated_examples": 0,
                    "model_forward_calls": 0}))
        grants = [operator.grant_run(TASK, REFERENCE_POLICY_ID, base) for _ in range(2)]
    return base, grants


def _never(*args, **kwargs):
    raise AssertionError("this call must not happen")


def test_directions_by_artifact_uri_must_belong_to_the_run(run_setup, workspace, handles, monkeypatch):
    base, grants = run_setup
    full = _sel((0, 16), (0, 32))
    with Engine(workspace, run_id=grants[0].run_id) as bound:
        eid = bound.create_experiment(grants[0].run_id, TASK, REFERENCE_POLICY_ID, "c").experiment_id
        svd = bound.analyze_model(eid, base, "svd_top_k", DOWN, full, parameters={"k": 2})
        uri = svd.result_uri
        stored = bound.read_analysis_result(uri)
        left, right = stored["left_vectors"], stored["right_vectors"]
        operations = [
            {"op": "project_out", "tensor": DOWN, "selection": full, "direction": f"{uri}#/left_vectors/0",
             "columns": [0, 5, 31]},
            {"op": "add_low_rank", "tensor": DOWN, "selection": full, "u": [f"{uri}#/left_vectors/1", [0.5] * 16],
             "v": [f"{uri}#/right_vectors/1", [0.25] * 32], "alpha": -0.5},
        ]
        spec = {"base_checkpoint_id": base, "hypothesis": "Removing the top directions changes the output.",
                "operations": operations}
        preview = bound.preview_patch(eid, spec, "p1")
        # Applying replays the stored patch; the references are not read again.
        with monkeypatch.context() as patched:
            patched.setattr(ExperimentService, "find_analysis_result", _never)
            candidate = bound.apply_patch(eid, preview.patch_id, "a1")
    # The same edits with the vectors inline give the same weights.
    inline = [{**operations[0], "direction": left[0]},
              {**operations[1], "u": [left[1], [0.5] * 16], "v": [right[1], [0.25] * 32]}]
    direct = materialize_patch(handles["llama_untied"], inline, policy=EDITS)
    assert candidate.weights_hash == preview.result_weights_hash == direct.patch.result_weights_hash
    # Each step records the reference, with the length and hash of the vector it named, instead of the vector.
    with Engine(workspace) as operator:
        artifact = operator.read_patch(eid, preview.patch_id)

    def record(pointer, vector):
        return {"reference": f"{uri}#{pointer}", "length": len(vector), "sha256": json_hash(vector)}

    first, second = artifact.patch.steps
    assert first.parameters == {"direction": record("/left_vectors/0", left[0]), "columns": [0, 5, 31], "alpha": 1.0}
    assert second.parameters == {"u": [record("/left_vectors/1", left[1]), [0.5] * 16],
                                 "v": [record("/right_vectors/1", right[1]), [0.25] * 32], "alpha": -0.5}
    assert [operation.parameters for operation in artifact.spec.operations] == [first.parameters, second.parameters]
    for managed, plain in zip(artifact.patch.steps, direct.patch.steps):
        assert managed.model_dump(exclude={"parameters"}) == plain.model_dump(exclude={"parameters"})
    # Another run's results are refused exactly like results that do not exist, before any weights are read, and
    # a refused preview charges nothing.
    missing = "wse://analyses/" + "0" * 64
    with Engine(workspace, run_id=grants[1].run_id) as other:
        other_eid = other.create_experiment(grants[1].run_id, TASK, REFERENCE_POLICY_ID, "c").experiment_id
        with monkeypatch.context() as patched:
            patched.setattr(Engine, "_open_checkpoint", _never)

            def refused(operation, key, hidden_uri):
                error = _expect(ErrorCode.NOT_FOUND, other.preview_patch, other_eid,
                                {**spec, "operations": [operation]}, key)
                assert error.details["operation_index"] == 0
                return error.message.replace(hidden_uri, "<uri>")

            absent = refused({**operations[0], "direction": f"{missing}#/left_vectors/0"}, "k1", missing)
            assert refused(operations[0], "k2", uri) == absent and "direction" in absent
            factors = {**operations[1], "u": [[0.5] * 16, [0.5] * 16]}
            absent = refused({**factors, "v": [f"{missing}#/right_vectors/1", [0.25] * 32]}, "k3", missing)
            assert refused(factors, "k4", uri) == absent and "v/0" in absent
        assert _consumed(other, other_eid, "patch_previews") == 0
        failures = [event for event in other.audit_log(other_eid) if event["event_type"] == "patch.preview.failed"]
        assert [event["payload"]["code"] for event in failures] == ["NOT_FOUND"] * 4
    # Direct use has no stored results to reference.
    for request in ({"op": "project_out", "direction": f"{uri}#/left_vectors/0", "columns": [0]},
                    {"op": "add_low_rank", "u": [[1.0] * 16], "v": [f"{uri}#/right_vectors/0"]}):
        error = _expect(ErrorCode.INVALID_ARGUMENT, apply_operator, handles["llama_untied"], request.pop("op"),
                        {"tensor": DOWN, "selection": full, **request}, policy=EDITS)
        assert "inline" in error.message
