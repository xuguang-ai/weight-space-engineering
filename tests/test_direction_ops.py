"""The direction edit operator ``project_out@1`` (M2 plan Section 5.1, Task 9).

``W'[:, C] = W[:, C] - alpha d (d^T W[:, C]) / (d^T d)`` for a two-dimensional
``[out_features, in_features]`` tensor, with ``d`` on the output axis and ``C``
a strictly increasing list of input-axis indices. The in-memory model is the
signed-sum fixture; BF16 and F32 open-weight tensors come from the tiny
offline fixtures of ``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from wse.api import Engine, apply_operator, load_model
from wse.operators.directions import MAX_DIRECTION_LENGTH, MIN_DIRECTION_NORM, ProjectOutOperator
from wse.patches import _derive, replay_patch
from wse.policy import ExperimentPolicy, builtin_experiment_policies, default_operation_policy
from wse.registry import builtin_registry
from wse.schemas import ErrorCode, PatchStep, UsageContract, WSEError

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DOWN = "model.layers.0.mlp.down_proj.weight"  # [16, 32] in the Llama (BF16) and Gemma 3 (F32) fixtures
W1 = "layers.0.weight"  # [4, 2] in the signed-sum model
POLICY = default_operation_policy().with_edit("project_out", "project_out@1")


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
               for name in ("llama_untied", "gemma3")}
    return workspace, ids


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


def test_alias_group_members_are_refused(f32_model):
    assert f32_model.architecture.alias_groups == [["lm_head.weight", "model.embed_tokens.weight"]]
    error = _expect(ErrorCode.UNSUPPORTED_CAPABILITY, _project, f32_model, "lm_head.weight",
                    _sel((0, 64), (0, 16)), direction=[1.0] * 64, columns=[0])
    assert "alias group" in error.message


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
