"""Analyses for large tensors: ``tensor_norms@1``, ``svd_top_k@1``, and ``direction_projection@1``.

M2 plan Section 5.2 and Task 10. Each analysis returns a summary and its top
entries inline; in managed use its vectors are stored as a run-scoped
``wse://analyses/<sha256>`` artifact, and ``direction_projection`` accepts a
vector of such a result as its direction. Known matrices come from one-layer
MLP models; BF16 and F16 open-weight tensors come from the tiny offline
fixtures of ``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from wse.adapters.hf_causal_lm import HFCausalLMAdapter
from wse.adapters.mlp import MLPAdapter
from wse.analysis import analyze_tensor, inline_size
from wse.api import Engine, load_model
from wse.operators.base import BaseAnalysisOperator, OperatorParameters, OperatorResult, TensorRegion, make_spec
from wse.operators.linalg import (
    DEFAULT_SVD_K,
    MAX_SVD_K,
    RANDOMIZED_SVD_POWER_ITERATIONS,
    SVD_EXACT_MAX_ELEMENTS,
    DirectionProjectionResult,
    SvdTopKOperator,
    SvdTopKResult,
    TensorNormsResult,
    randomized_svd_samples,
)
from wse.patches import _derive
from wse.policy import ExperimentPolicy, builtin_experiment_policies, default_operation_policy, json_size
from wse.registry import Registry, builtin_registry
from wse.schemas import AxisSlice, ErrorCode, UsageContract, WSEError, canonical_json, json_hash

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
TASK = "tiny_lm_v1"
DOWN = "model.layers.0.mlp.down_proj.weight"  # [16, 32]: BF16 in the Llama fixture, F16 in the Qwen3 fixture
EMBED = "model.embed_tokens.weight"  # [64, 16]
NORM = "model.norm.weight"  # [16]
W = "layers.0.weight"  # the single weight of a one-layer MLP built by _matrix
KINDS = ("tensor_norms", "svd_top_k", "direction_projection")
ARTIFACT_FIELDS = {"tensor_norms": {"norms"}, "svd_top_k": {"left_vectors", "right_vectors"},
                   "direction_projection": {"direction", "projections", "cosines"}}
LIMIT = 31 * 1024  # the default inline-result limit: 32 KiB less the 1 KiB response envelope
KNOWN = [[3.0, 4.0, 0.0], [0.0, -5.0, 0.0], [1.0, 2.0, 2.0], [-6.0, 8.0, 0.0]]


def _sel(*ranges):
    return [{"start": start, "stop": stop, "step": 1} for start, stop in ranges]


def _full(*shape):
    return _sel(*[(0, size) for size in shape])


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _operations(**limits):
    policy = default_operation_policy()
    for kind in KINDS:
        policy = policy.with_analysis(kind, f"{kind}@1")
    return policy.with_limits(**limits) if limits else policy


POLICY = _operations()


def _experiment_policy(policy_id: str, *, analysis_calls: int = 50) -> ExperimentPolicy:
    return ExperimentPolicy(policy_id=policy_id, mode="weights_only", operations=POLICY,
                            budget={"candidate_commits": 1, "analysis_calls": analysis_calls, "evaluation_batches": 0,
                                    "examples_per_batch": 0, "evaluated_examples": 0, "model_forward_calls": 0})


def _matrix(values):
    """A one-layer MLP whose only weight ``layers.0.weight`` is ``values`` ([rows, columns], F32)."""
    array = np.asarray(values, dtype=np.float32)
    config = {"layer_sizes": [array.shape[1], array.shape[0]], "activation": "relu", "bias": False}
    return load_model({W: torch.from_numpy(array.copy())}, "mlp@1", config)


def _analyze(model, kind, selection, tensor=W, **parameters):
    return analyze_tensor(model, kind, tensor, selection, parameters=parameters, policy=POLICY)


def _region(values) -> TensorRegion:
    array = np.ascontiguousarray(values, dtype=np.float32)
    array.flags.writeable = False
    rows, cols = array.shape
    return TensorRegion(tensor="w", dtype="float32", axes=("out_features", "in_features"), tensor_shape=(rows, cols),
                        selection=(AxisSlice(start=0, stop=rows, step=1), AxisSlice(start=0, stop=cols, step=1)),
                        values=array)


def _consumed(engine, experiment_id, counter):
    return engine.get_experiment(experiment_id).budget[counter].consumed


@pytest.fixture(scope="module")
def hf_workspace(hf_snapshots, tmp_path_factory):
    workspace = tmp_path_factory.mktemp("linalg") / "workspace"
    with Engine(workspace) as engine:
        ids = {name: engine.import_hf_snapshot(hf_snapshots[name], repo=REPO, revision=REVISION).checkpoint_id
               for name in ("llama_untied", "qwen3")}
    return workspace, ids


@pytest.fixture(scope="module")
def bf16_model(hf_workspace):
    workspace, ids = hf_workspace
    with Engine(workspace) as engine:
        return engine.open_checkpoint(ids["llama_untied"])


@pytest.fixture
def run_setup(workspace, hf_snapshots):
    """A Llama (BF16) base checkpoint in a fresh workspace, a policy granting the three kinds, and two run grants."""
    with Engine(workspace) as operator:
        base = operator.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION).checkpoint_id
        operator.register_policy(_experiment_policy("linalg_v1"))
        grants = [operator.grant_run(TASK, "linalg_v1", base) for _ in range(2)]
    return base, grants


# -- registration -----------------------------------------------------------------------------------------------


def test_linalg_analyses_are_registered_through_operator_specs():
    registry = builtin_registry()
    for kind in KINDS:
        operator = registry.resolve_ref("operator", f"{kind}@1")
        spec = operator.spec
        assert spec.effect == "analysis" and spec.required_capabilities == ["weights.read"]
        assert spec.usage == UsageContract(counter="analysis_calls", units_per_call=1, model_forward_calls=0)
        assert spec.input_schema == operator.parameters_model.model_json_schema()
        assert spec.output_schema == operator.result_model.model_json_schema()
        assert set(operator.artifact_fields) == ARTIFACT_FIELDS[kind]
        # The vector fields are optional, so the inline result (which omits them) is valid against the output schema.
        assert not any(operator.result_model.model_fields[name].is_required() for name in operator.artifact_fields)
    assert registry.resolve_ref("operator", "direction_projection@1").reference_parameters == ("direction",)
    assert (MAX_SVD_K, DEFAULT_SVD_K, SVD_EXACT_MAX_ELEMENTS) == (16, 8, 4 * 1024 * 1024)
    # The built-in policies do not grant them; a policy grants each under a kind of its choice.
    assert set(default_operation_policy().analyses) == {"stats", "norm", "top_k"}
    for policy in builtin_experiment_policies().values():
        assert not set(KINDS) & set(policy.operations.analyses)


class _VectorResult(OperatorResult):
    vector: list[float]


def _custom(*, artifact_fields=(), reference_parameters=()):
    class Custom(BaseAnalysisOperator):
        result_model = _VectorResult
        spec = make_spec(operator_id="custom_vector", version="1", effect="analysis", description="a vector",
                         required_capabilities=["weights.read"], parameters_model=OperatorParameters,
                         result_model=_VectorResult, usage=UsageContract(counter="analysis_calls"))

        def analyze(self, region, parameters):  # pragma: no cover - never registered
            return _VectorResult(vector=[])

    Custom.artifact_fields = artifact_fields
    Custom.reference_parameters = reference_parameters
    return Custom()


@pytest.mark.parametrize("declaration, fragment", [
    ({"artifact_fields": ("vector",)}, "optional"),  # a required field cannot be left out of the inline result
    ({"artifact_fields": ("missing",)}, "missing"),
    ({"reference_parameters": ("direction",)}, "direction"),
])
def test_artifact_fields_and_reference_parameters_are_checked_at_registration(declaration, fragment):
    with pytest.raises(WSEError) as excinfo:
        Registry().register("operator", "custom_vector", "1", _custom(**declaration))
    assert excinfo.value.code == ErrorCode.INVALID_ARGUMENT and fragment in excinfo.value.message


# -- the plan's tests ---------------------------------------------------------------------------------------------


def test_tensor_norms_on_a_known_matrix():
    model = _matrix(KNOWN)
    rows = _analyze(model, "tensor_norms", _full(4, 3), axis="rows")
    assert rows.parameters == {"axis": "rows", "top": 16}
    assert rows.result == {
        "axis": "rows", "axis_name": "out_features", "offset": 0,
        "summary": {"count": 4, "min": 3.0, "max": 10.0, "mean": 5.75, "std": math.sqrt(6.6875)},
        # Descending norm, then ascending index for ties.
        "top": [{"index": 3, "value": 10.0}, {"index": 0, "value": 5.0}, {"index": 1, "value": 5.0},
                {"index": 2, "value": 3.0}],
        "norms": [5.0, 5.0, 3.0, 10.0],  # direct use keeps the vector inline
    }
    assert rows.result_uri is None and rows.usage.model_forward_calls == 0
    columns = _analyze(model, "tensor_norms", _full(4, 3), axis="columns", top=2).result
    assert columns["axis_name"] == "in_features" and columns["summary"]["count"] == 3
    assert columns["norms"] == [math.sqrt(46), math.sqrt(109), 2.0]
    assert columns["top"] == [{"index": 1, "value": math.sqrt(109)}, {"index": 0, "value": math.sqrt(46)}]
    # A slice reports tensor indices, and its vector starts at the slice's offset.
    part = _analyze(model, "tensor_norms", _sel((1, 4), (1, 3)), axis="rows", top=64).result
    assert part["offset"] == 1 and part["norms"] == [5.0, math.sqrt(8), 8.0]
    assert part["top"] == [{"index": 3, "value": 8.0}, {"index": 1, "value": 5.0}, {"index": 2, "value": math.sqrt(8)}]
    TensorNormsResult.model_validate(part)


def test_svd_top_k_matches_numpy_on_a_small_matrix_up_to_sign():
    values = np.random.default_rng(5).normal(0.0, 1.0, size=(7, 5)).astype(np.float32)
    model = _matrix(values)
    result = _analyze(model, "svd_top_k", _full(7, 5), k=4)
    assert result.parameters == {"k": 4, "top": 8, "method": "exact", "seed": None}
    found = result.result
    u, s, vt = np.linalg.svd(values.astype(np.float64), full_matrices=False)
    assert found["method"] == "exact" and found["k"] == 4 and found["shape"] == [7, 5]
    np.testing.assert_allclose(found["singular_values"], s[:4], rtol=1e-5)
    np.testing.assert_allclose(found["energy_fractions"], s[:4] ** 2 / np.sum(s ** 2), rtol=1e-5)
    assert found["frobenius_norm"] == pytest.approx(float(np.linalg.norm(values.astype(np.float64))), rel=1e-12)
    for i in range(4):
        left, right = np.array(found["left_vectors"][i]), np.array(found["right_vectors"][i])
        sign = np.sign(left @ u[:, i])
        np.testing.assert_allclose(left, sign * u[:, i], atol=1e-5)
        np.testing.assert_allclose(right, sign * vt[i], atol=1e-5)  # u and v flip together
        # Sign convention: the largest-magnitude entry of each left vector is positive.
        peak = int(np.argmax(np.abs(left)))
        assert left[peak] > 0
        assert found["left_top"][i][0] == {"index": peak, "value": found["left_vectors"][i][peak]}
        assert len(found["left_top"][i]) == 7 and len(found["right_top"][i]) == 5  # at most top=8 entries
        magnitudes = [abs(entry["value"]) for entry in found["right_top"][i]]
        assert magnitudes == sorted(magnitudes, reverse=True)
    # The default k is min(8, rows, columns); all components rebuild the matrix.
    full = _analyze(model, "svd_top_k", _full(7, 5)).result
    assert full["k"] == 5 and _analyze(model, "svd_top_k", _full(7, 5)).parameters["k"] is None
    rebuilt = sum(value * np.outer(left, right) for value, left, right in
                  zip(full["singular_values"], full["left_vectors"], full["right_vectors"]))
    np.testing.assert_allclose(rebuilt, values, atol=1e-5)
    assert sum(full["energy_fractions"]) == pytest.approx(1.0, abs=1e-6)
    # A slice reports tensor indices.
    part = _analyze(model, "svd_top_k", _sel((2, 7), (1, 5)), k=2, top=3).result
    assert part["shape"] == [5, 4] and all(len(entries) == 3 for entries in part["left_top"] + part["right_top"])
    assert {e["index"] for entries in part["left_top"] for e in entries} <= set(range(2, 7))
    assert {e["index"] for entries in part["right_top"] for e in entries} <= set(range(1, 5))
    np.testing.assert_allclose(part["singular_values"],
                               np.linalg.svd(values[2:7, 1:5].astype(np.float64), compute_uv=False)[:2], rtol=1e-5)
    SvdTopKResult.model_validate(part)


def _planted(rows, cols, spectrum, seed):
    """A matrix with known singular values ``spectrum`` and singular vectors, plus noise far below them."""
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.standard_normal((rows, len(spectrum))))
    right, _ = np.linalg.qr(rng.standard_normal((cols, len(spectrum))))
    noise = rng.standard_normal((rows, cols), dtype=np.float32) * np.float32(1e-6)
    return ((left * spectrum) @ right.T).astype(np.float32) + noise, left, right


def test_randomized_svd_is_used_above_the_threshold_and_is_reproducible_with_its_seed(run_setup, workspace):
    operator = SvdTopKOperator()
    at = operator.resolve_parameters({"k": 4}, SVD_EXACT_MAX_ELEMENTS)
    above = operator.resolve_parameters({"k": 4}, SVD_EXACT_MAX_ELEMENTS + 1)
    assert (at.method, at.seed) == ("exact", None)  # the seed of an exact SVD is not recorded
    assert (above.method, above.seed) == ("randomized", 0)  # the default seed is 0
    assert operator.resolve_parameters({"k": 4, "seed": 7}, SVD_EXACT_MAX_ELEMENTS + 1).seed == 7
    assert operator.resolve_parameters({"k": 4, "method": "randomized"}, 12).method == "randomized"
    error = _expect(ErrorCode.INVALID_ARGUMENT, operator.resolve_parameters, {"k": 4, "method": "exact"},
                    SVD_EXACT_MAX_ELEMENTS + 1)
    assert str(SVD_EXACT_MAX_ELEMENTS) in error.message.replace(",", "")
    assert randomized_svd_samples(4, 2048, 2049) == 24 and randomized_svd_samples(16, 16, 32) == 16
    assert RANDOMIZED_SVD_POWER_ITERATIONS == 10

    # A matrix just above the threshold uses the randomized method and recovers the planted components.
    spectrum = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25])
    values, left, right = _planted(2048, 2049, spectrum, seed=3)
    assert values.size == SVD_EXACT_MAX_ELEMENTS + 2048
    region = _region(values)
    first = operator.analyze(region, above)
    assert first.method == "randomized" and first.k == 4
    np.testing.assert_allclose(first.singular_values, spectrum[:4], rtol=1e-5)
    for i in range(4):
        assert abs(np.dot(first.left_vectors[i], left[:, i])) > 1 - 1e-6
        assert abs(np.dot(first.right_vectors[i], right[:, i])) > 1 - 1e-6
    # The same seed gives the same result; another seed gives another sketch, equally accurate.
    assert operator.analyze(region, above) == first
    other = operator.analyze(region, operator.resolve_parameters({"k": 4, "seed": 1}, values.size))
    np.testing.assert_allclose(other.singular_values, spectrum[:4], rtol=1e-5)
    assert other.right_vectors != first.right_vectors

    # Through the managed path the seed is part of the recorded parameters and the input hash.
    base, grants = run_setup
    with Engine(workspace) as engine:
        eid = engine.create_experiment(grants[0].run_id, TASK, "linalg_v1", "c").experiment_id
        request = {"k": 4, "method": "randomized", "seed": 3}
        once = engine.analyze_model(eid, base, "svd_top_k", DOWN, _full(16, 32), parameters=request)
        again = engine.analyze_model(eid, base, "svd_top_k", DOWN, _full(16, 32), parameters=request)
        assert once.parameters == {"k": 4, "top": 8, "method": "randomized", "seed": 3}
        assert (again.result_uri, again.input_hash) == (once.result_uri, once.input_hash)
        reseeded = engine.analyze_model(eid, base, "svd_top_k", DOWN, _full(16, 32), parameters={**request, "seed": 4})
        assert reseeded.input_hash != once.input_hash


def test_direction_projection_on_known_vectors():
    # Columns: c0 = [3, 4, 0], c1 = 0, c2 = [6, 8, 30], c3 = [-4, 3, 0] (orthogonal to d), c4 = -c0.
    model = _matrix([[3.0, 0.0, 6.0, -4.0, -3.0], [4.0, 0.0, 8.0, 3.0, -4.0], [0.0, 0.0, 30.0, 0.0, 0.0]])
    cosine_c2 = 50.0 / (5.0 * math.sqrt(1000.0))
    columns = _analyze(model, "direction_projection", _full(3, 5), direction=[3.0, 4.0, 0.0], axis="columns")
    assert columns.parameters == {"direction": {"length": 3, "sha256": json_hash([3.0, 4.0, 0.0])},
                                  "axis": "columns", "measure": "projection", "top": 16}
    found = columns.result
    assert found["projections"] == [5.0, 0.0, 10.0, 0.0, -5.0]  # (d . c) / |d|
    assert found["cosines"][:2] == [1.0, 0.0] and found["cosines"][3:] == [0.0, -1.0]
    assert found["cosines"][2] == pytest.approx(cosine_c2, rel=1e-15)
    assert (found["direction"], found["direction_norm"], found["direction_length"]) == ([3.0, 4.0, 0.0], 5.0, 3)
    assert (found["axis"], found["axis_name"], found["offset"], found["zero_norm_count"]) == ("columns", "in_features",
                                                                                               0, 1)
    # Ranked by |projection|, then ascending index; each entry carries both measures.
    assert [(e["index"], e["projection"]) for e in found["top"]] == [(2, 10.0), (0, 5.0), (4, -5.0), (1, 0.0),
                                                                      (3, 0.0)]
    assert found["summary"] == {"count": 5, "min": -5.0, "max": 10.0, "mean": 2.0, "std": math.sqrt(26.0)}
    # Ranked by |cosine| instead: c0 and c4 are exactly aligned, c2 is long but only partly aligned.
    by_cosine = _analyze(model, "direction_projection", _full(3, 5), direction=[6.0, 8.0, 0.0], axis="columns",
                         measure="cosine", top=3).result
    assert [e["index"] for e in by_cosine["top"]] == [0, 4, 2]
    assert by_cosine["top"][0] == {"index": 0, "projection": 5.0, "cosine": 1.0}  # independent of the scale of d
    assert by_cosine["summary"]["max"] == 1.0 and by_cosine["summary"]["min"] == -1.0
    # Against the rows, d has one value per column.
    rows = _analyze(model, "direction_projection", _sel((0, 3), (1, 5)), direction=[0.0, 1.0, 0.0, 0.0],
                    axis="rows").result
    assert rows["projections"] == [6.0, 8.0, 30.0] and rows["offset"] == 0 and rows["axis_name"] == "out_features"
    assert rows["cosines"] == pytest.approx([6.0 / math.sqrt(61.0), 8.0 / math.sqrt(89.0), 1.0], rel=1e-15)
    DirectionProjectionResult.model_validate(rows)


def _worst_case(result, inline, parameters):
    """``result`` with ``inline`` as its inline result and ``parameters``: the response that would be returned."""
    return result.model_copy(update={"result": inline, "parameters": parameters})


def test_vector_results_return_a_summary_and_a_run_scoped_uri_within_31_kib(run_setup, workspace):
    base, grants = run_setup
    # Shortest float32 digits, the form in which a direction is stored.
    direction = [float(str(value)) for value in np.random.default_rng(1).normal(0.0, 1.0, size=16).astype(np.float32)]
    requests = [
        ("tensor_norms", EMBED, _full(64, 16), {"axis": "rows", "top": 64}),
        ("svd_top_k", DOWN, _full(16, 32), {"k": 16, "top": 16}),
        ("direction_projection", DOWN, _full(16, 32), {"direction": direction, "axis": "columns", "top": 32}),
    ]
    uris = []
    with Engine(workspace, run_id=grants[0].run_id) as bound:
        eid = bound.create_experiment(grants[0].run_id, TASK, "linalg_v1", "c").experiment_id
        for kind, tensor, selection, parameters in requests:
            result = bound.analyze_model(eid, base, kind, tensor, selection, parameters=parameters)
            assert json_size(result.model_dump(mode="json")) <= LIMIT
            assert result.result_uri is not None and result.result_uri.startswith("wse://analyses/")
            stored = bound.read_analysis_result(result.result_uri)
            # The response carries the summary and top entries; the stored result adds the vectors.
            vectors = ARTIFACT_FIELDS[kind]
            assert vectors <= set(stored) and not vectors & set(result.result)
            assert result.result == {key: value for key, value in stored.items() if key not in vectors}
            assert result.result_hash == json_hash(stored) == "sha256:" + result.result_uri.removeprefix(
                "wse://analyses/")
            assert result.usage.artifact_bytes == result.result_bytes == len(canonical_json(stored).encode("utf-8"))
            uris.append(result.result_uri)
            if kind == "tensor_norms":
                assert len(stored["norms"]) == 64 and len(result.result["top"]) == 64
            elif kind == "svd_top_k":
                assert [len(v) for v in stored["left_vectors"]] == [16] * 16
                assert [len(v) for v in stored["right_vectors"]] == [32] * 16
            else:
                assert len(stored["projections"]) == len(stored["cosines"]) == 32 and len(result.result["top"]) == 32
                assert stored["direction"] == direction
        # The inline parts are bounded whatever the tensor: worst-case digits and ten-digit indices still fit.
        last = bound.analyze_model(eid, base, "svd_top_k", DOWN, _full(16, 32), parameters={"k": 16, "top": 16})
    f32, f64, big = -1.2345679e-05, -1.2345678901234567e-300, 2 ** 31 - 1
    assert len(json.dumps(f32)) == 14 and len(json.dumps(f64)) == 24
    svd = SvdTopKResult(method="randomized", k=16, shape=[big, big], singular_values=[f32] * 16, frobenius_norm=f64,
                        energy_fractions=[f64] * 16, left_top=[[{"index": big, "value": f32}] * 16] * 16,
                        right_top=[[{"index": big, "value": f32}] * 16] * 16)
    summary = {"count": big, "min": f64, "max": f64, "mean": f64, "std": f64}
    norms = TensorNormsResult(axis="columns", axis_name="in_features", offset=big, summary=summary,
                              top=[{"index": big, "value": f64}] * 64)
    projection = DirectionProjectionResult(
        axis="columns", axis_name="in_features", measure="projection", offset=big, direction_length=2048,
        direction_norm=f64, zero_norm_count=big, summary=summary,
        top=[{"index": big, "projection": f64, "cosine": f64}] * 64)
    worst_parameters = {
        "svd_top_k": {"k": 16, "top": 16, "method": "randomized", "seed": 2 ** 63 - 1},
        "tensor_norms": {"axis": "columns", "top": 64},
        "direction_projection": {"direction": {"length": 2048, "sha256": json_hash([f32] * 2048)}, "axis": "columns",
                                 "measure": "projection", "top": 64},
    }
    for kind, worst in (("svd_top_k", svd), ("tensor_norms", norms), ("direction_projection", projection)):
        inline = worst.model_dump(mode="json", exclude=ARTIFACT_FIELDS[kind])
        assert inline_size(_worst_case(last, inline, worst_parameters[kind])) <= LIMIT
    # Another run cannot read these results, exactly as if they did not exist.
    with Engine(workspace, run_id=grants[1].run_id) as other:
        other.create_experiment(grants[1].run_id, TASK, "linalg_v1", "c")
        missing = "wse://analyses/" + "0" * 64
        absent = _expect(ErrorCode.NOT_FOUND, other.read_analysis_result, missing).message.replace(missing, "<uri>")
        for uri in uris:
            assert _expect(ErrorCode.NOT_FOUND, other.read_analysis_result, uri).message.replace(uri, "<uri>") == absent
    # A 2,048-value inline direction is recorded by its length and hash, so its response stays small.
    long_direction = [f32] * 2048
    model = _matrix(np.ones((2048, 2), dtype=np.float32))
    wide = _analyze(model, "direction_projection", _full(2048, 2), direction=long_direction, axis="columns")
    assert wide.parameters["direction"] == {"length": 2048, "sha256": json_hash(long_direction)}
    assert json_size({**wide.model_dump(mode="json"), "result": {
        key: value for key, value in wide.result.items() if key not in ARTIFACT_FIELDS["direction_projection"]}}) < 8192


def test_analyses_never_call_forward(hf_workspace, bf16_model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an analysis must never run the target network")

    for cls, name in ((MLPAdapter, "forward"), (HFCausalLMAdapter, "forward"), (HFCausalLMAdapter, "runtime"),
                      (torch.nn.Module, "__call__")):
        monkeypatch.setattr(cls, name, forbidden)
    workspace, ids = hf_workspace
    requests = [("tensor_norms", {"axis": "rows"}), ("svd_top_k", {"k": 3}),
                ("svd_top_k", {"k": 3, "method": "randomized"}),
                ("direction_projection", {"direction": [1.0] * 16, "axis": "columns"})]
    known = _matrix(np.arange(16 * 32, dtype=np.float32).reshape(16, 32))
    with Engine(workspace, policy=POLICY) as engine:
        for kind, parameters in requests:
            for checkpoint in ids.values():
                managed = engine.analyze(checkpoint, kind, DOWN, _full(16, 32), parameters=parameters)
                assert managed.usage.model_forward_calls == 0 and managed.result_uri is not None
            assert _analyze(bf16_model, kind, _full(16, 32), tensor=DOWN, **parameters).usage.model_forward_calls == 0
            assert _analyze(known, kind, _full(16, 32), **parameters).usage.model_forward_calls == 0


def test_analysis_usage_is_metered(workspace, hf_snapshots):
    with Engine(workspace) as engine:
        base = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION).checkpoint_id
        engine.register_policy(_experiment_policy("linalg_metered_v1", analysis_calls=4))
        grant = engine.grant_run(TASK, "linalg_metered_v1", base)
        eid = engine.create_experiment(grant.run_id, TASK, "linalg_metered_v1", "c").experiment_id
        requests = [("tensor_norms", {"axis": "columns"}), ("svd_top_k", {"k": 2}),
                    ("direction_projection", {"direction": [0.5] * 16, "axis": "columns"})]
        for count, (kind, parameters) in enumerate(requests, start=1):
            before = _consumed(engine, eid, "compute_seconds")
            result = engine.analyze_model(eid, base, kind, DOWN, _full(16, 32), parameters=parameters)
            assert result.usage.analysis_calls == 1 and result.usage.model_forward_calls == 0
            assert result.usage.tensor_read_calls is None and result.usage.candidate_commits is None
            assert result.usage.compute_seconds == result.elapsed_seconds
            assert result.usage.artifact_bytes == result.result_bytes
            assert _consumed(engine, eid, "analysis_calls") == count
            assert _consumed(engine, eid, "compute_seconds") > before
        # A refused request charges nothing.
        _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze_model, eid, base, "svd_top_k", DOWN, _full(16, 32),
                parameters={"k": 17})
        _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze_model, eid, base, "direction_projection", DOWN,
                _full(16, 32), parameters={"direction": [1.0] * 32, "axis": "columns"})
        assert _consumed(engine, eid, "analysis_calls") == 3
        engine.analyze_model(eid, base, "tensor_norms", EMBED, _full(64, 16), parameters={"axis": "rows"})
        error = _expect(ErrorCode.BUDGET_EXCEEDED, engine.analyze_model, eid, base, "tensor_norms", EMBED,
                        _full(64, 16), parameters={"axis": "rows"})
        assert "analysis_calls" in error.message
        assert _consumed(engine, eid, "analysis_calls") == 4
        assert _consumed(engine, eid, "tensor_read_calls") == 0 and _consumed(engine, eid, "model_forward_calls") == 0


# -- open-weight dtypes, references, and refusals -----------------------------------------------------------------


def test_bf16_and_f16_regions_are_analyzed_as_their_float32_values(hf_workspace, bf16_model):
    planted = torch.zeros(16, 32, dtype=torch.bfloat16)
    planted[0, 0], planted[0, 1], planted[5, 7] = 3.0, 4.0, -0.5
    model = _derive(bf16_model, {**bf16_model._tensors, DOWN: planted})
    norms = _analyze(model, "tensor_norms", _full(16, 32), tensor=DOWN, axis="rows", top=2)
    assert norms.result["top"] == [{"index": 0, "value": 5.0}, {"index": 5, "value": 0.5}]
    # M1's analyses accept BF16 regions too.
    stats = analyze_tensor(model, "stats", DOWN, _full(16, 32)).result
    widened = planted.to(torch.float32).numpy()
    assert stats == {"count": 512, "mean": 6.5 / 512, "std": float(np.std(widened, dtype=np.float64)), "min": -0.5,
                     "max": 4.0, "nonfinite_count": 0}
    workspace, ids = hf_workspace
    with Engine(workspace, policy=POLICY) as engine:
        for name, dtype in (("llama_untied", "bfloat16"), ("qwen3", "float16")):
            checkpoint = ids[name]
            assert engine.load_manifest(checkpoint).tensor_info(DOWN).dtype == dtype
            reference = engine.open_checkpoint(checkpoint).tensor(DOWN).to(torch.float32).numpy().astype(np.float64)
            columns = engine.analyze(checkpoint, "tensor_norms", DOWN, _full(16, 32), parameters={"axis": "columns"})
            stored = engine.read_analysis_result(columns.result_uri)
            np.testing.assert_allclose(stored["norms"], np.linalg.norm(reference, axis=0), rtol=1e-12)
            svd = engine.analyze(checkpoint, "svd_top_k", DOWN, _full(16, 32), parameters={"k": 4})
            np.testing.assert_allclose(svd.result["singular_values"], np.linalg.svd(reference, compute_uv=False)[:4],
                                       rtol=1e-5)
        # A BF16 read returns the shortest digits of each value's float32 widening.
        read = engine.read_tensor_slice(ids["llama_untied"], DOWN, _sel((0, 1), (0, 8)))
        first_row = engine.open_checkpoint(ids["llama_untied"]).tensor(DOWN)[0, :8].to(torch.float32).numpy()
        assert read.dtype == "bfloat16" and read.values == [float(str(value)) for value in first_row]


def test_direction_references_resolve_vectors_of_the_runs_own_results(run_setup, workspace, bf16_model):
    base, grants = run_setup
    full = _full(16, 32)
    with Engine(workspace, run_id=grants[0].run_id) as bound:
        eid = bound.create_experiment(grants[0].run_id, TASK, "linalg_v1", "c").experiment_id
        svd = bound.analyze_model(eid, base, "svd_top_k", DOWN, full, parameters={"k": 2})
        stored = bound.read_analysis_result(svd.result_uri)
        s0, u0, v0 = stored["singular_values"][0], np.array(stored["left_vectors"][0]), stored["right_vectors"][0]
        reference = f"{svd.result_uri}#/right_vectors/0"
        rows = bound.analyze_model(eid, base, "direction_projection", DOWN, full,
                                   parameters={"direction": reference, "axis": "rows"})
        assert rows.parameters["direction"] == reference  # recorded as given; the URI is content-addressed
        projected = bound.read_analysis_result(rows.result_uri)
        assert projected["direction"] == v0
        np.testing.assert_allclose(projected["projections"], s0 * u0, rtol=1e-4, atol=1e-5 * s0)  # W v0 = s0 u0
        # The stored direction of that result is itself a vector that can be referenced.
        again = bound.analyze_model(eid, base, "direction_projection", DOWN, full,
                                    parameters={"direction": f"{rows.result_uri}#/direction", "axis": "rows"})
        assert bound.read_analysis_result(again.result_uri)["projections"] == projected["projections"]
        columns = bound.analyze_model(eid, base, "direction_projection", DOWN, full,
                                      parameters={"direction": f"{svd.result_uri}#/left_vectors/0", "axis": "columns"})
        np.testing.assert_allclose(bound.read_analysis_result(columns.result_uri)["projections"], np.array(v0) * s0,
                                   rtol=1e-4, atol=1e-5 * s0)  # W^T u0 = s0 v0

        def refused(direction, code=ErrorCode.INVALID_ARGUMENT, axis="columns"):
            before = _consumed(bound, eid, "analysis_calls")
            error = _expect(code, bound.analyze_model, eid, base, "direction_projection", DOWN, full,
                            parameters={"direction": direction, "axis": axis})
            assert _consumed(bound, eid, "analysis_calls") == before
            return error.message

        assert "16" in refused(reference)  # a 32-value vector against 16-value columns
        assert "list of numbers" in refused(f"{svd.result_uri}#/singular_values/0")
        assert "list of numbers" in refused(f"{svd.result_uri}#/left_top/0")
        assert "no_such_field" in refused(f"{svd.result_uri}#/no_such_field")
        assert "right_vectors/7" in refused(f"{svd.result_uri}#/right_vectors/7", axis="rows")
        refused(svd.result_uri)  # no JSON pointer
        refused(f"{svd.result_uri}#/right_vectors/00", axis="rows")  # not a canonical pointer
        missing = "wse://analyses/" + "0" * 64
        absent = refused(f"{missing}#/norms", ErrorCode.NOT_FOUND).replace(missing, "<uri>")
    with Engine(workspace, run_id=grants[1].run_id) as other:
        other_eid = other.create_experiment(grants[1].run_id, TASK, "linalg_v1", "c").experiment_id
        hidden = _expect(ErrorCode.NOT_FOUND, other.analyze_model, other_eid, base, "direction_projection", DOWN, full,
                         parameters={"direction": f"{svd.result_uri}#/norms", "axis": "rows"})
        assert hidden.message.replace(svd.result_uri, "<uri>") == absent
    # Direct use has no stored results to reference.
    error = _expect(ErrorCode.INVALID_ARGUMENT, _analyze, bf16_model, "direction_projection", full, tensor=DOWN,
                    direction=reference, axis="rows")
    assert "inline" in error.message


@pytest.mark.parametrize("kind, tensor, selection, parameters, fragment", [
    ("tensor_norms", NORM, _full(16), {"axis": "rows"}, "two-dimensional"),
    ("tensor_norms", DOWN, _full(16, 32), {}, "axis"),
    ("tensor_norms", DOWN, _full(16, 32), {"axis": "diagonal"}, "axis"),
    ("tensor_norms", DOWN, _full(16, 32), {"axis": "rows", "top": 65}, "top"),
    ("tensor_norms", DOWN, _full(16, 32), {"axis": "rows", "k": 3}, "Accepted parameters"),
    ("svd_top_k", DOWN, _full(16, 32), {"k": 17}, "k"),
    ("svd_top_k", DOWN, _full(16, 32), {"k": 0}, "k"),
    ("svd_top_k", DOWN, _sel((0, 16), (0, 3)), {"k": 4}, "min(16, rows, columns) = 3"),
    ("svd_top_k", DOWN, _full(16, 32), {"k": 2, "top": 17}, "top"),
    ("svd_top_k", DOWN, _full(16, 32), {"k": 2, "method": "lanczos"}, "method"),
    ("svd_top_k", DOWN, _full(16, 32), {"k": 2, "seed": -1}, "seed"),
    ("svd_top_k", NORM, _full(16), {"k": 1}, "two-dimensional"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [1.0] * 16}, "axis"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [1.0] * 32, "axis": "columns"}, "16"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [1.0] * 16, "axis": "rows"}, "32"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [0.0] * 16, "axis": "columns"}, "norm"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [1e-9] * 16, "axis": "columns"}, "norm"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [math.inf] + [1.0] * 15, "axis": "columns"},
     "finite"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [1.0] * 2049, "axis": "columns"}, "2048"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [], "axis": "columns"}, "direction"),
    ("direction_projection", DOWN, _full(16, 32), {"direction": [1.0] * 16, "axis": "columns", "measure": "dot"},
     "measure"),
    ("direction_projection", NORM, _full(16), {"direction": [1.0] * 16, "axis": "columns"}, "two-dimensional"),
])
def test_invalid_requests_are_refused_and_charge_nothing(run_setup, workspace, kind, tensor, selection, parameters,
                                                         fragment):
    base, grants = run_setup
    with Engine(workspace) as engine:
        eid = engine.create_experiment(grants[0].run_id, TASK, "linalg_v1", "c").experiment_id
        error = _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze_model, eid, base, kind, tensor, selection,
                        parameters=parameters)
        assert fragment in error.message
        assert _consumed(engine, eid, "analysis_calls") == 0 and _consumed(engine, eid, "compute_seconds") == 0


def test_nonfinite_selections_are_refused():
    # Models never hold nonfinite weights (imports and edits refuse them), so the operators are called directly.
    values = np.ones((16, 32), dtype=np.float32)
    values[3, 4], values[7, 1] = np.inf, np.nan
    region = _region(values)
    registry = builtin_registry()
    for kind, parameters in (("tensor_norms", {"axis": "rows"}), ("svd_top_k", {"k": 2}),
                             ("svd_top_k", {"k": 2, "method": "randomized"}),
                             ("direction_projection", {"direction": [1.0] * 16, "axis": "columns"})):
        operator = registry.resolve_ref("operator", f"{kind}@1")
        resolved = operator.resolve_parameters(parameters, region.element_count)
        error = _expect(ErrorCode.INVALID_ARGUMENT, operator.analyze, region, resolved)
        assert "2 nonfinite" in error.message and error.details["nonfinite_count"] == 2
