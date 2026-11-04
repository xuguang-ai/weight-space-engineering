"""Bounded weight reading and registered analyses (stats, norm, top-k)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from wse.adapters.base import CAP_FORWARD_LOGITS, CAP_WEIGHTS_READ, AdapterCapabilities
from wse.adapters.mlp import MLPAdapter
from wse.analysis import analyze_tensor, read_tensor_slice
from wse.api import Engine, load_model
from wse.operators.base import BaseAnalysisOperator, OperatorParameters, OperatorResult, make_spec
from wse.policy import (
    DEFAULT_TOP_K,
    MAX_INLINE_JSON_BYTES,
    MAX_READ_ELEMENTS,
    MAX_TOP_K,
    default_operation_policy,
    json_size,
)
from wse.registry import builtin_registry
from wse.schemas import AnalysisResult, ErrorCode, UsageContract, WSEError

WIDE_CONFIG = {"layer_sizes": [64, 64, 2], "activation": "relu", "bias": True}
LONG_ROW_CONFIG = {"layer_sizes": [4096, 1, 2], "activation": "relu", "bias": True}
SMALL_CONFIG = {"layer_sizes": [2, 2], "activation": "relu", "bias": False}


def _sel(*ranges):
    return [{"start": start, "stop": stop, "step": 1} for start, stop in ranges]


def _full(shape):
    return _sel(*[(0, size) for size in shape])


def _expect(code, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _mlp_tensors(config, fill):
    sizes = config["layer_sizes"]
    tensors = {}
    for index, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
        tensors[f"layers.{index}.weight"] = torch.as_tensor(fill((fan_out, fan_in)), dtype=torch.float32)
        if config["bias"]:
            tensors[f"layers.{index}.bias"] = torch.as_tensor(fill((fan_out,)), dtype=torch.float32)
    return tensors


def _normal(seed=0):
    rng = np.random.default_rng(seed)
    return lambda shape: rng.normal(0.0, 0.02, size=shape).astype(np.float32)


def _huge(seed=0):
    # Magnitudes >= 1e15 serialize as 19-20 characters each, so 2,048 values exceed 32 KiB.
    rng = np.random.default_rng(seed)
    return lambda shape: (-rng.uniform(1e15, 9e15, size=shape)).astype(np.float32)


@pytest.fixture
def task_ckpt(engine, bundles):
    return engine.import_bundle(bundles["task"])


@pytest.fixture
def wide_ckpt(engine):
    return engine.import_model({"adapter": "mlp@1", "config": WIDE_CONFIG}, _mlp_tensors(WIDE_CONFIG, _normal()))


@pytest.fixture
def huge_ckpt(engine):
    return engine.import_model({"adapter": "mlp@1", "config": WIDE_CONFIG}, _mlp_tensors(WIDE_CONFIG, _huge()))


# -- inspect ------------------------------------------------------------------------------


def test_inspect_returns_the_ready_manifest(engine, task_ckpt):
    assert engine.inspect(task_ckpt.checkpoint_id) == engine.load_manifest(task_ckpt.checkpoint_id)
    _expect(ErrorCode.NOT_FOUND, engine.inspect, "ckpt_" + "f" * 32)
    _expect(ErrorCode.NOT_FOUND, engine.inspect, "../metadata.sqlite")


# -- selection boundaries -----------------------------------------------------------------


@pytest.mark.parametrize(
    "selection",
    [
        _sel((0, 5), (0, 2)),  # stop beyond the axis
        _sel((-1, 2), (0, 2)),  # negative start
        _sel((2, 2), (0, 2)),  # empty
        _sel((3, 1), (0, 2)),  # reversed
        [{"start": 0, "stop": 4, "step": 2}, {"start": 0, "stop": 2, "step": 1}],  # step != 1
        _sel((0, 4)),  # too few axes
        _sel((0, 4), (0, 2), (0, 1)),  # too many axes
        [{"start": True, "stop": 4, "step": 1}, {"start": 0, "stop": 2, "step": 1}],  # bool
        [{"start": 0.0, "stop": 4, "step": 1}, {"start": 0, "stop": 2, "step": 1}],  # float
        [{"start": "0", "stop": 4, "step": 1}, {"start": 0, "stop": 2, "step": 1}],  # string
        [{"start": 0, "stop": 4, "step": 1, "expr": "x"}, {"start": 0, "stop": 2, "step": 1}],
        "0:4, 0:2",
        "[:, :]",
        None,
        {"start": 0, "stop": 4},
    ],
)
def test_invalid_selections_are_rejected_for_reads_and_analyses(engine, task_ckpt, selection):
    ckpt = task_ckpt.checkpoint_id
    _expect(ErrorCode.INVALID_ARGUMENT, engine.read_tensor_slice, ckpt, "layers.0.weight", selection)
    for kind in ("stats", "norm", "top_k"):
        _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze, ckpt, kind, "layers.0.weight", selection)


def test_selection_edges_are_inclusive_of_the_last_element(engine, task_ckpt, signed_sum):
    ckpt = task_ckpt.checkpoint_id
    weight = signed_sum.task_tensors()["layers.1.weight"]
    corner = engine.read_tensor_slice(ckpt, "layers.1.weight", _sel((1, 2), (3, 4)))
    assert corner.shape == [1, 1] and corner.values == [float(weight[1, 3])]
    full = engine.read_tensor_slice(ckpt, "layers.1.weight", _full([2, 4]))
    assert full.shape == [2, 4] and full.element_count == 8
    assert full.values == [float(v) for v in weight.reshape(-1)]
    assert full.axes == ["out_features", "in_features"]
    assert full.tensor_shape == [2, 4]
    assert full.checkpoint_id == ckpt and full.weights_hash == task_ckpt.weights_hash


def test_tensor_names_and_kinds_must_match_exactly(engine, task_ckpt):
    ckpt = task_ckpt.checkpoint_id
    for name in ["layers.0.weight[0]", "LAYERS.0.WEIGHT", " layers.0.weight", "layers.0.*", "layers.0"]:
        _expect(ErrorCode.NOT_FOUND, engine.read_tensor_slice, ckpt, name, _sel((0, 1), (0, 1)))
        _expect(ErrorCode.NOT_FOUND, engine.analyze, ckpt, "stats", name, _sel((0, 1), (0, 1)))
    for kind in ["stats()", "top-k", "TOP_K", "svd", "__import__('os')", "", None]:
        error = _expect(ErrorCode.POLICY_DENIED, engine.analyze, ckpt, kind, "layers.0.weight", _full([4, 2]))
        assert error.details["permitted_kinds"] == ["norm", "stats", "top_k"]


# -- the 2,048-element limit ----------------------------------------------------------------


def test_read_of_exactly_2048_elements_succeeds(engine, wide_ckpt):
    rows = MAX_READ_ELEMENTS // 64
    result = engine.read_tensor_slice(wide_ckpt.checkpoint_id, "layers.0.weight", _sel((0, rows), (0, 64)))
    assert result.element_count == MAX_READ_ELEMENTS == 2048
    assert json_size(result.model_dump(mode="json")) <= MAX_INLINE_JSON_BYTES
    assert result.usage.tensor_read_calls == 1 and result.usage.model_forward_calls == 0
    assert result.management == "managed"


def test_oversized_read_reports_the_largest_permitted_selection(engine, wide_ckpt):
    ckpt = wide_ckpt.checkpoint_id
    error = _expect(ErrorCode.INVALID_ARGUMENT, engine.read_tensor_slice, ckpt, "layers.0.weight",
                    _sel((10, 64), (0, 64)))
    suggestion = error.details["largest_permitted_selection"]
    assert suggestion == _sel((10, 42), (0, 64))
    assert error.details["max_elements"] == 2048 and error.details["requested_elements"] == 54 * 64
    assert json.dumps(suggestion, separators=(",", ":")) in error.message
    assert engine.read_tensor_slice(ckpt, "layers.0.weight", suggestion).element_count == 2048


def test_oversized_single_row_shrinks_the_inner_axis(engine):
    tensors = _mlp_tensors(LONG_ROW_CONFIG, _normal(1))
    ckpt = engine.import_model({"adapter": "mlp@1", "config": LONG_ROW_CONFIG}, tensors).checkpoint_id
    error = _expect(ErrorCode.INVALID_ARGUMENT, engine.read_tensor_slice, ckpt, "layers.0.weight",
                    _sel((0, 1), (100, 4096)))
    assert error.details["largest_permitted_selection"] == _sel((0, 1), (100, 2148))


def test_bounded_read_tensor_returns_exact_values_and_enforces_the_limit(engine, wide_ckpt):
    ckpt = wide_ckpt.checkpoint_id
    region = engine.read_tensor(ckpt, "layers.0.weight", _sel((0, 32), (0, 64)))
    assert region.shape == (32, 64) and region.dtype == torch.float32
    _expect(ErrorCode.INVALID_ARGUMENT, engine.read_tensor, ckpt, "layers.0.weight", _full([64, 64]))


# -- the 32 KiB canonical-JSON limit ------------------------------------------------------


def test_normal_weights_at_2048_elements_fit_and_round_trip_bitwise(engine, wide_ckpt):
    ckpt = wide_ckpt.checkpoint_id
    result = engine.read_tensor_slice(ckpt, "layers.0.weight", _sel((0, 32), (0, 64)))
    text = result.canonical_json()
    assert len(text.encode()) <= MAX_INLINE_JSON_BYTES
    stored = engine.store.load_tensors(ckpt)["layers.0.weight"][:32].numpy()
    decoded = np.array(json.loads(text)["values"], dtype=np.float32).reshape(32, 64)
    assert decoded.tobytes() == stored.tobytes()
    # Shortest float32 digits, not the float64 text of .tolist(): about 25 KiB, not about 43 KiB.
    assert len(text.encode()) < 27 * 1024
    assert len(json.dumps(stored.reshape(-1).tolist(), separators=(",", ":"))) > 40 * 1024


def test_2048_values_of_14_characters_each_fit(engine):
    rng = np.random.default_rng(3)

    def fourteen(shape):
        # Eight-digit decimals such as -1.2345678e-05 parse to float32 values whose
        # shortest representation has at most eight digits, i.e. at most 14 characters.
        mantissas = rng.integers(11_000_001, 19_999_999, size=shape)
        text = [f"-{m // 10_000_000}.{m % 10_000_000:07d}e-05" for m in mantissas.reshape(-1)]
        return np.array(text, dtype=np.float32).reshape(shape)

    tensors = _mlp_tensors(WIDE_CONFIG, fourteen)
    ckpt = engine.import_model({"adapter": "mlp@1", "config": WIDE_CONFIG}, tensors).checkpoint_id
    result = engine.read_tensor_slice(ckpt, "layers.0.weight", _sel((0, 32), (0, 64)))
    lengths = [len(json.dumps(value)) for value in result.values]
    assert max(lengths) == 14 and sum(lengths) / len(lengths) > 13.5
    assert 29 * 1024 < len(result.canonical_json().encode()) <= MAX_INLINE_JSON_BYTES


def test_values_use_the_shortest_float32_digits():
    tensors = {"layers.0.weight": torch.tensor([[0.1, -0.2], [1e-8, 3.4028235e38]], dtype=torch.float32)}
    model = load_model(tensors, "mlp@1", SMALL_CONFIG)
    result = read_tensor_slice(model, "layers.0.weight", _full([2, 2]))
    text = result.canonical_json()
    assert '"values":[0.1,-0.2,1e-08,3.4028235e+38]' in text
    assert result.management == "unmanaged" and result.checkpoint_id is None


def test_read_exceeding_32_kib_reports_the_largest_fitting_selection(engine, huge_ckpt):
    ckpt = huge_ckpt.checkpoint_id
    error = _expect(ErrorCode.INVALID_ARGUMENT, engine.read_tensor_slice, ckpt, "layers.0.weight",
                    _sel((0, 32), (0, 64)))
    assert error.details["limit"] == "max_inline_json_bytes"
    assert error.details["max_inline_json_bytes"] == MAX_INLINE_JSON_BYTES
    suggestion = error.details["largest_permitted_selection"]
    assert suggestion[1] == {"start": 0, "stop": 64, "step": 1}
    rows = suggestion[0]["stop"]
    assert 0 < rows < 32
    fitted = engine.read_tensor_slice(ckpt, "layers.0.weight", suggestion)
    assert len(fitted.canonical_json().encode()) <= MAX_INLINE_JSON_BYTES
    larger = _sel((0, rows + 1), (0, 64))
    again = _expect(ErrorCode.INVALID_ARGUMENT, engine.read_tensor_slice, ckpt, "layers.0.weight", larger)
    assert again.details["limit"] == "max_inline_json_bytes"


def test_element_limit_suggestion_also_respects_the_json_limit(engine, huge_ckpt):
    ckpt = huge_ckpt.checkpoint_id
    error = _expect(ErrorCode.INVALID_ARGUMENT, engine.read_tensor_slice, ckpt, "layers.0.weight",
                    _full([64, 64]))
    assert error.details["limit"] == "max_read_elements"
    suggestion = error.details["largest_permitted_selection"]
    assert suggestion[0]["stop"] < 32
    engine.read_tensor_slice(ckpt, "layers.0.weight", suggestion)


# -- top-k parameter rules ------------------------------------------------------------------


def test_default_top_k_succeeds_on_tensors_with_fewer_than_16_elements(engine, task_ckpt, signed_sum):
    ckpt = task_ckpt.checkpoint_id
    for info in task_ckpt.tensors:  # every tensor of the signed-sum fixture has < 16 elements
        assert info.element_count < DEFAULT_TOP_K
        result = engine.analyze(ckpt, "top_k", info.name, _full(info.shape))
        assert result.parameters == {"k": info.element_count}
        assert result.result["k"] == info.element_count
        assert len(result.result["entries"]) == info.element_count


def test_default_top_k_is_16_on_larger_selections(engine, wide_ckpt):
    result = engine.analyze(wide_ckpt.checkpoint_id, "top_k", "layers.0.weight", _full([64, 64]))
    assert result.parameters == {"k": 16} and len(result.result["entries"]) == 16


@pytest.mark.parametrize("k", [0, -1, MAX_TOP_K + 1, 1000])
def test_out_of_range_top_k_is_rejected(engine, wide_ckpt, k):
    error = _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze, wide_ckpt.checkpoint_id, "top_k",
                    "layers.0.weight", _full([64, 64]), k=k)
    assert error.details["max_k"] == MAX_TOP_K


def test_top_k_cannot_exceed_the_selection_size(engine, task_ckpt):
    ckpt = task_ckpt.checkpoint_id
    error = _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze, ckpt, "top_k", "layers.1.bias",
                    _full([2]), k=3)
    assert error.details["max_k"] == 2
    assert engine.analyze(ckpt, "top_k", "layers.1.bias", _full([2]), k=2).result["k"] == 2


def test_top_k_accepts_the_maximum_k(engine, wide_ckpt):
    result = engine.analyze(wide_ckpt.checkpoint_id, "top_k", "layers.0.weight", _full([64, 64]), k=64)
    assert len(result.result["entries"]) == 64


@pytest.mark.parametrize("k", ["4", 4.0, True, [4], "min(16, n)"])
def test_non_integer_k_is_rejected(engine, wide_ckpt, k):
    _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze, wide_ckpt.checkpoint_id, "top_k",
            "layers.0.weight", _full([64, 64]), k=k)


@pytest.mark.parametrize("kind", ["stats", "norm"])
def test_other_kinds_do_not_accept_k(engine, task_ckpt, kind):
    error = _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze, task_ckpt.checkpoint_id, kind,
                    "layers.0.weight", _full([4, 2]), k=2)
    assert error.details["accepted_parameters"] == []


# -- known matrices -------------------------------------------------------------------------


def _known_model():
    weight = torch.tensor([[3.0, -3.0], [1.0, -4.0]], dtype=torch.float32)
    return load_model({"layers.0.weight": weight}, "mlp@1", SMALL_CONFIG)


def test_stats_on_a_known_matrix():
    result = analyze_tensor(_known_model(), "stats", "layers.0.weight", _full([2, 2]))
    values = np.array([3.0, -3.0, 1.0, -4.0])
    assert result.result == {
        "count": 4,
        "mean": pytest.approx(values.mean()),
        "std": pytest.approx(values.std(ddof=0)),
        "min": -4.0,
        "max": 3.0,
        "nonfinite_count": 0,
    }
    assert result.result["std"] == pytest.approx(math.sqrt(np.mean((values - values.mean()) ** 2)))
    assert result.result["std"] != pytest.approx(values.std(ddof=1))


def test_norms_on_a_known_matrix():
    result = analyze_tensor(_known_model(), "norm", "layers.0.weight", _full([2, 2]))
    assert result.result == {"l1": 11.0, "l2": pytest.approx(math.sqrt(35.0)), "linf": 4.0}
    row = analyze_tensor(_known_model(), "norm", "layers.0.weight", _sel((1, 2), (0, 2)))
    assert row.result == {"l1": 5.0, "l2": pytest.approx(math.sqrt(17.0)), "linf": 4.0}


def test_top_k_orders_by_magnitude_then_flat_index():
    result = analyze_tensor(_known_model(), "top_k", "layers.0.weight", _full([2, 2]), k=3)
    assert result.result["entries"] == [
        {"flat_index": 3, "index": [1, 1], "value": -4.0},
        {"flat_index": 0, "index": [0, 0], "value": 3.0},  # tie with index 1: lower index first
        {"flat_index": 1, "index": [0, 1], "value": -3.0},
    ]


def test_top_k_indices_are_reported_in_tensor_coordinates(engine, wide_ckpt):
    ckpt = wide_ckpt.checkpoint_id
    stored = engine.store.load_tensors(ckpt)["layers.0.weight"].numpy()
    result = engine.analyze(ckpt, "top_k", "layers.0.weight", _sel((5, 20), (7, 30)), k=10)
    region = stored[5:20, 7:30]
    flat = region.reshape(-1)
    expected = sorted(range(flat.size), key=lambda i: (-abs(float(flat[i])), i))[:10]
    entries = result.result["entries"]
    assert [entry["flat_index"] for entry in entries] == expected
    for entry in entries:
        row, col = entry["index"]
        assert np.float32(entry["value"]) == stored[row, col]
        assert (row - 5) * 23 + (col - 7) == entry["flat_index"]


def test_top_k_ties_include_signed_zeros_in_index_order():
    weight = torch.tensor([[0.0, -0.0], [2.0, -2.0]], dtype=torch.float32)
    model = load_model({"layers.0.weight": weight}, "mlp@1", SMALL_CONFIG)
    result = analyze_tensor(model, "top_k", "layers.0.weight", _full([2, 2]))
    assert [entry["flat_index"] for entry in result.result["entries"]] == [2, 3, 0, 1]
    assert math.copysign(1.0, result.result["entries"][3]["value"]) == -1.0


def test_known_matrix_stats_match_through_the_managed_path(engine, signed_sum, task_ckpt):
    result = engine.analyze(task_ckpt.checkpoint_id, "stats", "layers.1.weight", _full([2, 4]))
    weight = signed_sum.task_tensors()["layers.1.weight"].numpy().astype(np.float64)
    assert result.result["mean"] == pytest.approx(weight.mean())
    assert result.result["std"] == pytest.approx(weight.std())
    norm = engine.analyze(task_ckpt.checkpoint_id, "norm", "layers.1.weight", _full([2, 4]))
    assert norm.result == {"l1": 8.0, "l2": pytest.approx(math.sqrt(8.0)), "linf": 1.0}


def test_stats_count_nonfinite_values_and_norm_rejects_them():
    from wse.operators.base import TensorRegion
    from wse.operators.tensor import NormOperator, StatsOperator
    from wse.schemas import AxisSlice

    values = np.array([1.0, np.nan, np.inf, 3.0], dtype=np.float32)
    values.flags.writeable = False
    region = TensorRegion(tensor="t", dtype="float32", axes=("x",), tensor_shape=(4,),
                          selection=(AxisSlice(start=0, stop=4),), values=values)
    stats = StatsOperator().analyze(region, OperatorParameters())
    assert stats.model_dump() == {"count": 4, "mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0,
                                  "nonfinite_count": 2}
    _expect(ErrorCode.INVALID_ARGUMENT, NormOperator().analyze, region, OperatorParameters())


# -- result contents, parity, and isolation from forward -----------------------------------


def test_analysis_result_records_operation_input_hash_and_elapsed_time(engine, task_ckpt):
    result = engine.analyze(task_ckpt.checkpoint_id, "top_k", "layers.1.weight", _full([2, 4]), k=2)
    assert isinstance(result, AnalysisResult)
    assert str(result.operator) == "tensor_top_k@1" and result.kind == "top_k"
    assert result.input_hash.startswith("sha256:") and result.result_hash.startswith("sha256:")
    assert result.elapsed_seconds >= 0
    assert result.usage.analysis_calls == 1 and result.usage.model_forward_calls == 0
    assert result.usage.compute_seconds == result.elapsed_seconds
    assert result.model == task_ckpt.identity and result.checkpoint_id == task_ckpt.checkpoint_id
    assert result.result_uri is None and result.management == "managed"
    assert result.result_bytes == len(json.dumps(result.result, sort_keys=True, separators=(",", ":")))
    # The input hash binds the parameters and the selection.
    other_k = engine.analyze(task_ckpt.checkpoint_id, "top_k", "layers.1.weight", _full([2, 4]), k=3)
    other_sel = engine.analyze(task_ckpt.checkpoint_id, "top_k", "layers.1.weight", _sel((0, 1), (0, 4)), k=2)
    assert len({result.input_hash, other_k.input_hash, other_sel.input_hash}) == 3
    same = engine.analyze(task_ckpt.checkpoint_id, "top_k", "layers.1.weight", _full([2, 4]), k=2)
    assert same.input_hash == result.input_hash and same.result == result.result


def test_direct_and_managed_analyses_agree(engine, task_ckpt, signed_sum):
    model = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    for kind in ("stats", "norm", "top_k"):
        for info in task_ckpt.tensors:
            managed = engine.analyze(task_ckpt.checkpoint_id, kind, info.name, _full(info.shape))
            direct = analyze_tensor(model, kind, info.name, _full(info.shape))
            assert direct.management == "unmanaged" and direct.checkpoint_id is None
            assert (direct.result, direct.input_hash, direct.operator) == (
                managed.result, managed.input_hash, managed.operator)
    direct_read = read_tensor_slice(model, "layers.0.weight", _full([4, 2]))
    managed_read = engine.read_tensor_slice(task_ckpt.checkpoint_id, "layers.0.weight", _full([4, 2]))
    assert direct_read.values == managed_read.values


def test_reads_and_analyses_never_call_forward(engine, task_ckpt, signed_sum, monkeypatch):
    def forbidden(self, tensors, inputs):
        raise AssertionError("adapter.forward must not be called by inspection or analysis")

    monkeypatch.setattr(MLPAdapter, "forward", forbidden)
    ckpt = task_ckpt.checkpoint_id
    model = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    engine.inspect(ckpt)
    for info in task_ckpt.tensors:
        engine.read_tensor_slice(ckpt, info.name, _full(info.shape))
        engine.read_tensor(ckpt, info.name, _full(info.shape))
        read_tensor_slice(model, info.name, _full(info.shape))
        for kind in ("stats", "norm", "top_k"):
            assert engine.analyze(ckpt, kind, info.name, _full(info.shape)).usage.model_forward_calls == 0
            analyze_tensor(model, kind, info.name, _full(info.shape))


def test_direct_analysis_does_not_mutate_or_copy_the_model(signed_sum):
    model = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
    before = model.weights_hash

    class Mutating(BaseAnalysisOperator):
        result_model = OperatorResult
        spec = make_spec(operator_id="mutating", version="1", effect="analysis", description="tries to write",
                         required_capabilities=[CAP_WEIGHTS_READ], parameters_model=OperatorParameters,
                         result_model=OperatorResult, usage=UsageContract(counter="analysis_calls"))

        def analyze(self, region, parameters):
            assert not region.values.flags.writeable
            assert np.shares_memory(region.values, model._tensors[region.tensor].numpy())
            with pytest.raises(ValueError):
                region.values[...] = 0
            return OperatorResult()

    registry = builtin_registry()
    registry.register("operator", "mutating", "1", Mutating())
    policy = default_operation_policy().with_analysis("mutating", "mutating@1")
    analyze_tensor(model, "mutating", "layers.0.weight", _full([4, 2]), registry=registry, policy=policy)
    from wse.tensors import weights_hash
    assert weights_hash(dict(model._tensors)) == before


# -- extension contract -------------------------------------------------------------------


class AbsMeanResult(OperatorResult):
    abs_mean: float


class AbsMeanOperator(BaseAnalysisOperator):
    """A separately supplied analysis: the mean absolute value."""

    result_model = AbsMeanResult
    spec = make_spec(
        operator_id="abs_mean", version="1", effect="analysis", description="Mean absolute value.",
        required_capabilities=[CAP_WEIGHTS_READ], parameters_model=OperatorParameters,
        result_model=AbsMeanResult, usage=UsageContract(counter="analysis_calls"),
    )

    def analyze(self, region, parameters):
        return AbsMeanResult(abs_mean=float(np.mean(np.abs(region.values), dtype=np.float64)))


class SpectralResult(OperatorResult):
    radius: float


class SpectralOperator(BaseAnalysisOperator):
    """Requires a capability that no registered adapter advertises."""

    result_model = SpectralResult
    spec = make_spec(
        operator_id="spectral_radius", version="1", effect="analysis", description="Spectral radius.",
        required_capabilities=[CAP_WEIGHTS_READ, "weights.spectral"], parameters_model=OperatorParameters,
        result_model=SpectralResult, usage=UsageContract(counter="analysis_calls"),
    )

    def analyze(self, region, parameters):  # pragma: no cover - must never run
        raise AssertionError("capability check must reject before analysis")


class DumpResult(OperatorResult):
    values: list[float]


class DumpOperator(BaseAnalysisOperator):
    """A large-output analysis used to exercise the large-object response policy."""

    result_model = DumpResult
    spec = make_spec(
        operator_id="dump", version="1", effect="analysis", description="Every value, squared.",
        required_capabilities=[CAP_WEIGHTS_READ], parameters_model=OperatorParameters,
        result_model=DumpResult, usage=UsageContract(counter="analysis_calls"),
    )

    def analyze(self, region, parameters):
        return DumpResult(values=(region.values.astype(np.float64).reshape(-1) ** 2).tolist())


def _extended(*operators):
    registry = builtin_registry()
    policy = default_operation_policy()
    for kind, operator in operators:
        registry.register("operator", operator.spec.operator_id, operator.spec.version, operator)
        policy = policy.with_analysis(kind, operator.spec.ref)
    return registry, policy


def test_separately_supplied_analysis_runs_without_dispatch_changes(workspace, signed_sum):
    registry, policy = _extended(("abs_mean", AbsMeanOperator()))
    with Engine(workspace, registry=registry, policy=policy) as engine:
        ckpt = engine.import_model({"adapter": "mlp@1", "config": signed_sum.CONFIG}, signed_sum.task_tensors())
        result = engine.analyze(ckpt.checkpoint_id, "abs_mean", "layers.1.weight", _full([2, 4]))
        assert result.result == {"abs_mean": 1.0} and str(result.operator) == "abs_mean@1"
        model = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
        direct = analyze_tensor(model, "abs_mean", "layers.1.weight", _full([2, 4]), registry=registry, policy=policy)
        assert direct.result == result.result
    # Registered but not permitted by the policy: denied.
    with Engine(workspace, registry=registry) as engine:
        _expect(ErrorCode.POLICY_DENIED, engine.analyze, ckpt.checkpoint_id, "abs_mean", "layers.1.weight",
                _full([2, 4]))
    # Permitted by the policy but not registered: not found, never substituted.
    with Engine(workspace, policy=policy) as engine:
        _expect(ErrorCode.NOT_FOUND, engine.analyze, ckpt.checkpoint_id, "abs_mean", "layers.1.weight",
                _full([2, 4]))


def test_missing_operator_capability_is_rejected_before_reading(workspace, signed_sum, monkeypatch):
    registry, policy = _extended(("spectral", SpectralOperator()))
    with Engine(workspace, registry=registry, policy=policy) as engine:
        ckpt = engine.import_model({"adapter": "mlp@1", "config": signed_sum.CONFIG}, signed_sum.task_tensors())

        def no_reads(*args, **kwargs):
            raise AssertionError("weights must not be read before the capability check")

        monkeypatch.setattr(engine.store, "read_slice", no_reads)
        error = _expect(ErrorCode.UNSUPPORTED_CAPABILITY, engine.analyze, ckpt.checkpoint_id, "spectral",
                        "layers.1.weight", _full([2, 4]))
        assert error.details["missing_capabilities"] == ["weights.spectral"]
        model = load_model(signed_sum.task_tensors(), "mlp@1", signed_sum.CONFIG)
        _expect(ErrorCode.UNSUPPORTED_CAPABILITY, analyze_tensor, model, "spectral", "layers.1.weight",
                _full([2, 4]), registry=registry, policy=policy)


class OpaqueAdapter(MLPAdapter):
    """An adapter that does not advertise weight reads."""

    adapter_id = "opaque_mlp"
    version = "1"

    def capabilities(self):
        return AdapterCapabilities(capabilities=frozenset({CAP_FORWARD_LOGITS}))

    def validate(self, manifest, tensors):
        from wse.adapters.base import check_tensors_match

        architecture = self.describe(manifest.config)
        check_tensors_match(architecture, tensors)
        return architecture


def test_adapter_without_weight_reads_rejects_reads_and_builtin_analyses(workspace, signed_sum):
    registry = builtin_registry()
    registry.register("adapter", "opaque_mlp", "1", OpaqueAdapter())
    with Engine(workspace, registry=registry) as engine:
        ckpt = engine.import_model({"adapter": "opaque_mlp@1", "config": signed_sum.CONFIG},
                                   signed_sum.task_tensors()).checkpoint_id
        for kind in ("stats", "norm", "top_k"):
            _expect(ErrorCode.UNSUPPORTED_CAPABILITY, engine.analyze, ckpt, kind, "layers.0.weight", _full([4, 2]))
        _expect(ErrorCode.UNSUPPORTED_CAPABILITY, engine.read_tensor_slice, ckpt, "layers.0.weight", _full([4, 2]))


def test_large_results_are_referenced_by_artifact_uri(workspace):
    registry, policy = _extended(("dump", DumpOperator()))
    tensors = _mlp_tensors(WIDE_CONFIG, _normal(2))
    with Engine(workspace, registry=registry, policy=policy) as engine:
        ckpt = engine.import_model({"adapter": "mlp@1", "config": WIDE_CONFIG}, tensors).checkpoint_id
        result = engine.analyze(ckpt, "dump", "layers.0.weight", _full([64, 64]))
        assert result.result is None
        assert result.result_uri is not None and result.result_uri.startswith("wse://analyses/")
        assert result.result_bytes > MAX_INLINE_JSON_BYTES
        assert json_size(result.model_dump(mode="json")) <= MAX_INLINE_JSON_BYTES
        stored = engine.read_analysis_result(result.result_uri)
        expected = (tensors["layers.0.weight"].numpy().astype(np.float64).reshape(-1) ** 2).tolist()
        assert stored == {"values": expected}
        assert engine.analyze(ckpt, "dump", "layers.0.weight", _full([64, 64])).result_uri == result.result_uri
        small = engine.analyze(ckpt, "dump", "layers.0.weight", _sel((0, 1), (0, 4)))
        assert small.result is not None and small.result_uri is None
        _expect(ErrorCode.NOT_FOUND, engine.read_analysis_result, "wse://analyses/" + "0" * 64)
        _expect(ErrorCode.INVALID_ARGUMENT, engine.read_analysis_result, "file:///etc/passwd")
        _expect(ErrorCode.INVALID_ARGUMENT, engine.read_analysis_result, "wse://analyses/../../metadata")
    # Direct use has no transport limit, so results stay inline.
    model = load_model(tensors, "mlp@1", WIDE_CONFIG)
    direct = analyze_tensor(model, "dump", "layers.0.weight", _full([64, 64]), registry=registry, policy=policy)
    assert direct.result_uri is None and direct.result is not None
    assert direct.result_hash == result.result_hash


def test_analysis_element_limit_is_enforced_before_reading(workspace, signed_sum):
    policy = default_operation_policy().with_limits(max_analysis_elements=4)
    with Engine(workspace, policy=policy) as engine:
        ckpt = engine.import_model({"adapter": "mlp@1", "config": signed_sum.CONFIG}, signed_sum.task_tensors())
        error = _expect(ErrorCode.INVALID_ARGUMENT, engine.analyze, ckpt.checkpoint_id, "stats",
                        "layers.0.weight", _full([4, 2]))
        assert error.details["largest_permitted_selection"] == _sel((0, 2), (0, 2))
        assert engine.analyze(ckpt.checkpoint_id, "stats", "layers.0.weight", _sel((0, 2), (0, 2))).result["count"] == 4
