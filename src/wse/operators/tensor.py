"""Built-in tensor operators.

Analyses ``tensor_stats@1``, ``tensor_norm@1``, and ``tensor_top_k@1`` read
only the region the engine passes in and accumulate in float64 without
copying it. Edits ``set_slice@1`` and ``scale_slice@1`` return new values for
that region, computed in FP32; the patch engine writes them back to the
stored dtype (BF16, F16, or F32) with round-to-nearest-even. None of them calls the model adapter's
``forward``. Element values in results use the shortest decimal digits that
round-trip the stored dtype.

A number is accepted where a float32 value is expected (``set_slice``
values, a ``scale_slice`` factor) when it is finite and rounds to a finite
float32 (``float32_finite``: its magnitude is below ``2^128 - 2^103``),
which includes the shortest digits of the largest float32,
``3.4028235e+38``. A flat list of ``set_slice`` values is accepted for a
selection with at most one axis longer than 1, and is recorded in the
selection's nesting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypeAliasType

from wse.adapters.base import CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT
from wse.operators.base import (
    BaseAnalysisOperator,
    BaseEditOperator,
    OperatorParameters,
    OperatorResult,
    TensorRegion,
    make_spec,
    parse_parameters,
)
from wse.policy import DEFAULT_TOP_K, MAX_TOP_K
from wse.schemas import ErrorCode, PatchStep, UsageContract, WSEError

_ANALYSIS_USAGE = UsageContract(counter="analysis_calls", units_per_call=1, model_forward_calls=0)


def shortest_floats(values: np.ndarray) -> list[float]:
    """Python floats whose ``repr`` uses the shortest digits of the stored dtype.

    For float32 these are NumPy's float32 digits rather than the float64 text
    that ``.tolist()`` would produce (``0.1`` instead of ``0.10000000149011612``).
    """
    if values.dtype == np.float64:
        return [float(value) for value in values.reshape(-1)]
    return [float(str(value)) for value in values.reshape(-1)]


def _require_finite(region: TensorRegion, operation: str) -> None:
    bad = int(np.count_nonzero(~np.isfinite(region.values)))
    if bad:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{operation} is undefined for tensor {region.tensor!r}: the selection contains {bad} "
            "nonfinite value(s); use the stats analysis to count them",
            details={"tensor": region.tensor, "nonfinite_count": bad},
        )


# -- stats ---------------------------------------------------------------------------


class StatsResult(OperatorResult):
    count: int = Field(description="number of elements in the selection")
    mean: float | None = Field(description="mean of the finite elements; null if none")
    std: float | None = Field(description="population standard deviation (ddof=0) of the finite elements")
    min: float | None
    max: float | None
    nonfinite_count: int


class StatsOperator(BaseAnalysisOperator):
    """count/mean/std/min/max/nonfinite_count; std is the population std (ddof=0)."""

    parameters_model = OperatorParameters
    result_model = StatsResult
    spec = make_spec(
        operator_id="tensor_stats",
        version="1",
        effect="analysis",
        description="Count, mean, population standard deviation (ddof=0), min, max, and "
        "nonfinite count of a tensor selection; statistics cover the finite elements.",
        required_capabilities=[CAP_WEIGHTS_READ],
        parameters_model=OperatorParameters,
        result_model=StatsResult,
        usage=_ANALYSIS_USAGE,
    )

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> StatsResult:
        values = region.values
        finite_mask = np.isfinite(values)
        nonfinite = int(values.size - np.count_nonzero(finite_mask))
        finite = values if nonfinite == 0 else values[finite_mask]
        if finite.size == 0:
            return StatsResult(count=int(values.size), mean=None, std=None, min=None, max=None,
                               nonfinite_count=nonfinite)
        extremes = shortest_floats(np.array([finite.min(), finite.max()], dtype=values.dtype))
        return StatsResult(
            count=int(values.size),
            mean=float(np.mean(finite, dtype=np.float64)),
            std=float(np.std(finite, dtype=np.float64, ddof=0)),
            min=extremes[0],
            max=extremes[1],
            nonfinite_count=nonfinite,
        )


# -- norm ----------------------------------------------------------------------------


class NormResult(OperatorResult):
    l1: float
    l2: float
    linf: float


class NormOperator(BaseAnalysisOperator):
    """L1, L2, and Linf norms of the flattened selection (float64 accumulation)."""

    parameters_model = OperatorParameters
    result_model = NormResult
    spec = make_spec(
        operator_id="tensor_norm",
        version="1",
        effect="analysis",
        description="L1, L2, and Linf norms of a tensor selection, treated as a flat vector.",
        required_capabilities=[CAP_WEIGHTS_READ],
        parameters_model=OperatorParameters,
        result_model=NormResult,
        usage=_ANALYSIS_USAGE,
    )

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> NormResult:
        _require_finite(region, "norm")
        values = region.values
        magnitude = np.abs(values)
        l1 = float(np.sum(magnitude, dtype=np.float64))
        l2 = math.sqrt(float(np.sum(np.square(magnitude, dtype=np.float64))))
        linf = shortest_floats(np.array([magnitude.max()], dtype=values.dtype))[0]
        return NormResult(l1=l1, l2=l2, linf=linf)


# -- top-k ---------------------------------------------------------------------------


class TopKParameters(OperatorParameters):
    k: int | None = Field(
        default=None,
        description=f"number of entries, 1 <= k <= min({MAX_TOP_K}, element_count); "
        f"omitted means min({DEFAULT_TOP_K}, element_count)",
    )


class TopKEntry(BaseModel):
    flat_index: int = Field(description="row-major index within the selection")
    index: list[int] = Field(description="coordinates within the full tensor")
    value: float


class TopKResult(OperatorResult):
    k: int
    entries: list[TopKEntry]


class TopKOperator(BaseAnalysisOperator):
    """Entries with the largest absolute values.

    Sorted by descending absolute value, then ascending flattened index for
    ties (so ``-0.0`` and ``0.0`` tie and keep index order).
    """

    parameters_model = TopKParameters
    result_model = TopKResult
    spec = make_spec(
        operator_id="tensor_top_k",
        version="1",
        effect="analysis",
        description="The k entries of a tensor selection with the largest absolute values, sorted "
        "by descending absolute value and then ascending flattened index.",
        required_capabilities=[CAP_WEIGHTS_READ],
        parameters_model=TopKParameters,
        result_model=TopKResult,
        usage=_ANALYSIS_USAGE,
    )

    def resolve_parameters(self, parameters: Mapping[str, Any], element_count: int) -> TopKParameters:
        parsed = parse_parameters(self.spec, TopKParameters, parameters)
        assert isinstance(parsed, TopKParameters)
        upper = min(MAX_TOP_K, element_count)
        if parsed.k is None:
            return TopKParameters(k=min(DEFAULT_TOP_K, element_count))
        if not 1 <= parsed.k <= upper:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"k must satisfy 1 <= k <= min({MAX_TOP_K}, element_count) = {upper} for this "
                f"selection of {element_count} element(s); got k={parsed.k}",
                details={"k": parsed.k, "max_k": upper, "element_count": element_count},
            )
        return parsed

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> TopKResult:
        assert isinstance(parameters, TopKParameters) and parameters.k is not None
        _require_finite(region, "top_k")
        flat = region.values.reshape(-1)
        magnitude = np.abs(flat)
        k = parameters.k
        if k < flat.size:
            # Every element tied with the k-th largest magnitude stays a candidate,
            # so the index tie-break is exact.
            threshold = np.partition(magnitude, flat.size - k)[flat.size - k]
            candidates = np.flatnonzero(magnitude >= threshold)
        else:
            candidates = np.arange(flat.size)
        order = candidates[np.lexsort((candidates, -magnitude[candidates].astype(np.float64)))][:k]
        values = shortest_floats(flat[order])
        return TopKResult(
            k=k,
            entries=[
                TopKEntry(flat_index=int(i), index=region.tensor_index(int(i)), value=value)
                for i, value in zip(order, values)
            ],
        )


BUILTIN_ANALYSES: dict[str, BaseAnalysisOperator] = {
    "stats": StatsOperator(),
    "norm": NormOperator(),
    "top_k": TopKOperator(),
}


# -- edits ---------------------------------------------------------------------------

_EDIT_USAGE = UsageContract(counter="candidate_commits", units_per_call=1, model_forward_calls=0)
_EDIT_CAPABILITIES = [CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT]
# Magnitudes at or above this round to infinity in float32: the largest float32 plus half its spacing.
FLOAT32_OVERFLOW = 2.0 ** 128 - 2.0 ** 103


def float32_finite(value: float) -> bool:
    """True when ``value`` is finite and rounds to a finite float32 (``3.4028235e+38`` does)."""
    return math.isfinite(value) and abs(value) < FLOAT32_OVERFLOW

# Nested lists of numbers, one nesting level per axis of the selection.
NestedValues = TypeAliasType("NestedValues", "list[float] | list[NestedValues]")


class SetSliceParameters(OperatorParameters):
    values: NestedValues = Field(
        description="replacement values as nested lists whose shape equals the selection's shape "
        "exactly (no broadcasting); a 1-axis selection takes a flat list"
    )


class SetSliceOperator(BaseEditOperator):
    """Replace a region with explicit values; shapes must match exactly."""

    parameters_model = SetSliceParameters
    spec = make_spec(
        operator_id="set_slice",
        version="1",
        effect="edit",
        description="Replace a tensor selection with explicit finite values. The values' nested-list "
        "shape must equal the selection's shape exactly; nothing is broadcast.",
        required_capabilities=_EDIT_CAPABILITIES,
        parameters_model=SetSliceParameters,
        result_model=PatchStep,
        usage=_EDIT_USAGE,
    )

    def resolve_parameters(self, parameters: Mapping[str, Any], region: TensorRegion) -> SetSliceParameters:
        """The checked values, in the selection's nesting.

        A flat list is accepted for a selection with at most one axis longer
        than 1 (for example ``[v, v, v, v]`` for one row of four columns) and
        is recorded in the selection's nesting (``[[v, v, v, v]]``), so both
        forms make the same patch.
        """
        parsed = parse_parameters(self.spec, SetSliceParameters, parameters)
        assert isinstance(parsed, SetSliceParameters)
        expected = tuple(region.values.shape)
        try:
            array = np.array(parsed.values, dtype=np.float64)
        except ValueError:
            array = None
        squeezable = sum(1 for extent in expected if extent != 1) <= 1
        if array is not None and array.ndim == 1 and array.shape != expected and squeezable \
                and array.size == math.prod(expected):
            array = array.reshape(expected)
            parsed = SetSliceParameters(values=array.tolist())
        if array is None or array.shape != expected:
            flat = (f"; a flat list of {math.prod(expected)} values is accepted too" if squeezable else
                    "; a flat list is accepted only when at most one axis of the selection is longer than 1")
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"set_slice values must be nested lists of shape {list(expected)} to match the selection "
                f"exactly ({_nesting(expected)}); got {_shape_text(parsed.values)}{flat}",
                details={"expected_shape": list(expected), "tensor": region.tensor},
            )
        _require_float32_range(array, "set_slice values", region.tensor)
        return parsed

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:
        assert isinstance(parameters, SetSliceParameters)
        return np.array(parameters.values, dtype=np.float32)


class ScaleSliceParameters(OperatorParameters):
    factor: float = Field(description="finite multiplier applied to every element of the selection")

    @field_validator("factor")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not float32_finite(value):
            raise ValueError("factor must be a finite number within the float32 range")
        return value


class ScaleSliceOperator(BaseEditOperator):
    """Multiply a region by a finite scalar in FP32; shape and dtype are preserved."""

    parameters_model = ScaleSliceParameters
    spec = make_spec(
        operator_id="scale_slice",
        version="1",
        effect="edit",
        description="Multiply every element of a tensor selection by a finite factor, computed in "
        "float32. Elements whose stored bytes do not change are not counted as changed.",
        required_capabilities=_EDIT_CAPABILITIES,
        parameters_model=ScaleSliceParameters,
        result_model=PatchStep,
        usage=_EDIT_USAGE,
    )

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:
        assert isinstance(parameters, ScaleSliceParameters)
        with np.errstate(over="ignore", invalid="ignore"):
            return region.values.astype(np.float32) * np.float32(parameters.factor)


def _shape_text(values: Any) -> str:
    try:
        return f"shape {list(np.array(values, dtype=np.float64).shape)}"
    except ValueError:
        return "ragged nested lists"


def _nesting(shape: tuple[int, ...]) -> str:
    """The nesting that ``shape`` needs, in words: ``a list of 2 lists of 4 numbers`` for ``(2, 4)``."""
    text = f"{shape[-1]} numbers"
    for extent in reversed(shape[:-1]):
        text = f"{extent} lists of {text}"
    return f"a list of {text}"


def _require_float32_range(array: np.ndarray, what: str, tensor: str) -> None:
    bad = int(np.count_nonzero(~np.isfinite(array) | (np.abs(array) >= FLOAT32_OVERFLOW)))
    if bad:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{what} for tensor {tensor!r} contain {bad} value(s) that are not finite float32 numbers",
            details={"tensor": tensor, "nonfinite_count": bad},
        )


BUILTIN_EDITS: dict[str, BaseEditOperator] = {
    "set_slice": SetSliceOperator(),
    "scale_slice": ScaleSliceOperator(),
}
