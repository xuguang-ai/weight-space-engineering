"""Built-in M2 edit operators (plan Section 5.1): ``project_out@1``, ``add_low_rank@1``, and ``fill_slice@1``.

Each requires a two-dimensional tensor with known axes (no ``unknown`` axis).
Like every edit, the result is written back to the stored dtype (BF16, F16,
or F32) by the patch engine with round-to-nearest-even, only inside the
selection; changed elements are those whose stored bits differ. No operator
here runs a matrix product (BLAS), so a result is the same on every platform.

``project_out@1`` removes an output-axis direction ``d`` from selected
input-axis indices (columns) ``C`` of an ``[out_features, in_features]``
tensor (the layout of a ``Linear.weight``)::

    W'[:, C] = W[:, C] - alpha * d (d^T W[:, C]) / (d^T d)

With ``alpha = 1`` every edited column becomes orthogonal to ``d`` (up to the
rounding of the stored dtype). Columns of the selection outside ``C`` keep
their stored bytes and are not counted as changed. ``d`` and the region are
taken as float32 values; every product ``d_i * W[i, c]`` is formed in float64,
where it is exact; each sum (``s_c = d^T W[:, c]`` and ``n = d^T d``) is the
correctly rounded ``math.fsum`` of those exact products; and each entry is
updated with elementwise IEEE float64 operations, ``t = d_i * (alpha * (s_c /
n))`` and ``W'[i, c] = W[i, c] - t``, except that an entry whose term ``t`` is
zero keeps its value (so ``-0.0`` stays ``-0.0``). The float64 result is
rounded to float32, and the patch engine rounds that to the stored dtype.

``add_low_rank@1`` adds ``alpha U V^T`` to the selected region ``W[R, C]``.
``U`` is given as ``r`` columns of one value per selected row and ``V`` as
``r`` columns of one value per selected column, with ``1 <= r <= 16``; for a
selection of the whole tensor, ``U`` is ``[out, r]`` and ``V`` is ``[in, r]``.
Factor values are taken as float32. Each entry's sum ``S_ij = sum_k U_ik
V_jk`` of exact float64 products is correctly rounded to float64 (one IEEE
addition for ``r <= 2``, which is correctly rounded, and ``math.fsum``
otherwise), so it depends neither on the platform nor on the order of the
factor columns. Then ``t = alpha * S_ij`` and ``W'_ij = W_ij + t`` in float64,
except that an entry whose ``t`` is zero keeps its value; the result is
rounded to float32 and then by the engine to the stored dtype, as the
article's low-rank rebuild rounds FP64 to FP32 to BF16. The article's factors
are FP64, so a published low-rank edit is reproduced by its exact patch
(``wse import-patch``), not by re-running this operator.

``fill_slice@1`` sets every element of the selection to one finite scalar,
for example to ablate an attention head. The value is rounded to float32 and
then to the stored dtype, exactly as the engine writes it, and recorded in
that form, in the digits of reads (the shortest digits of a BF16 value's
float32 widening, of an F16 value, or of an F32 value); a value that rounds to
infinity in the stored dtype is refused.

**Vectors.** ``project_out``'s ``direction`` and each column of
``add_low_rank``'s ``u`` and ``v`` are given inline or as a vector reference
``wse://analyses/<sha256>#<JSON pointer>`` to a stored analysis result of the
run (``reference_parameters``), which only managed previews resolve. A
recorded step keeps an inline vector's values (shortest float32 digits) and
records a reference as ``{reference, length, sha256}``: the reference and the
length and SHA-256 (``json_hash`` of the float32 values' shortest digits) of
the vector it named. Replay needs neither, because the step records its
after-values. An inline direction holds at most ``MAX_DIRECTION_LENGTH``
(2,048) values, the size of one bounded read. Inline factor columns hold at
most ``MAX_INLINE_FACTOR_VALUES`` (16,384) values in all, a quarter of the
65,536 inline elements that one patch may carry, so that one request stays
well inside that limit (16,384 values are at most about 245 KB of JSON); a
rank-2 update of a ``[1024, 3584]`` tensor (9,216 values) fits inline, and
longer factors go by reference. These values, and
``project_out``'s ``columns``, count toward the patch's inline elements
(``inline_parameters``).

**Numbers.** A value is accepted where a float32 one is expected when it is
finite and rounds to a finite float32 (its magnitude is below ``2^128 -
2^103``). This includes the shortest digits of the largest float32,
``3.4028235e+38``, which exceed that largest value as a float64.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wse.adapters.base import CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT
from wse.operators.base import (
    BaseEditOperator,
    OperatorParameters,
    TensorRegion,
    VectorReference,
    make_spec,
    parse_parameters,
)
from wse.operators.tensor import shortest_floats
from wse.policy import MAX_PATCH_ELEMENTS
from wse.schemas import ErrorCode, PatchStep, UsageContract, WSEError, json_hash
from wse.tensors import bits_to_values, values_to_bits

__all__ = [
    "BUILTIN_DIRECTION_EDITS",
    "MAX_DIRECTION_LENGTH",
    "MAX_INLINE_FACTOR_VALUES",
    "MAX_LOW_RANK",
    "MIN_DIRECTION_NORM",
    "PROJECT_OUT_AXES",
    "AddLowRankOperator",
    "AddLowRankParameters",
    "FillSliceOperator",
    "FillSliceParameters",
    "ProjectOutOperator",
    "ProjectOutParameters",
    "VectorRecord",
    "vector_summary",
]

# An inline direction holds at most this many values (one per output row).
MAX_DIRECTION_LENGTH = 2048
# A direction whose Euclidean norm (of its float32 values) is below this is refused: the
# projection divides by d^T d, and a direction this small is almost certainly a mistake.
MIN_DIRECTION_NORM = 1e-6
# The axis layout the projection requires: axis 0 is the output axis that d spans.
PROJECT_OUT_AXES = ("out_features", "in_features")
# The largest rank of add_low_rank (M2 plan Section 5.1).
MAX_LOW_RANK = 16
# Inline factor columns of one add_low_rank operation hold at most this many values in all.
MAX_INLINE_FACTOR_VALUES = 16_384
# The adapters' name for an axis whose meaning they do not know (wse.adapters.hf_causal_lm).
UNKNOWN_AXIS = "unknown"

_EDIT_USAGE = UsageContract(counter="candidate_commits", units_per_call=1, model_forward_calls=0)
_EDIT_CAPABILITIES = [CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT]
# Magnitudes at or above this round to infinity in float32: the largest float32 plus half its spacing.
_FLOAT32_OVERFLOW = 2.0 ** 128 - 2.0 ** 103
# Low-rank sums are formed over row blocks of about this many elements, which bounds their memory.
_LOW_RANK_BLOCK_ELEMENTS = 1 << 16
# The largest finite value of each stored dtype (BF16 0x7F7F and F16 0x7BFF).
_LARGEST_FINITE = {"bfloat16": float(bits_to_values(np.array([0x7F7F], dtype=np.uint16), "bfloat16")[0]),
                   "float16": float(np.finfo(np.float16).max), "float32": float(np.finfo(np.float32).max)}


def _float32_finite(value: float) -> bool:
    """True when ``value`` is finite and rounds to a finite float32."""
    return math.isfinite(value) and abs(value) < _FLOAT32_OVERFLOW


def _bad_numbers(values: list[float]) -> int:
    return sum(1 for entry in values if not _float32_finite(entry))


def vector_summary(values: list[float]) -> dict[str, Any]:
    """``{length, sha256}`` of a vector given by its recorded values: the form agent-facing views echo."""
    return {"length": len(values), "sha256": json_hash(values)}


class VectorRecord(BaseModel):
    """How a step records a vector given by reference: the reference, and the length and hash of the vector it named.

    ``sha256`` is ``json_hash`` of the vector's float32 values in their
    shortest digits, the hash that agent-facing views give an inline vector,
    so equal vectors have equal hashes however they were given.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: str
    length: int
    sha256: str

    @classmethod
    def of(cls, reference: str, vector: np.ndarray) -> VectorRecord:
        return cls(reference=reference, **vector_summary(shortest_floats(vector)))


def _referenced(region: TensorRegion, key: str) -> np.ndarray:
    """The resolved vector of reference parameter ``key`` (``direction``, ``u/0``, ...)."""
    vector = region.inputs.get(key)
    if vector is None:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"the {key} reference was not resolved; references resolve only in managed previews",
                       details={"parameter": key})
    return vector


def _vector(region: TensorRegion, value: list[float] | VectorRecord, key: str) -> np.ndarray:
    """The float32 values of a recorded vector parameter: inline values, or the vector its reference named."""
    if isinstance(value, VectorRecord):
        return _referenced(region, key)
    return np.asarray(value, dtype=np.float32)


def _require_two_known_axes(region: TensorRegion, operation: str) -> None:
    tensor, shape = region.tensor, list(region.tensor_shape)
    if len(shape) != 2:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"{operation} requires a two-dimensional tensor with known axes; {tensor!r} has shape {shape}",
                       details={"tensor": tensor, "shape": shape})
    if UNKNOWN_AXIS in region.axes:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"{operation} requires a two-dimensional tensor with known axes; the adapter does not know the "
                       f"axes of {tensor!r} ({list(region.axes)})",
                       details={"tensor": tensor, "axes": list(region.axes)})


def _extent(region: TensorRegion, axis: int) -> int:
    entry = region.selection[axis]
    return entry.stop - entry.start


# -- project_out -----------------------------------------------------------------------------------------------------

InlineDirection = Annotated[list[float], Field(min_length=1, max_length=MAX_DIRECTION_LENGTH)]


class ProjectOutParameters(OperatorParameters):
    direction: InlineDirection | VectorReference = Field(
        description="the output-axis direction d, one value per row of the tensor: inline, at most "
        f"{MAX_DIRECTION_LENGTH} finite float32 values (rounded to float32 and recorded in their shortest digits), "
        "or a reference <result_uri>#<JSON pointer> to a vector of a stored analysis result of this run, for "
        "example wse://analyses/<sha256>#/left_vectors/0. Its norm must be at least "
        f"{MIN_DIRECTION_NORM}",
    )
    columns: list[int] = Field(
        min_length=1,
        max_length=MAX_PATCH_ELEMENTS,
        description="the input-axis indices C to edit: strictly increasing absolute column indices, all "
        "inside the selection's column range (so at most as many as it has columns); other columns of the "
        "selection are left unchanged",
    )
    alpha: float = Field(
        default=1.0,
        description="finite scale of the removed component; 1 removes it, 0 changes nothing",
    )

    @field_validator("direction")
    @classmethod
    def _finite_direction(cls, value: list[float] | str) -> list[float] | str:
        if isinstance(value, list):
            bad = _bad_numbers(value)
            if bad:
                raise ValueError(f"direction values must be finite numbers within the float32 range; {bad} are not")
        return value

    @field_validator("columns")
    @classmethod
    def _increasing_columns(cls, value: list[int]) -> list[int]:
        if any(later <= earlier for earlier, later in zip(value, value[1:])):
            raise ValueError("columns must be strictly increasing (sorted, without duplicates)")
        return value

    @field_validator("alpha")
    @classmethod
    def _finite_alpha(cls, value: float) -> float:
        if not _float32_finite(value):
            raise ValueError("alpha must be a finite number within the float32 range")
        return value


class ProjectOutRecord(BaseModel):
    """The recorded parameters of a ``project_out`` step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: list[float] | VectorRecord
    columns: list[int]
    alpha: float


class ProjectOutOperator(BaseEditOperator):
    """Remove an output-axis direction from selected columns of an ``[out, in]`` tensor."""

    parameters_model = ProjectOutParameters
    reference_parameters = ("direction",)
    inline_parameters = ("direction", "columns")
    spec = make_spec(
        operator_id="project_out",
        version="1",
        effect="edit",
        description="Remove the output-axis direction d from selected input-axis columns C of a "
        "two-dimensional [out_features, in_features] tensor: W'[:, C] = W[:, C] - alpha d (d^T W[:, C]) / "
        "(d^T d). The selection must cover every row; columns of the selection outside C are unchanged. "
        "d is inline or a reference to a vector of a stored analysis result of this run. Sums are "
        "exact-product, correctly rounded float64 sums, so the result is the same on every platform.",
        required_capabilities=_EDIT_CAPABILITIES,
        parameters_model=ProjectOutParameters,
        result_model=PatchStep,
        usage=_EDIT_USAGE,
    )

    def resolve_parameters(self, parameters: Mapping[str, Any], region: TensorRegion) -> ProjectOutRecord:
        parsed = parse_parameters(self.spec, ProjectOutParameters, parameters)
        assert isinstance(parsed, ProjectOutParameters)
        tensor, shape = region.tensor, list(region.tensor_shape)
        if len(shape) != 2:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"project_out requires a two-dimensional [out_features, in_features] tensor; {tensor!r} has "
                f"shape {shape}",
                details={"tensor": tensor, "shape": shape},
            )
        if tuple(region.axes) != PROJECT_OUT_AXES:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"project_out requires axes {list(PROJECT_OUT_AXES)} (a Linear weight, whose axis 0 is the "
                f"output axis); {tensor!r} has axes {list(region.axes)}",
                details={"tensor": tensor, "axes": list(region.axes)},
            )
        rows, cols = region.selection
        if (rows.start, rows.stop) != (0, shape[0]):
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"project_out needs a selection that covers every row, because d spans the output axis: use "
                f"{{start: 0, stop: {shape[0]}, step: 1}} on axis 0 (got start={rows.start}, stop={rows.stop})",
                details={"tensor": tensor, "rows": shape[0]},
            )
        if isinstance(parsed.direction, str):
            direction = _referenced(region, "direction")
        else:
            direction = np.asarray(parsed.direction, dtype=np.float32)
        if direction.size != shape[0]:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"direction must hold one value per row of {tensor!r} ({shape[0]}); got {direction.size}",
                details={"tensor": tensor, "expected_length": shape[0], "length": int(direction.size)},
            )
        wide = direction.astype(np.float64)
        norm = math.sqrt(math.fsum((wide * wide).tolist()))
        if not norm >= MIN_DIRECTION_NORM:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"the direction's norm (of its float32 values) is {norm:.3g}; project_out requires a norm of at "
                f"least {MIN_DIRECTION_NORM}",
                details={"tensor": tensor, "norm": norm, "min_norm": MIN_DIRECTION_NORM},
            )
        width = cols.stop - cols.start
        if len(parsed.columns) > width:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"columns holds {len(parsed.columns)} indices, but the selection of {tensor!r} has {width} columns "
                f"[{cols.start}, {cols.stop}); give at most {width}",
                details={"tensor": tensor, "column_start": cols.start, "column_stop": cols.stop,
                         "columns": len(parsed.columns)},
            )
        outside = [c for c in parsed.columns if not cols.start <= c < cols.stop]
        if outside:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"columns must lie inside the selection's column range [{cols.start}, {cols.stop}) of "
                f"{tensor!r}; {len(outside)} do not, for example {outside[0]}",
                details={"tensor": tensor, "column_start": cols.start, "column_stop": cols.stop,
                         "outside": outside[:16]},
            )
        recorded = (VectorRecord.of(parsed.direction, direction) if isinstance(parsed.direction, str)
                    else shortest_floats(direction))
        return ProjectOutRecord(direction=recorded, columns=parsed.columns, alpha=parsed.alpha)

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:
        assert isinstance(parameters, ProjectOutRecord)
        values = np.array(region.values, dtype=np.float32)
        local = np.asarray(parameters.columns, dtype=np.int64) - region.selection[1].start
        d = _vector(region, parameters.direction, "direction").astype(np.float64)
        block = values[:, local].astype(np.float64)
        # float32 x float32 products are exact in float64; fsum rounds each sum once.
        norm_squared = math.fsum((d * d).tolist())
        sums = np.array([math.fsum(column) for column in (d[:, None] * block).T.tolist()], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            terms = d[:, None] * (parameters.alpha * (sums / norm_squared))[None, :]
            updated = np.where(terms == 0.0, block, block - terms)
            values[:, local] = updated.astype(np.float32)
        return values


# -- add_low_rank ----------------------------------------------------------------------------------------------------

FactorColumn = Annotated[list[float], Field(min_length=1)] | VectorReference


class AddLowRankParameters(OperatorParameters):
    u: list[FactorColumn] = Field(
        min_length=1,
        max_length=MAX_LOW_RANK,
        description=f"the r columns of U (1 <= r <= {MAX_LOW_RANK}), each one value per selected row: inline finite "
        "float32 values (rounded to float32), or a reference <result_uri>#<JSON pointer> to a vector of a stored "
        "analysis result of this run, for example wse://analyses/<sha256>#/left_vectors/0",
    )
    v: list[FactorColumn] = Field(
        min_length=1,
        max_length=MAX_LOW_RANK,
        description="the r columns of V, as many as U has, each one value per selected column: inline or a "
        "reference, for example wse://analyses/<sha256>#/right_vectors/0. Inline columns of u and v hold at most "
        f"{MAX_INLINE_FACTOR_VALUES} values in all; give longer ones by reference",
    )
    alpha: float = Field(default=1.0, description="finite scale of the update alpha U V^T; 0 changes nothing")

    @field_validator("u", "v")
    @classmethod
    def _finite_columns(cls, value: list[list[float] | str]) -> list[list[float] | str]:
        bad = sum(_bad_numbers(column) for column in value if isinstance(column, list))
        if bad:
            raise ValueError(f"factor values must be finite numbers within the float32 range; {bad} are not")
        return value

    @field_validator("alpha")
    @classmethod
    def _finite_alpha(cls, value: float) -> float:
        if not _float32_finite(value):
            raise ValueError("alpha must be a finite number within the float32 range")
        return value

    @model_validator(mode="after")
    def _same_rank_and_bounded(self) -> AddLowRankParameters:
        if len(self.u) != len(self.v):
            raise ValueError(f"u and v must hold the same number of columns (the rank r); u has {len(self.u)} and "
                             f"v has {len(self.v)}")
        inline = sum(len(column) for column in (*self.u, *self.v) if isinstance(column, list))
        if inline > MAX_INLINE_FACTOR_VALUES:
            raise ValueError(f"inline factor columns hold at most {MAX_INLINE_FACTOR_VALUES} values in all; these "
                             f"hold {inline}. Give longer columns as references to vectors of stored analysis results")
        return self


class AddLowRankRecord(BaseModel):
    """The recorded parameters of an ``add_low_rank`` step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    u: list[list[float] | VectorRecord]
    v: list[list[float] | VectorRecord]
    alpha: float


def _low_rank_sums(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``S[i, j] = sum_k u[i, k] v[j, k]`` correctly rounded to float64, for float32 values held as float64.

    Every product is exact in float64 (24 + 24 significant bits). One product
    needs no rounding, and one IEEE addition of two is correctly rounded; for
    three or more, ``math.fsum`` rounds each sum once.
    """
    rank = u.shape[1]
    products = [u[:, k:k + 1] * v[:, k][None, :] for k in range(min(rank, 2))]
    if rank == 1:
        return products[0]
    if rank == 2:
        return products[0] + products[1]
    terms = [product.reshape(-1).tolist() for product in products]
    terms += [(u[:, k:k + 1] * v[:, k][None, :]).reshape(-1).tolist() for k in range(2, rank)]
    return np.array([math.fsum(entry) for entry in zip(*terms)], dtype=np.float64).reshape(products[0].shape)


class AddLowRankOperator(BaseEditOperator):
    """Add a low-rank update ``alpha U V^T`` to a two-dimensional selection."""

    parameters_model = AddLowRankParameters
    reference_parameters = ("u", "v")
    inline_parameters = ("u", "v")
    spec = make_spec(
        operator_id="add_low_rank",
        version="1",
        effect="edit",
        description="Add alpha U V^T to the selection of a two-dimensional tensor with known axes: W'[R, C] = "
        f"W[R, C] + alpha U V^T, with U given as r <= {MAX_LOW_RANK} columns of one value per selected row and V as "
        "r columns of one value per selected column (U: [out, r] and V: [in, r] for the whole tensor). Each column "
        "is inline or a reference to a vector of a stored analysis result of this run. Each entry's sum of exact "
        "float64 products is correctly rounded, so the result is the same on every platform and for every order of "
        "the factor columns; entries whose update is zero keep their bytes.",
        required_capabilities=_EDIT_CAPABILITIES,
        parameters_model=AddLowRankParameters,
        result_model=PatchStep,
        usage=_EDIT_USAGE,
    )

    def resolve_parameters(self, parameters: Mapping[str, Any], region: TensorRegion) -> AddLowRankRecord:
        parsed = parse_parameters(self.spec, AddLowRankParameters, parameters)
        assert isinstance(parsed, AddLowRankParameters)
        _require_two_known_axes(region, "add_low_rank")
        factors: dict[str, list[list[float] | VectorRecord]] = {}
        for name, columns, axis, unit in (("u", parsed.u, 0, "row"), ("v", parsed.v, 1, "column")):
            expected = _extent(region, axis)
            recorded: list[list[float] | VectorRecord] = []
            for index, column in enumerate(columns):
                key = f"{name}/{index}"
                vector = _referenced(region, key) if isinstance(column, str) else np.asarray(column, dtype=np.float32)
                if vector.size != expected:
                    raise WSEError(
                        ErrorCode.INVALID_ARGUMENT,
                        f"{key} must hold one value per selected {unit} of {region.tensor!r} ({expected}, along axis "
                        f"{region.axes[axis]}); got {vector.size}",
                        details={"tensor": region.tensor, "parameter": key, "expected_length": expected,
                                 "length": int(vector.size)},
                    )
                recorded.append(VectorRecord.of(column, vector) if isinstance(column, str) else shortest_floats(vector))
            factors[name] = recorded
        return AddLowRankRecord(u=factors["u"], v=factors["v"], alpha=parsed.alpha)

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:
        assert isinstance(parameters, AddLowRankRecord)
        u = np.stack([_vector(region, column, f"u/{k}") for k, column in enumerate(parameters.u)], axis=1)
        v = np.stack([_vector(region, column, f"v/{k}") for k, column in enumerate(parameters.v)], axis=1)
        u, v = u.astype(np.float64), v.astype(np.float64)
        values = np.array(region.values, dtype=np.float32)
        rows, cols = values.shape
        step = max(1, _LOW_RANK_BLOCK_ELEMENTS // max(cols, 1))
        for start in range(0, rows, step):
            stop = min(rows, start + step)
            block = values[start:stop].astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                terms = parameters.alpha * _low_rank_sums(u[start:stop], v)
                values[start:stop] = np.where(terms == 0.0, block, block + terms).astype(np.float32)
        return values


# -- fill_slice ------------------------------------------------------------------------------------------------------


class FillSliceParameters(OperatorParameters):
    value: float = Field(
        description="the finite scalar written to every element of the selection. It is rounded to float32 and then "
        "to the tensor's stored dtype (to nearest even) and recorded as stored; a value that rounds to infinity in "
        "the stored dtype is refused",
    )

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not _float32_finite(value):
            raise ValueError("value must be a finite number within the float32 range")
        return value


class FillSliceOperator(BaseEditOperator):
    """Set every element of a two-dimensional selection to one finite scalar."""

    parameters_model = FillSliceParameters
    spec = make_spec(
        operator_id="fill_slice",
        version="1",
        effect="edit",
        description="Set every element of a selection of a two-dimensional tensor with known axes to one finite "
        "value, for example 0 to ablate an attention head's slice. The value is recorded as the engine stores it "
        "(rounded to float32 and then to the stored dtype); elements that already hold it are not counted as "
        "changed.",
        required_capabilities=_EDIT_CAPABILITIES,
        parameters_model=FillSliceParameters,
        result_model=PatchStep,
        usage=_EDIT_USAGE,
    )

    def resolve_parameters(self, parameters: Mapping[str, Any], region: TensorRegion) -> FillSliceParameters:
        parsed = parse_parameters(self.spec, FillSliceParameters, parameters)
        assert isinstance(parsed, FillSliceParameters)
        _require_two_known_axes(region, "fill_slice")
        dtype = region.dtype
        bits = values_to_bits(np.array([parsed.value], dtype=np.float32), dtype)
        stored = bits_to_values(bits, dtype)
        if not np.isfinite(stored).all():  # only a 16-bit dtype: the value rounds to a finite float32
            largest = _LARGEST_FINITE[dtype]
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"the value {parsed.value!r} rounds to infinity in {dtype}, the stored dtype of {region.tensor!r}, "
                f"whose largest finite value is {largest:.8g}",
                details={"tensor": region.tensor, "dtype": dtype, "largest_finite": largest},
            )
        recorded = shortest_floats(stored)[0]
        if not np.array_equal(values_to_bits(np.array([recorded], dtype=np.float32), dtype), bits):
            recorded = float(stored[0])  # the exact value, should its shortest digits not round back to it
        return FillSliceParameters(value=recorded)

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:
        assert isinstance(parameters, FillSliceParameters)
        return np.full(region.values.shape, parameters.value, dtype=np.float32)


BUILTIN_DIRECTION_EDITS: dict[str, BaseEditOperator] = {
    "project_out": ProjectOutOperator(),
    "add_low_rank": AddLowRankOperator(),
    "fill_slice": FillSliceOperator(),
}
