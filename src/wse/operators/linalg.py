"""Built-in analyses for large tensors (M2 plan Section 5.2).

* ``tensor_norms@1``: the L2 norm of every row or every column of a
  two-dimensional selection.
* ``svd_top_k@1``: the top ``k <= 16`` singular values and vectors of a
  two-dimensional selection, computed in FP32.
* ``direction_projection@1``: the scalar projection ``(d . w) / |d|`` and the
  cosine ``(d . w) / (|d| |w|)`` of a direction ``d`` against every row or
  every column ``w`` of a two-dimensional selection.

Each result carries a summary and its top entries, and declares its vectors as
``artifact_fields``: in managed use the engine stores the complete result as a
run-scoped ``wse://analyses/<sha256>`` artifact and returns the rest inline
with the URI (``wse.analysis``), so a response stays within the 31 KiB result
limit whatever the tensor's size. A vector of a stored result is named
``<result_uri>#<JSON pointer>``, for example ``#/right_vectors/0`` of an SVD
or ``#/direction`` of a projection; ``direction_projection`` accepts such a
reference as its direction (``reference_parameters``). Entry indices are
indices of the full tensor; a stored vector starts at the selection's
``offset`` on its axis.

Numerics. A region arrives as float32, float16, or (BF16) its exact float32
widening. Norms, projections, cosines, and the Frobenius norm accumulate
float64 products over fixed blocks of rows, so memory stays bounded for the
largest tensors (a [248320, 1024] embedding), and the same input gives the
same result on one machine and library version. Nonfinite selections are
refused, as by ``tensor_norm@1``. A row or column of zero norm has cosine 0.

SVD. A selection of at most ``SVD_EXACT_MAX_ELEMENTS`` (4,194,304 = 2^22)
elements uses LAPACK's divide-and-conquer SVD (``numpy.linalg.svd``) in
float32; at that size it took at most about 1 s on the operator's Apple M4 Pro
([2048, 2048]: 1.05 s; [1024, 3584]: 0.45 s). A larger selection uses the
randomized SVD of Halko, Martinsson, and Tropp (2011, Algorithm 4.4): a
Gaussian test matrix of ``randomized_svd_samples(k, rows, columns) = min(2k +
16, rows, columns)`` columns, ``RANDOMIZED_SVD_POWER_ITERATIONS`` (10) power
iterations re-orthonormalized by QR, and an exact SVD of the projected matrix,
all in float32. On Qwen3.5-0.8B's [248320, 1024] embedding these settings gave
the top 16 singular values within a relative 3e-4 of an FP64 reference.
``method`` may also request the randomized SVD at any size; ``exact`` is
refused above the threshold, because its cost grows with the cube of the
smaller dimension.

Seeds. The randomized SVD draws its test matrix from
``numpy.random.default_rng(seed)`` (float32 normals). ``seed`` defaults to 0
and is recorded in the resolved parameters, and so in the input hash: the same
seed, selection, and machine give the same result and the same artifact URI,
and another seed gives another, equally accurate sketch. An exact SVD uses no
randomness and records ``seed`` as null. Each singular pair is determined up
to sign; it is flipped so that the largest-magnitude entry of its left vector
(the first, on ties) is positive.

Checks that need the selection's shape (two axes, ``k <= min(rows, columns)``,
the direction's length) run when the region arrives, before any computation;
a refused call is not charged. None of these analyses runs the target network.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_serializer, field_validator

from wse.adapters.base import CAP_WEIGHTS_READ
from wse.operators.base import (
    BaseAnalysisOperator,
    OperatorParameters,
    OperatorResult,
    TensorRegion,
    VectorReference,
    make_spec,
    parse_parameters,
)
from wse.operators.directions import MAX_DIRECTION_LENGTH, MIN_DIRECTION_NORM
from wse.operators.tensor import shortest_floats
from wse.policy import DEFAULT_TOP_K, MAX_TOP_K
from wse.schemas import ErrorCode, UsageContract, WSEError, json_hash

__all__ = [
    "BUILTIN_LINALG_ANALYSES",
    "DEFAULT_SVD_K",
    "DEFAULT_SVD_TOP",
    "DEFAULT_VECTOR_TOP",
    "MAX_SVD_K",
    "MAX_SVD_TOP",
    "MAX_VECTOR_TOP",
    "RANDOMIZED_SVD_POWER_ITERATIONS",
    "SVD_EXACT_MAX_ELEMENTS",
    "DirectionProjectionOperator",
    "DirectionProjectionParameters",
    "DirectionProjectionResult",
    "IndexedValue",
    "ProjectionEntry",
    "SvdTopKOperator",
    "SvdTopKParameters",
    "SvdTopKResult",
    "TensorNormsOperator",
    "TensorNormsParameters",
    "TensorNormsResult",
    "VectorSummary",
    "randomized_svd_samples",
]

MAX_SVD_K = 16
DEFAULT_SVD_K = 8
# Inline entries per singular vector: 16 components x 2 vectors x 16 entries stay within the result limit.
MAX_SVD_TOP = 16
DEFAULT_SVD_TOP = 8
# Inline entries of a norms or projection vector (M1's top-k bounds).
MAX_VECTOR_TOP = MAX_TOP_K
DEFAULT_VECTOR_TOP = DEFAULT_TOP_K
SVD_EXACT_MAX_ELEMENTS = 4 * 1024 * 1024
RANDOMIZED_SVD_POWER_ITERATIONS = 10
MAX_SEED = 2**63 - 1
# Rows are converted to float64 in blocks of about this many elements (8 MiB).
_BLOCK_ELEMENTS = 1 << 20

_ANALYSIS_USAGE = UsageContract(counter="analysis_calls", units_per_call=1, model_forward_calls=0)
_FLOAT32_MAX = float(np.finfo(np.float32).max)

Axis = Literal["rows", "columns"]
_AXIS_DESCRIPTION = ("rows: one value per row of the selection (axis 0), over the selected columns; columns: one value "
                     "per column (axis 1), over the selected rows")


def randomized_svd_samples(k: int, rows: int, columns: int) -> int:
    """Columns of the randomized SVD's Gaussian test matrix: ``min(2k + 16, rows, columns)``."""
    return min(2 * k + 16, rows, columns)


# -- shared result parts -----------------------------------------------------------------------------------------


class IndexedValue(BaseModel):
    index: int = Field(description="index along the vector's axis in the full tensor")
    value: float


class VectorSummary(BaseModel):
    count: int = Field(description="number of values in the vector")
    min: float
    max: float
    mean: float
    std: float = Field(description="population standard deviation (ddof=0)")


def _summary(vector: np.ndarray) -> VectorSummary:
    return VectorSummary(count=int(vector.size), min=float(vector.min()), max=float(vector.max()),
                         mean=float(np.mean(vector)), std=float(np.std(vector)))


def _top_order(magnitude: np.ndarray, count: int) -> np.ndarray:
    """Positions of the ``count`` largest magnitudes: descending, then ascending position for ties."""
    size = magnitude.size
    count = min(count, size)
    if count < size:
        # Every element tied with the count-th largest stays a candidate, so the tie-break is exact.
        threshold = np.partition(magnitude, size - count)[size - count]
        candidates = np.flatnonzero(magnitude >= threshold)
    else:
        candidates = np.arange(size)
    return candidates[np.lexsort((candidates, -magnitude[candidates].astype(np.float64)))][:count]


def _matrix(region: TensorRegion, operation: str) -> np.ndarray:
    values = region.values
    if values.ndim != 2:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{operation} requires a two-dimensional selection (rows and columns); tensor {region.tensor!r} has "
            f"{values.ndim} axis/axes {list(region.axes)}",
            details={"tensor": region.tensor, "axes": list(region.axes)},
        )
    return values


def _row_blocks(rows: int, columns: int) -> Iterator[tuple[int, int]]:
    step = max(1, _BLOCK_ELEMENTS // max(columns, 1))
    for start in range(0, rows, step):
        yield start, min(rows, start + step)


def _squares(values: np.ndarray, axis: Axis) -> np.ndarray:
    """Sums of squares (float64) of every row or every column."""
    rows, columns = values.shape
    out = np.empty(rows, dtype=np.float64) if axis == "rows" else np.zeros(columns, dtype=np.float64)
    for start, stop in _row_blocks(rows, columns):
        block = values[start:stop].astype(np.float64)
        if axis == "rows":
            out[start:stop] = np.einsum("ij,ij->i", block, block)
        else:
            out += np.einsum("ij,ij->j", block, block)
    return out


def _refuse_nonfinite(region: TensorRegion, operation: str, sums: np.ndarray | float) -> None:
    """Refuse a selection whose float64 sums of squares are not finite (a nonfinite element, or float64 overflow)."""
    if bool(np.all(np.isfinite(sums))):
        return
    bad = int(np.count_nonzero(~np.isfinite(region.values)))
    reason = (f"the selection contains {bad} nonfinite value(s); use the stats analysis to count them" if bad
              else "its squared values overflow float64")
    raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{operation} is undefined for tensor {region.tensor!r}: {reason}",
                   details={"tensor": region.tensor, "nonfinite_count": bad})


def _offset(region: TensorRegion, axis: Axis) -> int:
    return region.selection[0 if axis == "rows" else 1].start


# -- tensor_norms ------------------------------------------------------------------------------------------------


class TensorNormsParameters(OperatorParameters):
    axis: Axis = Field(description=_AXIS_DESCRIPTION)
    top: int = Field(default=DEFAULT_VECTOR_TOP, ge=1, le=MAX_VECTOR_TOP,
                     description=f"inline entries: at most this many of the largest norms (1-{MAX_VECTOR_TOP})")


class TensorNormsResult(OperatorResult):
    axis: Axis
    axis_name: str = Field(description="the manifest's name of the reported axis, for example num_embeddings")
    offset: int = Field(description="tensor index of the first norm: the selection's start on the axis")
    summary: VectorSummary
    top: list[IndexedValue] = Field(description="the largest norms, descending, then ascending index for ties")
    norms: list[float] | None = Field(
        default=None, description="every norm, in index order from offset; stored behind result_uri in managed use")


class TensorNormsOperator(BaseAnalysisOperator):
    """L2 norms (float64 accumulation) of every row or column of a two-dimensional selection."""

    parameters_model = TensorNormsParameters
    result_model = TensorNormsResult
    artifact_fields = ("norms",)
    spec = make_spec(
        operator_id="tensor_norms",
        version="1",
        effect="analysis",
        description="L2 norms of every row or every column of a two-dimensional tensor selection, accumulated in "
        "float64: a summary and the largest norms inline, and every norm in the stored result (#/norms).",
        required_capabilities=[CAP_WEIGHTS_READ],
        parameters_model=TensorNormsParameters,
        result_model=TensorNormsResult,
        usage=_ANALYSIS_USAGE,
    )

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> TensorNormsResult:
        assert isinstance(parameters, TensorNormsParameters)
        values = _matrix(region, "tensor_norms")
        squares = _squares(values, parameters.axis)
        _refuse_nonfinite(region, "tensor_norms", squares)
        norms = np.sqrt(squares)
        offset = _offset(region, parameters.axis)
        return TensorNormsResult(
            axis=parameters.axis,
            axis_name=region.axes[0 if parameters.axis == "rows" else 1],
            offset=offset,
            summary=_summary(norms),
            top=[IndexedValue(index=offset + int(i), value=float(norms[i]))
                 for i in _top_order(norms, parameters.top)],
            norms=norms.tolist(),
        )


# -- svd_top_k ---------------------------------------------------------------------------------------------------


class SvdTopKParameters(OperatorParameters):
    k: int | None = Field(
        default=None, ge=1, le=MAX_SVD_K,
        description=f"number of singular triplets, 1 <= k <= min({MAX_SVD_K}, rows, columns); omitted means "
        f"min({DEFAULT_SVD_K}, rows, columns)")
    top: int = Field(default=DEFAULT_SVD_TOP, ge=1, le=MAX_SVD_TOP,
                     description=f"inline entries per singular vector: at most this many of its largest-magnitude "
                     f"entries (1-{MAX_SVD_TOP})")
    method: Literal["exact", "randomized"] | None = Field(
        default=None,
        description=f"omitted: exact for selections of at most {SVD_EXACT_MAX_ELEMENTS:,} elements, randomized "
        "above; exact is refused above that size")
    seed: int | None = Field(
        default=None, ge=0, le=MAX_SEED,
        description="seed of the randomized SVD's Gaussian test matrix; omitted means 0; recorded as null for the "
        "exact SVD, which uses no randomness")


class SvdTopKResult(OperatorResult):
    method: Literal["exact", "randomized"]
    k: int
    shape: list[int] = Field(description="the selection's shape [rows, columns]")
    singular_values: list[float] = Field(description="the k largest singular values, descending (float32)")
    frobenius_norm: float = Field(description="Frobenius norm of the selection (float64 accumulation)")
    energy_fractions: list[float] = Field(description="singular_value**2 / frobenius_norm**2 for each component")
    left_top: list[list[IndexedValue]] = Field(
        description="per component, the largest-magnitude entries of its left vector; index is a row of the tensor")
    right_top: list[list[IndexedValue]] = Field(
        description="per component, the largest-magnitude entries of its right vector; index is a column")
    left_vectors: list[list[float]] | None = Field(
        default=None, description="k left singular vectors (one value per selected row; reference #/left_vectors/i); "
        "stored behind result_uri in managed use")
    right_vectors: list[list[float]] | None = Field(
        default=None, description="k right singular vectors (one value per selected column; reference "
        "#/right_vectors/i); stored behind result_uri in managed use")


def _exact_svd(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return u[:, :k], s[:k], vt[:k]


def _randomized_svd(a: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko, Martinsson, and Tropp (2011), Algorithm 4.4, in float32."""
    rows, columns = a.shape
    sketch = np.random.default_rng(seed).standard_normal((columns, randomized_svd_samples(k, rows, columns)),
                                                         dtype=np.float32)
    basis, _ = np.linalg.qr(a @ sketch)
    for _ in range(RANDOMIZED_SVD_POWER_ITERATIONS):
        right, _ = np.linalg.qr(a.T @ basis)
        basis, _ = np.linalg.qr(a @ right)
    small_u, s, vt = np.linalg.svd(basis.T @ a, full_matrices=False)
    return basis @ small_u[:, :k], s[:k], vt[:k]


class SvdTopKOperator(BaseAnalysisOperator):
    """Top-k singular values and vectors in FP32: exact up to the size threshold, randomized above it."""

    parameters_model = SvdTopKParameters
    result_model = SvdTopKResult
    artifact_fields = ("left_vectors", "right_vectors")
    spec = make_spec(
        operator_id="svd_top_k",
        version="1",
        effect="analysis",
        description="The top k <= 16 singular values and vectors of a two-dimensional tensor selection, in float32: "
        f"an exact SVD up to {SVD_EXACT_MAX_ELEMENTS:,} elements and a seeded randomized SVD above. Inline: the "
        "singular values, energy fractions, and the largest entries of each vector; stored: every vector "
        "(#/left_vectors/i, #/right_vectors/i). Signs make each left vector's largest entry positive.",
        required_capabilities=[CAP_WEIGHTS_READ],
        parameters_model=SvdTopKParameters,
        result_model=SvdTopKResult,
        usage=_ANALYSIS_USAGE,
    )

    def resolve_parameters(self, parameters: Mapping[str, Any], element_count: int) -> SvdTopKParameters:
        parsed = parse_parameters(self.spec, SvdTopKParameters, parameters)
        assert isinstance(parsed, SvdTopKParameters)
        method = parsed.method or ("exact" if element_count <= SVD_EXACT_MAX_ELEMENTS else "randomized")
        if method == "exact" and element_count > SVD_EXACT_MAX_ELEMENTS:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"the exact SVD covers at most {SVD_EXACT_MAX_ELEMENTS:,} elements, but this selection has "
                f"{element_count:,}; omit method (or pass randomized) for a randomized SVD, or select less",
                details={"max_exact_elements": SVD_EXACT_MAX_ELEMENTS, "element_count": element_count},
            )
        seed = None if method == "exact" else (0 if parsed.seed is None else parsed.seed)
        return SvdTopKParameters(k=parsed.k, top=parsed.top, method=method, seed=seed)

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> SvdTopKResult:
        assert isinstance(parameters, SvdTopKParameters) and parameters.method is not None
        values = _matrix(region, "svd_top_k")
        rows, columns = values.shape
        limit = min(MAX_SVD_K, rows, columns)
        k = parameters.k if parameters.k is not None else min(DEFAULT_SVD_K, rows, columns)
        if k > limit:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"k must be at most min({MAX_SVD_K}, rows, columns) = {limit} for this selection of shape "
                f"[{rows}, {columns}]; got k={k}",
                details={"k": k, "max_k": limit, "shape": [rows, columns]},
            )
        squared_norm = float(np.sum(_squares(values, "rows")))
        _refuse_nonfinite(region, "svd_top_k", squared_norm)
        matrix = np.ascontiguousarray(values, dtype=np.float32)
        if parameters.method == "exact":
            u, s, vt = _exact_svd(matrix, k)
        else:
            assert parameters.seed is not None
            u, s, vt = _randomized_svd(matrix, k, parameters.seed)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(s)) and np.all(np.isfinite(vt))):
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"the float32 SVD of tensor {region.tensor!r} overflowed; its values are too large",
                           details={"tensor": region.tensor})
        # Flip each pair so that the largest-magnitude entry of the left vector is positive.
        peaks = np.argmax(np.abs(u), axis=0)
        signs = np.where(u[peaks, np.arange(k)] < 0, -1.0, 1.0).astype(np.float32)
        u, vt = u * signs, vt * signs[:, None]
        left = [shortest_floats(np.ascontiguousarray(u[:, i])) for i in range(k)]
        right = [shortest_floats(np.ascontiguousarray(vt[i])) for i in range(k)]
        row_offset, column_offset = region.offsets

        def entries(vector: np.ndarray, digits: list[float], offset: int) -> list[IndexedValue]:
            return [IndexedValue(index=offset + int(i), value=digits[i])
                    for i in _top_order(np.abs(vector), parameters.top)]

        singular = s.astype(np.float32)
        return SvdTopKResult(
            method=parameters.method,
            k=k,
            shape=[rows, columns],
            singular_values=shortest_floats(singular),
            frobenius_norm=math.sqrt(squared_norm),
            energy_fractions=[float(value) ** 2 / squared_norm if squared_norm > 0 else 0.0
                              for value in singular.astype(np.float64)],
            left_top=[entries(u[:, i], left[i], row_offset) for i in range(k)],
            right_top=[entries(vt[i], right[i], column_offset) for i in range(k)],
            left_vectors=left,
            right_vectors=right,
        )


# -- direction_projection ----------------------------------------------------------------------------------------

InlineDirection = Annotated[list[float], Field(min_length=1, max_length=MAX_DIRECTION_LENGTH)]


class DirectionProjectionParameters(OperatorParameters):
    direction: InlineDirection | VectorReference = Field(
        description="the direction d: inline, at most 2048 finite float32 values (rounded to float32), or a "
        "reference <result_uri>#<JSON pointer> to a vector of a stored analysis result of this run, for example "
        "wse://analyses/<sha256>#/right_vectors/0. It holds one value per selected row (axis columns) or column "
        f"(axis rows), and its norm must be at least {MIN_DIRECTION_NORM}. Results record an inline direction as "
        "its length and the SHA-256 of its float32 values; the stored result keeps the values (#/direction)")
    axis: Axis = Field(description="columns: compare d with every column (d has one value per selected row); rows: "
                       "compare d with every row (d has one value per selected column)")
    measure: Literal["projection", "cosine"] = Field(
        default="projection",
        description="the measure that ranks the inline entries by magnitude; both are always reported")
    top: int = Field(default=DEFAULT_VECTOR_TOP, ge=1, le=MAX_VECTOR_TOP,
                     description=f"inline entries: at most this many, by descending |measure| (1-{MAX_VECTOR_TOP})")

    @field_validator("direction")
    @classmethod
    def _finite_direction(cls, value: list[float] | str) -> list[float] | str:
        if isinstance(value, list):
            bad = sum(1 for entry in value if not (math.isfinite(entry) and abs(entry) <= _FLOAT32_MAX))
            if bad:
                raise ValueError(f"direction values must be finite numbers within the float32 range; {bad} are not")
        return value

    @field_serializer("direction", when_used="json")
    def _record_direction(self, value: list[float] | str) -> dict[str, Any] | str:
        if isinstance(value, str):
            return value
        return {"length": len(value), "sha256": json_hash(value)}


class ProjectionEntry(BaseModel):
    index: int = Field(description="index of the row or column in the full tensor")
    projection: float = Field(description="(d . w) / |d|")
    cosine: float = Field(description="(d . w) / (|d| |w|); 0 when |w| = 0")


class DirectionProjectionResult(OperatorResult):
    axis: Axis
    axis_name: str = Field(description="the manifest's name of the axis whose rows or columns were compared")
    measure: Literal["projection", "cosine"]
    offset: int = Field(description="tensor index of the first compared row or column")
    direction_length: int
    direction_norm: float = Field(description="Euclidean norm of the float32 direction (correctly rounded)")
    zero_norm_count: int = Field(description="rows or columns with zero norm, whose cosine is reported as 0")
    summary: VectorSummary = Field(description="summary of the chosen measure over every row or column")
    top: list[ProjectionEntry] = Field(description="descending |measure|, then ascending index for ties")
    direction: list[float] | None = Field(
        default=None, description="the float32 direction used; stored behind result_uri in managed use")
    projections: list[float] | None = Field(
        default=None, description="every projection, in index order from offset; stored behind result_uri")
    cosines: list[float] | None = Field(
        default=None, description="every cosine, in index order from offset; stored behind result_uri")


class DirectionProjectionOperator(BaseAnalysisOperator):
    """Scalar projections and cosines of a direction against every row or column, in float64."""

    parameters_model = DirectionProjectionParameters
    result_model = DirectionProjectionResult
    artifact_fields = ("direction", "projections", "cosines")
    reference_parameters = ("direction",)
    spec = make_spec(
        operator_id="direction_projection",
        version="1",
        effect="analysis",
        description="Projections (d . w) / |d| and cosines of a direction d against every row or every column w of "
        "a two-dimensional tensor selection, accumulated in float64. d is inline or a reference to a vector of a "
        "stored analysis result of this run. Inline: a summary and the entries of largest |measure|; stored: every "
        "projection and cosine (#/projections, #/cosines) and d (#/direction).",
        required_capabilities=[CAP_WEIGHTS_READ],
        parameters_model=DirectionProjectionParameters,
        result_model=DirectionProjectionResult,
        usage=_ANALYSIS_USAGE,
    )

    def resolve_parameters(self, parameters: Mapping[str, Any],
                           element_count: int) -> DirectionProjectionParameters:
        parsed = parse_parameters(self.spec, DirectionProjectionParameters, parameters)
        assert isinstance(parsed, DirectionProjectionParameters)
        if isinstance(parsed.direction, str):
            return parsed
        direction = np.asarray(parsed.direction, dtype=np.float32)
        _require_norm(direction)
        return parsed.model_copy(update={"direction": shortest_floats(direction)})

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> DirectionProjectionResult:
        assert isinstance(parameters, DirectionProjectionParameters)
        values = _matrix(region, "direction_projection")
        rows, columns = values.shape
        if isinstance(parameters.direction, str):
            direction = region.inputs.get("direction")
            if direction is None:
                raise WSEError(ErrorCode.INVALID_ARGUMENT,
                               "the direction reference was not resolved; references resolve only through the engine")
        else:
            direction = np.asarray(parameters.direction, dtype=np.float32)
        norm = _require_norm(direction)
        expected, unit = (rows, "row") if parameters.axis == "columns" else (columns, "column")
        if direction.size != expected:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"against the {parameters.axis}, the direction must hold one value per selected {unit} ({expected}); "
                f"got {direction.size}",
                details={"axis": parameters.axis, "expected_length": expected, "length": int(direction.size)},
            )
        d = direction.astype(np.float64)
        count = columns if parameters.axis == "columns" else rows
        dots = np.zeros(count, dtype=np.float64)
        squares = np.zeros(count, dtype=np.float64)
        for start, stop in _row_blocks(rows, columns):
            block = values[start:stop].astype(np.float64)
            if parameters.axis == "columns":
                dots += np.einsum("i,ij->j", d[start:stop], block)
                squares += np.einsum("ij,ij->j", block, block)
            else:
                dots[start:stop] = np.einsum("ij,j->i", block, d)
                squares[start:stop] = np.einsum("ij,ij->i", block, block)
        _refuse_nonfinite(region, "direction_projection", squares)
        lengths = np.sqrt(squares)
        projections = dots / norm
        zero = lengths == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            cosines = np.where(zero, 0.0, np.clip(dots / (norm * lengths), -1.0, 1.0))
        ranked = projections if parameters.measure == "projection" else cosines
        offset = _offset(region, parameters.axis)
        return DirectionProjectionResult(
            axis=parameters.axis,
            axis_name=region.axes[0 if parameters.axis == "rows" else 1],
            measure=parameters.measure,
            offset=offset,
            direction_length=int(direction.size),
            direction_norm=norm,
            zero_norm_count=int(np.count_nonzero(zero)),
            summary=_summary(ranked),
            top=[ProjectionEntry(index=offset + int(i), projection=float(projections[i]), cosine=float(cosines[i]))
                 for i in _top_order(np.abs(ranked), parameters.top)],
            direction=shortest_floats(direction),
            projections=projections.tolist(),
            cosines=cosines.tolist(),
        )


def _require_norm(direction: np.ndarray) -> float:
    """The correctly rounded norm of a float32 direction; refuse one below ``MIN_DIRECTION_NORM``."""
    wide = direction.astype(np.float64)
    norm = math.sqrt(math.fsum((wide * wide).tolist()))
    if not norm >= MIN_DIRECTION_NORM:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"the direction's norm (of its float32 values) is {norm:.3g}; direction_projection requires a norm of "
            f"at least {MIN_DIRECTION_NORM}",
            details={"norm": norm, "min_norm": MIN_DIRECTION_NORM},
        )
    return norm


BUILTIN_LINALG_ANALYSES: dict[str, BaseAnalysisOperator] = {
    "tensor_norms": TensorNormsOperator(),
    "svd_top_k": SvdTopKOperator(),
    "direction_projection": DirectionProjectionOperator(),
}
