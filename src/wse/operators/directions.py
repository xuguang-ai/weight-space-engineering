"""Built-in direction edit operators.

``project_out@1`` removes an output-axis direction ``d`` from selected
input-axis indices (columns) ``C`` of a two-dimensional
``[out_features, in_features]`` tensor (the layout of a ``Linear.weight``)::

    W'[:, C] = W[:, C] - alpha * d (d^T W[:, C]) / (d^T d)

With ``alpha = 1`` every edited column becomes orthogonal to ``d`` (up to the
rounding of the stored dtype). Columns of the selection outside ``C`` keep
their stored bytes and are not counted as changed.

The result is the same on every platform. ``d`` and the region are taken as
float32 values; every product ``d_i * W[i, c]`` is formed in float64, where it
is exact; each sum (``s_c = d^T W[:, c]`` and ``n = d^T d``) is the correctly
rounded ``math.fsum`` of those exact products; and each entry is updated with
elementwise IEEE float64 operations, ``t = d_i * (alpha * (s_c / n))`` and
``W'[i, c] = W[i, c] - t``, except that an entry whose term ``t`` is zero keeps
its value (so ``-0.0`` stays ``-0.0``). The float64 result is rounded to float32,
and the patch engine rounds that to the stored dtype (BF16, F16, or F32) with
round-to-nearest-even, as for every edit. No matrix product (BLAS) reaches the
output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from wse.adapters.base import CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT
from wse.operators.base import BaseEditOperator, OperatorParameters, TensorRegion, make_spec, parse_parameters
from wse.operators.tensor import shortest_floats
from wse.schemas import ErrorCode, PatchStep, UsageContract, WSEError

__all__ = [
    "BUILTIN_DIRECTION_EDITS",
    "MAX_DIRECTION_LENGTH",
    "MIN_DIRECTION_NORM",
    "PROJECT_OUT_AXES",
    "ProjectOutOperator",
    "ProjectOutParameters",
]

# An inline direction holds at most this many values (one per output row).
MAX_DIRECTION_LENGTH = 2048
# A direction whose Euclidean norm (of its float32 values) is below this is refused: the
# projection divides by d^T d, and a direction this small is almost certainly a mistake.
MIN_DIRECTION_NORM = 1e-6
# The axis layout the projection requires: axis 0 is the output axis that d spans.
PROJECT_OUT_AXES = ("out_features", "in_features")

_EDIT_USAGE = UsageContract(counter="candidate_commits", units_per_call=1, model_forward_calls=0)
_EDIT_CAPABILITIES = [CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT]
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _float32_finite(value: float) -> bool:
    return math.isfinite(value) and abs(value) <= _FLOAT32_MAX


class ProjectOutParameters(OperatorParameters):
    direction: list[float] = Field(
        min_length=1,
        max_length=MAX_DIRECTION_LENGTH,
        description="the output-axis direction d: one finite float32 value per row of the tensor, at most "
        f"{MAX_DIRECTION_LENGTH}; its norm must be at least {MIN_DIRECTION_NORM}. Values are rounded to float32 "
        "and recorded in their shortest float32 digits",
    )
    columns: list[int] = Field(
        min_length=1,
        description="the input-axis indices C to edit: strictly increasing absolute column indices, all "
        "inside the selection's column range; other columns of the selection are left unchanged",
    )
    alpha: float = Field(
        default=1.0,
        description="finite scale of the removed component; 1 removes it, 0 changes nothing",
    )

    @field_validator("direction")
    @classmethod
    def _finite_direction(cls, value: list[float]) -> list[float]:
        bad = sum(1 for entry in value if not _float32_finite(entry))
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


class ProjectOutOperator(BaseEditOperator):
    """Remove an output-axis direction from selected columns of an ``[out, in]`` tensor."""

    parameters_model = ProjectOutParameters
    spec = make_spec(
        operator_id="project_out",
        version="1",
        effect="edit",
        description="Remove the output-axis direction d from selected input-axis columns C of a "
        "two-dimensional [out_features, in_features] tensor: W'[:, C] = W[:, C] - alpha d (d^T W[:, C]) / "
        "(d^T d). The selection must cover every row; columns of the selection outside C are unchanged. "
        "Sums are exact-product, correctly rounded float64 sums, so the result is the same on every platform.",
        required_capabilities=_EDIT_CAPABILITIES,
        parameters_model=ProjectOutParameters,
        result_model=PatchStep,
        usage=_EDIT_USAGE,
    )

    def resolve_parameters(self, parameters: Mapping[str, Any], region: TensorRegion) -> ProjectOutParameters:
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
        if len(parsed.direction) != shape[0]:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"direction must hold one value per row of {tensor!r} ({shape[0]}); got {len(parsed.direction)}",
                details={"tensor": tensor, "expected_length": shape[0], "length": len(parsed.direction)},
            )
        direction = np.asarray(parsed.direction, dtype=np.float32)
        wide = direction.astype(np.float64)
        norm = math.sqrt(math.fsum((wide * wide).tolist()))
        if not norm >= MIN_DIRECTION_NORM:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"the direction's norm (of its float32 values) is {norm:.3g}; project_out requires a norm of at "
                f"least {MIN_DIRECTION_NORM}",
                details={"tensor": tensor, "norm": norm, "min_norm": MIN_DIRECTION_NORM},
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
        return ProjectOutParameters(direction=shortest_floats(direction), columns=parsed.columns,
                                    alpha=parsed.alpha)

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:
        assert isinstance(parameters, ProjectOutParameters)
        values = np.array(region.values, dtype=np.float32)
        local = np.asarray(parameters.columns, dtype=np.int64) - region.selection[1].start
        d = np.asarray(parameters.direction, dtype=np.float32).astype(np.float64)
        block = values[:, local].astype(np.float64)
        # float32 x float32 products are exact in float64; fsum rounds each sum once.
        norm_squared = math.fsum((d * d).tolist())
        sums = np.array([math.fsum(column) for column in (d[:, None] * block).T.tolist()], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            terms = d[:, None] * (parameters.alpha * (sums / norm_squared))[None, :]
            updated = np.where(terms == 0.0, block, block - terms)
            values[:, local] = updated.astype(np.float32)
        return values


BUILTIN_DIRECTION_EDITS: dict[str, BaseEditOperator] = {
    "project_out": ProjectOutOperator(),
}
