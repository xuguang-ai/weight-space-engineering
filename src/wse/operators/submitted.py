"""``replace_tensor@1``: write tensor values that a run computed in its sandbox (candidate submission).

The values come from a stored tensor file of the run: a ``.safetensors``
file that the run's sandboxed code saved to ``outputs/``, which WSE stores
when the call ends and names ``wse://tensors/<sha256>``. ``source`` names one
tensor of it, ``wse://tensors/<sha256>#<tensor name in the file>``. A
managed preview resolves the reference before any weights are read
(``wse.patches.resolve_tensor_references``): the file must belong to the
run's own experiment and still have its SHA-256, and the tensor must have the
target's stored dtype and the selection's shape, with every value finite.
The operator writes the values unchanged, so the edit is exact; protected
roles, alias groups, and the other checks of every edit apply as usual.
The recorded parameters keep the reference with the tensor's shape, dtype,
and the SHA-256 of its values; replay writes the recorded after-values and
never reads the file. Direct use has no stored tensor files, so it refuses
the operator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wse.adapters.base import CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT
from wse.operators.base import BaseEditOperator, OperatorParameters, TensorRegion, make_spec, parse_parameters
from wse.schemas import ErrorCode, PatchStep, UsageContract, WSEError

__all__ = ["BUILTIN_SUBMITTED_EDITS", "TENSOR_REFERENCE_PATTERN", "ReplaceTensorOperator", "ReplaceTensorRecord",
           "TensorRecord", "values_sha256"]

# wse://tensors/<sha256 of the file>#<tensor name in it>
TENSOR_REFERENCE_PATTERN = r"^wse://tensors/[0-9a-f]{64}#[^#\s]{1,512}$"


def values_sha256(values: np.ndarray) -> str:
    """The SHA-256 of a tensor's values as contiguous float32 bytes (exact for float32, float16, and bfloat16)."""
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


class ReplaceTensorParameters(OperatorParameters):
    source: str = Field(
        pattern=TENSOR_REFERENCE_PATTERN,
        description="wse://tensors/<sha256>#<tensor name>: a tensor of a .safetensors file that this run's sandbox "
        "saved to outputs/, in the target tensor's stored dtype and the selection's shape",
    )


class TensorRecord(BaseModel):
    """How a step records its source tensor: the reference, and the shape, dtype, and values' SHA-256 it named."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: str
    shape: list[int]
    dtype: str
    sha256: str


class ReplaceTensorRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: TensorRecord


class ReplaceTensorOperator(BaseEditOperator):
    """Write the values of a stored tensor of the run into the selection (module docstring)."""

    parameters_model = ReplaceTensorParameters
    tensor_parameters = ("source",)
    spec = make_spec(
        operator_id="replace_tensor",
        version="1",
        effect="edit",
        description="Write a tensor that this run's sandbox computed into the selection, unchanged: source names a "
        "tensor of a .safetensors file the sandbox saved to outputs/ (wse://tensors/<sha256>#<name>), in the target's "
        "stored dtype and the selection's shape, with finite values. wse_submit_candidate builds these operations "
        "for every tensor of a file.",
        required_capabilities=[CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT],
        parameters_model=ReplaceTensorParameters,
        result_model=PatchStep,
        usage=UsageContract(counter="candidate_commits", units_per_call=1, model_forward_calls=0),
    )

    def resolve_parameters(self, parameters: Mapping[str, Any], region: TensorRegion) -> ReplaceTensorRecord:
        parsed = parse_parameters(self.spec, ReplaceTensorParameters, parameters)
        assert isinstance(parsed, ReplaceTensorParameters)
        values = region.inputs.get("source")
        if values is None:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           "replace_tensor's source names a stored tensor file of a run, which only that run's "
                           "managed previews resolve", details={"parameter": "source"})
        if tuple(values.shape) != tuple(region.values.shape):
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"the source tensor has shape {list(values.shape)}; the selection of {region.tensor!r} has "
                           f"shape {list(region.values.shape)}",
                           details={"tensor": region.tensor, "source_shape": list(values.shape),
                                    "selection_shape": list(region.values.shape)})
        return ReplaceTensorRecord(source=TensorRecord(reference=parsed.source, shape=list(values.shape),
                                                       dtype=region.dtype, sha256=values_sha256(values)))

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:
        assert isinstance(parameters, ReplaceTensorRecord)
        return np.array(region.inputs["source"], dtype=np.float32)


BUILTIN_SUBMITTED_EDITS: dict[str, BaseEditOperator] = {"replace_tensor": ReplaceTensorOperator()}
