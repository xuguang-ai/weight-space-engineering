"""The operator contract: ``OperatorSpec`` plus an implementation protocol.

An operator is registered under the exact ``operator_id@version`` named by its
spec. It declares Pydantic models for its parameters and results; the spec's
``input_schema`` and ``output_schema`` are generated from those models, so the
advertised schemas cannot drift from what the engine validates.

Analysis operators receive a read-only ``TensorRegion`` that the engine has
already selected and bounded; they never read checkpoints, files, or other
tensors themselves, and never run the target network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from wse.schemas import (
    AxisSlice,
    ErrorCode,
    OperatorEffect,
    OperatorSpec,
    UsageContract,
    WSEError,
)

__all__ = [
    "AnalysisOperator",
    "BaseAnalysisOperator",
    "EFFECT_COUNTERS",
    "EFFECT_METHODS",
    "OperatorParameters",
    "OperatorResult",
    "OperatorSpec",
    "TensorRegion",
    "UsageContract",
    "make_spec",
    "validate_operator",
]

# The method each effect category must implement. Probe and edit operators are
# introduced by later milestones; their methods are reserved here.
EFFECT_METHODS: dict[str, str] = {"analysis": "analyze", "probe": "probe", "edit": "apply"}

# The budget counters each effect category may be charged under. Analyses and
# edits never run the target network, so they must declare zero forward calls.
EFFECT_COUNTERS: dict[str, frozenset[str]] = {
    "analysis": frozenset({"analysis_calls", "tensor_read_calls"}),
    "edit": frozenset({"candidate_commits"}),
    "probe": frozenset({"model_forward_calls", "evaluation_batches", "evaluated_examples"}),
}


class OperatorParameters(BaseModel):
    """Base for operator parameter models: strict, immutable, no unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class OperatorResult(BaseModel):
    """Base for operator result models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True)
class TensorRegion:
    """A read-only view of one selected region of a tensor.

    ``values`` has the region's shape and the stored dtype and is not
    writeable. ``selection`` locates it inside the full tensor.
    """

    tensor: str
    dtype: str
    axes: tuple[str, ...]
    tensor_shape: tuple[int, ...]
    selection: tuple[AxisSlice, ...]
    values: np.ndarray

    @property
    def element_count(self) -> int:
        return int(self.values.size)

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(entry.start for entry in self.selection)

    def tensor_index(self, flat_index: int) -> list[int]:
        """Coordinates in the full tensor of a row-major index into the region."""
        local = np.unravel_index(flat_index, self.values.shape) if self.values.ndim else ()
        return [int(offset + position) for offset, position in zip(self.offsets, local)]


@runtime_checkable
class AnalysisOperator(Protocol):
    """Contract for registered analysis operators."""

    spec: OperatorSpec
    parameters_model: type[BaseModel]
    result_model: type[BaseModel]

    def resolve_parameters(self, parameters: Mapping[str, Any], element_count: int) -> BaseModel:
        """Validate parameters and fill defaults that depend on the region size."""
        ...

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> BaseModel:
        """Compute results from the region; must not modify it."""
        ...


def make_spec(
    *,
    operator_id: str,
    version: str,
    effect: OperatorEffect,
    description: str,
    required_capabilities: list[str],
    parameters_model: type[BaseModel],
    result_model: type[BaseModel],
    usage: UsageContract,
) -> OperatorSpec:
    """Build a spec whose schemas are generated from the operator's models."""
    return OperatorSpec(
        operator_id=operator_id,
        version=version,
        effect=effect,
        description=description,
        required_capabilities=required_capabilities,
        input_schema=parameters_model.model_json_schema(),
        output_schema=result_model.model_json_schema(),
        usage=usage,
    )


class BaseAnalysisOperator:
    """Convenience base for analysis operators.

    Subclasses set ``spec``, ``parameters_model``, and ``result_model`` and
    implement ``analyze``. The default ``resolve_parameters`` validates the
    parameters strictly and rejects unknown names.
    """

    spec: ClassVar[OperatorSpec]
    parameters_model: ClassVar[type[BaseModel]] = OperatorParameters
    result_model: ClassVar[type[BaseModel]]

    def resolve_parameters(self, parameters: Mapping[str, Any], element_count: int) -> BaseModel:
        return parse_parameters(self.spec, self.parameters_model, parameters)

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> BaseModel:  # pragma: no cover
        raise NotImplementedError


def parse_parameters(
    spec: OperatorSpec, model: type[BaseModel], parameters: Mapping[str, Any]
) -> BaseModel:
    """Validate ``parameters`` against ``model``; raise ``INVALID_ARGUMENT``."""
    if not isinstance(parameters, Mapping):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"parameters of {spec.ref} must be an object")
    try:
        return model.model_validate(dict(parameters))
    except ValidationError as exc:
        accepted = sorted(model.model_fields)
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'parameters'}: {error['msg']}"
            for error in exc.errors(include_url=False)
        )
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"invalid parameters for {spec.ref}: {problems}. Accepted parameters: {accepted}",
            details={"operator": str(spec.ref), "accepted_parameters": accepted},
        ) from exc


def validate_operator(extension_id: str, version: str, implementation: Any) -> None:
    """Registry check: the implementation matches its spec and effect contract."""
    spec = getattr(implementation, "spec", None)
    if not isinstance(spec, OperatorSpec):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"operator {extension_id}@{version} must expose an OperatorSpec as .spec",
        )
    if (spec.operator_id, spec.version) != (extension_id, version):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"operator spec declares {spec.ref} but is being registered as {extension_id}@{version}",
        )
    for attribute in ("parameters_model", "result_model"):
        model = getattr(implementation, attribute, None)
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"operator {spec.ref} must declare {attribute} as a Pydantic model class",
            )
    if spec.input_schema != implementation.parameters_model.model_json_schema():
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"operator {spec.ref} input_schema does not match parameters_model")
    if spec.output_schema != implementation.result_model.model_json_schema():
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"operator {spec.ref} output_schema does not match result_model")
    allowed = EFFECT_COUNTERS[spec.effect]
    if spec.usage.counter not in allowed:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{spec.effect} operator {spec.ref} declares usage counter {spec.usage.counter!r}; "
            f"{spec.effect} operators are charged under one of {sorted(allowed)}",
        )
    if spec.effect in ("analysis", "edit") and spec.usage.model_forward_calls != 0:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{spec.effect} operator {spec.ref} must declare model_forward_calls=0; "
            f"{spec.effect} operators never run the target network",
        )
    method = EFFECT_METHODS[spec.effect]
    required = [method] + (["resolve_parameters"] if spec.effect in ("analysis", "edit") else [])
    missing = [name for name in required if not callable(getattr(implementation, name, None))]
    if missing:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{spec.effect} operator {spec.ref} does not implement {missing}",
            details={"missing_methods": missing},
        )
