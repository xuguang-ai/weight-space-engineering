"""The operator contract: ``OperatorSpec`` plus an implementation protocol.

An operator is registered under the exact ``operator_id@version`` named by its
spec. It declares Pydantic models for its parameters and results; the spec's
``input_schema`` and ``output_schema`` are generated from those models, so the
advertised schemas cannot drift from what the engine validates.

Analysis operators receive a read-only ``TensorRegion`` that the engine has
already selected and bounded; they never read checkpoints, files, or other
tensors themselves, and never run the target network.

Two optional declarations serve analyses whose results hold vectors (M2):

* ``artifact_fields`` names result fields that hold vectors. In managed use
  the engine stores the complete result as a run-scoped
  ``wse://analyses/<sha256>`` artifact and returns the result without those
  fields inline, so the response carries the summary and top entries and the
  artifact's URI. The fields must be optional, so that the inline result is
  valid against the advertised output schema. Direct use keeps them inline.
* ``reference_parameters`` names parameters that may be given as a
  ``VectorReference``: ``<result URI>#<JSON pointer>``, a vector inside a
  stored analysis result. The engine reads the result (a run reads only the
  results of its own experiment), checks that the pointer names a list of
  finite numbers, and hands the vector to the operator in
  ``TensorRegion.inputs`` as a read-only float32 array. The recorded
  parameters keep the reference, which is content-addressed.

Edit operators receive the same read-only region and return its new values
as an array of the region's shape. The engine computes in FP32, casts back to
the stored dtype, writes only that region, checks shapes and finiteness, and
records the result in the standard materialized representation
(``PatchStep``), which is therefore every edit operator's output schema.
Replaying a materialized patch never runs the operator again.

Edit operators may declare two optional lists of parameter names (M2):

* ``reference_parameters`` names parameters that hold a vector, or a list of
  vectors, each given inline (a list of numbers) or as a ``VectorReference``.
  Only managed previews resolve references: the engine reads them through the
  run-scoped reader before any weights are read and hands each vector to the
  operator in ``TensorRegion.inputs``, keyed by the parameter name or, for an
  entry of a list, ``<name>/<index>`` (``reference_entries``). Direct use
  refuses references. The recorded parameters keep each reference with the
  length and SHA-256 of the vector it named, and agent-facing patch views
  echo each inline vector of these parameters as ``{length, sha256}``.
* ``inline_parameters`` names parameters whose inline numbers (vectors and
  index lists) count toward a patch's limit on inline elements.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from wse.schemas import (
    AxisSlice,
    ErrorCode,
    OperatorEffect,
    OperatorSpec,
    PatchStep,
    UsageContract,
    WSEError,
)

__all__ = [
    "AnalysisOperator",
    "BaseAnalysisOperator",
    "BaseEditOperator",
    "EditOperator",
    "EFFECT_COUNTERS",
    "EFFECT_METHODS",
    "OperatorParameters",
    "OperatorResult",
    "OperatorSpec",
    "TensorRegion",
    "UsageContract",
    "MAX_VECTOR_REFERENCE_LENGTH",
    "VECTOR_REFERENCE_PATTERN",
    "VectorReference",
    "make_spec",
    "reference_entries",
    "validate_operator",
]

# ``wse://analyses/<sha256>#<pointer>``: a JSON pointer (RFC 6901, URI fragment form) into a stored
# analysis result, whose segments are field names or list indices without leading zeros, for example
# ``#/right_vectors/0``.
VECTOR_REFERENCE_PATTERN = r"^wse://analyses/[0-9a-f]{64}#(?:/(?:[A-Za-z_][A-Za-z0-9_]*|0|[1-9][0-9]*))+$"
# A reference is at most this long: the 80-character URI and a pointer of a few short segments. Longer strings are
# refused before they are matched or echoed, so no error or audit record can grow with them.
MAX_VECTOR_REFERENCE_LENGTH = 256
VectorReference = Annotated[str, StringConstraints(max_length=MAX_VECTOR_REFERENCE_LENGTH,
                                                   pattern=VECTOR_REFERENCE_PATTERN)]
_VECTOR_REFERENCE = re.compile(VECTOR_REFERENCE_PATTERN)

# A parameter error message lists at most this many problems and counts the rest, so that an invalid long vector
# cannot produce an unbounded message (each problem names one entry).
MAX_REPORTED_PROBLEMS = 5

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
    writeable; a BF16 region, which NumPy cannot hold, arrives as its exact
    float32 widening (``dtype`` still names the stored dtype). ``selection``
    locates it inside the full tensor. ``inputs`` holds the vectors of an
    operator's ``reference_parameters`` that were given as references, as
    read-only float32 arrays, keyed by parameter name (``<name>/<index>`` for
    an entry of a list parameter of an edit operator); it is empty otherwise.
    """

    tensor: str
    dtype: str
    axes: tuple[str, ...]
    tensor_shape: tuple[int, ...]
    selection: tuple[AxisSlice, ...]
    values: np.ndarray
    inputs: Mapping[str, np.ndarray] = field(default_factory=lambda: MappingProxyType({}))

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


@runtime_checkable
class EditOperator(Protocol):
    """Contract for registered edit operators.

    ``result_model`` is always ``PatchStep``: the engine, not the operator,
    produces the materialized record of an edit.
    """

    spec: OperatorSpec
    parameters_model: type[BaseModel]
    result_model: type[BaseModel]

    def resolve_parameters(self, parameters: Mapping[str, Any], region: TensorRegion) -> BaseModel:
        """Validate parameters against the selected region (shape, dtype)."""
        ...

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:
        """Return the region's new values (same shape); must not modify the region."""
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
    parameters strictly and rejects unknown names. ``artifact_fields`` and
    ``reference_parameters`` are the optional declarations described in the
    module docstring.
    """

    spec: ClassVar[OperatorSpec]
    parameters_model: ClassVar[type[BaseModel]] = OperatorParameters
    result_model: ClassVar[type[BaseModel]]
    artifact_fields: ClassVar[tuple[str, ...]] = ()
    reference_parameters: ClassVar[tuple[str, ...]] = ()

    def resolve_parameters(self, parameters: Mapping[str, Any], element_count: int) -> BaseModel:
        return parse_parameters(self.spec, self.parameters_model, parameters)

    def analyze(self, region: TensorRegion, parameters: BaseModel) -> BaseModel:  # pragma: no cover
        raise NotImplementedError


class BaseEditOperator:
    """Convenience base for edit operators.

    Subclasses set ``spec`` and ``parameters_model`` and implement ``apply``.
    ``reference_parameters`` and ``inline_parameters`` are the optional
    declarations described in the module docstring.
    """

    spec: ClassVar[OperatorSpec]
    parameters_model: ClassVar[type[BaseModel]] = OperatorParameters
    result_model: ClassVar[type[BaseModel]] = PatchStep
    reference_parameters: ClassVar[tuple[str, ...]] = ()
    inline_parameters: ClassVar[tuple[str, ...]] = ()

    def resolve_parameters(self, parameters: Mapping[str, Any], region: TensorRegion) -> BaseModel:
        return parse_parameters(self.spec, self.parameters_model, parameters)

    def apply(self, region: TensorRegion, parameters: BaseModel) -> np.ndarray:  # pragma: no cover
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
        errors = exc.errors(include_url=False, include_input=False, include_context=False)
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'parameters'}: {error['msg']}"
            for error in errors[:MAX_REPORTED_PROBLEMS]
        )
        if len(errors) > MAX_REPORTED_PROBLEMS:
            problems += f"; and {len(errors) - MAX_REPORTED_PROBLEMS} more problems"
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"invalid parameters for {spec.ref}: {problems}. Accepted parameters: {accepted}",
            details={"operator": str(spec.ref), "accepted_parameters": accepted, "error_count": len(errors)},
        ) from exc


def reference_entries(name: str, value: Any) -> list[tuple[str, str]]:
    """The ``(key, reference)`` pairs of a reference parameter's raw value.

    The value itself when it is a ``VectorReference`` (key ``name``), or each
    entry of a list that is one (key ``<name>/<index>``). Other values,
    including strings that are not references, yield nothing: the operator's
    parameter validation reports them.
    """
    if isinstance(value, str):
        return [(name, value)] if _VECTOR_REFERENCE.fullmatch(value) else []
    if isinstance(value, list):
        return [(f"{name}/{index}", entry) for index, entry in enumerate(value)
                if isinstance(entry, str) and _VECTOR_REFERENCE.fullmatch(entry)]
    return []


def _validate_vector_declarations(spec: OperatorSpec, implementation: Any) -> None:
    """``artifact_fields`` (analyses) are optional result fields; the parameter declarations are parameter fields."""
    if spec.effect == "analysis":
        result_fields = implementation.result_model.model_fields
        for name in tuple(getattr(implementation, "artifact_fields", ()) or ()):
            declared = result_fields.get(name)
            if declared is None or declared.is_required():
                raise WSEError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"analysis operator {spec.ref} declares artifact field {name!r}, which must be an optional field "
                    f"of its result_model (the inline result omits it); result fields: {sorted(result_fields)}",
                    details={"artifact_field": name},
                )
    parameter_fields = implementation.parameters_model.model_fields
    declarations = ["reference_parameters"] + (["inline_parameters"] if spec.effect == "edit" else [])
    for declaration in declarations:
        for name in tuple(getattr(implementation, declaration, ()) or ()):
            if name not in parameter_fields:
                raise WSEError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"{spec.effect} operator {spec.ref} declares {declaration[:-1].replace('_', ' ')} {name!r}, which "
                    f"is not a field of its parameters_model; parameters: {sorted(parameter_fields)}",
                    details={declaration[:-1]: name},
                )


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
    if spec.effect == "edit" and implementation.result_model is not PatchStep:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"edit operator {spec.ref} must declare result_model = PatchStep, the standard "
            "materialized patch representation",
        )
    if spec.effect in ("analysis", "edit"):
        _validate_vector_declarations(spec, implementation)
    allowed = EFFECT_COUNTERS[spec.effect]
    if spec.usage.counter not in allowed:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{spec.effect} operator {spec.ref} declares usage counter {spec.usage.counter!r}; "
            f"{spec.effect} operators are charged under one of {sorted(allowed)}",
        )
    if spec.effect == "edit" and spec.usage.units_per_call != 1:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"edit operator {spec.ref} declares units_per_call={spec.usage.units_per_call}; edit operators "
            "must declare units_per_call=1, because the managed path charges exactly one candidate commit "
            "per applied patch, whatever its operators",
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
