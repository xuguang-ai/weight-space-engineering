"""Shared Pydantic contracts, canonical encodings, and domain error codes.

These contracts are application-level: they never import MCP request types or
agent-SDK types and carry no transport or session state. Every persisted or
exported object carries ``schema_version``. Canonical JSON (sorted keys,
compact separators, no NaN/Infinity) is the only encoding used for hashing, so
identities never depend on Python dictionary insertion order.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1"

CHECKPOINT_ID_PATTERN = re.compile(r"^ckpt_[0-9a-f]{32}$")
EXTENSION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
VERSION_PATTERN = re.compile(r"^[0-9]{1,6}(\.[0-9]{1,6}){0,2}$")
HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorCode(StrEnum):
    """Domain error codes shared by the Python API, CLI, and (later) MCP."""

    NOT_FOUND = "NOT_FOUND"
    POLICY_DENIED = "POLICY_DENIED"
    INVALID_PATCH = "INVALID_PATCH"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    INCOMPATIBLE_REPORTS = "INCOMPATIBLE_REPORTS"
    WORKER_FAILED = "WORKER_FAILED"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INVALID_MODEL = "INVALID_MODEL"
    ARTIFACT_CORRUPT = "ARTIFACT_CORRUPT"


class DomainError(BaseModel):
    """Serializable error body: ``code, message, retryable, details``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class WSEError(Exception):
    """A domain failure with an actionable message and a stable error code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = message
        self.retryable = retryable
        self.details = dict(details or {})

    def to_domain_error(self) -> DomainError:
        return DomainError(
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            details=self.details,
        )

    def __repr__(self) -> str:
        return f"WSEError({self.code.value}: {self.message})"


# ---------------------------------------------------------------------------
# Canonical encodings
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Return the canonical compact JSON text of ``value``.

    Raises ``WSEError(INVALID_ARGUMENT)`` for values that are not plain JSON or
    that contain non-finite floats.
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"value is not canonical JSON (plain JSON types, finite numbers only): {exc}",
        ) from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(data: bytes) -> str:
    """Return the ``sha256:<hex>`` content hash of raw bytes."""
    return "sha256:" + sha256_hex(data)


def json_hash(value: Any) -> str:
    """Return the content hash of the canonical JSON encoding of ``value``."""
    return content_hash(canonical_json(value).encode("utf-8"))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


def _check_json_object(value: dict[str, Any]) -> dict[str, Any]:
    # Round-trip through canonical JSON so stored configs are plain, finite JSON.
    return json.loads(canonical_json(value))


# ---------------------------------------------------------------------------
# Extension references
# ---------------------------------------------------------------------------


class ExtensionRef(_Frozen):
    """An exact extension identity, written ``<extension_id>@<version>``.

    Versions are always explicit: the registry never substitutes the latest
    implementation for a missing or unknown version.
    """

    extension_id: str
    version: str

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._split(value)
        return value

    @staticmethod
    def _split(text: str) -> dict[str, str]:
        if text.count("@") != 1:
            raise ValueError(
                f"extension reference {text!r} must be '<id>@<version>', for example 'mlp@1'"
            )
        extension_id, version = text.split("@")
        return {"extension_id": extension_id, "version": version}

    @field_validator("extension_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not EXTENSION_ID_PATTERN.fullmatch(value):
            raise ValueError(
                f"extension id {value!r} must match {EXTENSION_ID_PATTERN.pattern}"
            )
        return value

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        if not VERSION_PATTERN.fullmatch(value):
            raise ValueError(f"version {value!r} must match {VERSION_PATTERN.pattern}")
        return value

    @classmethod
    def parse(cls, value: str | ExtensionRef) -> ExtensionRef:
        """Parse ``value``; raise ``WSEError(INVALID_ARGUMENT)`` when malformed."""
        if isinstance(value, ExtensionRef):
            return value
        try:
            return cls.model_validate(value)
        except ValueError as exc:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"invalid extension reference {value!r}; use '<id>@<version>', for example 'mlp@1'",
                details={"reference": str(value)},
            ) from exc

    def __str__(self) -> str:
        return f"{self.extension_id}@{self.version}"


# ---------------------------------------------------------------------------
# Tensor selections
# ---------------------------------------------------------------------------


class AxisSlice(_Frozen):
    """One axis of a selection: ``start/stop/step`` with integer fields only."""

    start: StrictInt
    stop: StrictInt
    step: StrictInt = 1


# ---------------------------------------------------------------------------
# Model manifests
# ---------------------------------------------------------------------------


class TensorInfo(_Frozen):
    """Name, shape, dtype, and per-axis semantics of one stored tensor."""

    name: str
    shape: list[int]
    dtype: str
    axes: list[str]
    element_count: int

    @model_validator(mode="after")
    def _check_axes(self) -> TensorInfo:
        if len(self.axes) != len(self.shape):
            raise ValueError("axes must name every dimension of shape")
        if self.element_count != math.prod(self.shape):
            raise ValueError("element_count must equal the product of shape")
        return self


class IOSpec(_Frozen):
    """Input or output semantics of a model (structural shape plus description)."""

    shape: list[int | str]
    dtype: str
    description: str = ""


class Provenance(_Frozen):
    """Where a checkpoint came from.

    ``source`` and ``notes`` are operator-supplied labels; WSE never fills them
    from filesystem paths. Agent-visible manifests carry this object, so it must
    describe origin neutrally and never reveal evaluation answers.
    """

    kind: str = Field(default="import", pattern=r"^[a-z][a-z_]{0,31}$")
    source: str = Field(default="", max_length=256)
    parent_checkpoint_id: str | None = None
    notes: str = Field(default="", max_length=1024)


class ImportProvenance(_Frozen):
    """Operator-supplied provenance labels accepted by an import request.

    Lineage (``parent_checkpoint_id``) is deliberately absent: the server
    derives it when it creates a checkpoint from another one and never trusts
    a client-asserted parent. Imports always have ``kind="import"``.
    """

    kind: Literal["import"] = "import"
    source: str = Field(default="", max_length=256)
    notes: str = Field(default="", max_length=1024)

    def to_provenance(self) -> Provenance:
        return Provenance(kind=self.kind, source=self.source, notes=self.notes)


class ImportManifest(_Frozen):
    """Operator-supplied description of a model to import.

    Identity fields (checkpoint ID, content and configuration hashes, tensor
    layout) are deliberately absent: the server derives them from the tensors
    and the registered adapter and never trusts client-supplied values.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    adapter: ExtensionRef
    config: dict[str, Any]
    provenance: ImportProvenance = Field(default_factory=ImportProvenance)
    license: str = Field(default="unspecified", min_length=1, max_length=256)
    input_description: str = Field(default="", max_length=1024)
    output_description: str = Field(default="", max_length=1024)

    @field_validator("config")
    @classmethod
    def _normalize_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _check_json_object(value)


class BundleManifest(_Frozen):
    """The ``wse_bundle.json`` file of an on-disk model bundle.

    ``weights`` and ``config`` are paths relative to the bundle directory and
    must stay inside it.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    adapter: ExtensionRef
    weights: str = Field(min_length=1, max_length=255)
    config: str = Field(min_length=1, max_length=255)
    provenance: ImportProvenance = Field(default_factory=ImportProvenance)
    license: str = Field(default="unspecified", min_length=1, max_length=256)
    input_description: str = Field(default="", max_length=1024)
    output_description: str = Field(default="", max_length=1024)

    def to_import_manifest(self, config: dict[str, Any]) -> ImportManifest:
        return ImportManifest(
            adapter=self.adapter,
            config=config,
            provenance=self.provenance,
            license=self.license,
            input_description=self.input_description,
            output_description=self.output_description,
        )


class ModelIdentity(_Frozen):
    """Behavioral model identity: architecture adapter, configuration, and weights."""

    adapter: ExtensionRef
    config_hash: str
    weights_hash: str


class ModelManifest(_Frozen):
    """Immutable description of a registered checkpoint."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    checkpoint_id: str = Field(pattern=CHECKPOINT_ID_PATTERN.pattern)
    weights_hash: str = Field(pattern=HASH_PATTERN.pattern)
    config_hash: str = Field(pattern=HASH_PATTERN.pattern)
    adapter: ExtensionRef
    config: dict[str, Any]
    tensors: list[TensorInfo]
    parameter_count: int
    input_semantics: IOSpec
    output_semantics: IOSpec
    alias_groups: list[list[str]] = Field(default_factory=list)
    provenance: Provenance
    license: str

    @field_validator("config")
    @classmethod
    def _normalize_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _check_json_object(value)

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            adapter=self.adapter,
            config_hash=self.config_hash,
            weights_hash=self.weights_hash,
        )

    def tensor_info(self, name: str) -> TensorInfo:
        """Return the tensor with exactly this name, or raise ``NOT_FOUND``."""
        for info in self.tensors:
            if info.name == name:
                return info
        raise WSEError(
            ErrorCode.NOT_FOUND,
            f"tensor {name!r} does not exist in checkpoint {self.checkpoint_id}; "
            "tensor names must match exactly",
            details={"available_tensors": [info.name for info in self.tensors]},
        )


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class MetricGate(_Frozen):
    """One threshold on a task-defined metric.

    Exactly one of ``min_value`` (absolute acceptance threshold), ``min_delta``
    (required improvement over the base model), or ``max_drop`` (permitted
    regression relative to the base model) is set.
    """

    metric: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,128}$")
    min_value: float | None = None
    min_delta: float | None = None
    max_drop: float | None = None

    def __init__(self, metric: str | None = None, /, **data: Any) -> None:
        if metric is not None:
            data["metric"] = metric
        super().__init__(**data)

    @model_validator(mode="after")
    def _exactly_one_threshold(self) -> MetricGate:
        values = [v for v in (self.min_value, self.min_delta, self.max_drop) if v is not None]
        if len(values) != 1:
            raise ValueError("set exactly one of min_value, min_delta, or max_drop")
        if not math.isfinite(values[0]):
            raise ValueError("gate thresholds must be finite")
        if self.max_drop is not None and self.max_drop < 0:
            raise ValueError("max_drop must be nonnegative")
        return self

    @property
    def kind(self) -> Literal["acceptance", "regression"]:
        return "regression" if self.max_drop is not None else "acceptance"


class ObjectiveSpec(_Frozen):
    """Binds a natural-language goal to task-defined metrics and thresholds."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    task_id: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,128}$")
    description: str = Field(min_length=1, max_length=4096)
    gates: list[MetricGate] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_gates(self) -> ObjectiveSpec:
        if not self.acceptance_gates:
            raise ValueError("an objective needs at least one acceptance gate")
        keys = [(gate.metric, gate.kind) for gate in self.gates]
        if len(keys) != len(set(keys)):
            raise ValueError("each metric may have at most one acceptance and one regression gate")
        return self

    @property
    def acceptance_gates(self) -> list[MetricGate]:
        return [gate for gate in self.gates if gate.kind == "acceptance"]

    @property
    def regression_gates(self) -> list[MetricGate]:
        return [gate for gate in self.gates if gate.kind == "regression"]


# ---------------------------------------------------------------------------
# Core result contracts (direct library use; no experiment IDs)
# ---------------------------------------------------------------------------


class Usage(_Frozen):
    """Observed resource usage. ``None`` means unavailable, never zero.

    ``unavailable`` maps a field name to the reason it is ``None``.
    """

    evaluation_batches: int | None = None
    evaluated_examples: int | None = None
    model_forward_calls: int | None = None
    compute_seconds: float | None = None
    analysis_calls: int | None = None
    tensor_read_calls: int | None = None
    candidate_commits: int | None = None
    artifact_bytes: int | None = None
    peak_memory_bytes: int | None = None
    unavailable: dict[str, str] = Field(default_factory=dict)


class TensorDiff(_Frozen):
    """Per-tensor difference between a base and a candidate model."""

    tensor: str
    before_hash: str
    after_hash: str
    changed_elements: int
    element_count: int
    l2_delta: float
    max_abs_delta: float
    all_finite: bool


class PatchRegion(_Frozen):
    """Exact after-values written to one contiguous region, in row-major order."""

    tensor: str
    selection: list[AxisSlice]
    dtype: str
    values: list[float]


class MaterializedPatch(_Frozen):
    """An exact patch: replayable from its base without running the editing plugin."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    base_weights_hash: str
    result_weights_hash: str
    operator: ExtensionRef
    parameters: dict[str, Any] = Field(default_factory=dict)
    regions: list[PatchRegion] = Field(default_factory=list)


Management = Literal["unmanaged", "managed"]
