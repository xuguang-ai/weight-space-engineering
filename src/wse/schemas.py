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
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"


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
    """Per-tensor difference between a base and a candidate model.

    ``changed_elements`` counts elements whose stored bytes differ after
    writing back to the original dtype (so ``-0.0`` replacing ``0.0`` counts,
    and a no-op write does not). ``relative_l2_delta`` is ``l2_delta`` divided
    by the base tensor's L2 norm, or ``None`` when that norm is zero.
    """

    tensor: str
    before_hash: str
    after_hash: str
    changed_elements: int
    element_count: int
    l2_delta: float
    max_abs_delta: float
    before_l2: float
    relative_l2_delta: float | None
    all_finite: bool


class PatchStep(_Frozen):
    """One materialized edit step: the exact after-values of one region.

    ``values`` are row-major, in the stored dtype's shortest round-trip
    digits, so replaying a step writes exactly these bytes without running
    the editing operator. ``before_hash``/``after_hash`` are the tensor's
    content hashes around this step.
    """

    op: str
    operator: ExtensionRef
    tensor: str
    selection: list[AxisSlice]
    parameters: dict[str, Any]
    dtype: str
    before_hash: str
    after_hash: str
    values: list[float]
    changed_elements: int


class MaterializedPatch(_Frozen):
    """An exact patch: replayable from its base without running the editing plugin.

    Steps apply in order to one snapshot. ``base`` is the full identity
    (adapter, configuration, weights) the patch was materialized against; an
    exact patch is specific to that base revision.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    base: ModelIdentity
    result_weights_hash: str
    steps: list[PatchStep]


Management = Literal["unmanaged", "managed"]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

OperatorEffect = Literal["analysis", "probe", "edit"]

UsageCounter = Literal[
    "analysis_calls",
    "tensor_read_calls",
    "candidate_commits",
    "model_forward_calls",
    "evaluation_batches",
    "evaluated_examples",
]


class UsageContract(_Frozen):
    """How one invocation of an operator is charged against a budget."""

    counter: UsageCounter
    units_per_call: int = Field(default=1, ge=1)
    model_forward_calls: Literal[0] | None = Field(
        default=0,
        description="0 when the operator never runs the target network; None when it may",
    )


class OperatorSpec(_Frozen):
    """Identity and contract of a registered operator.

    ``input_schema`` describes the operator's own parameters (for example
    ``k`` of top-k); the tensor and selection are common to every analysis.
    The engine dispatches by this spec; interface adapters contain no
    operator-specific algorithms.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    operator_id: str = Field(pattern=EXTENSION_ID_PATTERN.pattern)
    version: str = Field(pattern=VERSION_PATTERN.pattern)
    effect: OperatorEffect
    description: str = Field(min_length=1, max_length=1024)
    required_capabilities: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    usage: UsageContract

    @field_validator("required_capabilities")
    @classmethod
    def _sorted_unique(cls, value: list[str]) -> list[str]:
        return sorted(set(value))

    @field_validator("input_schema", "output_schema")
    @classmethod
    def _plain_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _check_json_object(value)

    @property
    def ref(self) -> ExtensionRef:
        return ExtensionRef(extension_id=self.operator_id, version=self.version)


# ---------------------------------------------------------------------------
# Bounded reads and analyses
# ---------------------------------------------------------------------------


class TensorSlice(_Frozen):
    """A bounded, JSON-serializable region of one stored tensor.

    ``values`` are row-major and use the shortest decimal digits that
    round-trip the stored dtype. ``checkpoint_id`` is ``None`` for direct use.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    checkpoint_id: str | None
    weights_hash: str
    tensor: str
    dtype: str
    axes: list[str]
    tensor_shape: list[int]
    selection: list[AxisSlice]
    shape: list[int]
    element_count: int
    values: list[float]
    usage: Usage
    management: Management


class AnalysisResult(_Frozen):
    """Result of one registered analysis of a tensor region.

    ``input_hash`` binds the operator, model identity, tensor, selection, and
    resolved parameters. ``result`` holds the numerical results inline; when
    they exceed the inline response limit of a managed call, ``result`` is
    ``None`` and ``result_uri`` references the stored content, whose hash is
    ``result_hash``. ``checkpoint_id`` is ``None`` for direct use.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    kind: str
    operator: ExtensionRef
    checkpoint_id: str | None
    model: ModelIdentity
    tensor: str
    selection: list[AxisSlice]
    element_count: int
    parameters: dict[str, Any]
    input_hash: str
    result: dict[str, Any] | None
    result_hash: str
    result_bytes: int
    result_uri: str | None = None
    elapsed_seconds: float
    usage: Usage
    management: Management


# ---------------------------------------------------------------------------
# Patches (managed declarative edits)
# ---------------------------------------------------------------------------

OBJECT_ID_PATTERNS = {
    "run": re.compile(r"^run_[0-9a-f]{32}$"),
    "experiment": re.compile(r"^exp_[0-9a-f]{32}$"),
    "patch": re.compile(r"^patch_[0-9a-f]{32}$"),
}
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
LABEL_PATTERN = r"^[A-Za-z0-9_.\-]{1,128}$"
_COMMON_OPERATION_FIELDS = ("op", "tensor", "selection")


class PatchOperation(BaseModel):
    """One declarative operation of a patch.

    ``op`` is a public operation alias permitted by the experiment policy
    (``set_slice`` or ``scale_slice`` by default). ``tensor`` is an exact
    tensor name and ``selection`` a list of ``{start, stop, step}`` objects,
    one per axis. Every other field is a parameter of the operator (for
    example ``factor`` of ``scale_slice`` or ``values`` of ``set_slice``) and
    is validated by that operator.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    op: str = Field(min_length=1, max_length=128)
    tensor: str = Field(min_length=1, max_length=256)
    selection: list[Any]

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    def operator_request(self) -> dict[str, Any]:
        """Tensor, selection, and parameters as passed to ``apply_operator``."""
        return {"tensor": self.tensor, "selection": list(self.selection), **self.parameters}


class PatchSpec(_Frozen):
    """A client-proposed patch: base checkpoint, ordered operations, and rationale.

    The base content hash is deliberately absent: the server resolves it from
    the immutable checkpoint and writes it into the frozen spec.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    base_checkpoint_id: str = Field(pattern=CHECKPOINT_ID_PATTERN.pattern)
    hypothesis: str = Field(min_length=1, max_length=2000)
    expected_behavior: str = Field(default="", max_length=2000)
    development_suite_id: str | None = Field(default=None, pattern=LABEL_PATTERN)
    operations: list[PatchOperation] = Field(min_length=1)


class FrozenOperation(_Frozen):
    """An operation resolved to an exact operator version."""

    op: str
    operator: ExtensionRef
    tensor: str
    selection: list[AxisSlice]
    parameters: dict[str, Any]


class FrozenPatchSpec(_Frozen):
    """The server-resolved, immutable form of a ``PatchSpec``."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    base_checkpoint_id: str
    base_weights_hash: str
    hypothesis: str
    expected_behavior: str
    development_suite_id: str | None
    operations: list[FrozenOperation]


class PatchArtifact(_Frozen):
    """Stored content of a preview: the frozen spec and its exact patch."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    spec: FrozenPatchSpec
    patch: MaterializedPatch


class PatchOperationSummary(_Frozen):
    index: int
    op: str
    operator: ExtensionRef
    tensor: str
    selection: list[AxisSlice]
    changed_elements: int


class StorageCost(_Frozen):
    """Bytes stored for the preview and for a candidate created from it.

    The MVP saves complete snapshots, so ``candidate_bytes`` is the size of
    every tensor of the model; identical weights are stored once.
    """

    patch_bytes: int
    candidate_bytes: int


class PatchPreview(_Frozen):
    """Immutable, validated preview of a patch; ``apply_patch`` accepts only its ID."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    patch_id: str
    experiment_id: str
    spec_hash: str
    base_checkpoint_id: str
    base_weights_hash: str
    result_weights_hash: str
    hypothesis: str
    expected_behavior: str
    development_suite_id: str | None
    operations: list[PatchOperationSummary]
    tensors: list[TensorDiff]
    changed_elements: int
    parameter_count: int
    changed_fraction: float
    l2_delta: float
    max_abs_delta: float
    relative_l2_delta: float | None
    all_finite: bool
    storage: StorageCost
    patch_uri: str
    usage: Usage
    management: Management = "managed"


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

ExperimentState = Literal["OPEN", "FROZEN", "CLOSED"]
ExperimentMode = Literal["weights_only", "bounded_eval"]


class RunGrant(_Frozen):
    """An operator-issued grant: one experiment for this task, policy, and base model."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    run_id: str
    task_id: str
    policy_id: str
    base_checkpoint_id: str
    created_at: str


class BudgetEntry(_Frozen):
    """One counter of an experiment's budget ledger."""

    limit: int | float
    consumed: int | float
    remaining: int | float


class Experiment(_Frozen):
    """Persistent state of one managed experiment.

    ``policy_id`` names an immutable policy whose ``mode`` and ``policy_hash``
    are recorded here. ``budget`` maps each ledger counter to its limit and
    consumption; consumed allowance is never refunded.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    experiment_id: str
    run_id: str
    task_id: str
    policy_id: str
    policy_hash: str
    mode: ExperimentMode
    base_checkpoint_id: str
    selected_checkpoint_id: str
    state: ExperimentState
    budget: dict[str, BudgetEntry]
    created_at: str
    ended_at: str | None = None


class ExperimentCheckpoint(_Frozen):
    """A checkpoint owned by an experiment: its base or one of its candidates."""

    checkpoint_id: str
    role: Literal["base", "candidate"]
    parent_checkpoint_id: str | None
    patch_id: str | None
    weights_hash: str
    created_at: str
