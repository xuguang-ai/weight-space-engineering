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


def _is_none(value: Any) -> bool:
    return value is None


def _optional(**kwargs: Any) -> Any:
    """An optional field that is omitted from serialized output while unset.

    M2 adds fields to M1 contracts this way, so M1 manifests, identities, and
    their canonical JSON stay byte-identical.
    """
    return Field(default=None, exclude_if=_is_none, **kwargs)


class TensorInfo(_Frozen):
    """Name, shape, dtype, and per-axis semantics of one stored tensor.

    ``module_name`` is the parameter name in the runtime module when it differs
    from the canonical (checkpoint) name or the model is imported from a
    snapshot; ``blob_id`` is the tensor's content-addressed blob (M2 plan
    Sections 2.1 and 3.1). Both are absent for M1 models.
    """

    name: str
    shape: list[int]
    dtype: str
    axes: list[str]
    element_count: int
    module_name: str | None = _optional(min_length=1)
    blob_id: str | None = _optional(pattern=r"^[0-9a-f]{64}$")

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


class GenerationProfile(_Frozen):
    """A registered, versioned sampling configuration for generation (M2 plan Section 4.2).

    The prompt is the problem followed by ``user_suffix``, rendered as the
    user turn of the model's chat template with ``chat_template_kwargs``
    (for example ``{"enable_thinking": true}``), after an optional system
    turn; with ``chat_template`` false the prompt is that text alone.
    Decoding is greedy unless ``do_sample``; sampling applies the presence
    penalty, then temperature, top-k (``None`` keeps every token), and top-p.
    Each (problem, sample) pair draws from its own seed, derived from
    ``seed``. ``stop_token_ids`` ``None`` stops at the model's end-of-sequence
    tokens. ``source`` says where the settings came from, and
    ``ignored_settings`` lists settings of that source that WSE does not apply.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    profile_id: str = Field(pattern=EXTENSION_ID_PATTERN.pattern)
    version: str = Field(pattern=VERSION_PATTERN.pattern)
    description: str = Field(default="", max_length=1024)
    chat_template: bool = True
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str | None = Field(default=None, max_length=8192)
    user_suffix: str = Field(default="", max_length=8192)
    do_sample: bool
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    top_k: StrictInt | None = Field(default=None, ge=1)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    presence_penalty: float = Field(default=0.0, allow_inf_nan=False)
    max_new_tokens: StrictInt = Field(ge=1, le=1_048_576)
    seed: StrictInt = Field(default=0, ge=0)
    stop_token_ids: list[StrictInt] | None = None
    source: str | None = Field(default=None, max_length=256)
    ignored_settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chat_template_kwargs", "ignored_settings")
    @classmethod
    def _plain(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _check_json_object(value)

    @property
    def ref(self) -> ExtensionRef:
        return ExtensionRef(extension_id=self.profile_id, version=self.version)

    @property
    def profile_hash(self) -> str:
        return json_hash(self.model_dump(mode="json"))


class ModelSource(_Frozen):
    """Where an open-weight model snapshot came from, and the hashes of what was read (M2 plan Section 2.1).

    ``files`` maps every snapshot file the importer read to its SHA-256.
    ``tokenizer_hash`` covers the tokenizer files present, the chat template
    hash covers the template text, and ``generation_config_hash`` the
    generation configuration; each is ``None`` when the snapshot has none.
    ``default_profile`` is the model's default generation profile, frozen from
    its ``generation_config.json`` at import (M2 plan Section 4.2).
    """

    repository: str = Field(min_length=1, max_length=256)
    revision: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._/-]+$")
    family: ExtensionRef | None = None
    files: dict[str, str]
    tokenizer_hash: str | None = Field(default=None, pattern=HASH_PATTERN.pattern)
    chat_template_hash: str | None = Field(default=None, pattern=HASH_PATTERN.pattern)
    generation_config_hash: str | None = Field(default=None, pattern=HASH_PATTERN.pattern)
    dropped_tensor_count: int = Field(default=0, ge=0)
    default_profile: GenerationProfile | None = _optional()


class ModelIdentity(_Frozen):
    """Behavioral model identity: architecture adapter, configuration, and weights.

    For an imported open-weight model it also comprises the tokenizer, chat
    template, and generation-configuration hashes and the source repository and
    revision (M1 plan Section 3, M2 plan Section 2.1). Those fields are absent
    for M1 models.
    """

    adapter: ExtensionRef
    config_hash: str
    weights_hash: str
    tokenizer_hash: str | None = _optional()
    chat_template_hash: str | None = _optional()
    generation_config_hash: str | None = _optional()
    source_repository: str | None = _optional()
    source_revision: str | None = _optional()


def model_identity(
    adapter: ExtensionRef, config_hash: str, weights_hash: str, source: ModelSource | None = None
) -> ModelIdentity:
    """The identity of a model; a snapshot import adds its source's hashes, repository, and revision."""
    if source is None:
        return ModelIdentity(adapter=adapter, config_hash=config_hash, weights_hash=weights_hash)
    return ModelIdentity(
        adapter=adapter,
        config_hash=config_hash,
        weights_hash=weights_hash,
        tokenizer_hash=source.tokenizer_hash,
        chat_template_hash=source.chat_template_hash,
        generation_config_hash=source.generation_config_hash,
        source_repository=source.repository,
        source_revision=source.revision,
    )


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
    source: ModelSource | None = _optional()

    @field_validator("config")
    @classmethod
    def _normalize_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _check_json_object(value)

    @property
    def identity(self) -> ModelIdentity:
        return model_identity(self.adapter, self.config_hash, self.weights_hash, self.source)

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
    patch_previews: int | None = None
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


_BLOB_ID = r"^[0-9a-f]{64}$"


class PatchBlock(_Frozen):
    """A binary block: the exact stored bits one step writes into one tensor (M2 plan Section 3.3).

    The block's bytes live in a binary patch artifact, keyed by ``block_id``,
    the SHA-256 hex digest of those bytes. A ``sparse`` block holds ``count``
    flat row-major indices into the tensor (little-endian ``uint64``) followed
    by as many bit patterns of the stored dtype (little-endian); a ``dense``
    block holds the bit patterns of the contiguous ``selection``, row-major.
    ``base_blob_id`` and ``result_blob_id`` bind the block to the tensor's
    bytes before and after it: replay refuses another base and verifies the
    result.
    """

    kind: Literal["sparse", "dense"]
    dtype: str
    tensor_shape: list[int]
    selection: list[AxisSlice] | None = None
    count: int = Field(ge=0)
    block_id: str = Field(pattern=_BLOB_ID)
    base_blob_id: str = Field(pattern=_BLOB_ID)
    result_blob_id: str = Field(pattern=_BLOB_ID)

    @model_validator(mode="after")
    def _selection_matches_kind(self) -> PatchBlock:
        if (self.kind == "dense") != (self.selection is not None):
            raise ValueError("a dense block has a selection and a sparse block has none")
        return self


class PatchStep(_Frozen):
    """One materialized edit step: the exact after-values of one region.

    A step records its result in one of two forms. M1 models keep inline
    ``values``: row-major, in the stored dtype's shortest round-trip digits.
    Open-weight models record a binary ``block`` (M2 plan Section 3.3) whose
    bytes are stored next to the patch. Either way, replaying a step writes
    exactly these bytes without running the editing operator.
    ``before_hash``/``after_hash`` are the tensor's content hashes around this
    step.
    """

    op: str
    operator: ExtensionRef
    tensor: str
    selection: list[AxisSlice]
    parameters: dict[str, Any]
    dtype: str
    before_hash: str
    after_hash: str
    values: list[float] | None = _optional()
    block: PatchBlock | None = _optional()
    changed_elements: int

    @model_validator(mode="after")
    def _one_form(self) -> PatchStep:
        if (self.values is None) == (self.block is None):
            raise ValueError("a patch step records either inline values or a binary block")
        return self


class ExternalPatchSource(_Frozen):
    """Where an imported patch came from: its format, repository, revision, run, and file hashes."""

    format: str
    repository: str | None = None
    revision: str | None = None
    run: str
    files: dict[str, str] = Field(description="file name -> SHA-256 hex digest of each file read")


class MaterializedPatch(_Frozen):
    """An exact patch: replayable from its base without running the editing plugin.

    Steps apply in order to one snapshot. ``base`` is the full identity
    (adapter, configuration, weights) the patch was materialized against; an
    exact patch is specific to that base revision. ``source`` describes an
    imported external patch.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    base: ModelIdentity
    result_weights_hash: str
    steps: list[PatchStep]
    source: ExternalPatchSource | None = _optional()


class ImportedPatchTensor(_Frozen):
    """One tensor of an imported patch, verified against its published hashes."""

    tensor: str
    form: str
    base_sha256: str
    edited_sha256: str
    changed_elements: int
    block: Literal["sparse", "dense"]


class ImportedPatch(_Frozen):
    """An external patch recorded as an exact WSE patch, with the candidate it produced."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    patch_id: str
    base_checkpoint_id: str
    checkpoint_id: str
    result_weights_hash: str
    source: ExternalPatchSource
    tensors: list[ImportedPatchTensor]
    changed_elements: int
    patch_bytes: int


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
    "job": re.compile(r"^job_[0-9a-f]{32}$"),
    "reservation": re.compile(r"^resv_[0-9a-f]{32}$"),
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
    """One counter of an experiment's budget ledger.

    ``reserved`` is held by queued or running jobs and not yet settled;
    ``remaining`` is ``limit - consumed - reserved``.
    """

    limit: int | float
    consumed: int | float
    reserved: int | float = 0
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
    frozen_at: str | None = None
    ended_at: str | None = None


class ExperimentCheckpoint(_Frozen):
    """A checkpoint owned by an experiment: its base or one of its candidates."""

    checkpoint_id: str
    role: Literal["base", "candidate"]
    parent_checkpoint_id: str | None
    patch_id: str | None
    weights_hash: str
    created_at: str


# ---------------------------------------------------------------------------
# Evaluation suites, budget reservations, and jobs
# ---------------------------------------------------------------------------

SuiteSplit = Literal["development", "final"]


class SuiteData(_Frozen):
    """Examples of a suite: row-major float inputs, optional class labels, and optional clusters.

    ``clusters`` gives the cluster ID of every example (for example the pair
    of an ``x, -x`` fixture); examples of one cluster are contiguous.
    ``cluster_unit`` names the unit (``"pair"``). Without clusters every
    example is its own cluster, as for independent examples.
    """

    inputs: list[list[float]] = Field(min_length=1)
    labels: list[StrictInt] | None = None
    clusters: list[StrictInt] | None = None
    cluster_unit: str | None = Field(default=None, pattern=r"^[a-z][a-z_]{0,31}$")

    @model_validator(mode="after")
    def _check_rows(self) -> SuiteData:
        widths = {len(row) for row in self.inputs}
        if len(widths) != 1 or 0 in widths:
            raise ValueError("every input row must have the same nonzero length")
        if self.labels is not None and len(self.labels) != len(self.inputs):
            raise ValueError("labels must have one entry per input row")
        for row in self.inputs:
            if not all(math.isfinite(value) for value in row):
                raise ValueError("inputs must be finite")
        if (self.clusters is None) != (self.cluster_unit is None):
            raise ValueError("clusters and cluster_unit must be given together")
        if self.clusters is not None:
            if len(self.clusters) != len(self.inputs):
                raise ValueError("clusters must have one entry per input row")
            seen: set[int] = set()
            for index, cluster in enumerate(self.clusters):
                if index and cluster == self.clusters[index - 1]:
                    continue
                if cluster in seen:
                    raise ValueError("the examples of each cluster must be contiguous")
                seen.add(cluster)
        return self

    @property
    def example_count(self) -> int:
        return len(self.inputs)

    def cluster_ids(self) -> list[int]:
        """The cluster of every example; independent examples are their own clusters."""
        return list(self.clusters) if self.clusters is not None else list(range(len(self.inputs)))

    def cuts_clusters(self, sample_count: int) -> bool:
        """Whether evaluating the first ``sample_count`` examples would split a cluster."""
        ids = self.cluster_ids()
        return 0 < sample_count < len(ids) and ids[sample_count - 1] == ids[sample_count]

    @property
    def data_hash(self) -> str:
        return json_hash(self.model_dump(mode="json"))


class DatasetManifest(_Frozen):
    """Description of a dataset held in evaluator storage, without its examples.

    The examples, and how they were generated (including any seed), stay in
    evaluator storage; suites, reports, and comparisons carry only this
    manifest and its ``data_hash``.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    data_hash: str = Field(pattern=HASH_PATTERN.pattern)
    example_count: int = Field(ge=1)
    input_width: int = Field(ge=1)
    label_classes: list[int] | None = None
    cluster_unit: str | None = None
    cluster_count: int = Field(ge=1)

    @classmethod
    def from_data(cls, data: SuiteData) -> DatasetManifest:
        return cls(
            data_hash=data.data_hash,
            example_count=data.example_count,
            input_width=len(data.inputs[0]),
            label_classes=sorted(set(data.labels)) if data.labels is not None else None,
            cluster_unit=data.cluster_unit,
            cluster_count=len(set(data.cluster_ids())),
        )


class TextItem(_Frozen):
    """One problem of a text suite: a stable ID, the problem text, and the reference its grader reads."""

    item_id: str = Field(min_length=1, max_length=256)
    problem: str = Field(min_length=1)
    reference: str

    @field_validator("problem")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a problem must not be blank")
        return value


class TextSuiteData(_Frozen):
    """Problems of a generation suite (M2 plan Section 4.3). Every problem is its own cluster."""

    kind: Literal["text"] = "text"
    items: list[TextItem] = Field(min_length=1)
    cluster_unit: Literal["problem"] = "problem"

    @model_validator(mode="after")
    def _unique(self) -> TextSuiteData:
        ids = [item.item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("item IDs must be unique")
        return self

    @property
    def example_count(self) -> int:
        return len(self.items)

    def cluster_ids(self) -> list[int]:
        return list(range(len(self.items)))

    def cuts_clusters(self, sample_count: int) -> bool:
        return False

    @property
    def data_hash(self) -> str:
        return json_hash(self.model_dump(mode="json"))


class TextDatasetManifest(_Frozen):
    """Description of a text dataset held in evaluator storage, without its problems.

    ``source_format`` and ``source_sha256`` name the JSON Lines file the
    problems were read from and its SHA-256.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    kind: Literal["text"] = "text"
    data_hash: str = Field(pattern=HASH_PATTERN.pattern)
    example_count: int = Field(ge=1)
    cluster_unit: Literal["problem"] = "problem"
    cluster_count: int = Field(ge=1)
    source_format: str | None = None
    source_sha256: str | None = Field(default=None, pattern=HASH_PATTERN.pattern)

    @classmethod
    def from_data(cls, data: TextSuiteData, *, source_format: str | None = None,
                  source_sha256: str | None = None) -> TextDatasetManifest:
        return cls(data_hash=data.data_hash, example_count=data.example_count, cluster_count=data.example_count,
                   source_format=source_format, source_sha256=source_sha256)


def dataset_manifest_matches(data: SuiteData | TextSuiteData, manifest: DatasetManifest | TextDatasetManifest) -> bool:
    """Whether ``manifest`` describes ``data`` (a text manifest's source fields are not part of the data)."""
    if isinstance(data, TextSuiteData) != isinstance(manifest, TextDatasetManifest):
        return False
    if isinstance(manifest, TextDatasetManifest):
        return TextDatasetManifest.from_data(data, source_format=manifest.source_format,
                                             source_sha256=manifest.source_sha256) == manifest
    return DatasetManifest.from_data(data) == manifest


class EvaluationSuite(_Frozen):
    """An operator-registered evaluation suite.

    ``sample_counts`` and ``seeds`` are the preregistered ``eval_config``
    values; a submission with any other value is rejected. A sample count of
    ``n`` evaluates the first ``n`` examples. ``evaluator`` is the exact
    registered evaluator that runs the suite, and ``parameters`` are passed
    to it unchanged. Only ``development`` suites accept agent submissions.

    The examples are either inline (``data``; development suites only) or
    held in evaluator storage and described by ``dataset``. A registered
    suite never carries both; the worker and final evaluator *resolve* a
    stored suite by attaching its verified examples in memory
    (``resolved``). Final suites always use evaluator storage.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    suite_id: str = Field(pattern=LABEL_PATTERN)
    task_id: str = Field(pattern=LABEL_PATTERN)
    split: SuiteSplit
    evaluator: ExtensionRef
    parameters: dict[str, Any] = Field(default_factory=dict)
    sample_counts: list[StrictInt] = Field(min_length=1)
    seeds: list[StrictInt] = Field(min_length=1)
    data: SuiteData | TextSuiteData | None = None
    dataset: DatasetManifest | TextDatasetManifest | None = None

    @field_validator("parameters")
    @classmethod
    def _plain_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _check_json_object(value)

    @model_validator(mode="after")
    def _check_preregistration(self) -> EvaluationSuite:
        if self.data is None and self.dataset is None:
            raise ValueError("a suite needs inline data or a dataset manifest")
        if self.data is not None and self.dataset is not None \
                and not dataset_manifest_matches(self.data, self.dataset):
            raise ValueError("the attached examples do not match the suite's dataset manifest")
        size = self.example_count
        if any(not 1 <= count <= size for count in self.sample_counts):
            raise ValueError(f"sample_counts must lie between 1 and the {size} examples of the suite")
        if len(set(self.sample_counts)) != len(self.sample_counts) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("sample_counts and seeds must not repeat")
        if self.data is not None and any(self.data.cuts_clusters(count) for count in self.sample_counts):
            raise ValueError("every preregistered sample count must end at a cluster boundary")
        return self

    @property
    def example_count(self) -> int:
        return self.dataset.example_count if self.dataset is not None else self.data.example_count

    @property
    def suite_hash(self) -> str:
        """Hash of the registered form (examples detached when they live in evaluator storage)."""
        dumped = self.model_dump(mode="json")
        if self.dataset is not None:
            dumped["data"] = None
        return json_hash(dumped)

    @property
    def data_hash(self) -> str:
        return self.dataset.data_hash if self.dataset is not None else self.data.data_hash

    def registered(self) -> EvaluationSuite:
        """The form stored at registration: stored examples are detached."""
        return self.model_copy(update={"data": None}) if self.dataset is not None else self

    def resolved(self, data: SuiteData | TextSuiteData) -> EvaluationSuite:
        """This suite with its stored examples attached (in memory only)."""
        return EvaluationSuite.model_validate({**self.model_dump(mode="json"), "data": data.model_dump(mode="json")})


class EvalConfig(_Frozen):
    """The MVP ``eval_config``: a preregistered sample count and seed.

    The server fixes the execution device and batch size.
    """

    sample_count: StrictInt = Field(ge=1)
    seed: StrictInt


ReservationState = Literal["HELD", "SETTLED"]
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
TerminalJobState = Literal["succeeded", "failed", "cancelled"]
TERMINAL_JOB_STATES: tuple[str, ...] = ("succeeded", "failed", "cancelled")


class Reservation(_Frozen):
    """Allowance held for one job until it settles.

    ``requested`` maps ledger counters to the reserved amounts. Settlement
    charges ``charged`` (never more than was reserved) and releases the rest.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    reservation_id: str
    experiment_id: str
    job_id: str
    state: ReservationState
    requested: dict[str, int | float]
    charged: dict[str, int | float] | None = None
    terminal_state: TerminalJobState | None = None
    created_at: str
    settled_at: str | None = None


class JobProgress(_Frozen):
    """Batches of a job: planned, started (charged if the job stops), and completed."""

    batches_total: int
    batches_started: int = 0
    batches_completed: int = 0
    examples_started: int = 0


class Job(_Frozen):
    """An asynchronous evaluation job.

    ``usage`` is the settled usage once the job is terminal and ``None``
    before. ``result`` carries the report summary (metrics, sample count,
    errors) inline; ``report_id`` references a persisted report once
    evaluation reports exist. ``cancel_requested`` is set when a running job
    was asked to stop; computation already performed remains charged.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    job_id: str
    experiment_id: str
    kind: Literal["evaluation"]
    checkpoint_id: str
    suite_id: str
    config: EvalConfig
    state: JobState
    cancel_requested: bool
    reservation: Reservation
    progress: JobProgress
    usage: Usage | None = None
    result: dict[str, Any] | None = None
    report_id: str | None = None
    error: DomainError | None = None
    attempts: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_JOB_STATES


# ---------------------------------------------------------------------------
# Tasks, evaluation reports, comparisons, and final evaluations
# ---------------------------------------------------------------------------

REPORT_ID_PATTERN = re.compile(r"^report_[0-9a-f]{32}$")
OBJECT_ID_PATTERNS["report"] = REPORT_ID_PATTERN
EVALUATION_DEVICE = "cpu"


class TaskDefinition(_Frozen):
    """The agent-visible definition of a task (``task.json``).

    It states the behavior to achieve and how it is measured, never how the
    task model was built or how to change it. ``target_accuracy`` is met only
    by a successful report of the task's own development suite covering all
    of its ``development_examples``.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    task_id: str = Field(pattern=LABEL_PATTERN)
    title: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=4096)
    input_description: str = Field(default="", max_length=1024)
    output_description: str = Field(default="", max_length=1024)
    objective: ObjectiveSpec
    target_accuracy: float | None = Field(default=None, ge=0, le=1)
    development_suite_id: str = Field(pattern=LABEL_PATTERN)
    development_examples: int = Field(ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> TaskDefinition:
        if self.objective.task_id != self.task_id:
            raise ValueError("the objective must name the same task_id")
        if self.target_accuracy is not None:
            gates = [g for g in self.objective.acceptance_gates if g.metric == "accuracy" and g.min_value is not None]
            if [g.min_value for g in gates] != [self.target_accuracy]:
                raise ValueError("target_accuracy must equal the objective's accuracy min_value gate")
        return self

    @property
    def task_hash(self) -> str:
        return json_hash(self.model_dump(mode="json"))


class ExecutionConfig(_Frozen):
    """Everything that determines how an evaluator scores a model, except the model and data.

    Its hash is a report's ``evaluator_config_hash``: two reports are
    comparable only when it, the data, and the suite agree.
    """

    evaluator: ExtensionRef
    parameters: dict[str, Any]
    eval_config: EvalConfig
    batch_size: int = Field(ge=1)
    # M1 evaluations run on the CPU; generation evaluations record the engine's device (M2 plan Section 4.1).
    device: Literal["cpu", "mps"] = EVALUATION_DEVICE

    @property
    def config_hash(self) -> str:
        return json_hash(self.model_dump(mode="json"))


class IntervalEstimate(_Frozen):
    """A 95% interval around ``estimate``.

    ``cluster_bootstrap``: percentile interval over ``resamples`` resamples of
    whole clusters with ``seed``. ``clopper_pearson``: exact binomial interval
    on ``successes`` of ``trials`` (fully correct clusters).
    """

    method: Literal["cluster_bootstrap", "clopper_pearson"]
    level: float = 0.95
    estimate: float
    lower: float
    upper: float
    cluster_unit: str
    resamples: int | None = None
    seed: int | None = None
    successes: int | None = None
    trials: int | None = None


class ClusterSummary(_Frozen):
    """Clusters covered by an evaluation and how many were scored fully correct."""

    unit: str
    count: int
    fully_correct: int


class SampleResult(_Frozen):
    """One generated sample of a problem: its seed, grade, length, and whether it reached the token limit.

    ``answer`` is the answer the grader read (at most 200 characters; ``None``
    when there was none). ``output_sha256`` identifies the generated token
    IDs, so identical generations can be recognized without the text.
    """

    sample: int = Field(ge=0)
    seed: int = Field(ge=0)
    correct: bool
    answer: str | None = None
    generated_tokens: int = Field(ge=0)
    finished: bool
    limit_hit: bool
    output_sha256: str = Field(pattern=HASH_PATTERN.pattern)


class ProblemResult(_Frozen):
    """The samples of one problem of a generation suite, in sample order.

    ``prompt_sha256`` identifies the prompt's token IDs as the model received
    them (rendered with the profile's chat template), so two evaluations that
    rendered a problem differently are recognized as incomparable.
    """

    index: int = Field(ge=0)
    item_id: str
    prompt_sha256: str = Field(pattern=HASH_PATTERN.pattern)
    samples: list[SampleResult] = Field(min_length=1)


class EvaluationReport(_Frozen):
    """A persisted, managed evaluation bound to exact model, data, and configuration identities.

    ``scores`` (per example, 1 = correct) and ``cluster_ids`` are kept for
    development reports so paired comparisons can be computed; final reports
    carry only aggregates. ``target_met`` is ``None`` for final reports and
    when the task declares no accuracy target; otherwise it is whether the
    report is of the task's own development suite, covers all of its
    ``development_examples``, and meets the target.

    Generation reports (M2) add ``problems`` (per-problem samples: grades,
    lengths, limit hits; development reports only) and ``runtime`` (the
    engine's device, stored dtypes, model class, and library versions).
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    report_id: str = Field(pattern=REPORT_ID_PATTERN.pattern)
    experiment_id: str
    job_id: str | None
    checkpoint_id: str
    model: ModelIdentity
    task_id: str
    task_hash: str | None
    suite_id: str
    suite_hash: str
    split: SuiteSplit
    data_hash: str
    evaluator: ExtensionRef
    evaluator_config_hash: str
    execution: ExecutionConfig
    seed: int
    sample_count: int
    metrics: dict[str, float]
    clusters: ClusterSummary | None
    intervals: dict[str, IntervalEstimate]
    scores: list[int] | None = None
    cluster_ids: list[int] | None = None
    target_met: bool | None = None
    elapsed_seconds: float
    usage: Usage
    errors: list[str]
    truncation_count: int = 0
    management: Management = "managed"
    created_at: str
    problems: list[ProblemResult] | None = _optional()
    runtime: dict[str, Any] | None = _optional()

    @property
    def report_uri(self) -> str:
        return f"wse://reports/{self.report_id}"

    def summary(self) -> dict[str, Any]:
        """The inline summary carried by a job: metrics, sample count, errors, and the reference."""
        return {"report_id": self.report_id, "report_uri": self.report_uri, "metrics": dict(self.metrics),
                "sample_count": self.sample_count, "errors": list(self.errors),
                "truncation_count": self.truncation_count, "target_met": self.target_met}


class GenerationComparison(_Frozen):
    """Paired candidate-minus-base differences of two generation evaluations on one suite and configuration.

    ``differences`` holds ``accuracy``, ``limit_hit_rate``, and
    ``mean_generated_tokens``, each with a paired cluster-bootstrap interval
    over problems.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    suite_id: str
    data_hash: str
    evaluator_config_hash: str
    base_model: ModelIdentity
    candidate_model: ModelIdentity
    problems: int
    samples: int
    base: dict[str, float]
    candidate: dict[str, float]
    differences: dict[str, IntervalEstimate]


class ComparisonReport(_Frozen):
    """Paired difference between two reports on the same examples and configuration.

    ``difference`` is candidate minus base accuracy, with a paired
    cluster-bootstrap interval; ``base_fully_correct`` and
    ``candidate_fully_correct`` are exact Clopper-Pearson intervals on the
    number of fully correct clusters. Clusters are counted as improved,
    worsened, or unchanged by their number of correct examples.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    experiment_id: str
    base_report_id: str
    candidate_report_id: str
    base_checkpoint_id: str
    candidate_checkpoint_id: str
    split: SuiteSplit
    suite_id: str
    suite_hash: str
    data_hash: str
    evaluator_config_hash: str
    sample_count: int
    seed: int
    metric: Literal["accuracy"] = "accuracy"
    base_value: float
    candidate_value: float
    difference: float
    difference_interval: IntervalEstimate
    base_fully_correct: IntervalEstimate
    candidate_fully_correct: IntervalEstimate
    clusters_improved: int
    clusters_worsened: int
    clusters_unchanged: int
    management: Management = "managed"


class FinalEvaluationReport(_Frozen):
    """The sealed result of an experiment: the frozen candidate against the base-checkpoint control.

    Both checkpoints are evaluated on the same final examples and
    configuration; when the frozen selection is the base checkpoint, one
    evaluation serves both (``same_checkpoint``). The final examples and
    their generation seed are never included.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    experiment_id: str
    run_id: str
    task_id: str
    task_hash: str | None
    policy_id: str
    suite_id: str
    suite_hash: str
    data_hash: str
    candidate_checkpoint_id: str
    control_checkpoint_id: str
    same_checkpoint: bool
    candidate: EvaluationReport
    control: EvaluationReport
    comparison: ComparisonReport
    attempts: int
    created_at: str
    management: Management = "managed"


# ---------------------------------------------------------------------------
# Experiment and model descriptions (agent-facing views)
# ---------------------------------------------------------------------------

ConstraintStatus = Literal["verified", "unverified"]


class SuiteSummary(_Frozen):
    """An evaluation suite as an experiment may use it: identity and preregistered configurations, no examples."""

    suite_id: str
    task_id: str
    split: SuiteSplit
    evaluator: ExtensionRef
    sample_counts: list[int]
    seeds: list[int]
    example_count: int
    cluster_unit: str | None = None
    cluster_count: int | None = None

    @classmethod
    def from_suite(cls, suite: EvaluationSuite) -> SuiteSummary:
        if suite.dataset is not None:
            unit, clusters = suite.dataset.cluster_unit, suite.dataset.cluster_count
        else:
            unit, clusters = suite.data.cluster_unit, len(set(suite.data.cluster_ids()))
        return cls(suite_id=suite.suite_id, task_id=suite.task_id, split=suite.split, evaluator=suite.evaluator,
                   sample_counts=list(suite.sample_counts), seeds=list(suite.seeds),
                   example_count=suite.example_count, cluster_unit=unit, cluster_count=clusters)


class OperationsSummary(_Frozen):
    """The operations an experiment's policy permits: public names mapped to exact operator versions."""

    analyses: dict[str, ExtensionRef]
    edits: dict[str, ExtensionRef]
    limits: dict[str, int]


class ModelSummary(BaseModel):
    """A registered checkpoint this run may use, without tensor data."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    adapter: ExtensionRef
    weights_hash: str
    parameter_count: int
    tensor_count: int
    input_description: str
    output_description: str
    manifest_uri: str


class ModelList(BaseModel):
    """One page of model summaries; pass ``next_cursor`` back as ``cursor`` for the next page."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelSummary]
    next_cursor: str | None


class ExperimentDescription(_Frozen):
    """Everything a controller needs to work on an experiment, inline.

    ``task`` is the registered task definition with its ``ObjectiveSpec``
    (``None`` when the operator registered none). ``suites`` lists the
    development suites the experiment may evaluate; ``evaluation_available``
    is false in ``weights_only`` mode. ``constraint_status`` is ``verified``
    only when the operator who started the interface attests that the
    controller has no other execution channel (Section 6.1 of the plan);
    otherwise it is ``unverified``.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    experiment: Experiment
    task: TaskDefinition | None
    operations: OperationsSummary
    evaluation_available: bool
    suites: list[SuiteSummary]
    checkpoints: list[ExperimentCheckpoint]
    policy_uri: str
    constraint_status: ConstraintStatus = "unverified"
    management: Management = "managed"


class TensorStatistics(_Frozen):
    """Whole-tensor statistics of one parameter, or why they are unavailable."""

    tensor: str
    shape: list[int]
    dtype: str
    statistics: dict[str, float | int] | None
    operator: ExtensionRef | None = None
    unavailable: str | None = None


class ModelInspection(_Frozen):
    """The manifest of a checkpoint an experiment owns, with whole-tensor statistics by layer."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    experiment_id: str
    checkpoint_id: str
    role: Literal["base", "candidate"]
    manifest: ModelManifest
    tensor_statistics: list[TensorStatistics]
    manifest_uri: str
    usage: Usage | None = None
    management: Management = "managed"


class PatchStepSummary(_Frozen):
    """One materialized step without its after-values: what it changed, not the resulting weights."""

    op: str
    operator: ExtensionRef
    tensor: str
    selection: list[AxisSlice]
    dtype: str
    before_hash: str
    after_hash: str
    changed_elements: int


class PatchDocument(_Frozen):
    """A stored patch as a run sees it: its preview, its frozen spec, and per-step summaries.

    The materialized after-values are omitted: with a known factor they
    reproduce the base weights of the edited region, so they would bypass
    the read budget. The exact patch remains an operator view
    (``Engine.read_patch``, ``wse patch``) for replay and export.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    preview: PatchPreview
    spec: FrozenPatchSpec
    steps: list[PatchStepSummary]
    result_weights_hash: str
