"""Operation, data, and resource restrictions.

``OperationPolicy`` maps public analysis kinds and edit operation aliases to
exact registered operator versions and bounds single reads, analyses, and
patches. ``ExperimentPolicy`` adds the immutable experiment mode and budget
limits; ``small_model_v1`` and ``small_model_weights_only_v1`` are built in.
Budget limits are recorded in every experiment's ledger; this milestone
enforces the candidate-commit limit, and the reservation layer enforces the
other counters.

Response-size rules (Section 4.1 of the plan): a read returns at most 2,048
elements, and the canonical compact JSON of a read or analysis result must not
exceed 32 KiB, because an MCP response carries it twice (``structuredContent``
and the text mirror) within a 64 KiB total. Explicit top-k values satisfy
``1 <= k <= min(64, element_count)``; an omitted ``k`` resolves to
``min(16, element_count)``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, model_validator

from wse.schemas import (
    ErrorCode,
    ExperimentMode,
    ExtensionRef,
    OperatorSpec,
    Usage,
    WSEError,
    _Frozen,
    canonical_json,
    json_hash,
)

MAX_READ_ELEMENTS = 2048
MAX_INLINE_JSON_BYTES = 32 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_TOP_K = 64
DEFAULT_TOP_K = 16
MAX_ANALYSIS_ELEMENTS = 16 * 1024 * 1024
MAX_PATCH_OPERATIONS = 32
MAX_PATCH_ELEMENTS = 65_536

ANALYSIS_KIND_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


class ResponseLimits(_Frozen):
    """Bounds on a single read, analysis, or patch.

    ``max_patch_elements`` bounds the total number of elements written by all
    operations of one patch.
    """

    max_read_elements: int = Field(default=MAX_READ_ELEMENTS, ge=1)
    max_inline_json_bytes: int = Field(default=MAX_INLINE_JSON_BYTES, ge=1024)
    max_analysis_elements: int = Field(default=MAX_ANALYSIS_ELEMENTS, ge=1)
    max_patch_operations: int = Field(default=MAX_PATCH_OPERATIONS, ge=1)
    max_patch_elements: int = Field(default=MAX_PATCH_ELEMENTS, ge=1)


def _default_edits() -> dict[str, ExtensionRef]:
    return {
        "set_slice": ExtensionRef.parse("set_slice@1"),
        "scale_slice": ExtensionRef.parse("scale_slice@1"),
    }


class OperationPolicy(_Frozen):
    """Permitted analysis kinds, edit operations, and response limits.

    ``analyses`` maps each public kind (for example ``"stats"``) and ``edits``
    each public operation alias (for example ``"scale_slice"``) to the exact
    registered operator version that implements it. Names absent from the
    maps are denied; versions are never substituted.
    """

    analyses: dict[str, ExtensionRef]
    edits: dict[str, ExtensionRef] = Field(default_factory=_default_edits)
    limits: ResponseLimits = Field(default_factory=ResponseLimits)

    def edit_operator(self, op: Any) -> ExtensionRef:
        if not isinstance(op, str) or op not in self.edits:
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"edit operation {op!r} is not permitted; permitted operations: {sorted(self.edits)}",
                details={"op": str(op), "permitted_operations": sorted(self.edits)},
            )
        return self.edits[op]

    def with_edit(self, op: str, operator: str | ExtensionRef) -> OperationPolicy:
        """Return a copy that also permits the edit alias ``op`` through ``operator``."""
        if not re.fullmatch(ANALYSIS_KIND_PATTERN, op):
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"edit operation {op!r} must match {ANALYSIS_KIND_PATTERN}")
        return self.model_copy(update={"edits": {**self.edits, op: ExtensionRef.parse(operator)}})

    def analysis_operator(self, kind: Any) -> ExtensionRef:
        if not isinstance(kind, str) or kind not in self.analyses:
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"analysis kind {kind!r} is not permitted; permitted kinds: {sorted(self.analyses)}",
                details={"kind": str(kind), "permitted_kinds": sorted(self.analyses)},
            )
        return self.analyses[kind]

    def with_analysis(self, kind: str, operator: str | ExtensionRef) -> OperationPolicy:
        """Return a copy that also permits ``kind`` through ``operator``."""
        if not re.fullmatch(ANALYSIS_KIND_PATTERN, kind):
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"analysis kind {kind!r} must match {ANALYSIS_KIND_PATTERN}")
        return self.model_copy(update={"analyses": {**self.analyses, kind: ExtensionRef.parse(operator)}})

    def with_limits(self, **limits: int) -> OperationPolicy:
        return self.model_copy(
            update={"limits": ResponseLimits.model_validate({**self.limits.model_dump(), **limits})}
        )


def default_operation_policy() -> OperationPolicy:
    """The M1 operation policy: stats, norm, and top-k; set_slice and scale_slice."""
    return OperationPolicy(
        analyses={
            "stats": ExtensionRef.parse("tensor_stats@1"),
            "norm": ExtensionRef.parse("tensor_norm@1"),
            "top_k": ExtensionRef.parse("tensor_top_k@1"),
        }
    )


def require_capabilities(
    capabilities: Any, spec: OperatorSpec, *, adapter: str, always: Sequence[str] = ()
) -> None:
    """Reject an operator whose capabilities the adapter does not advertise.

    ``always`` names capabilities the engine requires for the operator's
    effect regardless of what the spec declares (for example ``weights.read``
    for every analysis, because the engine reads the weights it passes in).
    """
    required = sorted(set(spec.required_capabilities) | set(always))
    missing = [name for name in required if not capabilities.supports(name)]
    if missing:
        raise WSEError(
            ErrorCode.UNSUPPORTED_CAPABILITY,
            f"operator {spec.ref} requires capabilities {missing} that adapter {adapter} does not support",
            details={"operator": str(spec.ref), "adapter": adapter, "missing_capabilities": missing},
        )


def json_size(payload: Mapping[str, Any]) -> int:
    """Size in bytes of the canonical compact JSON of ``payload``."""
    return len(canonical_json(payload).encode("utf-8"))


def contract_usage(spec: OperatorSpec, *, compute_seconds: float | None = None, calls: int = 1) -> Usage:
    """Per-call usage charged under ``spec.usage``: its counter and forward calls."""
    return Usage.model_validate(
        {
            spec.usage.counter: spec.usage.units_per_call * calls,
            "model_forward_calls": spec.usage.model_forward_calls,
            "compute_seconds": compute_seconds,
        }
    )


# ---------------------------------------------------------------------------
# Experiment policies and budget limits
# ---------------------------------------------------------------------------

POLICY_ID_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
SMALL_MODEL_POLICY_ID = "small_model_v1"
WEIGHTS_ONLY_POLICY_ID = "small_model_weights_only_v1"

# Ledger counters, in the order they are reported. ``examples_per_batch`` and
# ``seconds_per_operation`` are per-operation limits, not cumulative counters.
LEDGER_COUNTERS = (
    "candidate_commits",
    "analysis_calls",
    "tensor_read_calls",
    "evaluation_batches",
    "evaluated_examples",
    "model_forward_calls",
    "compute_seconds",
)


class BudgetLimits(_Frozen):
    """Per-experiment allowances (Section 6.2 of the plan)."""

    candidate_commits: int = Field(default=20, ge=0)
    analysis_calls: int = Field(default=50, ge=0)
    tensor_read_calls: int = Field(default=100, ge=0)
    evaluation_batches: int = Field(default=10, ge=0)
    examples_per_batch: int = Field(default=64, ge=0)
    evaluated_examples: int = Field(default=640, ge=0)
    model_forward_calls: int = Field(default=10, ge=0)
    seconds_per_operation: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    compute_seconds: float = Field(default=300.0, ge=0, allow_inf_nan=False)

    def ledger_limits(self) -> dict[str, int | float]:
        return {counter: getattr(self, counter) for counter in LEDGER_COUNTERS}


class ExperimentPolicy(_Frozen):
    """An immutable experiment policy: mode, budget limits, and permitted operations.

    ``mode`` cannot be overridden when an experiment is created. A
    ``weights_only`` policy must grant no evaluation batches, examples, or
    forward calls.
    """

    schema_version: Literal["1"] = "1"
    policy_id: str = Field(pattern=POLICY_ID_PATTERN)
    mode: ExperimentMode
    budget: BudgetLimits = Field(default_factory=BudgetLimits)
    operations: OperationPolicy = Field(default_factory=lambda: default_operation_policy())

    @model_validator(mode="after")
    def _weights_only_has_no_inference(self) -> ExperimentPolicy:
        if self.mode == "weights_only":
            granted = {
                name: getattr(self.budget, name)
                for name in ("evaluation_batches", "examples_per_batch", "evaluated_examples", "model_forward_calls")
                if getattr(self.budget, name) != 0
            }
            if granted:
                raise ValueError(f"weights_only policies must not grant inference allowances: {granted}")
        return self

    @property
    def policy_hash(self) -> str:
        return json_hash(self.model_dump(mode="json"))


def builtin_experiment_policies() -> dict[str, ExperimentPolicy]:
    """``small_model_v1`` (bounded_eval) and ``small_model_weights_only_v1`` (weights_only)."""
    bounded = ExperimentPolicy(policy_id=SMALL_MODEL_POLICY_ID, mode="bounded_eval")
    weights_only = ExperimentPolicy(
        policy_id=WEIGHTS_ONLY_POLICY_ID,
        mode="weights_only",
        budget=BudgetLimits(
            candidate_commits=1,
            evaluation_batches=0,
            examples_per_batch=0,
            evaluated_examples=0,
            model_forward_calls=0,
        ),
    )
    return {policy.policy_id: policy for policy in (bounded, weights_only)}
