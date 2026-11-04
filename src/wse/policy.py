"""Operation, data, and resource restrictions.

This module holds the parts of a policy that bound weight reading and
analysis: which public analysis kinds map to which registered operator
versions, the response-size rules, and capability checks. Later milestones
add experiment modes and budget limits here.

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
from typing import Any

from pydantic import Field

from wse.schemas import (
    ErrorCode,
    ExtensionRef,
    OperatorSpec,
    Usage,
    WSEError,
    _Frozen,
    canonical_json,
)

MAX_READ_ELEMENTS = 2048
MAX_INLINE_JSON_BYTES = 32 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_TOP_K = 64
DEFAULT_TOP_K = 16
MAX_ANALYSIS_ELEMENTS = 16 * 1024 * 1024

ANALYSIS_KIND_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


class ResponseLimits(_Frozen):
    """Bounds on a single read or analysis."""

    max_read_elements: int = Field(default=MAX_READ_ELEMENTS, ge=1)
    max_inline_json_bytes: int = Field(default=MAX_INLINE_JSON_BYTES, ge=1024)
    max_analysis_elements: int = Field(default=MAX_ANALYSIS_ELEMENTS, ge=1)


class OperationPolicy(_Frozen):
    """Permitted analysis kinds and response limits.

    ``analyses`` maps each public kind (for example ``"stats"``) to the exact
    registered operator version that implements it. Kinds absent from the map
    are denied; versions are never substituted.
    """

    analyses: dict[str, ExtensionRef]
    limits: ResponseLimits = Field(default_factory=ResponseLimits)

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
    """The M1 analysis policy: stats, norm, and top-k only."""
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
