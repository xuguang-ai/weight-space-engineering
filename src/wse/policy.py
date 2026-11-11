"""Operation, data, and resource restrictions.

``OperationPolicy`` maps public analysis kinds and edit operation aliases to
exact registered operator versions and bounds single reads, analyses, and
patches. ``ExperimentPolicy`` adds the immutable experiment mode and budget
limits; ``small_model_v1``, ``small_model_weights_only_v1``, and
``qwen_loop_v1`` are built in.
Budget limits are recorded in every experiment's ledger and enforced by
``wse.budget``: candidate commits, tensor reads, and analyses are charged when
they run; evaluation batches, examples, forward calls, and compute seconds
are reserved before a job is queued and settled by the worker. Every
synchronous managed operation also charges its elapsed compute seconds.
``examples_per_batch`` and ``seconds_per_operation`` bound each job.

Language models (M2 plan Section 5.5) add four ledger counters:
``generation_passes`` and ``generated_tokens`` (development evaluations of
generation suites, reserved and settled like M1's jobs), and ``probe_calls``
and ``teacher_forced_tokens`` (activation probes). Their per-operation limits
are ``problems_per_generation_pass`` and ``new_tokens_per_output`` for a pass,
and ``texts_per_probe_call`` and ``tokens_per_probe_text`` for a probe. All
eight are optional fields: unset, they are absent from a policy's JSON, so the
M1 policies keep their hashes, and an unset counter is recorded in the ledger
with limit 0, which permits nothing. A generation pass is metered in passes
and generated tokens (one decoding step of one output is one token), not in
M1's evaluation batches, examples, or forward calls.

Response-size rules (Section 4.1 of the plan): a read returns at most 2,048
elements, and the canonical compact JSON of a read or analysis result must not
exceed 32 KiB, because an MCP response carries it twice (``structuredContent``
and the text mirror) within a 64 KiB total. Those two copies also carry the
tool-response envelope (summary, URIs, cumulative usage) and the JSON-RPC
framing, so a result may use at most ``max_inline_json_bytes`` less
``RESPONSE_ENVELOPE_BYTES`` (1 KiB): 31 KiB under the default policy, which
keeps the largest permitted response within 64 KiB. Because the text copy
escapes quotes and backslashes, the bounded views of M2 (previews,
inspections, candidates, and reports of open-weight models) measure both
copies (``response_size``) against twice that limit. Explicit top-k values satisfy
``1 <= k <= min(64, element_count)``; an omitted ``k`` resolves to
``min(16, element_count)``.

Protections (M2 plan Sections 2.2, 5.1, and 5.5): every policy refuses edits
of a tensor whose role the model's family descriptor protects
(``require_unprotected``, ``POLICY_DENIED``), and edits of alias-group members
(``wse.patches``). Protections are the descriptor's data, not policy fields,
so the built-in policies and their hashes are unchanged. One protection is a
policy option, because it depends on the operation rather than the tensor:
``OperationPolicy.refuse_whole_tensor_scaling`` refuses edits that scale a
whole tensor by one constant, as the article's ``save_edit`` did. A
``scale_slice`` whose selection covers its entire tensor is refused at once
(``require_partial_scaling``); managed previews also refuse any patch whose
net effect is such a scaling of a tensor of the experiment's base
checkpoint, however its operations or earlier candidates produce it
(``ExperimentService._refuse_net_whole_tensor_scaling``, with
``constant_factor_bounds``), and name no factor. It is off by default and,
while unset, absent from the policy's JSON, so stored policies keep their
hashes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

from wse.schemas import (
    AxisSlice,
    ErrorCode,
    ExperimentMode,
    ExtensionRef,
    OperatorSpec,
    Usage,
    WSEError,
    _Frozen,
    _optional,
    canonical_json,
    json_hash,
)

if TYPE_CHECKING:
    from wse.adapters.families import FamilyDescriptor

MAX_READ_ELEMENTS = 2048
MAX_INLINE_JSON_BYTES = 32 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
RESPONSE_ENVELOPE_BYTES = 1024
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
    max_inline_json_bytes: int = Field(default=MAX_INLINE_JSON_BYTES, ge=2048)
    max_analysis_elements: int = Field(default=MAX_ANALYSIS_ELEMENTS, ge=1)
    max_patch_operations: int = Field(default=MAX_PATCH_OPERATIONS, ge=1)
    max_patch_elements: int = Field(default=MAX_PATCH_ELEMENTS, ge=1)

    @property
    def max_result_json_bytes(self) -> int:
        """Largest canonical JSON of one read or analysis result: the inline limit less the envelope allowance."""
        return self.max_inline_json_bytes - RESPONSE_ENVELOPE_BYTES


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
    maps are denied; versions are never substituted. A managed preview also
    accepts an operation named by the exact reference of an operator that
    ``edits`` lists (``"scale_slice@1"``), and no other.

    ``refuse_whole_tensor_scaling``, when true, refuses edits that scale a
    whole tensor by one constant (``require_partial_scaling``). It is unset,
    and so off, by default; unset, it is absent from the policy's JSON.
    """

    analyses: dict[str, ExtensionRef]
    edits: dict[str, ExtensionRef] = Field(default_factory=_default_edits)
    limits: ResponseLimits = Field(default_factory=ResponseLimits)
    refuse_whole_tensor_scaling: bool | None = _optional(
        description="true refuses a scale_slice whose selection covers its entire tensor (POLICY_DENIED); unset or "
        "false permits it")

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


def require_unprotected(tensor: str, family: FamilyDescriptor | None) -> None:
    """Refuse an edit of ``tensor`` when its role is one that ``family`` protects (``POLICY_DENIED``).

    Every run of an edit operator applies the protections of the model's
    family descriptor, under every policy and through every entry point that
    runs one: direct ``apply_operator`` and ``materialize_patch``, managed
    previews, and the tools built on them. Writing recorded values runs no
    edit operator and checks no protection: exact replay (``replay_patch``,
    which ``apply_patch`` and export replay use) and the operator-only import
    of a published patch (``Engine.import_patch``) write the stored values as
    recorded. That exemption is deliberate, so that published edits are
    reproduced exactly.

    A model without a descriptor (Level 0 or M1) has no roles and so no
    protected tensors. An operator who needs to edit a protected role
    registers a descriptor version without that protection and imports the
    snapshot with it; the new version is recorded in the model's identity.
    """
    if family is None:
        return
    found = family.protected_role(tensor)
    if found is None:
        return
    role, layer = found
    protected = sorted(family.protected_roles)
    editable = sorted({rule.role for rule in family.roles} - set(protected))
    where = "" if layer is None else f" (layer {layer})"
    advice = (f"Edit a tensor of an unprotected role instead: {', '.join(editable)}" if editable
              else "Every role of this family is protected")
    raise WSEError(
        ErrorCode.POLICY_DENIED,
        f"edits of tensor {tensor!r} are refused: its role {role}{where} is protected by family {family.ref}, "
        f"which protects {', '.join(protected)}. {advice}.",
        details={"tensor": tensor, "role": role, "layer": layer, "family": str(family.ref),
                 "protected_roles": protected, "editable_roles": editable},
    )


def require_partial_scaling(policy: OperationPolicy, operator: ExtensionRef, tensor: str, shape: Sequence[int],
                            selection: Sequence[AxisSlice]) -> None:
    """Refuse a whole-tensor ``scale_slice`` when the policy sets ``refuse_whole_tensor_scaling`` (``POLICY_DENIED``).

    A ``scale_slice`` (any version) whose selection covers every element of
    its tensor scales the tensor by one constant, whatever the factor, which
    changes only the tensor's overall scale (M2 plan Section 5.5). Any proper
    part of the tensor may still be scaled, and no other operator is affected.
    """
    if not policy.refuse_whole_tensor_scaling or operator.extension_id != "scale_slice":
        return
    if any((entry.start, entry.stop) != (0, size) for entry, size in zip(selection, shape)):
        return
    others = sorted(alias for alias, ref in policy.edits.items() if ref.extension_id != "scale_slice")
    alternatives = f", or use another permitted operation: {', '.join(others)}" if others else ""
    raise WSEError(
        ErrorCode.POLICY_DENIED,
        f"scaling the whole of tensor {tensor!r} {list(shape)} by one factor is refused by this experiment's operation "
        f"policy (refuse_whole_tensor_scaling): it changes only the tensor's overall scale. Scale a proper part of "
        f"the tensor instead, such as some of its rows or columns or one attention head's range{alternatives}.",
        details={"tensor": tensor, "shape": list(shape), "operator": str(operator),
                 "policy_option": "refuse_whole_tensor_scaling", "permitted_operations": sorted(policy.edits)},
    )


def constant_factor_bounds(base: Any, candidate: Any) -> tuple[float, float] | None:
    """The factors ``c`` with ``candidate == round(c * base)`` in every element, as ``(lower, upper)``, or ``None``.

    ``base`` and ``candidate`` are tensors of one shape; rounding is to
    ``candidate``'s dtype (BF16, F16, or F32). Each element admits the
    interval of factors whose product rounds to its stored value (half-way to
    its neighbors, ties included, and widened by a float32 rounding, as the
    edit operators multiply in float32 first); the tensor is a constant
    scaling exactly when the intervals of all its elements intersect, and
    every factor of the intersection reproduces it. A zero base element
    admits any factor if its candidate is zero and none otherwise; an all-zero
    base admits none. Computed in float64, so a factor that no single element
    pins down (for example 0.7 in BF16) is still found.
    """
    import torch

    values = candidate.detach().reshape(-1).cpu()
    original = base.detach().reshape(-1).cpu().to(torch.float64)
    if values.numel() == 0 or values.numel() != original.numel():
        return None
    exact = values.to(torch.float64)
    if not bool(torch.isfinite(exact).all()) or not bool(torch.isfinite(original).all()):
        return None
    up = torch.nextafter(values, torch.full_like(values, math.inf)).to(torch.float64)
    down = torch.nextafter(values, torch.full_like(values, -math.inf)).to(torch.float64)
    slack = exact.abs() * 2.0 ** -24
    low, high = (exact + down) / 2 - slack, (exact + up) / 2 + slack
    zero = original == 0
    if bool((zero & (exact != 0)).any()):
        return None  # c * 0 is 0 for every finite c
    if bool(zero.all()):
        return None
    scale, low, high = original[~zero], low[~zero], high[~zero]
    lower = float(torch.where(scale > 0, low / scale, high / scale).max())
    upper = float(torch.where(scale > 0, high / scale, low / scale).min())
    return (lower, upper) if lower <= upper else None


def json_size(payload: Mapping[str, Any]) -> int:
    """Size in bytes of the canonical compact JSON of ``payload``."""
    return len(canonical_json(payload).encode("utf-8"))


def response_size(payload: Any) -> int:
    """The bytes ``payload`` takes in an MCP tool result, which carries it twice.

    Once as its canonical compact JSON (``structuredContent``) and once as
    that text inside a JSON string (the text mirror), where every quote,
    backslash, and control character is escaped. For numbers the two copies
    are about equal, as M1's rule (twice ``max_result_json_bytes``) assumes;
    text-heavy results, such as tails of generated outputs or manifests, take
    more. The bounded agent-facing views (``wse.experiments``,
    ``wse.evaluation``) keep it within twice ``max_result_json_bytes``.
    """
    text = canonical_json(payload)
    return len(text.encode("utf-8")) + len(json.dumps(text, ensure_ascii=False).encode("utf-8"))


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
QWEN_LOOP_POLICY_ID = "qwen_loop_v1"

# Ledger counters, in the order they are reported. ``examples_per_batch``,
# ``seconds_per_operation``, and the per-pass and per-probe limits are
# per-operation limits, not cumulative counters.
LEDGER_COUNTERS = (
    "candidate_commits",
    "patch_previews",
    "analysis_calls",
    "tensor_read_calls",
    "evaluation_batches",
    "evaluated_examples",
    "model_forward_calls",
    "compute_seconds",
    "generation_passes",
    "generated_tokens",
    "probe_calls",
    "teacher_forced_tokens",
)
# Each cumulative counter of language models and the per-operation limits that a policy granting it must set.
GENERATION_LIMITS = ("problems_per_generation_pass", "new_tokens_per_output", "generated_tokens")
PROBE_LIMITS = ("texts_per_probe_call", "tokens_per_probe_text", "teacher_forced_tokens")
INFERENCE_LIMITS = ("evaluation_batches", "examples_per_batch", "evaluated_examples", "model_forward_calls",
                    "generation_passes", *GENERATION_LIMITS, "probe_calls", *PROBE_LIMITS)


class BudgetLimits(_Frozen):
    """Per-experiment allowances (Section 6.2 of the plan; M2 plan Section 5.5).

    ``patch_previews`` is counted separately because a preview's deltas
    reveal the base values of the edited region: two single-element previews
    determine one weight exactly.

    The language-model fields are optional, and unset (``None``) means not
    permitted: ``generation_passes`` and ``generated_tokens`` are cumulative
    counters, and a pass evaluates at most ``problems_per_generation_pass``
    problems, each output at most ``new_tokens_per_output`` new tokens;
    ``probe_calls`` and ``teacher_forced_tokens`` likewise, with at most
    ``texts_per_probe_call`` texts of ``tokens_per_probe_text`` tokens per
    probe. A budget that grants passes or probes must set their limits.
    """

    candidate_commits: int = Field(default=20, ge=0)
    patch_previews: int = Field(default=50, ge=0)
    analysis_calls: int = Field(default=50, ge=0)
    tensor_read_calls: int = Field(default=100, ge=0)
    evaluation_batches: int = Field(default=10, ge=0)
    examples_per_batch: int = Field(default=64, ge=0)
    evaluated_examples: int = Field(default=640, ge=0)
    model_forward_calls: int = Field(default=10, ge=0)
    seconds_per_operation: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    compute_seconds: float = Field(default=300.0, ge=0, allow_inf_nan=False)
    generation_passes: int | None = _optional(ge=0, description="development evaluations of generation suites")
    problems_per_generation_pass: int | None = _optional(ge=0, description="problems one generation pass may "
                                                         "evaluate")
    new_tokens_per_output: int | None = _optional(ge=0, description="new tokens one generated output may have "
                                                  "(the suite's max_new_tokens)")
    generated_tokens: int | None = _optional(ge=0, description="new tokens generated by all passes")
    probe_calls: int | None = _optional(ge=0, description="activation probes")
    texts_per_probe_call: int | None = _optional(ge=0, description="texts one probe may teacher-force")
    tokens_per_probe_text: int | None = _optional(ge=0, description="tokens one probed text may have")
    teacher_forced_tokens: int | None = _optional(ge=0, description="tokens teacher-forced by all probes")

    @model_validator(mode="after")
    def _limits_of_granted_operations(self) -> BudgetLimits:
        for counter, limits in (("generation_passes", GENERATION_LIMITS), ("probe_calls", PROBE_LIMITS)):
            missing = [name for name in limits if not getattr(self, counter) in (None, 0) and not getattr(self, name)]
            if missing:
                raise ValueError(f"a budget that grants {counter} must also set {', '.join(missing)} above 0")
        return self

    def limit(self, name: str) -> int | float:
        """The limit ``name`` sets; an unset (optional) limit is 0, which permits nothing."""
        value = getattr(self, name)
        return 0 if value is None else value

    def ledger_limits(self) -> dict[str, int | float]:
        """The limit of every ledger counter, as recorded in an experiment's ledger (unset counters are 0)."""
        return {counter: self.limit(counter) for counter in LEDGER_COUNTERS}


class ExperimentPolicy(_Frozen):
    """An immutable experiment policy: mode, budget limits, and permitted operations.

    ``mode`` cannot be overridden when an experiment is created. A
    ``weights_only`` policy must grant no evaluation batches, examples,
    forward calls, generation passes, or probes.
    """

    schema_version: Literal["1"] = "1"
    policy_id: str = Field(pattern=POLICY_ID_PATTERN)
    mode: ExperimentMode
    budget: BudgetLimits = Field(default_factory=BudgetLimits)
    operations: OperationPolicy = Field(default_factory=lambda: default_operation_policy())

    @model_validator(mode="after")
    def _weights_only_has_no_inference(self) -> ExperimentPolicy:
        if self.mode == "weights_only":
            granted = {name: getattr(self.budget, name) for name in INFERENCE_LIMITS
                       if getattr(self.budget, name) not in (None, 0)}
            if granted:
                raise ValueError(f"weights_only policies must not grant inference allowances: {granted}")
        return self

    @property
    def policy_hash(self) -> str:
        return json_hash(self.model_dump(mode="json"))

    @property
    def evaluation_available(self) -> bool:
        """Whether runs may submit development evaluations: forward-batch suites, generation suites, or both."""
        return self.mode == "bounded_eval" and (self.budget.evaluation_batches > 0
                                                or self.budget.limit("generation_passes") > 0)


# The policy of the Qwen loop task (M2 plan Section 5.5); Task 13 may lower its limits from Task 7's measurements.
QWEN_LOOP_GENERATION_PASSES = 10
QWEN_LOOP_PROBLEMS_PER_PASS = 32
QWEN_LOOP_NEW_TOKENS = 8192
QWEN_LOOP_PROBE_CALLS = 25
QWEN_LOOP_TEXTS_PER_PROBE = 64
QWEN_LOOP_TOKENS_PER_TEXT = 1024
# 2**28 elements: Qwen3.5-0.8B's largest tensor, its tied embedding (248,320 x 1,024 = 254,279,680), and
# Llama-3.2-1B-Instruct's (128,256 x 2,048 = 262,668,288) can be analyzed whole.
QWEN_LOOP_MAX_ANALYSIS_ELEMENTS = 1 << 28


def qwen_loop_policy() -> ExperimentPolicy:
    """``qwen_loop_v1``: generation passes, probes, and the M2 edits and analyses (M2 plan Section 5.5).

    * 10 generation passes of at most 32 practice problems, each output at
      most 8,192 new tokens, so at most 10 x 32 x 8,192 generated tokens;
    * 25 probe calls of at most 64 texts of at most 1,024 tokens, so at most
      25 x 64 x 1,024 teacher-forced tokens;
    * 50 analyses, 100 slice reads, 50 previews, and 20 candidate commits;
    * 6 hours of computation, at most 2 per operation;
    * no M1 forward-batch evaluation: a language model is evaluated by
      generation passes, and its forward passes are metered as tokens;
    * edits ``scale_slice``, ``set_slice``, ``fill_slice``, ``project_out``,
      and ``add_low_rank``; analyses ``stats``, ``norm``, ``top_k``,
      ``tensor_norms``, ``svd_top_k``, and ``direction_projection``, of
      tensors of up to 2**28 elements;
    * whole-tensor scaling refused. Alias groups and the roles a family
      descriptor protects (embeddings, output head, final norm) are refused
      under every policy.
    """
    operations = OperationPolicy(
        analyses={
            "stats": ExtensionRef.parse("tensor_stats@1"),
            "norm": ExtensionRef.parse("tensor_norm@1"),
            "top_k": ExtensionRef.parse("tensor_top_k@1"),
            "tensor_norms": ExtensionRef.parse("tensor_norms@1"),
            "svd_top_k": ExtensionRef.parse("svd_top_k@1"),
            "direction_projection": ExtensionRef.parse("direction_projection@1"),
        },
        edits={
            "scale_slice": ExtensionRef.parse("scale_slice@1"),
            "set_slice": ExtensionRef.parse("set_slice@1"),
            "fill_slice": ExtensionRef.parse("fill_slice@1"),
            "project_out": ExtensionRef.parse("project_out@1"),
            "add_low_rank": ExtensionRef.parse("add_low_rank@1"),
        },
        limits=ResponseLimits(max_analysis_elements=QWEN_LOOP_MAX_ANALYSIS_ELEMENTS),
        refuse_whole_tensor_scaling=True,
    )
    budget = BudgetLimits(
        candidate_commits=20,
        patch_previews=50,
        analysis_calls=50,
        tensor_read_calls=100,
        evaluation_batches=0,
        examples_per_batch=0,
        evaluated_examples=0,
        model_forward_calls=0,
        seconds_per_operation=2 * 3600.0,
        compute_seconds=6 * 3600.0,
        generation_passes=QWEN_LOOP_GENERATION_PASSES,
        problems_per_generation_pass=QWEN_LOOP_PROBLEMS_PER_PASS,
        new_tokens_per_output=QWEN_LOOP_NEW_TOKENS,
        generated_tokens=QWEN_LOOP_GENERATION_PASSES * QWEN_LOOP_PROBLEMS_PER_PASS * QWEN_LOOP_NEW_TOKENS,
        probe_calls=QWEN_LOOP_PROBE_CALLS,
        texts_per_probe_call=QWEN_LOOP_TEXTS_PER_PROBE,
        tokens_per_probe_text=QWEN_LOOP_TOKENS_PER_TEXT,
        teacher_forced_tokens=QWEN_LOOP_PROBE_CALLS * QWEN_LOOP_TEXTS_PER_PROBE * QWEN_LOOP_TOKENS_PER_TEXT,
    )
    return ExperimentPolicy(policy_id=QWEN_LOOP_POLICY_ID, mode="bounded_eval", budget=budget, operations=operations)


def builtin_experiment_policies() -> dict[str, ExperimentPolicy]:
    """``small_model_v1`` (bounded_eval), ``small_model_weights_only_v1`` (weights_only), and ``qwen_loop_v1``."""
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
    return {policy.policy_id: policy for policy in (bounded, weights_only, qwen_loop_policy())}
