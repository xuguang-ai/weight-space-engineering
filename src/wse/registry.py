"""Explicit, versioned extension registration.

The operator registers each adapter, operator, evaluator, strategy, family
descriptor, generation profile, or grader under an exact ``(kind, extension_id, version)``. Resolution is exact: an unknown or
missing version is rejected rather than replaced by the latest one. A
reference without a version (``"mlp"``) resolves only through an alias the
operator declares with ``register_alias`` (the built-in registry maps ``mlp``
to ``mlp@1``), never to the latest registered version; callers record the
resolved exact reference. Agents invoke registered extensions; they never
install them.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Literal, get_args

from wse.schemas import EXTENSION_ID_PATTERN, ErrorCode, ExtensionRef, WSEError

ExtensionKind = Literal["adapter", "operator", "evaluator", "strategy", "family", "profile", "grader"]
EXTENSION_KINDS: tuple[str, ...] = get_args(ExtensionKind)

Validator = Callable[[str, str, Any], None]


def _validate_adapter(extension_id: str, version: str, implementation: Any) -> None:
    from wse.adapters.base import REQUIRED_ADAPTER_METHODS

    missing = [
        name for name in REQUIRED_ADAPTER_METHODS if not callable(getattr(implementation, name, None))
    ]
    if missing:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"adapter {extension_id}@{version} does not implement the ModelAdapter contract; "
            f"missing methods: {missing}",
            details={"missing_methods": missing},
        )
    declared = (getattr(implementation, "adapter_id", None), getattr(implementation, "version", None))
    if declared != (extension_id, version):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"adapter declares {declared[0]}@{declared[1]} but is being registered as "
            f"{extension_id}@{version}",
        )
    from wse.adapters.base import AdapterCapabilities

    if not isinstance(implementation.capabilities(), AdapterCapabilities):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"adapter {extension_id}@{version}.capabilities() must return AdapterCapabilities",
        )


def _validate_operator(extension_id: str, version: str, implementation: Any) -> None:
    from wse.operators.base import validate_operator

    validate_operator(extension_id, version, implementation)


def _validate_evaluator(extension_id: str, version: str, implementation: Any) -> None:
    from wse.evaluators.base import validate_evaluator

    validate_evaluator(extension_id, version, implementation)


def _validate_strategy(extension_id: str, version: str, implementation: Any) -> None:
    from wse.strategies.base import validate_strategy

    validate_strategy(extension_id, version, implementation)


def _validate_family(extension_id: str, version: str, implementation: Any) -> None:
    from wse.adapters.families import validate_family

    validate_family(extension_id, version, implementation)


def _validate_profile(extension_id: str, version: str, implementation: Any) -> None:
    from wse.schemas import GenerationProfile

    if not isinstance(implementation, GenerationProfile):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"profile {extension_id}@{version} must be a GenerationProfile")
    if (implementation.profile_id, implementation.version) != (extension_id, version):
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"profile declares {implementation.ref} but is being registered as {extension_id}@{version}")


def _validate_grader(extension_id: str, version: str, implementation: Any) -> None:
    from wse.evaluators.graders import validate_grader

    validate_grader(extension_id, version, implementation)


# Per-kind compatibility checks run at registration time.
KIND_VALIDATORS: dict[str, Validator] = {
    "adapter": _validate_adapter,
    "operator": _validate_operator,
    "evaluator": _validate_evaluator,
    "strategy": _validate_strategy,
    "family": _validate_family,
    "profile": _validate_profile,
    "grader": _validate_grader,
}


class Registry:
    """A small in-process registry with explicit registrations."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], Any] = {}
        self._aliases: dict[tuple[str, str], ExtensionRef] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(kind: str, extension_id: str, version: str) -> tuple[str, str, str]:
        if kind not in EXTENSION_KINDS:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"unknown extension kind {kind!r}; expected one of {list(EXTENSION_KINDS)}",
            )
        ref = ExtensionRef.parse(f"{extension_id}@{version}")
        return kind, ref.extension_id, ref.version

    def register(self, kind: str, extension_id: str, version: str, implementation: Any) -> None:
        """Register ``implementation``; duplicate registrations are rejected."""
        key = self._key(kind, extension_id, version)
        validator = KIND_VALIDATORS.get(kind)
        if validator is not None:
            validator(key[1], key[2], implementation)
        with self._lock:
            if key in self._entries:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"{kind} {extension_id}@{version} is already registered; "
                    "register a new version instead of replacing an existing one",
                )
            self._entries[key] = implementation

    def resolve(self, kind: str, extension_id: str, version: str) -> Any:
        """Return the implementation registered under exactly this version."""
        key = self._key(kind, extension_id, version)
        with self._lock:
            implementation = self._entries.get(key)
            if implementation is not None:
                return implementation
            versions = sorted(v for (k, i, v) in self._entries if k == kind and i == extension_id)
        if versions:
            message = (
                f"{kind} {extension_id}@{version} is not registered; registered versions: "
                f"{versions}. Versions are never substituted."
            )
        else:
            message = f"{kind} {extension_id!r} is not registered"
        raise WSEError(
            ErrorCode.NOT_FOUND,
            message,
            details={"kind": kind, "extension_id": extension_id, "version": version,
                     "registered_versions": versions},
        )

    def register_alias(self, kind: str, alias: str, target: str | ExtensionRef) -> None:
        """Declare that the version-less reference ``alias`` means exactly ``target``.

        ``target`` must be an exact, registered reference. Aliases are
        immutable: redeclaring one with a different target is rejected.
        """
        self._key(kind, "x", "1")  # validates the kind
        if not isinstance(alias, str) or not EXTENSION_ID_PATTERN.fullmatch(alias):
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"alias {alias!r} must match {EXTENSION_ID_PATTERN.pattern}")
        ref = ExtensionRef.parse(target)
        self.resolve(kind, ref.extension_id, ref.version)
        with self._lock:
            existing = self._aliases.get((kind, alias))
            if existing is not None and existing != ref:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"{kind} alias {alias!r} already means {existing}; aliases cannot be redirected",
                    details={"alias": alias, "target": str(existing)},
                )
            self._aliases[(kind, alias)] = ref

    def aliases(self, kind: str) -> dict[str, ExtensionRef]:
        self._key(kind, "x", "1")
        with self._lock:
            return {alias: ref for (k, alias), ref in sorted(self._aliases.items()) if k == kind}

    def resolve_reference(self, kind: str, reference: str | ExtensionRef) -> ExtensionRef:
        """Return the exact, registered reference that ``reference`` names.

        An exact reference (``"mlp@1"``) must be registered; a version-less
        one (``"mlp"``) must be a declared alias. Unknown aliases and unknown
        versions are rejected with ``NOT_FOUND``; nothing is substituted.
        """
        if isinstance(reference, str) and "@" not in reference and EXTENSION_ID_PATTERN.fullmatch(reference):
            self._key(kind, reference, "1")
            with self._lock:
                ref = self._aliases.get((kind, reference))
                declared = sorted(alias for (k, alias) in self._aliases if k == kind)
            if ref is None:
                raise WSEError(
                    ErrorCode.NOT_FOUND,
                    f"{kind} {reference!r} has no declared alias; use an exact reference '<id>@<version>' "
                    f"or one of the declared aliases {declared}",
                    details={"kind": kind, "reference": reference, "declared_aliases": declared},
                )
        else:
            ref = ExtensionRef.parse(reference)
        self.resolve(kind, ref.extension_id, ref.version)
        return ref

    def resolve_ref(self, kind: str, ref: str | ExtensionRef) -> Any:
        parsed = ExtensionRef.parse(ref)
        return self.resolve(kind, parsed.extension_id, parsed.version)

    def list(self, kind: str) -> list[ExtensionRef]:
        if kind not in EXTENSION_KINDS:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"unknown extension kind {kind!r}")
        with self._lock:
            keys = sorted(k for k in self._entries if k[0] == kind)
        return [ExtensionRef(extension_id=i, version=v) for (_, i, v) in keys]


def builtin_registry() -> Registry:
    """Return a new registry containing the built-in extensions.

    M1's extensions, plus the generic Hugging Face causal-LM adapter
    ``hf_causal_lm@1``, the built-in family descriptors, the generation
    evaluator ``generation@1``, the profile ``qwen35think@1``, and the graders
    ``math_boxed@1`` and ``gsm8k_numeric@1``, the edit operators
    ``project_out@1``, ``add_low_rank@1``, and ``fill_slice@1``, and the
    analyses ``tensor_norms@1``, ``svd_top_k@1``, and
    ``direction_projection@1`` (M2). None of them imports
    ``transformers`` or ``math_verify`` until it is used.
    """
    from wse.adapters.families import BUILTIN_FAMILIES
    from wse.adapters.hf_causal_lm import HFCausalLMAdapter
    from wse.adapters.mlp import MLPAdapter
    from wse.evaluators.classification import ClassificationEvaluator
    from wse.evaluators.forward_batch import ForwardBatchEvaluator
    from wse.evaluators.generation import GenerationEvaluator
    from wse.evaluators.graders import BUILTIN_GRADERS
    from wse.evaluators.sampling import BUILTIN_PROFILES
    from wse.operators.directions import BUILTIN_DIRECTION_EDITS
    from wse.operators.linalg import BUILTIN_LINALG_ANALYSES
    from wse.operators.tensor import BUILTIN_ANALYSES, BUILTIN_EDITS
    from wse.strategies.scripted import ScriptedSearchStrategy

    registry = Registry()
    adapter = MLPAdapter()
    registry.register("adapter", adapter.adapter_id, adapter.version, adapter)
    registry.register_alias("adapter", adapter.adapter_id, f"{adapter.adapter_id}@{adapter.version}")
    hf_adapter = HFCausalLMAdapter(registry)
    registry.register("adapter", hf_adapter.adapter_id, hf_adapter.version, hf_adapter)
    for descriptor in BUILTIN_FAMILIES:
        registry.register("family", descriptor.family_id, descriptor.version, descriptor)
    for operator in [*BUILTIN_ANALYSES.values(), *BUILTIN_LINALG_ANALYSES.values(), *BUILTIN_EDITS.values(),
                     *BUILTIN_DIRECTION_EDITS.values()]:
        registry.register("operator", operator.spec.operator_id, operator.spec.version, operator)
    for evaluator in (ForwardBatchEvaluator(), ClassificationEvaluator(), GenerationEvaluator()):
        registry.register("evaluator", evaluator.evaluator_id, evaluator.version, evaluator)
    for profile in BUILTIN_PROFILES:
        registry.register("profile", profile.profile_id, profile.version, profile)
    for grader in BUILTIN_GRADERS:
        registry.register("grader", grader.grader_id, grader.version, grader)
    registry.register("strategy", ScriptedSearchStrategy.strategy_id, ScriptedSearchStrategy.version,
                      ScriptedSearchStrategy)
    return registry
