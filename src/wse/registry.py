"""Explicit, versioned extension registration.

The operator registers each adapter, operator, evaluator, or strategy under an
exact ``(kind, extension_id, version)``. Resolution is exact: an unknown or
missing version is rejected rather than replaced by the latest one. Agents
invoke registered extensions; they never install them.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Literal, get_args

from wse.schemas import ErrorCode, ExtensionRef, WSEError

ExtensionKind = Literal["adapter", "operator", "evaluator", "strategy"]
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


# Per-kind compatibility checks run at registration time. Later milestones add
# validators for evaluators and strategies here.
KIND_VALIDATORS: dict[str, Validator] = {"adapter": _validate_adapter, "operator": _validate_operator}


class Registry:
    """A small in-process registry with explicit registrations."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], Any] = {}
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
    """Return a new registry containing the built-in M1 extensions."""
    from wse.adapters.mlp import MLPAdapter
    from wse.operators.tensor import BUILTIN_ANALYSES

    registry = Registry()
    adapter = MLPAdapter()
    registry.register("adapter", adapter.adapter_id, adapter.version, adapter)
    for operator in BUILTIN_ANALYSES.values():
        registry.register("operator", operator.spec.operator_id, operator.spec.version, operator)
    return registry
