"""Family descriptors: registered data that generic import cannot infer (M2 plan Section 2.2).

A family descriptor is an extension of kind ``family``, selected at import by
``config.json``'s ``model_type``. It holds only data:

* a name map from checkpoint names to the parameter names of the
  ``transformers`` module, including rules that drop tensors the language
  model does not use (vision towers, multi-token-prediction heads);
* for multimodal wrappers, the key of the text configuration and the causal-LM
  class built from it;
* semantic roles, head groupings, and default protections (Q2).

Canonical tensor names stay the checkpoint names, so published hashes and edit
recipes refer to the same names; the module names come from the map. Adding a
family registers a descriptor and changes no core code. This module never
imports ``transformers``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wse.schemas import EXTENSION_ID_PATTERN, VERSION_PATTERN, ErrorCode, ExtensionRef, WSEError

if TYPE_CHECKING:
    from wse.registry import Registry

ROLES = (
    "embed", "lm_head", "final_norm", "attn.q", "attn.k", "attn.v", "attn.o", "linear_attn.in",
    "linear_attn.out", "mlp.gate", "mlp.up", "mlp.down", "norm.input", "norm.post_attention",
)


class _Data(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PrefixRule(_Data):
    """Checkpoint names that start with ``checkpoint_prefix`` map to ``module_prefix`` + the rest.

    A ``module_prefix`` of ``None`` drops the matching tensors: they are neither
    stored nor checked against the module.
    """

    checkpoint_prefix: str = Field(min_length=1, max_length=256)
    module_prefix: str | None = Field(max_length=256)


class HeadGrouping(_Data):
    """How a role's tensor splits into heads: config keys for the head count and size, and the head axis."""

    num_heads_key: str = Field(min_length=1, max_length=128)
    head_dim_key: str = Field(min_length=1, max_length=128)
    axis: int = Field(ge=0, le=7)


class RoleRule(_Data):
    """A semantic role for canonical names that match ``pattern`` (with a named group ``layer``)."""

    pattern: str = Field(min_length=1, max_length=512)
    role: str
    heads: HeadGrouping | None = None

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            compiled = re.compile(value)
        except re.error as exc:
            raise ValueError(f"role pattern {value!r} is not a regular expression: {exc}") from exc
        if "layer" not in compiled.groupindex:
            raise ValueError(f"role pattern {value!r} must have a named group 'layer'")
        return value

    @field_validator("role")
    @classmethod
    def _known_role(cls, value: str) -> str:
        if value not in ROLES:
            raise ValueError(f"role {value!r} is not one of {list(ROLES)}")
        return value


class FamilyDescriptor(_Data):
    """A registered, versioned description of one model family."""

    family_id: str
    version: str
    model_types: list[str] = Field(min_length=1)
    text_config_key: str | None = None
    causal_lm_class: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    name_map: list[PrefixRule] = Field(default_factory=list)
    roles: list[RoleRule] = Field(default_factory=list)
    protected_roles: list[str] = Field(default_factory=list)

    @field_validator("family_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not EXTENSION_ID_PATTERN.fullmatch(value):
            raise ValueError(f"family id {value!r} must match {EXTENSION_ID_PATTERN.pattern}")
        return value

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        if not VERSION_PATTERN.fullmatch(value):
            raise ValueError(f"version {value!r} must match {VERSION_PATTERN.pattern}")
        return value

    @model_validator(mode="after")
    def _check(self) -> FamilyDescriptor:
        if (self.text_config_key is None) != (self.causal_lm_class is None):
            raise ValueError("text_config_key and causal_lm_class are given together (multimodal wrappers) or not at all")
        unknown = sorted(set(self.protected_roles) - set(ROLES))
        if unknown:
            raise ValueError(f"protected roles {unknown} are not roles")
        return self

    @property
    def ref(self) -> ExtensionRef:
        return ExtensionRef(extension_id=self.family_id, version=self.version)

    # -- name maps ------------------------------------------------------------

    def _rule(self, name: str) -> PrefixRule | None:
        for rule in self.name_map:
            if name.startswith(rule.checkpoint_prefix):
                return rule
        return None

    def module_name(self, checkpoint_name: str) -> str | None:
        """The module parameter name of a checkpoint tensor, or ``None`` when the map drops it.

        The first rule whose checkpoint prefix matches applies; names that no
        rule matches keep their name.
        """
        rule = self._rule(checkpoint_name)
        if rule is None:
            return checkpoint_name
        if rule.module_prefix is None:
            return None
        return rule.module_prefix + checkpoint_name[len(rule.checkpoint_prefix):]

    def canonical_name(self, module_name: str) -> str:
        """The checkpoint (canonical) name of a module parameter that the checkpoint does not store.

        Used for members of a tied group stored once, such as a tied output
        head: the first mapping rule whose module prefix matches is reversed.
        """
        for rule in self.name_map:
            if rule.module_prefix is not None and module_name.startswith(rule.module_prefix):
                candidate = rule.checkpoint_prefix + module_name[len(rule.module_prefix):]
                if self.module_name(candidate) == module_name:
                    return candidate
        return module_name


def map_names(descriptor: FamilyDescriptor | None, checkpoint_names: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """Map checkpoint names to module names; return ``(canonical -> module, dropped)``.

    Without a descriptor (Level 0) every name maps to itself.
    """
    mapped: dict[str, str] = {}
    dropped: list[str] = []
    for name in sorted(checkpoint_names):
        module = name if descriptor is None else descriptor.module_name(name)
        if module is None:
            dropped.append(name)
        else:
            mapped[name] = module
    return mapped, dropped


def validate_family(extension_id: str, version: str, implementation: Any) -> None:
    """Registration check for kind ``family``: a descriptor registered under its own ID and version."""
    if not isinstance(implementation, FamilyDescriptor):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"family {extension_id}@{version} must be a FamilyDescriptor, not {type(implementation).__name__}",
        )
    if (implementation.family_id, implementation.version) != (extension_id, version):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"family descriptor declares {implementation.ref} but is being registered as {extension_id}@{version}",
        )


def select_family(
    registry: Registry, model_type: str, family: str | ExtensionRef | None = None
) -> tuple[ExtensionRef, FamilyDescriptor] | None:
    """The family descriptor for ``model_type``, or ``None`` when no registered descriptor accepts it.

    With ``family`` (an exact reference) that descriptor is used and must
    accept ``model_type``. Without it, exactly one registered descriptor may
    accept ``model_type``; several are an ambiguity the operator resolves by
    naming one. A version is never substituted.
    """
    if family is not None:
        ref = registry.resolve_reference("family", family)
        descriptor = registry.resolve("family", ref.extension_id, ref.version)
        if model_type not in descriptor.model_types:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"family {ref} accepts model types {descriptor.model_types}, not {model_type!r}",
                details={"family": str(ref), "model_type": model_type},
            )
        return ref, descriptor
    matches = [ref for ref in registry.list("family")
               if model_type in registry.resolve("family", ref.extension_id, ref.version).model_types]
    if not matches:
        return None
    if len(matches) > 1:
        names = [str(ref) for ref in matches]
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"several family descriptors accept model type {model_type!r}: {names}; name one explicitly",
            details={"model_type": model_type, "families": names},
        )
    ref = matches[0]
    return ref, registry.resolve("family", ref.extension_id, ref.version)


# Qwen3.5 checkpoints are multimodal wrappers (Qwen3_5ForConditionalGeneration):
# the language model lives under model.language_model.*, beside a vision tower
# (model.visual.*) and multi-token-prediction tensors (mtp.*). Qwen3_5ForCausalLM,
# built from the text configuration, names the language model's parameters
# model.*, which the fixture test pins. Roles arrive in Task 8.
QWEN3_5 = FamilyDescriptor(
    family_id="qwen3_5",
    version="1",
    model_types=["qwen3_5"],
    text_config_key="text_config",
    causal_lm_class="Qwen3_5ForCausalLM",
    name_map=[
        PrefixRule(checkpoint_prefix="model.language_model.", module_prefix="model."),
        PrefixRule(checkpoint_prefix="model.visual.", module_prefix=None),
        PrefixRule(checkpoint_prefix="mtp.", module_prefix=None),
    ],
)

BUILTIN_FAMILIES: tuple[FamilyDescriptor, ...] = (QWEN3_5,)
