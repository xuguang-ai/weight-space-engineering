"""Family descriptors: registered data that generic import cannot infer (M2 plan Section 2.2).

A family descriptor is an extension of kind ``family``, selected at import by
``config.json``'s ``model_type``. It holds only data:

* a name map from checkpoint names to the parameter names of the
  ``transformers`` module, including rules that drop tensors the language
  model does not use (vision towers, multi-token-prediction heads);
* for multimodal wrappers, the key of the text configuration and the causal-LM
  class built from it;
* semantic roles (``RoleRule``), head groupings (``HeadGrouping``), and the
  roles that every policy protects from edits (Level 1).

Canonical tensor names stay the checkpoint names, so published hashes and edit
recipes refer to the same names; the module names come from the map. Adding a
family registers a descriptor and changes no core code. This module never
imports ``transformers``.

**Roles.** A role rule names the role of every canonical tensor name that its
pattern matches in full. Per-layer roles capture the layer index in a named
group ``layer``; the model-wide roles ``embed``, ``lm_head``, and
``final_norm`` have none. At most one rule may match a tensor, and at most one
tensor may hold a role in a layer. A head grouping splits a role's tensor into
heads along one axis, with the head count and size read from configuration
attributes. Import resolves every role against the checkpoint and refuses a
descriptor that does not fit it.

**Listing roles.** ``model_roles(model)`` lists the roles of a model (a
``ModelHandle`` or a ``ModelManifest``) as ``TensorRole`` entries, and
``role_tensor(model, "mlp.down", layer=15)`` returns one of them: the canonical
tensor to name as an edit target, its head ranges, and the module whose input
or output is a probe's role site. Models without a descriptor (Level 0 and M1
models) have no roles.

**Protections.** Every policy refuses edits of tensors whose role the model's
descriptor protects (``wse.policy.require_unprotected``); the built-in
descriptors protect ``embed``, ``lm_head``, and ``final_norm``. Exact replay
and the operator's import of a published patch write recorded values and
check no protection, so published edits reproduce exactly.

**Versions.** The descriptor version is recorded in the model's configuration,
and so in its identity. A released descriptor is therefore frozen: changing
its roles or protections needs a new version, and a test pins the content of
every built-in descriptor.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wse.schemas import (
    EXTENSION_ID_PATTERN,
    VERSION_PATTERN,
    ErrorCode,
    ExtensionRef,
    ModelManifest,
    TensorInfo,
    WSEError,
)

if TYPE_CHECKING:
    from wse.api import ModelHandle
    from wse.registry import Registry

ROLES = (
    "embed", "lm_head", "final_norm", "attn.q", "attn.k", "attn.v", "attn.o", "linear_attn.in",
    "linear_attn.out", "mlp.gate", "mlp.up", "mlp.down", "norm.input", "norm.post_attention",
)
MODEL_WIDE_ROLES = ("embed", "lm_head", "final_norm")  # one tensor per model; every other role is per layer


def _index(value: Any) -> int | None:
    """``value`` as an ``int`` when it is an integer (NumPy integers included) other than a bool, else ``None``."""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


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
    """How a role's tensor splits into heads along ``axis``.

    ``num_heads_key`` and ``head_dim_key`` name attributes of the model's
    configuration (its text configuration for multimodal wrappers). Head ``h``
    spans entries ``[h * size, (h + 1) * size)`` of the axis, where ``size`` is
    the head dimension times ``head_dim_factor``. A factor of 2 describes a
    gated query projection that stores each head's query and output gate side
    by side (Qwen3.5's ``q_proj``).
    """

    num_heads_key: str = Field(min_length=1, max_length=128)
    head_dim_key: str = Field(min_length=1, max_length=128)
    axis: int = Field(ge=0, le=7)
    head_dim_factor: int = Field(default=1, ge=1, le=8)


class RoleRule(_Data):
    """A semantic role for the canonical tensor names that ``pattern`` matches in full.

    The pattern of a per-layer role has a named group ``layer`` that captures
    the layer index; the pattern of a model-wide role (``MODEL_WIDE_ROLES``)
    has none.
    """

    pattern: str = Field(min_length=1, max_length=512)
    role: str
    heads: HeadGrouping | None = None

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"role pattern {value!r} is not a regular expression: {exc}") from exc
        return value

    @field_validator("role")
    @classmethod
    def _known_role(cls, value: str) -> str:
        if value not in ROLES:
            raise ValueError(f"role {value!r} is not one of {list(ROLES)}")
        return value

    @model_validator(mode="after")
    def _layer_group(self) -> RoleRule:
        has_layer = "layer" in re.compile(self.pattern).groupindex
        if self.role in MODEL_WIDE_ROLES and has_layer:
            raise ValueError(f"role {self.role} is model-wide: its pattern {self.pattern!r} must not have a group "
                             "'layer'")
        if self.role not in MODEL_WIDE_ROLES and not has_layer:
            raise ValueError(f"role {self.role} is per layer: its pattern {self.pattern!r} must have a named group "
                             "'layer' that captures the layer index")
        return self

    def match(self, name: str) -> tuple[bool, int | None]:
        """Whether ``name`` matches in full, and the layer index it captures (``None`` when not decimal)."""
        found = re.fullmatch(self.pattern, name)
        if found is None:
            return False, None
        text = found.groupdict().get("layer")
        return True, int(text) if isinstance(text, str) and text.isdecimal() and text.isascii() else None


class HeadLayout(_Data):
    """The heads of one tensor: head ``h`` spans entries ``[h * size, (h + 1) * size)`` of ``axis``."""

    axis: int
    num_heads: int
    head_dim: int
    size: int  # entries per head along the axis: head_dim times the grouping's head_dim_factor

    def head_range(self, head: int) -> tuple[int, int]:
        """The ``(start, stop)`` entries of ``head`` along the head axis."""
        index = _index(head)
        if index is None or not 0 <= index < self.num_heads:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"head must be an integer from 0 to {self.num_heads - 1}; got {head!r}",
                           details={"head": repr(head), "num_heads": self.num_heads})
        return index * self.size, (index + 1) * self.size


class TensorRole(_Data):
    """One tensor's semantic role in a model (``model_roles``).

    ``tensor`` is the canonical name, which edits and reads take;
    ``module_name`` is the parameter name in the ``transformers`` module.
    ``layer`` is ``None`` for the model-wide roles. Every edit operator refuses
    ``protected`` tensors; exact replay and patch imports write recorded values.
    """

    role: str
    layer: int | None
    tensor: str
    module_name: str
    shape: list[int]
    axes: list[str]
    heads: HeadLayout | None = None
    protected: bool = False

    @property
    def site(self) -> str:
        """The canonical name of the module that holds the tensor (``model.layers.15.mlp.down_proj``).

        A probe's role site, such as the input of ``mlp.down``, is this
        module's input or output.
        """
        return self.tensor.rpartition(".")[0]

    def selection(self, head: int | None = None) -> list[dict[str, int]]:
        """An edit or read selection: the whole tensor, or with ``head`` that head's range of the head axis."""
        selection = [{"start": 0, "stop": size, "step": 1} for size in self.shape]
        if head is None:
            return selection
        if self.heads is None:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"role {self.role} of tensor {self.tensor!r} has no head grouping; omit head",
                           details={"tensor": self.tensor, "role": self.role})
        start, stop = self.heads.head_range(head)
        selection[self.heads.axis] = {"start": start, "stop": stop, "step": 1}
        return selection


def _order(role: TensorRole) -> tuple[int, int]:
    """Model-wide roles first, then layer by layer, each in ``ROLES`` order."""
    return (-1 if role.layer is None else role.layer), ROLES.index(role.role)


def _positive(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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
        unruled = sorted(set(self.protected_roles) - {rule.role for rule in self.roles})
        if unruled:
            raise ValueError(f"protected roles {unruled} have no role rule, so they would protect nothing")
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

    # -- roles ----------------------------------------------------------------

    @property
    def config_keys(self) -> list[str]:
        """The configuration attributes that the head groupings read."""
        return sorted({key for rule in self.roles if rule.heads is not None
                       for key in (rule.heads.num_heads_key, rule.heads.head_dim_key)})

    def protected_role(self, name: str) -> tuple[str, int | None] | None:
        """The protected role of canonical tensor ``name`` and its layer, or ``None`` when it has none."""
        for rule in self.roles:
            if rule.role in self.protected_roles:
                matched, layer = rule.match(name)
                if matched:
                    return rule.role, layer
        return None

    def _misfit(self, message: str, **details: Any) -> WSEError:
        return WSEError(ErrorCode.INVALID_MODEL, f"family {self.ref} does not fit this model: {message}",
                        details={"family": str(self.ref), **details})

    def _heads(self, rule: RoleRule, info: TensorInfo, config: Mapping[str, Any]) -> HeadLayout | None:
        grouping = rule.heads
        if grouping is None:
            return None
        keys = (grouping.num_heads_key, grouping.head_dim_key)
        values = {key: config.get(key) for key in keys}
        if not all(_positive(value) for value in values.values()):
            raise self._misfit(f"the head grouping of role {rule.role} reads the configuration attributes "
                               f"{list(keys)}, which must be positive integers; the configuration gives {values}",
                               role=rule.role, tensor=info.name)
        if grouping.axis >= len(info.shape):
            raise self._misfit(f"the head grouping of role {rule.role} indexes heads on axis {grouping.axis}, but "
                               f"tensor {info.name!r} has {len(info.shape)} axes", role=rule.role, tensor=info.name)
        num_heads, head_dim = values[grouping.num_heads_key], values[grouping.head_dim_key]
        size = head_dim * grouping.head_dim_factor
        if num_heads * size != info.shape[grouping.axis]:
            factor = f" x {grouping.head_dim_factor}" if grouping.head_dim_factor != 1 else ""
            raise self._misfit(f"role {rule.role}: {num_heads} heads ({grouping.num_heads_key}) of {size} entries "
                               f"({grouping.head_dim_key} {head_dim}{factor}) do not partition axis {grouping.axis} "
                               f"of tensor {info.name!r}, which has {info.shape[grouping.axis]} entries",
                               role=rule.role, tensor=info.name)
        return HeadLayout(axis=grouping.axis, num_heads=num_heads, head_dim=head_dim, size=size)

    def resolve_roles(self, tensors: Iterable[TensorInfo], config: Mapping[str, Any]) -> tuple[TensorRole, ...]:
        """Resolve every role rule against a model's tensors.

        ``config`` maps the attributes in ``config_keys`` to their values in the
        model's configuration. Raises ``INVALID_MODEL`` when the descriptor does
        not fit the model: a tensor that two rules match, a role held by two
        tensors of one layer, a layer group that captures no decimal index, or
        a head grouping that does not partition its axis.
        """
        found: dict[tuple[str, int | None], TensorRole] = {}
        for info in sorted(tensors, key=lambda info: info.name):
            matches = [(rule, layer) for rule in self.roles for matched, layer in [rule.match(info.name)] if matched]
            if not matches:
                continue
            if len(matches) > 1:
                raise self._misfit(f"tensor {info.name!r} matches the rules of roles "
                                   f"{[rule.role for rule, _ in matches]}; at most one rule may match a tensor",
                                   tensor=info.name)
            rule, layer = matches[0]
            if rule.role not in MODEL_WIDE_ROLES and layer is None:
                raise self._misfit(f"the pattern {rule.pattern!r} of role {rule.role} matches tensor {info.name!r} "
                                   "but its group 'layer' captures no decimal layer index", tensor=info.name)
            key = (rule.role, layer)
            if key in found:
                where = "" if layer is None else f" in layer {layer}"
                raise self._misfit(f"tensors {found[key].tensor!r} and {info.name!r} both have role {rule.role}{where}",
                                   tensors=[found[key].tensor, info.name])
            found[key] = TensorRole(
                role=rule.role, layer=layer, tensor=info.name, module_name=info.module_name or info.name,
                shape=list(info.shape), axes=list(info.axes), heads=self._heads(rule, info, config),
                protected=rule.role in self.protected_roles,
            )
        return tuple(sorted(found.values(), key=_order))


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


# ---------------------------------------------------------------------------
# Roles of a model
# ---------------------------------------------------------------------------


def _adapter_and_tensors(model: ModelHandle | ModelManifest, registry: Registry | None) -> tuple[Any, list[TensorInfo]]:
    if isinstance(model, ModelManifest):
        from wse.registry import builtin_registry

        adapter = (registry or builtin_registry()).resolve_ref("adapter", model.adapter)
        return adapter, list(model.tensors)
    architecture = getattr(model, "architecture", None)
    if architecture is None or not hasattr(model, "adapter"):
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"model must be a ModelHandle or a ModelManifest, not {type(model).__name__}")
    return model.adapter, list(architecture.tensors)


def model_family(model: ModelHandle | ModelManifest, *, registry: Registry | None = None) -> FamilyDescriptor | None:
    """The family descriptor that a model's configuration records, or ``None`` (Level 0 and M1 models).

    ``registry`` resolves the adapter of a manifest (default: the built-in
    registry); a handle carries its adapter.
    """
    adapter, _ = _adapter_and_tensors(model, registry)
    family = getattr(adapter, "family", None)
    return family(model.config) if callable(family) else None


def model_roles(model: ModelHandle | ModelManifest, *, registry: Registry | None = None) -> tuple[TensorRole, ...]:
    """The roles of a model's tensors: model-wide roles first, then layer by layer in ``ROLES`` order.

    A manifest suffices, so no weight is loaded. Tensors without a role are
    omitted; a model without a family descriptor has no roles and returns
    ``()``. ``registry`` resolves the adapter of a manifest.
    """
    adapter, tensors = _adapter_and_tensors(model, registry)
    roles = getattr(adapter, "roles", None)
    return tuple(roles(model.config, tensors)) if callable(roles) else ()


def role_tensor(model: ModelHandle | ModelManifest, role: str, layer: int | None = None, *,
                registry: Registry | None = None) -> TensorRole:
    """The tensor that holds ``role`` in ``layer`` (no layer for ``embed``, ``lm_head``, and ``final_norm``).

    Raises ``INVALID_ARGUMENT`` for an unknown role or a layer argument that
    does not suit the role, and ``NOT_FOUND`` when the model has no such
    tensor, with the layers that hold the role in ``details["layers"]``.
    """
    if role not in ROLES:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"role {role!r} is not one of {list(ROLES)}",
                       details={"role": str(role)})
    if role in MODEL_WIDE_ROLES and layer is not None:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"role {role} is model-wide; omit layer",
                       details={"role": role})
    if role not in MODEL_WIDE_ROLES:
        index = _index(layer)
        if index is None or index < 0:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"role {role} is per layer; give a layer index (an integer "
                           f">= 0), not {layer!r}", details={"role": role})
        layer = index
    descriptor = model_family(model, registry=registry)
    if descriptor is None:
        raise WSEError(ErrorCode.NOT_FOUND, "the model has no family descriptor (Level 0), so its tensors have no "
                       "roles; name tensors by their canonical names", details={"role": role})
    roles = model_roles(model, registry=registry)
    for candidate in roles:
        if (candidate.role, candidate.layer) == (role, layer):
            return candidate
    layers = [candidate.layer for candidate in roles if candidate.role == role]
    if role not in {rule.role for rule in descriptor.roles}:
        message = f"family {descriptor.ref} has no role {role}"
    elif layer is None:
        message = f"the model has no tensor with role {role}"
    else:
        message = f"layer {layer} has no tensor with role {role}; layers with {role}: {layers}"
    raise WSEError(ErrorCode.NOT_FOUND, message,
                   details={"family": str(descriptor.ref), "role": role, "layer": layer, "layers": layers})


# ---------------------------------------------------------------------------
# Built-in descriptors
# ---------------------------------------------------------------------------


def _decoder_roles(prefix: str, layer_roles: Sequence[tuple[str, str, HeadGrouping | None]]) -> list[RoleRule]:
    """Role rules for a decoder whose language model's parameters start with ``prefix`` (canonical names).

    ``layer_roles`` holds ``(name within a layer, role, head grouping)``. The
    output head is ``lm_head.weight`` in every built-in family.
    """
    escaped = re.escape(prefix)
    rules = [
        RoleRule(pattern=rf"{escaped}embed_tokens\.weight", role="embed"),
        RoleRule(pattern=r"lm_head\.weight", role="lm_head"),
        RoleRule(pattern=rf"{escaped}norm\.weight", role="final_norm"),
    ]
    rules += [RoleRule(pattern=rf"{escaped}layers\.(?P<layer>\d+)\.{re.escape(name)}", role=role, heads=heads)
              for name, role, heads in layer_roles]
    return rules


def _heads(num_heads_key: str, axis: int, *, head_dim_key: str = "head_dim", factor: int = 1) -> HeadGrouping:
    return HeadGrouping(num_heads_key=num_heads_key, head_dim_key=head_dim_key, axis=axis, head_dim_factor=factor)


# The roles every built-in family protects: the embeddings, the output head, and the final norm.
PROTECTED_ROLES = ("embed", "final_norm", "lm_head")

# Layers of LlamaForCausalLM: grouped-query attention (q_proj, k_proj, v_proj, o_proj, whose
# heads are contiguous blocks of head_dim) and a gated MLP. Llama checkpoints name the module's
# parameters, so the descriptor needs no name map.
_LLAMA_LAYER = [
    ("self_attn.q_proj.weight", "attn.q", _heads("num_attention_heads", 0)),
    ("self_attn.k_proj.weight", "attn.k", _heads("num_key_value_heads", 0)),
    ("self_attn.v_proj.weight", "attn.v", _heads("num_key_value_heads", 0)),
    ("self_attn.o_proj.weight", "attn.o", _heads("num_attention_heads", 1)),
    ("mlp.gate_proj.weight", "mlp.gate", None),
    ("mlp.up_proj.weight", "mlp.up", None),
    ("mlp.down_proj.weight", "mlp.down", None),
    ("input_layernorm.weight", "norm.input", None),
    ("post_attention_layernorm.weight", "norm.post_attention", None),
]

LLAMA = FamilyDescriptor(
    family_id="llama",
    version="1",
    model_types=["llama"],
    roles=_decoder_roles("model.", _LLAMA_LAYER),
    protected_roles=list(PROTECTED_ROLES),
)

# Qwen3.5 checkpoints are multimodal wrappers (Qwen3_5ForConditionalGeneration):
# the language model lives under model.language_model.*, beside a vision tower
# (model.visual.*) and multi-token-prediction tensors (mtp.*). Qwen3_5ForCausalLM,
# built from the text configuration, names the language model's parameters
# model.*, which the fixture test pins. Its layers mix full attention, whose
# q_proj holds each head's query and output gate side by side (2 x head_dim rows
# per head), with Gated DeltaNet linear attention: in_proj_qkv projects the
# queries, keys, and values, and out_proj reads the value heads. The other
# linear-attention parameters (in_proj_z, in_proj_a, in_proj_b, conv1d, dt_bias,
# A_log, and the gated norm) and the per-head q_norm and k_norm have no role.
_QWEN3_5_LAYER = [
    ("self_attn.q_proj.weight", "attn.q", _heads("num_attention_heads", 0, factor=2)),
    ("self_attn.k_proj.weight", "attn.k", _heads("num_key_value_heads", 0)),
    ("self_attn.v_proj.weight", "attn.v", _heads("num_key_value_heads", 0)),
    ("self_attn.o_proj.weight", "attn.o", _heads("num_attention_heads", 1)),
    ("linear_attn.in_proj_qkv.weight", "linear_attn.in", None),
    ("linear_attn.out_proj.weight", "linear_attn.out",
     _heads("linear_num_value_heads", 1, head_dim_key="linear_value_head_dim")),
    ("mlp.gate_proj.weight", "mlp.gate", None),
    ("mlp.up_proj.weight", "mlp.up", None),
    ("mlp.down_proj.weight", "mlp.down", None),
    ("input_layernorm.weight", "norm.input", None),
    ("post_attention_layernorm.weight", "norm.post_attention", None),
]

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
    roles=_decoder_roles("model.language_model.", _QWEN3_5_LAYER),
    protected_roles=list(PROTECTED_ROLES),
)

BUILTIN_FAMILIES: tuple[FamilyDescriptor, ...] = (LLAMA, QWEN3_5)
