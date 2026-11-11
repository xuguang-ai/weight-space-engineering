"""The generic Hugging Face causal-LM adapter ``hf_causal_lm@1`` (M2 plan Section 2.1, Level 0).

It imports any causal language model whose architecture the pinned
``transformers`` implements natively, from a local snapshot of safetensors
weights in BF16, F16, or F32. Architectures that need remote code, quantized
formats, and encoder-decoder models are rejected.

* **Configuration.** The adapter configuration is ``{"model_config": <the
  snapshot's config.json>, "family": "<id>@<version>" | null}``; its hash is the
  model's ``config_hash``.
* **Names.** Canonical tensor names are the checkpoint names. At Level 0 they
  must equal the parameter names of the ``transformers`` module; a registered
  family descriptor (``wse.adapters.families``) may map them and drop tensors
  the language model does not use. Each manifest entry records its module name.
* **Roles (Level 1).** A family descriptor also names the semantic roles of
  tensors and their head groupings, which read attributes of the
  ``transformers`` configuration (``family_roles``). Import refuses a
  descriptor whose roles do not fit the checkpoint; ``roles`` and ``family``
  serve ``wse.adapters.families.model_roles`` and the edit protections.
* **Axes come from module types, never from tensor names.** The module is
  instantiated on the meta device: an ``nn.Linear`` weight is ``[out_features,
  in_features]`` and its bias ``[out_features]``, an ``nn.Embedding`` weight is
  ``[num_embeddings, embedding_dim]``, an ``nn.Conv1d`` weight is
  ``[out_channels, in_channels_per_group, kernel]``, a one-dimensional
  parameter of a normalization module is ``[features]``, and anything else is
  ``unknown``.
* **Alias groups.** Parameters that are one object in the module (tied input
  and output embeddings) form an alias group. The checkpoint stores the group
  once; the manifest lists every member, and all members share one blob.
* **Identity.** The weights hash is over blob IDs in canonical name order
  (``wse.tensors.blob_weights_hash``); the manifest's ``source`` records the
  repository, revision, every file read with its SHA-256, and the tokenizer,
  chat-template, and generation-configuration hashes.
* **Generation.** The import freezes the model's default generation profile
  from its ``generation_config.json`` (``wse.evaluators.sampling``) into the
  manifest's ``source``, and returns the bytes of the tokenizer, chat-template,
  and generation-configuration files for the store to keep.

``transformers`` is imported lazily, only when a layout is built, so ``import
wse`` and the registry work without the ``models-hf`` extra.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pydantic import ValidationError

from wse.adapters.base import (
    CAP_FORWARD_LOGITS,
    CAP_GENERATE_TEXT,
    CAP_WEIGHTS_READ,
    CAP_WEIGHTS_SLICE_EDIT,
    AdapterCapabilities,
    Architecture,
)
from wse.adapters.families import FamilyDescriptor, TensorRole, map_names, select_family
from wse.evaluators.sampling import profile_from_generation_config
from wse.hf_snapshot import CONFIG_FILE, GENERATION_CONFIG_FILE, STORED_DTYPES, SnapshotReader, TensorHeader
from wse.schemas import (
    ErrorCode,
    ExtensionRef,
    ImportManifest,
    IOSpec,
    ModelManifest,
    ModelSource,
    TensorInfo,
    WSEError,
)
from wse.tensors import WEIGHTS_HASH_BLOBS, blob_id, dtype_name, require_finite

if TYPE_CHECKING:
    from wse.engines.hf_torch import HFTorchModel
    from wse.registry import Registry

ADAPTER_ID = "hf_causal_lm"
ADAPTER_VERSION = "1"
DEFAULT_MAX_HF_MODEL_BYTES = 16 * 1024 ** 3
_TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_STORED_DTYPE_NAMES = {"bfloat16", "float16", "float32"}
_REVISION = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")
INPUT_SEMANTICS = IOSpec(shape=["batch", "sequence"], dtype="int64", description="token IDs")
OUTPUT_SEMANTICS = IOSpec(shape=["batch", "sequence", "vocab"], dtype="float32",
                          description="next-token logits over the vocabulary")


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_MODEL, message, details=details)


def _transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise WSEError(
            ErrorCode.UNSUPPORTED_CAPABILITY,
            "hf_causal_lm@1 needs the models-hf extra (transformers); install it with "
            "'uv sync --extra models-hf'",
        ) from exc
    return transformers


# ---------------------------------------------------------------------------
# Configuration checks
# ---------------------------------------------------------------------------


def _find_key(value: Any, key: str, path: str = "") -> str | None:
    """The JSON path of the first ``key`` anywhere in ``value``, or ``None``."""
    if isinstance(value, dict):
        for name in sorted(value):
            here = f"{path}.{name}" if path else name
            if name == key:
                return here
            found = _find_key(value[name], key, here)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_key(item, key, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def check_model_config(config: Any) -> str:
    """Reject configurations outside M2's supported models; return the ``model_type``."""
    if not isinstance(config, dict):
        raise _invalid(f"{CONFIG_FILE} must be a JSON object")
    found = _find_key(config, "auto_map")
    if found is not None:
        raise _invalid(
            f"{CONFIG_FILE} declares {found}: the architecture needs remote code (trust_remote_code), which WSE "
            "never executes; only architectures that transformers implements natively are supported",
            key=found,
        )
    found = _find_key(config, "quantization_config")
    if found is not None:
        raise _invalid(f"{CONFIG_FILE} declares {found}: quantized models (GGUF, GPTQ, AWQ, MXFP4, ...) are not "
                       "supported; import BF16, F16, or F32 safetensors weights", key=found)
    if config.get("is_encoder_decoder") is True:
        raise _invalid("encoder-decoder models are not supported; hf_causal_lm@1 imports causal language models")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise _invalid(f"{CONFIG_FILE} has no model_type")
    return model_type


# ---------------------------------------------------------------------------
# Module layout (meta device)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleParameter:
    """One parameter of the instantiated module."""

    module_name: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    group: int  # parameters with equal groups are one object (an alias group)


@dataclass(frozen=True)
class ModuleLayout:
    model_class: str
    parameters: dict[str, ModuleParameter]
    unique_parameter_count: int
    nominal_dtype: str = "float32"

    @property
    def alias_groups(self) -> list[list[str]]:
        groups: dict[int, list[str]] = {}
        for parameter in self.parameters.values():
            groups.setdefault(parameter.group, []).append(parameter.module_name)
        return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _is_norm(module: torch.nn.Module) -> bool:
    nn = torch.nn
    norms = (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)
    return isinstance(module, norms) or "Norm" in type(module).__name__


def _axes(module: torch.nn.Module, local_name: str, parameter: torch.Tensor) -> tuple[str, ...]:
    nn = torch.nn
    dim = parameter.dim()
    if isinstance(module, nn.Linear) and local_name in ("weight", "bias"):
        axes: tuple[str, ...] = ("out_features", "in_features") if local_name == "weight" else ("out_features",)
    elif isinstance(module, nn.Embedding) and local_name == "weight":
        axes = ("num_embeddings", "embedding_dim")
    elif isinstance(module, nn.Conv1d) and local_name in ("weight", "bias"):
        axes = ("out_channels", "in_channels_per_group", "kernel") if local_name == "weight" else ("out_channels",)
    elif _is_norm(module) and dim == 1:
        axes = ("features",)
    else:
        axes = ("unknown",) * dim
    return axes if len(axes) == dim else ("unknown",) * dim


def causal_lm_class(model_config: dict[str, Any], descriptor: FamilyDescriptor | None) -> tuple[Any, Any]:
    """The ``transformers`` causal-LM class and its configuration object for ``model_config``."""
    transformers = _transformers()
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    version = transformers.__version__
    model_type = model_config["model_type"]
    if descriptor is not None and descriptor.causal_lm_class is not None:
        model_class = getattr(transformers, descriptor.causal_lm_class, None)
        if model_class is None:
            raise _invalid(f"family {descriptor.ref} names {descriptor.causal_lm_class}, which transformers "
                           f"{version} does not provide")
        text_config = model_config.get(descriptor.text_config_key or "")
        if not isinstance(text_config, dict):
            raise _invalid(f"{CONFIG_FILE} has no {descriptor.text_config_key!r} object, which family "
                           f"{descriptor.ref} requires")
        config_class = model_class.config_class
        source = text_config
    else:
        if model_type not in CONFIG_MAPPING:
            raise _invalid(f"transformers {version} does not implement model type {model_type!r}; "
                           "hf_causal_lm@1 imports only architectures it implements natively",
                           model_type=model_type)
        config_class, model_class, source = CONFIG_MAPPING[model_type], None, model_config
    try:
        # A deep copy: transformers fills defaults into nested dicts in place (Llama's rope_scaling gains
        # rope_theta), and the stored, hashed configuration must stay the snapshot's config.json.
        config = config_class.from_dict(copy.deepcopy(dict(source)))
    except Exception as exc:  # transformers validates configurations with many exception types
        raise _invalid(f"{CONFIG_FILE} is not a valid {config_class.__name__}: {exc}") from exc
    if getattr(config, "is_encoder_decoder", False):
        raise _invalid(f"model type {model_type!r} is an encoder-decoder model; hf_causal_lm@1 imports causal "
                       "language models")
    if model_class is None:
        try:
            model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(config)]
        except KeyError:
            raise _invalid(f"transformers {version} has no causal-LM class for model type {model_type!r}",
                           model_type=model_type) from None
    return model_class, config


def module_layout(model_config: dict[str, Any], descriptor: FamilyDescriptor | None) -> ModuleLayout:
    """Instantiate the causal LM on the meta device and describe every parameter."""
    model_class, config = causal_lm_class(model_config, descriptor)
    try:
        with torch.device("meta"):
            module = model_class(config)
    except Exception as exc:  # construction errors come from arbitrary model code paths
        raise _invalid(f"{model_class.__name__} cannot be built from {CONFIG_FILE}: {exc}") from exc
    parameters: dict[str, ModuleParameter] = {}
    groups: dict[int, int] = {}
    for prefix, owner in module.named_modules(remove_duplicate=False):
        for local_name, parameter in owner.named_parameters(recurse=False, remove_duplicate=False):
            name = f"{prefix}.{local_name}" if prefix else local_name
            if name in parameters:
                continue
            group = groups.setdefault(id(parameter), len(groups))
            parameters[name] = ModuleParameter(name, tuple(int(s) for s in parameter.shape),
                                               _axes(owner, local_name, parameter), group)
    unique = sum(p.numel() for p in module.parameters())
    dtype = model_config.get("dtype") or model_config.get("torch_dtype")
    nominal = dtype if dtype in _STORED_DTYPE_NAMES else "float32"
    return ModuleLayout(type(module).__name__, parameters, unique, nominal)


def family_roles(model_config: dict[str, Any], descriptor: FamilyDescriptor,
                 tensors: Sequence[TensorInfo]) -> tuple[TensorRole, ...]:
    """The roles of ``tensors`` under ``descriptor`` (``FamilyDescriptor.resolve_roles``).

    Head groupings read their attributes from the ``transformers``
    configuration that the module is built from (the text configuration of a
    multimodal wrapper), so defaults the configuration class fills in, such as
    Llama's ``head_dim``, apply as they do to the module's shapes.
    """
    if not descriptor.roles:
        return ()
    _, config = causal_lm_class(model_config, descriptor)
    return descriptor.resolve_roles(tensors, {key: getattr(config, key, None) for key in descriptor.config_keys})


# ---------------------------------------------------------------------------
# Snapshot import
# ---------------------------------------------------------------------------


@dataclass
class SnapshotImport:
    """Everything an import derives from a snapshot, before it is stored."""

    config: dict[str, Any]
    tensors: dict[str, torch.Tensor]  # canonical name -> tensor; alias-group members share one tensor
    infos: list[TensorInfo]
    alias_groups: list[list[str]]
    parameter_count: int
    source: ModelSource
    read_log: list[str] = field(default_factory=list)
    model_files: dict[str, bytes] = field(default_factory=dict)  # tokenizer, chat template, generation config


def _canonical(descriptor: FamilyDescriptor | None, module_name: str) -> str:
    return module_name if descriptor is None else descriptor.canonical_name(module_name)


def _check_names(table: Mapping[str, TensorHeader], mapped: dict[str, str], layout: ModuleLayout,
                 descriptor: FamilyDescriptor | None) -> dict[str, str]:
    """Check the mapped checkpoint against the module; return module name -> canonical name."""
    by_module: dict[str, str] = {}
    for canonical, module in mapped.items():
        if module in by_module:
            raise _invalid(f"checkpoint tensors {by_module[module]!r} and {canonical!r} both map to module "
                           f"parameter {module!r}", tensors=[by_module[module], canonical])
        by_module[module] = canonical
    unmapped = sorted(canonical for canonical, module in mapped.items() if module not in layout.parameters)
    present_groups = {layout.parameters[m].group for m in by_module if m in layout.parameters}
    missing = sorted(_canonical(descriptor, name) for name, parameter in layout.parameters.items()
                     if parameter.group not in present_groups)
    if unmapped or missing:
        parts = []
        if unmapped:
            parts.append(f"checkpoint tensors that are not parameters of {layout.model_class}: {unmapped}")
        if missing:
            parts.append(f"parameters of {layout.model_class} that the checkpoint lacks: {missing}")
        hint = ("; register a family descriptor that maps or drops these names" if unmapped else "")
        raise _invalid("checkpoint names do not match the module; " + "; ".join(parts) + hint,
                       unmapped=unmapped, missing=missing, model_class=layout.model_class)
    for module, canonical in by_module.items():
        header = table[canonical]
        if header.dtype not in STORED_DTYPES:
            raise _invalid(f"tensor {canonical!r} has dtype {header.dtype}; M2 stores {list(STORED_DTYPES)} only",
                           tensor=canonical, dtype=header.dtype)
        expected = layout.parameters[module].shape
        if header.shape != expected:
            raise _invalid(f"tensor {canonical!r} has shape {list(header.shape)}; {layout.model_class} expects "
                           f"{list(expected)} for {module!r}", tensor=canonical, shape=list(header.shape),
                           expected_shape=list(expected))
    return by_module


def read_snapshot(
    snapshot: str | Path,
    *,
    repo: str,
    revision: str,
    registry: Registry,
    family: str | ExtensionRef | None = None,
    max_bytes: int = DEFAULT_MAX_HF_MODEL_BYTES,
) -> SnapshotImport:
    """Read and validate a snapshot for import through ``hf_causal_lm@1``.

    Only allow-listed files are read (``SnapshotReader``); every check runs
    before any tensor data is loaded.
    """
    if not isinstance(repo, str) or not 0 < len(repo) <= 256:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "repo must be the source repository ID (1-256 characters)")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"revision {revision!r} must be a commit SHA or revision name "
                       f"matching {_REVISION.pattern}")
    reader = SnapshotReader(snapshot)
    model_config = reader.read_json(CONFIG_FILE)
    model_type = check_model_config(model_config)
    selected = select_family(registry, model_type, family)
    family_ref, descriptor = selected if selected is not None else (None, None)
    layout = module_layout(model_config, descriptor)

    table = reader.tensor_table()
    mapped, dropped = map_names(descriptor, table)
    by_module = _check_names(table, mapped, layout, descriptor)
    size = sum(table[canonical].size_bytes for canonical in by_module.values())
    if size > max_bytes:
        raise _invalid(f"the language-model tensors total {size} bytes; the limit is {max_bytes} bytes",
                       size_bytes=size, max_bytes=max_bytes)

    if descriptor is not None:  # a descriptor whose roles do not fit the checkpoint is refused (INVALID_MODEL)
        family_roles(model_config, descriptor, [
            TensorInfo(name=by_module.get(module) or _canonical(descriptor, module), shape=list(parameter.shape),
                       dtype=layout.nominal_dtype, axes=list(parameter.axes), element_count=_count(parameter.shape),
                       module_name=module)
            for module, parameter in layout.parameters.items()])

    loaded: dict[str, torch.Tensor] = {}
    for file in sorted({table[canonical].file for canonical in by_module.values()}):
        with reader.open_weights(file) as handle:
            for canonical in sorted(by_module.values()):
                if table[canonical].file == file:
                    loaded[canonical] = handle.get_tensor(canonical)
    for canonical, tensor in loaded.items():
        if tensor.dtype != _TORCH_DTYPES[table[canonical].dtype]:  # pragma: no cover - safetensors guarantees it
            raise _invalid(f"tensor {canonical!r} did not load in its stored dtype")
    require_finite(loaded)

    # Alias groups: members the checkpoint stores must agree; absent members share the stored tensor.
    tensors: dict[str, torch.Tensor] = {}
    module_names: dict[str, str] = {}
    blob_ids: dict[int, str] = {}
    canonical_groups: list[list[str]] = []
    groups: dict[int, list[str]] = {}
    for name, parameter in layout.parameters.items():
        groups.setdefault(parameter.group, []).append(name)
    for group in groups.values():
        present = [by_module[m] for m in group if m in by_module]
        ids = {canonical: blob_id(loaded[canonical]) for canonical in present}
        if len(set(ids.values())) > 1:
            raise _invalid(f"tensors {sorted(ids)} form one alias group (one parameter of {layout.model_class}) "
                           "but differ in the checkpoint", alias_group=sorted(ids))
        blob_ids.update({id(loaded[canonical]): ids[canonical] for canonical in present})
        shared = loaded[present[0]]
        members = []
        for module in group:
            canonical = by_module.get(module) or _canonical(descriptor, module)
            if canonical in tensors or (module not in by_module and canonical in table):
                raise _invalid(f"module parameter {module!r} maps back to checkpoint name {canonical!r}, which "
                               "names another tensor", tensor=canonical)
            tensors[canonical] = loaded[canonical] if canonical in loaded else shared
            module_names[canonical] = module
            members.append(canonical)
        if len(members) > 1:
            canonical_groups.append(sorted(members))

    infos = []
    for name in sorted(tensors):
        tensor = tensors[name]
        parameter = layout.parameters[module_names[name]]
        infos.append(TensorInfo(
            name=name, shape=list(parameter.shape), dtype=dtype_name(tensor), axes=list(parameter.axes),
            element_count=tensor.numel(), module_name=module_names[name],
            blob_id=blob_ids.get(id(tensor)) or blob_id(tensor),
        ))

    tokenizer_hash = reader.tokenizer_hash()
    chat_template_hash = reader.chat_template_hash()
    generation_config_hash = reader.generation_config_hash()
    generation_config = reader.read_json(GENERATION_CONFIG_FILE) if generation_config_hash is not None else None
    default_profile = profile_from_generation_config(
        repo, generation_config, has_chat_template=chat_template_hash is not None,
        source=f"{GENERATION_CONFIG_FILE} {generation_config_hash}" if generation_config_hash else None)
    try:
        source = ModelSource(
            repository=repo, revision=revision, family=family_ref, files=dict(sorted(reader.files.items())),
            tokenizer_hash=tokenizer_hash, chat_template_hash=chat_template_hash,
            generation_config_hash=generation_config_hash, dropped_tensor_count=len(dropped),
            default_profile=default_profile,
        )
    except ValidationError as exc:  # pragma: no cover - inputs are checked above
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid model source: {exc}") from exc
    return SnapshotImport(
        config={"model_config": model_config, "family": str(family_ref) if family_ref is not None else None},
        tensors=tensors, infos=infos, alias_groups=sorted(canonical_groups),
        parameter_count=layout.unique_parameter_count, source=source, read_log=reader.read_log,
        model_files=dict(sorted(reader.contents.items())),
    )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class HFCausalLMAdapter:
    """Level 0 adapter for causal LMs that ``transformers`` implements natively.

    ``registry`` resolves the family descriptor a configuration names. Forward
    passes and generation run through ``wse.engines.hf_torch``: ``runtime``
    builds the module for a configuration and its stored tensors. The M1
    ``forward(tensors, inputs)`` contract carries no configuration, so model
    handles call ``runtime`` instead (``ModelHandle.forward``).
    """

    adapter_id = ADAPTER_ID
    version = ADAPTER_VERSION
    weights_hash_scheme = WEIGHTS_HASH_BLOBS

    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            capabilities=frozenset({CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT, CAP_FORWARD_LOGITS, CAP_GENERATE_TEXT}),
            shared_parameters=True,
        )

    def _parse(self, config: Mapping[str, Any]) -> tuple[dict[str, Any], FamilyDescriptor | None]:
        if not isinstance(config, Mapping) or set(config) != {"model_config", "family"}:
            raise _invalid("hf_causal_lm@1 configuration must be {\"model_config\": {...}, \"family\": "
                           "\"<id>@<version>\" | null}")
        model_config = config["model_config"]
        model_type = check_model_config(model_config)
        if config["family"] is None:
            return model_config, None
        if self._registry is None:
            raise _invalid("this hf_causal_lm@1 adapter has no registry to resolve family descriptors")
        ref = ExtensionRef.parse(config["family"])
        selected = select_family(self._registry, model_type, ref)
        assert selected is not None
        return model_config, selected[1]

    def canonical_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        self._parse(config)
        return dict(config)

    def family(self, config: Mapping[str, Any]) -> FamilyDescriptor | None:
        """The family descriptor that ``config`` records, resolved exactly; ``None`` at Level 0."""
        return self._parse(config)[1]

    def roles(self, config: Mapping[str, Any], tensors: Sequence[TensorInfo]) -> tuple[TensorRole, ...]:
        """The roles of ``tensors`` (a manifest's or an architecture's) under the family ``config`` records."""
        model_config, descriptor = self._parse(config)
        return () if descriptor is None else family_roles(model_config, descriptor, tensors)

    def describe(self, config: Mapping[str, Any]) -> Architecture:
        model_config, descriptor = self._parse(config)
        layout = module_layout(model_config, descriptor)
        tensors = [
            TensorInfo(name=_canonical(descriptor, name), shape=list(p.shape), dtype=layout.nominal_dtype,
                       axes=list(p.axes), element_count=_count(p.shape), module_name=name)
            for name, p in layout.parameters.items()
        ]
        groups = [sorted(_canonical(descriptor, name) for name in group) for group in layout.alias_groups]
        return Architecture(tensors=sorted(tensors, key=lambda info: info.name), input_semantics=INPUT_SEMANTICS,
                            output_semantics=OUTPUT_SEMANTICS, alias_groups=sorted(groups))

    def validate(self, manifest: ImportManifest | ModelManifest, tensors: Mapping[str, torch.Tensor]) -> Architecture:
        if str(manifest.adapter) != f"{self.adapter_id}@{self.version}":
            raise _invalid(f"manifest names adapter {manifest.adapter}, not {self.adapter_id}@{self.version}")
        architecture = self.describe(manifest.config)
        expected = {info.name: info for info in architecture.tensors}
        missing, unexpected = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
        if missing or unexpected:
            raise _invalid("tensor names do not match the architecture"
                           + (f"; missing: {missing}" if missing else "")
                           + (f"; unexpected: {unexpected}" if unexpected else ""),
                           missing=missing, unexpected=unexpected)
        recorded = {info.name: info.dtype for info in manifest.tensors} if isinstance(manifest, ModelManifest) else {}
        infos = []
        for name in sorted(expected):
            tensor, info = tensors[name], expected[name]
            dtype = dtype_name(tensor)
            if list(tensor.shape) != info.shape:
                raise _invalid(f"tensor {name!r} has shape {list(tensor.shape)}; the architecture requires {info.shape}",
                               tensor=name)
            if dtype not in _STORED_DTYPE_NAMES or recorded.get(name, dtype) != dtype:
                raise _invalid(f"tensor {name!r} has dtype {dtype}; expected {recorded.get(name, 'bf16/f16/f32')}",
                               tensor=name)
            infos.append(info.model_copy(update={"dtype": dtype}))
        require_finite(tensors)
        return architecture.model_copy(update={"tensors": infos})

    def runtime(self, config: Mapping[str, Any], tensors: Mapping[str, torch.Tensor], architecture: Architecture,
                *, device: str | torch.device | None = None) -> HFTorchModel:
        """Build the module for ``config`` with ``tensors`` (canonical names) on ``device``."""
        from wse.engines.hf_torch import build_model

        model_config, descriptor = self._parse(config)
        # Members of an alias group are one parameter: the module gets one tensor for all of them.
        source = {name: group[0] for group in architecture.alias_groups for name in group}
        by_module = {info.module_name or info.name: tensors[source.get(info.name, info.name)]
                     for info in architecture.tensors}
        return build_model(model_config, descriptor, by_module, device=device)

    def forward(self, tensors: Mapping[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                       "hf_causal_lm@1 needs the model configuration to run; call ModelHandle.forward or "
                       "ModelHandle.runtime, which build the module through wse.engines.hf_torch")


def _count(shape: tuple[int, ...]) -> int:
    count = 1
    for size in shape:
        count *= size
    return count
