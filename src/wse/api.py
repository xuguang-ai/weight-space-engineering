"""Direct library functions and the managed ``Engine`` facade.

Direct use (``load_model``) works in memory: it needs no service, database, or
MCP, and its handles carry no experiment IDs. Results of direct use are
labeled ``unmanaged`` and cannot support managed-enforcement claims.

The ``Engine`` wraps the same implementations with persistence: it imports
models into the content-addressed store, assigns checkpoint IDs, and serves
only READY checkpoints. Storage modules are imported lazily so that direct use
never loads SQLite.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from pydantic import ValidationError

from wse.adapters.base import Architecture, ModelAdapter
from wse.registry import Registry, builtin_registry
from wse.schemas import (
    BundleManifest,
    ErrorCode,
    ExtensionRef,
    ImportManifest,
    Management,
    MaterializedPatch,
    ModelIdentity,
    ModelManifest,
    TensorDiff,
    Usage,
    WSEError,
    canonical_json,
    json_hash,
)
from wse.tensors import normalize_tensors, resolve_selection, weights_hash

if TYPE_CHECKING:
    from wse.store import ArtifactStore, RecoveryReport

DEFAULT_MAX_MODEL_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
BUNDLE_MANIFEST_NAME = "wse_bundle.json"


# ---------------------------------------------------------------------------
# Core handles and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelHandle:
    """An in-memory model with content identity and adapter metadata.

    The handle owns private tensor copies; accessors return copies, so callers
    cannot change the weights behind the recorded hash.
    """

    adapter_ref: ExtensionRef
    config_json: str
    config_hash: str
    weights_hash: str
    architecture: Architecture
    adapter: ModelAdapter = field(repr=False)
    _tensors: Mapping[str, torch.Tensor] = field(repr=False)

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            adapter=self.adapter_ref, config_hash=self.config_hash, weights_hash=self.weights_hash
        )

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self.config_json)

    @property
    def parameter_count(self) -> int:
        return self.architecture.parameter_count

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(info.name for info in self.architecture.tensors)

    def tensor(self, name: str) -> torch.Tensor:
        if name not in self._tensors:
            raise WSEError(
                ErrorCode.NOT_FOUND,
                f"tensor {name!r} does not exist; tensor names must match exactly",
                details={"available_tensors": list(self.tensor_names())},
            )
        return self._tensors[name].clone()

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: self._tensors[name].clone() for name in self.tensor_names()}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Unmanaged inference through the adapter (direct library use only)."""
        return self.adapter.forward(self._tensors, inputs)


@dataclass(frozen=True)
class EditResult:
    """Outcome of a direct edit: the immutable candidate, exact patch, and diff."""

    candidate: ModelHandle
    patch: MaterializedPatch
    diff: tuple[TensorDiff, ...]
    management: Management = "unmanaged"


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of a direct evaluation, bound to its input and config identities."""

    metrics: Mapping[str, float]
    sample_count: int
    model: ModelIdentity
    evaluator: ExtensionRef
    evaluator_config_hash: str
    data_hash: str
    usage: Usage
    management: Management = "unmanaged"


def _resolve_adapter(registry: Registry, adapter: str | ExtensionRef) -> tuple[ExtensionRef, ModelAdapter]:
    ref = ExtensionRef.parse(adapter)
    return ref, registry.resolve("adapter", ref.extension_id, ref.version)


def _build_handle(
    adapter_ref: ExtensionRef,
    adapter: ModelAdapter,
    manifest: ImportManifest | ModelManifest,
    tensors: dict[str, torch.Tensor],
) -> ModelHandle:
    architecture = adapter.validate(manifest, tensors)
    ordered = {info.name: tensors[info.name] for info in architecture.tensors}
    return ModelHandle(
        adapter_ref=adapter_ref,
        config_json=canonical_json(manifest.config),
        config_hash=json_hash(manifest.config),
        weights_hash=weights_hash(ordered),
        architecture=architecture,
        adapter=adapter,
        _tensors=MappingProxyType(ordered),
    )


def _read_json_file(path: Path, *, what: str) -> Any:
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise WSEError(ErrorCode.INVALID_MODEL, f"{what} {path.name} exceeds {MAX_CONFIG_BYTES} bytes")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WSEError(ErrorCode.INVALID_MODEL, f"{what} {path.name} is not valid JSON: {exc}") from exc


def _load_safetensors(path: Path, max_bytes: int) -> dict[str, torch.Tensor]:
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    size = path.stat().st_size
    if size > max_bytes:
        raise WSEError(
            ErrorCode.INVALID_MODEL,
            f"weights file {path.name} is {size} bytes; the limit is {max_bytes} bytes",
            details={"size_bytes": size, "max_bytes": max_bytes},
        )
    try:
        return load_file(str(path))
    except (SafetensorError, OSError, ValueError) as exc:
        raise WSEError(ErrorCode.INVALID_MODEL, f"cannot read safetensors file {path.name}: {exc}") from exc


def _as_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "a model configuration (mapping or JSON path) is required")
    if isinstance(config, (str, Path)):
        config = _read_json_file(Path(config), what="configuration")
    if not isinstance(config, Mapping):
        raise WSEError(ErrorCode.INVALID_MODEL, "a model configuration must be a JSON object")
    return dict(config)


def load_model(
    source: Mapping[str, Any] | str | Path,
    adapter: str | ExtensionRef,
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    registry: Registry | None = None,
    max_bytes: int = DEFAULT_MAX_MODEL_BYTES,
) -> ModelHandle:
    """Load a model in memory through a registered adapter (direct use).

    ``source`` is a mapping of tensor name to ``torch.Tensor``/``numpy.ndarray``
    or the path of a ``.safetensors`` file. ``adapter`` is an exact reference
    such as ``"mlp@1"``; unknown adapters and versions are rejected.
    ``config`` is the adapter configuration or the path of a JSON file.
    Requires neither SQLite nor MCP.
    """
    registry = registry or builtin_registry()
    adapter_ref, implementation = _resolve_adapter(registry, adapter)
    if isinstance(source, (str, Path)):
        tensors: Mapping[str, Any] = _load_safetensors(Path(source), max_bytes)
    else:
        tensors = source
    manifest = _import_manifest({"adapter": adapter_ref, "config": _as_config(config)})
    return _build_handle(adapter_ref, implementation, manifest, normalize_tensors(tensors))


def _import_manifest(value: ImportManifest | Mapping[str, Any]) -> ImportManifest:
    if isinstance(value, ImportManifest):
        return value
    try:
        return ImportManifest.model_validate(dict(value))
    except ValidationError as exc:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"invalid import manifest: {exc}",
            details={"errors": json.loads(exc.json(include_url=False, include_input=False))},
        ) from exc


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def _bundle_member(root: Path, relative: str, *, what: str) -> Path:
    """Resolve a bundle-relative path, rejecting any escape from the bundle."""
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or "\x00" in relative
        or pure.is_absolute()
        or any(part in ("..", "") for part in relative.split("/"))
        or relative.startswith("~")
    ):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{what} path {relative!r} must be a plain relative path inside the bundle directory",
            details={"path": relative},
        )
    candidate = root / pure
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        raise WSEError(ErrorCode.NOT_FOUND, f"{what} file {relative!r} does not exist in the bundle") from None
    if not resolved.is_relative_to(root):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{what} path {relative!r} resolves outside the bundle directory",
            details={"path": relative},
        )
    if not resolved.is_file():
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"{what} path {relative!r} is not a regular file")
    return resolved


def read_bundle(path: str | Path, *, max_bytes: int = DEFAULT_MAX_MODEL_BYTES) -> tuple[ImportManifest, dict[str, torch.Tensor]]:
    """Read a bundle directory (or its ``wse_bundle.json``) into an import request.

    The bundle's filesystem location is never recorded in the manifest;
    provenance comes only from the bundle file's operator-supplied labels.
    """
    path = Path(path)
    if not path.exists():
        raise WSEError(
            ErrorCode.NOT_FOUND,
            f"bundle {path} does not exist; pass a directory containing {BUNDLE_MANIFEST_NAME}",
        )
    if path.is_dir():
        path = path / BUNDLE_MANIFEST_NAME
    try:
        root = path.parent.resolve(strict=True)
    except FileNotFoundError:
        raise WSEError(ErrorCode.NOT_FOUND, f"bundle directory {path.parent} does not exist") from None
    bundle_file = _bundle_member(root, path.name, what="bundle manifest")
    try:
        bundle = BundleManifest.model_validate(_read_json_file(bundle_file, what="bundle manifest"))
    except ValidationError as exc:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"invalid {BUNDLE_MANIFEST_NAME}: {exc}") from exc
    config = _as_config(_bundle_member(root, bundle.config, what="config"))
    tensors = _load_safetensors(_bundle_member(root, bundle.weights, what="weights"), max_bytes)
    return _import_manifest(bundle.to_import_manifest(config)), tensors


# ---------------------------------------------------------------------------
# Managed engine
# ---------------------------------------------------------------------------


class Engine:
    """Managed facade over the artifact store and registry.

    Opening an engine applies database migrations and runs the startup
    recovery scan.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        registry: Registry | None = None,
        recover: bool = True,
        max_bytes: int = DEFAULT_MAX_MODEL_BYTES,
    ) -> None:
        from wse.store import ArtifactStore

        self.registry = registry or builtin_registry()
        self.max_bytes = max_bytes
        self.store: ArtifactStore = ArtifactStore(workspace)
        self.last_recovery: RecoveryReport | None = self.store.recover() if recover else None

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def import_model(
        self, manifest: ImportManifest | Mapping[str, Any], tensors: Mapping[str, Any]
    ) -> ModelManifest:
        """Validate and store a model; return its READY manifest.

        Importing an identical request again returns the existing checkpoint;
        identical weights are stored once.
        """
        request = _import_manifest(manifest)
        adapter_ref, adapter = _resolve_adapter(self.registry, request.adapter)
        normalized = normalize_tensors(tensors)
        size = sum(t.numel() * t.element_size() for t in normalized.values())
        if size > self.max_bytes:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"model tensors total {size} bytes; the limit is {self.max_bytes} bytes",
            )
        handle = _build_handle(adapter_ref, adapter, request, normalized)
        architecture = handle.architecture
        if architecture.alias_groups:
            raise WSEError(ErrorCode.INVALID_MODEL, "shared parameters are not supported in this milestone")
        dedup_key = json_hash(
            {"import": request.model_dump(mode="json"), "weights_hash": handle.weights_hash}
        )
        model_manifest = ModelManifest(
            checkpoint_id=self.store.new_checkpoint_id(),
            weights_hash=handle.weights_hash,
            config_hash=handle.config_hash,
            adapter=adapter_ref,
            config=request.config,
            tensors=architecture.tensors,
            parameter_count=architecture.parameter_count,
            input_semantics=architecture.input_semantics.model_copy(
                update={"description": request.input_description or architecture.input_semantics.description}
            ),
            output_semantics=architecture.output_semantics.model_copy(
                update={"description": request.output_description or architecture.output_semantics.description}
            ),
            alias_groups=architecture.alias_groups,
            provenance=request.provenance,
            license=request.license,
        )
        return self.store.commit_checkpoint(
            model_manifest,
            dict(handle._tensors),
            dedup_key=dedup_key,
            event_type="checkpoint.imported",
        )

    def import_bundle(self, path: str | Path) -> ModelManifest:
        manifest, tensors = read_bundle(path, max_bytes=self.max_bytes)
        return self.import_model(manifest, tensors)

    def load_manifest(self, checkpoint_id: str) -> ModelManifest:
        """Return the manifest of a READY checkpoint."""
        return self.store.load_manifest(checkpoint_id)

    def list_models(self) -> list[ModelManifest]:
        """Manifests of all READY checkpoints, oldest first."""
        manifests = []
        for checkpoint_id in self.store.list_ready():
            try:
                manifests.append(self.store.load_manifest(checkpoint_id))
            except WSEError as exc:
                if exc.code != ErrorCode.ARTIFACT_CORRUPT:
                    raise
        return manifests

    def read_tensor(self, checkpoint_id: str, tensor: str, selection: Any) -> torch.Tensor:
        """Read an exact-name tensor region from a READY checkpoint."""
        manifest = self.store.load_manifest(checkpoint_id)
        info = manifest.tensor_info(tensor)
        slices = resolve_selection(selection, info.shape, tensor=tensor)
        return self.store.read_slice(checkpoint_id, tensor, slices)

    def open_checkpoint(self, checkpoint_id: str) -> ModelHandle:
        """Load a READY checkpoint into an in-memory handle (hash-verified)."""
        manifest = self.store.load_manifest(checkpoint_id)
        adapter_ref, adapter = _resolve_adapter(self.registry, manifest.adapter)
        handle = _build_handle(adapter_ref, adapter, manifest, self.store.load_tensors(checkpoint_id))
        if handle.identity != manifest.identity:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"checkpoint {checkpoint_id} identity mismatch")
        return handle
