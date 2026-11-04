"""Edits as registered operators, materialized patches, and exact replay.

Core functions work on in-memory ``ModelHandle`` objects and need no service,
database, or MCP:

* ``apply_operator(model, operator_id, parameters) -> EditResult`` runs one
  registered edit operator on one region and returns an immutable candidate,
  the materialized patch, and the per-tensor diff.
* ``materialize_patch(model, operations)`` applies several operations in list
  order to one snapshot through ``apply_operator``; if any step fails, no
  candidate is produced.
* ``replay_patch(model, patch)`` writes a materialized patch's stored
  after-values onto its base without running any editing operator, verifying
  every tensor hash along the way.

Edits are computed in FP32 and cast back to the stored dtype; only the
selected region is written, and bytes outside it are preserved. Changed
elements are those whose stored bytes differ after the write.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pydantic import BaseModel

from wse.adapters.base import CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT
from wse.operators.base import TensorRegion
from wse.operators.tensor import shortest_floats
from wse.policy import OperationPolicy, default_operation_policy, require_capabilities
from wse.schemas import (
    AxisSlice,
    ErrorCode,
    ExtensionRef,
    ImportManifest,
    MaterializedPatch,
    PatchStep,
    TensorDiff,
    TensorInfo,
    WSEError,
)
from wse.tensors import resolve_selection, tensor_hash, weights_hash

if TYPE_CHECKING:
    from wse.api import EditResult, ModelHandle
    from wse.registry import Registry

EDITABLE_DTYPES = {"float32": np.float32, "float16": np.float16}
_EDIT_CAPABILITIES = (CAP_WEIGHTS_READ, CAP_WEIGHTS_SLICE_EDIT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _info(model: ModelHandle, tensor: Any) -> TensorInfo:
    for info in model.architecture.tensors:
        if info.name == tensor:
            return info
    raise WSEError(
        ErrorCode.NOT_FOUND,
        f"tensor {tensor!r} does not exist; tensor names must match exactly",
        details={"tensor": str(tensor), "available_tensors": list(model.tensor_names())},
    )


def _bits(array: np.ndarray) -> np.ndarray:
    """The stored bit patterns of ``array`` (bytes, not values, are compared)."""
    flat = np.ascontiguousarray(array).reshape(-1)
    return flat.view(f"u{flat.dtype.itemsize}")


def _changed(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.count_nonzero(_bits(before) != _bits(after)))


def _derive(model: ModelHandle, tensors: Mapping[str, torch.Tensor]) -> ModelHandle:
    """A new handle with the same adapter and configuration and new weights.

    The adapter re-validates shapes, dtypes, and finiteness of every tensor.
    Unchanged tensors are shared with the parent; handles never mutate them.
    """
    from wse.api import ModelHandle

    request = ImportManifest(adapter=model.adapter_ref, config=model.config)
    architecture = model.adapter.validate(request, tensors)
    ordered = {info.name: tensors[info.name] for info in architecture.tensors}
    return ModelHandle(
        adapter_ref=model.adapter_ref,
        config_json=model.config_json,
        config_hash=model.config_hash,
        weights_hash=weights_hash(ordered),
        architecture=architecture,
        adapter=model.adapter,
        _tensors=MappingProxyType(ordered),
    )


def _resolve_operator(
    operator_id: str | ExtensionRef, registry: Registry, policy: OperationPolicy
) -> tuple[str, ExtensionRef, Any]:
    """Resolve an exact ``id@version`` or a policy alias to an edit operator."""
    if isinstance(operator_id, ExtensionRef) or (isinstance(operator_id, str) and "@" in operator_id):
        ref = ExtensionRef.parse(operator_id)
        label = str(ref)
    else:
        ref = policy.edit_operator(operator_id)
        label = operator_id
    operator = registry.resolve("operator", ref.extension_id, ref.version)
    if operator.spec.effect != "edit":
        raise WSEError(
            ErrorCode.POLICY_DENIED,
            f"operator {ref} is a {operator.spec.effect} operator, not an edit operator",
            details={"operator": str(ref)},
        )
    return label, ref, operator


def tensor_diffs(base: ModelHandle, candidate: ModelHandle) -> tuple[TensorDiff, ...]:
    """Per-tensor differences between two handles of the same architecture."""
    diffs = []
    for name in base.tensor_names():
        before = base._tensors[name].numpy()
        after = candidate._tensors[name].numpy()
        delta = after.astype(np.float64) - before.astype(np.float64)
        before_l2 = math.sqrt(float(np.sum(np.square(before, dtype=np.float64))))
        l2 = math.sqrt(float(np.sum(np.square(delta))))
        diffs.append(
            TensorDiff(
                tensor=name,
                before_hash=tensor_hash(name, base._tensors[name]),
                after_hash=tensor_hash(name, candidate._tensors[name]),
                changed_elements=_changed(before, after),
                element_count=int(before.size),
                l2_delta=l2,
                max_abs_delta=float(np.max(np.abs(delta))) if delta.size else 0.0,
                before_l2=before_l2,
                relative_l2_delta=(l2 / before_l2) if before_l2 > 0 else None,
                all_finite=bool(np.all(np.isfinite(after))),
            )
        )
    return tuple(diffs)


def _wrap_step_error(index: int, op: Any, exc: WSEError) -> WSEError:
    """Report a failed operation as ``INVALID_PATCH`` with its position.

    Policy, capability, and registration failures keep their own codes.
    """
    details = {**exc.details, "operation_index": index, "op": str(op)}
    if exc.code in (ErrorCode.INVALID_ARGUMENT, ErrorCode.NOT_FOUND, ErrorCode.INVALID_MODEL):
        details["cause"] = exc.code.value
        if exc.code == ErrorCode.NOT_FOUND and "tensor" not in exc.details:
            return WSEError(exc.code, f"operation {index} ({op}): {exc.message}", details=details)
        return WSEError(ErrorCode.INVALID_PATCH, f"operation {index} ({op}): {exc.message}", details=details)
    return WSEError(exc.code, f"operation {index} ({op}): {exc.message}", retryable=exc.retryable, details=details)


# ---------------------------------------------------------------------------
# Core edits
# ---------------------------------------------------------------------------


def _edit_step(
    model: ModelHandle,
    operator_id: str | ExtensionRef,
    parameters: Mapping[str, Any],
    *,
    registry: Registry,
    policy: OperationPolicy,
) -> tuple[ModelHandle, PatchStep]:
    if not isinstance(parameters, Mapping):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "edit parameters must be an object with tensor and selection")
    raw = dict(parameters)
    missing = [name for name in ("tensor", "selection") if name not in raw]
    if missing:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"edit parameters are missing {missing}")
    tensor, selection = raw.pop("tensor"), raw.pop("selection")
    label, ref, operator = _resolve_operator(operator_id, registry, policy)
    capabilities = model.adapter.capabilities()
    require_capabilities(capabilities, operator.spec, adapter=str(model.adapter_ref), always=_EDIT_CAPABILITIES)
    if capabilities.shared_parameters or model.architecture.alias_groups:
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, "edits of models with shared parameters are not supported")
    info = _info(model, tensor)
    if info.dtype not in EDITABLE_DTYPES:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"tensor {info.name!r} has dtype {info.dtype}; edits support {sorted(EDITABLE_DTYPES)}",
            details={"tensor": info.name, "dtype": info.dtype},
        )
    slices = resolve_selection(selection, info.shape, tensor=info.name)
    entries = [AxisSlice(start=s.start, stop=s.stop, step=1) for s in slices]
    current = model._tensors[info.name]
    before = current.numpy()[slices]
    view = before.view()
    view.flags.writeable = False
    region = TensorRegion(
        tensor=info.name,
        dtype=info.dtype,
        axes=tuple(info.axes),
        tensor_shape=tuple(info.shape),
        selection=tuple(entries),
        values=view,
    )
    resolved = operator.resolve_parameters(raw, region)
    output = operator.apply(region, resolved)
    try:
        computed = np.asarray(output, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise WSEError(ErrorCode.WORKER_FAILED, f"operator {ref} returned non-numeric values: {exc}") from exc
    if computed.shape != before.shape:
        raise WSEError(
            ErrorCode.WORKER_FAILED,
            f"operator {ref} returned shape {list(computed.shape)} for a region of shape {list(before.shape)}",
        )
    with np.errstate(over="ignore", invalid="ignore"):
        after = computed.astype(EDITABLE_DTYPES[info.dtype])
    bad = int(np.count_nonzero(~np.isfinite(after)))
    if bad:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{label} would write {bad} nonfinite value(s) into tensor {info.name!r} (after casting to "
            f"{info.dtype}); every weight must stay finite",
            details={"tensor": info.name, "nonfinite_count": bad},
        )
    updated = current.clone()
    updated.numpy()[slices] = after
    tensors = dict(model._tensors)
    tensors[info.name] = updated
    candidate = _derive(model, tensors)
    resolved_parameters = resolved.model_dump(mode="json") if isinstance(resolved, BaseModel) else {}
    step = PatchStep(
        op=label,
        operator=ref,
        tensor=info.name,
        selection=entries,
        parameters=resolved_parameters,
        dtype=info.dtype,
        before_hash=tensor_hash(info.name, current),
        after_hash=tensor_hash(info.name, updated),
        values=shortest_floats(after),
        changed_elements=_changed(before, after),
    )
    return candidate, step


def _result(base: ModelHandle, candidate: ModelHandle, steps: list[PatchStep]) -> EditResult:
    from wse.api import EditResult

    patch = MaterializedPatch(base=base.identity, result_weights_hash=candidate.weights_hash, steps=steps)
    return EditResult(candidate=candidate, patch=patch, diff=tensor_diffs(base, candidate))


def apply_operator(
    model: ModelHandle,
    operator_id: str | ExtensionRef,
    parameters: Mapping[str, Any],
    *,
    registry: Registry | None = None,
    policy: OperationPolicy | None = None,
) -> EditResult:
    """Apply one registered edit operator to an in-memory model (direct use).

    ``operator_id`` is an exact reference such as ``"scale_slice@1"`` or a
    public alias permitted by ``policy`` (``"scale_slice"``). ``parameters``
    holds ``tensor`` (exact name), ``selection`` (one ``{start, stop, step}``
    object per axis), and the operator's own parameters. The input model is
    never modified; the result carries a new immutable candidate.
    """
    from wse.registry import builtin_registry

    candidate, step = _edit_step(
        model,
        operator_id,
        parameters,
        registry=registry or builtin_registry(),
        policy=policy or default_operation_policy(),
    )
    return _result(model, candidate, [step])


def materialize_patch(
    model: ModelHandle,
    operations: Sequence[Mapping[str, Any]],
    *,
    registry: Registry | None = None,
    policy: OperationPolicy | None = None,
) -> EditResult:
    """Apply ``operations`` in list order to one snapshot through ``apply_operator``.

    Each operation is ``{"op": alias_or_ref, "tensor": ..., "selection": ...,
    <operator parameters>}``. Shapes and finiteness are checked after every
    step and for the final result. Any failure raises (``INVALID_PATCH`` for
    invalid content, with ``details["operation_index"]``) and produces no
    candidate. Patch size limits come from ``policy.limits``.
    """
    from wse.registry import builtin_registry

    registry = registry or builtin_registry()
    policy = policy or default_operation_policy()
    limits = policy.limits
    if isinstance(operations, (str, bytes, Mapping)) or not isinstance(operations, Sequence) or not operations:
        raise WSEError(ErrorCode.INVALID_PATCH, "a patch needs a nonempty list of operations")
    if len(operations) > limits.max_patch_operations:
        raise WSEError(
            ErrorCode.INVALID_PATCH,
            f"a patch may contain at most {limits.max_patch_operations} operations; got {len(operations)}",
            details={"max_patch_operations": limits.max_patch_operations},
        )
    current = model
    steps: list[PatchStep] = []
    written = 0
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping) or "op" not in operation:
            raise WSEError(ErrorCode.INVALID_PATCH, f"operation {index} must be an object with an 'op' field",
                           details={"operation_index": index})
        request = dict(operation)
        op = request.pop("op")
        try:
            current, step = _edit_step(current, op, request, registry=registry, policy=policy)
        except WSEError as exc:
            raise _wrap_step_error(index, op, exc) from exc
        written += len(step.values)
        if written > limits.max_patch_elements:
            raise WSEError(
                ErrorCode.INVALID_PATCH,
                f"a patch may write at most {limits.max_patch_elements} elements in total; operation "
                f"{index} brings it to {written}",
                details={"operation_index": index, "max_patch_elements": limits.max_patch_elements},
            )
        steps.append(step)
    return _result(model, current, steps)


def replay_patch(model: ModelHandle, patch: MaterializedPatch) -> ModelHandle:
    """Recreate a patch's candidate from its stored after-values.

    No editing operator runs: each step writes its recorded values, and the
    base identity, every per-step tensor hash, and the final weights hash are
    verified, so a replay reproduces identical tensor bytes or fails.
    """
    if model.identity != patch.base:
        raise WSEError(
            ErrorCode.VERSION_CONFLICT,
            "this exact patch was materialized against a different base model; "
            "an exact patch applies only to its own base revision",
            details={"patch_base": patch.base.model_dump(mode="json"),
                     "model": model.identity.model_dump(mode="json")},
        )
    tensors = dict(model._tensors)
    for index, step in enumerate(patch.steps):
        info = _info(model, step.tensor)
        if step.dtype != info.dtype or step.dtype not in EDITABLE_DTYPES:
            raise WSEError(ErrorCode.INVALID_PATCH, f"step {index} has dtype {step.dtype}, tensor has {info.dtype}")
        current = tensors[info.name]
        if tensor_hash(info.name, current) != step.before_hash:
            raise WSEError(ErrorCode.INVALID_PATCH, f"step {index}: tensor {info.name!r} does not match the "
                           "recorded before-hash", details={"step": index})
        slices = resolve_selection(step.selection, info.shape, tensor=info.name)
        shape = tuple(s.stop - s.start for s in slices)
        values = np.array(step.values, dtype=EDITABLE_DTYPES[step.dtype])
        if values.size != math.prod(shape):
            raise WSEError(ErrorCode.INVALID_PATCH, f"step {index} records {values.size} values for a region of "
                           f"shape {list(shape)}", details={"step": index})
        updated = current.clone()
        updated.numpy()[slices] = values.reshape(shape)
        if tensor_hash(info.name, updated) != step.after_hash:
            raise WSEError(ErrorCode.INVALID_PATCH, f"step {index}: replayed tensor {info.name!r} does not match "
                           "the recorded after-hash", details={"step": index})
        tensors[info.name] = updated
    candidate = _derive(model, tensors)
    if candidate.weights_hash != patch.result_weights_hash:
        raise WSEError(ErrorCode.INVALID_PATCH, "replayed weights do not match the patch's result hash")
    return candidate
