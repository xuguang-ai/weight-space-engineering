"""Edits as registered operators, materialized patches, and exact replay.

Core functions work on in-memory ``ModelHandle`` objects and need no service,
database, or MCP:

* ``apply_operator(model, operator_id, parameters) -> EditResult`` runs one
  registered edit operator on one region and returns an immutable candidate,
  the materialized patch, and the per-tensor diff.
* ``materialize_patch(model, operations)`` applies several operations in list
  order to one snapshot through ``apply_operator``; if any step fails, no
  candidate is produced.
* ``replay_patch(model, patch, blocks=...)`` writes a materialized patch's
  stored after-values onto its base without running any editing operator,
  verifying every tensor hash along the way.

Edits support the stored dtypes BF16, F16, and F32 (M2 plan Section 3.2).
Operators compute in FP32 (a BF16 region arrives as its exact float32
widening), and results are written back to the stored dtype with
round-to-nearest-even; only the selected region is written, and bytes outside
it are preserved. Changed elements are those whose stored bits differ after
the write.

A step records its result in one of two forms. M1 models (raw-bytes weights
hash) keep M1's inline ``values``. Open-weight models (blob weights hash)
record a binary block (``wse.patch_blocks``, M2 plan Section 3.3), whose bytes
travel next to the patch in ``EditResult.blocks`` and in binary patch files.
M1's limit on elements per patch counts inline values and inline
``set_slice`` values, not operator results recorded as blocks.
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
from wse.patch_blocks import apply_block, make_block
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
from wse.tensors import (
    WEIGHTS_HASH_BLOBS,
    bits_to_values,
    hash_weights,
    resolve_selection,
    tensor_bits,
    tensor_hash,
    values_to_bits,
    weights_hash_scheme,
)

if TYPE_CHECKING:
    from wse.api import EditResult, ModelHandle
    from wse.registry import Registry

# Stored dtypes that edits support, with the NumPy dtype of M1's inline values (BF16 records only blocks).
EDITABLE_DTYPES = {"float32": np.float32, "float16": np.float16, "bfloat16": None}
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


def _changed(before: np.ndarray, after: np.ndarray) -> int:
    """Elements whose stored bit patterns differ (bits, not values, are compared)."""
    return int(np.count_nonzero(np.asarray(before).reshape(-1) != np.asarray(after).reshape(-1)))


_SQUARE_CHUNK = 1 << 22


def _sum_of_squares(bits: np.ndarray, dtype: str, other: np.ndarray | None = None) -> float:
    """Sum of squares in float64 of a tensor's values, or of the difference to ``other``, in chunks."""
    flat = bits.reshape(-1)
    other_flat = None if other is None else other.reshape(-1)
    total = 0.0
    for start in range(0, flat.size, _SQUARE_CHUNK):
        values = bits_to_values(flat[start:start + _SQUARE_CHUNK], dtype).astype(np.float64)
        if other_flat is not None:
            values = bits_to_values(other_flat[start:start + _SQUARE_CHUNK], dtype).astype(np.float64) - values
        total += float(np.sum(np.square(values)))
    return total


def records_blocks(model: ModelHandle) -> bool:
    """True when ``model``'s patches record binary blocks: open-weight models (the blob weights hash)."""
    return weights_hash_scheme(model.adapter) == WEIGHTS_HASH_BLOBS


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
        weights_hash=hash_weights(ordered, weights_hash_scheme(model.adapter)),
        architecture=architecture,
        adapter=model.adapter,
        _tensors=MappingProxyType(ordered),
        source=model.source,
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
        info = _info(base, name)
        before_tensor, after_tensor = base._tensors[name], candidate._tensors[name]
        if info.dtype not in EDITABLE_DTYPES:
            before_bits = before_tensor.numpy()
            after_bits = after_tensor.numpy()
            before = before_bits.astype(np.float64)
            after = after_bits.astype(np.float64)
            changed = _changed(before_bits.view(f"u{before_bits.itemsize}"), after_bits.view(f"u{after_bits.itemsize}"))
            before_square = float(np.sum(np.square(before)))
            delta = after - before
            delta_square = float(np.sum(np.square(delta)))
            max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
            finite = bool(np.all(np.isfinite(after)))
            before_hash = tensor_hash(name, before_tensor)
            after_hash = tensor_hash(name, after_tensor)
        else:
            before_bits, after_bits = tensor_bits(before_tensor), tensor_bits(after_tensor)
            before_square = _sum_of_squares(before_bits, info.dtype)
            before_hash = tensor_hash(name, before_tensor)
            if after_tensor is before_tensor:  # shared, unchanged
                changed, delta_square, max_abs, finite, after_hash = 0, 0.0, 0.0, True, before_hash
            else:
                differs = np.flatnonzero(before_bits.reshape(-1) != after_bits.reshape(-1))
                changed = int(differs.size)
                delta_square = _sum_of_squares(before_bits, info.dtype, other=after_bits)
                changed_after = bits_to_values(after_bits.reshape(-1)[differs], info.dtype).astype(np.float64)
                changed_before = bits_to_values(before_bits.reshape(-1)[differs], info.dtype).astype(np.float64)
                max_abs = float(np.max(np.abs(changed_after - changed_before))) if changed else 0.0
                finite = bool(np.all(np.isfinite(changed_after)))
                after_hash = tensor_hash(name, after_tensor)
        before_l2 = math.sqrt(before_square)
        l2 = math.sqrt(delta_square)
        diffs.append(
            TensorDiff(
                tensor=name,
                before_hash=before_hash,
                after_hash=after_hash,
                changed_elements=changed,
                element_count=int(before_tensor.numel()),
                l2_delta=l2,
                max_abs_delta=max_abs,
                before_l2=before_l2,
                relative_l2_delta=(l2 / before_l2) if before_l2 > 0 else None,
                all_finite=finite,
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
) -> tuple[ModelHandle, PatchStep, dict[str, bytes]]:
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
    info = _info(model, tensor)
    # Adapters with shared parameters list every alias group in their architecture
    # (AdapterCapabilities.shared_parameters); only members of a group are refused.
    if any(info.name in group for group in model.architecture.alias_groups):
        raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY,
                       f"tensor {info.name!r} belongs to an alias group (one shared parameter); edits of shared "
                       "parameters are not supported", details={"tensor": info.name})
    if info.dtype not in EDITABLE_DTYPES:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"tensor {info.name!r} has dtype {info.dtype}; edits support {sorted(EDITABLE_DTYPES)}",
            details={"tensor": info.name, "dtype": info.dtype},
        )
    slices = resolve_selection(selection, info.shape, tensor=info.name)
    entries = [AxisSlice(start=s.start, stop=s.stop, step=1) for s in slices]
    current = model._tensors[info.name]
    before_bits = tensor_bits(current)[slices]
    before = bits_to_values(before_bits, info.dtype)  # the stored values; BF16 as its exact float32 widening
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
    after_bits = values_to_bits(computed, info.dtype)  # round to nearest even in the stored dtype
    after = bits_to_values(after_bits, info.dtype)
    bad = int(np.count_nonzero(~np.isfinite(after)))
    if bad:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"{label} would write {bad} nonfinite value(s) into tensor {info.name!r} (after casting to "
            f"{info.dtype}); every weight must stay finite",
            details={"tensor": info.name, "nonfinite_count": bad},
        )
    updated = current.clone(memory_format=torch.contiguous_format)
    tensor_bits(updated)[slices] = after_bits
    tensors = dict(model._tensors)
    tensors[info.name] = updated
    candidate = _derive(model, tensors)
    resolved_parameters = resolved.model_dump(mode="json") if isinstance(resolved, BaseModel) else {}
    blocks: dict[str, bytes] = {}
    if records_blocks(model):
        block, data = make_block(current, updated, selection=entries)
        blocks[block.block_id] = data
        form: dict[str, Any] = {"block": block}
    else:
        form = {"values": shortest_floats(after)}
    step = PatchStep(
        op=label,
        operator=ref,
        tensor=info.name,
        selection=entries,
        parameters=resolved_parameters,
        dtype=info.dtype,
        before_hash=tensor_hash(info.name, current),
        after_hash=tensor_hash(info.name, updated),
        changed_elements=_changed(before_bits, after_bits),
        **form,
    )
    return candidate, step, blocks


def _result(base: ModelHandle, candidate: ModelHandle, steps: list[PatchStep],
            blocks: Mapping[str, bytes]) -> EditResult:
    from wse.api import EditResult

    patch = MaterializedPatch(base=base.identity, result_weights_hash=candidate.weights_hash, steps=steps)
    return EditResult(candidate=candidate, patch=patch, diff=tensor_diffs(base, candidate),
                      blocks=MappingProxyType(dict(blocks)))


def _inline_elements(step: PatchStep) -> int:
    """Elements that count toward M1's per-patch limit: inline values (M1 steps and ``set_slice`` input)."""
    if step.values is not None:
        return len(step.values)
    if "values" in step.parameters:
        return math.prod(entry.stop - entry.start for entry in step.selection)
    return 0


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

    candidate, step, blocks = _edit_step(
        model,
        operator_id,
        parameters,
        registry=registry or builtin_registry(),
        policy=policy or default_operation_policy(),
    )
    return _result(model, candidate, [step], blocks)


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
    blocks: dict[str, bytes] = {}
    written = 0
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping) or "op" not in operation:
            raise WSEError(ErrorCode.INVALID_PATCH, f"operation {index} must be an object with an 'op' field",
                           details={"operation_index": index})
        request = dict(operation)
        op = request.pop("op")
        try:
            current, step, step_blocks = _edit_step(current, op, request, registry=registry, policy=policy)
        except WSEError as exc:
            raise _wrap_step_error(index, op, exc) from exc
        blocks.update(step_blocks)
        written += _inline_elements(step)
        if written > limits.max_patch_elements:
            raise WSEError(
                ErrorCode.INVALID_PATCH,
                f"a patch may write at most {limits.max_patch_elements} inline elements in total; operation "
                f"{index} brings it to {written}",
                details={"operation_index": index, "max_patch_elements": limits.max_patch_elements},
            )
        steps.append(step)
    return _result(model, current, steps, blocks)


def replay_patch(
    model: ModelHandle, patch: MaterializedPatch, *, blocks: Mapping[str, bytes] | None = None
) -> ModelHandle:
    """Recreate a patch's candidate from its stored after-values.

    No editing operator runs: each step writes its recorded values or its
    binary block (from ``blocks``, block ID -> bytes, as stored with the
    patch), and the base identity, every per-step tensor hash and blob ID, and
    the final weights hash are verified, so a replay reproduces identical
    tensor bytes or fails.
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
        if step.block is not None:
            data = (blocks or {}).get(step.block.block_id)
            if data is None:
                raise WSEError(ErrorCode.INVALID_PATCH, f"step {index}: the binary block {step.block.block_id} is "
                               "missing; replay reads the blocks stored with the patch", details={"step": index})
            try:
                updated = apply_block(current, step.block, data)
            except WSEError as exc:
                raise WSEError(ErrorCode.INVALID_PATCH, f"step {index}: {exc.message}",
                               details={**exc.details, "step": index}) from exc
            if tensor_hash(info.name, updated) != step.after_hash:
                raise WSEError(ErrorCode.INVALID_PATCH, f"step {index}: replayed tensor {info.name!r} does not "
                               "match the recorded after-hash", details={"step": index})
            tensors[info.name] = updated
            continue
        if EDITABLE_DTYPES[step.dtype] is None or step.values is None:
            raise WSEError(ErrorCode.INVALID_PATCH, f"step {index} records no values for a {step.dtype} tensor")
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
