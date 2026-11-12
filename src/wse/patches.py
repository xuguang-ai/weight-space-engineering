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
M1's limit on elements per patch (``max_patch_elements``) counts what is
inline: M1 steps' values, ``set_slice`` values, and the inline numbers of an
operator's ``inline_parameters`` (``project_out``'s direction and columns,
``add_low_rank``'s inline factor columns); operator results recorded as blocks
count nothing.

Operators are resolved through the policy. A public alias must be one the
policy lists. In direct use an exact reference (``"project_out@1"``) names any
registered edit operator; in a managed preview (``managed=True``) it must be
one that the policy lists, and any other is refused exactly like an alias the
policy does not list (``POLICY_DENIED``).

Every edit refuses members of an alias group (``UNSUPPORTED_CAPABILITY``) and,
first, tensors whose role the model's family descriptor protects
(``POLICY_DENIED``, ``wse.policy.require_unprotected``). A policy that sets
``refuse_whole_tensor_scaling`` also refuses a ``scale_slice`` of a whole
tensor (``wse.policy.require_partial_scaling``). Replay writes a recorded
patch as recorded and checks no protection.

Vector references (``wse.operators.base``): parameters that an operator
declares in ``reference_parameters`` may name a vector of a stored analysis
result. A managed preview resolves every reference of a patch with
``resolve_references`` through its run-scoped reader before any weights are
read, and passes the vectors to ``materialize_patch``; direct use has no
stored results and refuses references. Replay never reads a reference, since
a step records its after-values. ``summarize_vector_parameters`` gives the
agent-facing form of recorded parameters, whose inline vectors it echoes as
``{length, sha256}``.

Tensor references (``wse.operators.submitted``): parameters that an operator
declares in ``tensor_parameters`` name a tensor of a stored tensor file of
the run, ``wse://tensors/<sha256>#<name>``. A managed preview resolves them
with ``resolve_tensor_references`` before any weights are read: the file
must belong to the run's experiment, and the tensor must have the target's
stored dtype and the selection's shape, with finite values.
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
from wse.adapters.families import model_family
from wse.operators.base import TensorRegion, reference_entries
from wse.operators.directions import vector_summary
from wse.patch_blocks import apply_block, make_block
from wse.operators.tensor import shortest_floats
from wse.policy import (
    OperationPolicy,
    default_operation_policy,
    require_capabilities,
    require_partial_scaling,
    require_unprotected,
)
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
    dtype_name,
    hash_weights,
    resolve_selection,
    tensor_bits,
    tensor_hash,
    values_to_bits,
    weights_hash_scheme,
)

if TYPE_CHECKING:
    from wse.analysis import ReadResult
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
    operator_id: str | ExtensionRef, registry: Registry, policy: OperationPolicy, *, managed: bool = False
) -> tuple[str, ExtensionRef, Any]:
    """Resolve an exact ``id@version`` or a policy alias to an edit operator.

    A ``managed`` edit accepts an exact reference only when the policy lists
    it; direct use accepts that of any registered edit operator.
    """
    if isinstance(operator_id, ExtensionRef) or (isinstance(operator_id, str) and "@" in operator_id):
        ref = ExtensionRef.parse(operator_id)
        label = str(ref)
        if managed and ref not in policy.edits.values():
            # Refused exactly like an alias that the policy does not list (OperationPolicy.edit_operator).
            raise WSEError(
                ErrorCode.POLICY_DENIED,
                f"edit operation {label!r} is not permitted; permitted operations: {sorted(policy.edits)}",
                details={"op": label, "permitted_operations": sorted(policy.edits)},
            )
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
# Vector parameters
# ---------------------------------------------------------------------------


def _declared(operator: Any, declaration: str) -> tuple[str, ...]:
    return tuple(getattr(operator, declaration, ()) or ())


def resolve_references(
    operations: Sequence[Mapping[str, Any]],
    read_result: ReadResult,
    *,
    registry: Registry | None = None,
    policy: OperationPolicy | None = None,
) -> dict[str, np.ndarray]:
    """Resolve the vector references of a patch's operations; return each vector by its reference.

    Managed previews call this before any weights are read, with their
    run-scoped reader: a run reads only results its own experiment produced,
    and any other URI fails exactly like one that does not exist
    (``NOT_FOUND``). A failure names the operation's position, as step
    failures do. Operations that are malformed or whose operator does not
    resolve are left for ``materialize_patch`` to report in order.
    """
    from wse.analysis import resolve_vector_reference
    from wse.registry import builtin_registry

    registry = registry or builtin_registry()
    policy = policy or default_operation_policy()
    resolved: dict[str, np.ndarray] = {}
    if (isinstance(operations, (str, bytes, Mapping)) or not isinstance(operations, Sequence)
            or len(operations) > policy.limits.max_patch_operations):
        return resolved
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping) or "op" not in operation:
            continue
        try:
            _, _, operator = _resolve_operator(operation["op"], registry, policy, managed=True)
        except WSEError:
            continue
        for name in _declared(operator, "reference_parameters"):
            for key, reference in reference_entries(name, operation.get(name)):
                if reference in resolved:
                    continue
                try:
                    resolved[reference] = resolve_vector_reference(
                        reference, read_result, parameter=key, max_length=policy.limits.max_analysis_elements)
                except WSEError as exc:
                    raise _wrap_step_error(index, operation["op"], exc) from exc
    return resolved


def resolve_tensor_references(
    operations: Sequence[Mapping[str, Any]],
    tensor_file: Any,
    manifest: Any,
    *,
    registry: Registry | None = None,
    policy: OperationPolicy | None = None,
) -> dict[str, np.ndarray]:
    """Resolve the tensor references of a patch's operations; return each tensor's float32 values by reference.

    ``tensor_file(uri)`` returns the path of a stored tensor file that the
    caller may read (a run reads only its own experiment's files, after a
    hash check) and fails with ``NOT_FOUND`` for any other. ``manifest`` is
    the base checkpoint's: each tensor must have its target's stored dtype
    and the shape of the operation's selection, and every value must be
    finite. Operations that are malformed, whose operator does not resolve,
    or whose target or selection is invalid are left for
    ``materialize_patch`` to report in order.
    """
    import re

    from safetensors import safe_open

    from wse.operators.submitted import TENSOR_REFERENCE_PATTERN
    from wse.registry import builtin_registry

    registry = registry or builtin_registry()
    policy = policy or default_operation_policy()
    resolved: dict[str, np.ndarray] = {}
    if (isinstance(operations, (str, bytes, Mapping)) or not isinstance(operations, Sequence)
            or len(operations) > policy.limits.max_patch_operations):
        return resolved
    infos = {info.name: info for info in manifest.tensors}
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping) or "op" not in operation:
            continue
        try:
            _, _, operator = _resolve_operator(operation["op"], registry, policy, managed=True)
        except WSEError:
            continue
        for name in _declared(operator, "tensor_parameters"):
            reference = operation.get(name)
            if reference in resolved:
                continue
            try:
                if not isinstance(reference, str) or not re.fullmatch(TENSOR_REFERENCE_PATTERN, reference):
                    raise WSEError(ErrorCode.INVALID_ARGUMENT,
                                   f"{name} must name a stored tensor of this run, "
                                   "wse://tensors/<sha256>#<tensor name>",
                                   details={"parameter": name})
                info = infos.get(operation.get("tensor"))
                if info is None:
                    continue
                try:
                    shape = [s.stop - s.start for s in resolve_selection(operation.get("selection"), info.shape,
                                                                         tensor=info.name)]
                except WSEError:
                    continue
                uri, _, tensor_name = reference.partition("#")
                path = tensor_file(uri)
                with safe_open(str(path), framework="pt") as handle:
                    names = list(handle.keys())
                    if tensor_name not in names:
                        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                                       f"{uri} holds no tensor {tensor_name!r}; it holds {sorted(names)[:16]}",
                                       details={"parameter": name, "tensor_file": uri, "tensors": sorted(names)[:64]})
                    tensor = handle.get_tensor(tensor_name)
                if dtype_name(tensor) != info.dtype:
                    raise WSEError(ErrorCode.INVALID_ARGUMENT,
                                   f"source tensor {tensor_name!r} has dtype {dtype_name(tensor)}; {info.name!r} is "
                                   f"stored as {info.dtype}: save the values in the stored dtype",
                                   details={"parameter": name, "source_dtype": dtype_name(tensor),
                                            "stored_dtype": info.dtype})
                if list(tensor.shape) != shape:
                    raise WSEError(ErrorCode.INVALID_ARGUMENT,
                                   f"source tensor {tensor_name!r} has shape {list(tensor.shape)}; the selection of "
                                   f"{info.name!r} has shape {shape}",
                                   details={"parameter": name, "source_shape": list(tensor.shape),
                                            "selection_shape": shape})
                values = tensor.to(torch.float32).numpy()
                bad = int(np.count_nonzero(~np.isfinite(values)))
                if bad:
                    raise WSEError(ErrorCode.INVALID_ARGUMENT,
                                   f"source tensor {tensor_name!r} holds {bad} nonfinite value(s); every weight must "
                                   "be finite", details={"parameter": name, "nonfinite_count": bad})
            except WSEError as exc:
                raise _wrap_step_error(index, operation["op"], exc) from exc
            values.flags.writeable = False
            resolved[reference] = values
    return resolved


def _reference_inputs(operator: Any, raw: Mapping[str, Any],
                      references: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    """The resolved vectors (and stored tensors) of the operator's reference parameters, by parameter key."""
    inputs: dict[str, np.ndarray] = {}
    for name in _declared(operator, "tensor_parameters"):
        reference = raw.get(name)
        values = None if references is None or not isinstance(reference, str) else references.get(reference)
        if values is None:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"{name} names a stored tensor file of a run, which only that run's managed previews "
                           "resolve", details={"parameter": name})
        inputs[name] = values
    for name in _declared(operator, "reference_parameters"):
        for key, reference in reference_entries(name, raw.get(name)):
            if references is None:
                raise WSEError(ErrorCode.INVALID_ARGUMENT,
                               f"{key} names a stored analysis result, and direct use has no stored results; pass "
                               f"{key} inline as a list of numbers",
                               details={"parameter": key})
            vector = references.get(reference)
            if vector is None:
                raise WSEError(ErrorCode.INVALID_ARGUMENT,
                               f"the {key} reference {reference!r} was not resolved before the patch was materialized",
                               details={"parameter": key})
            inputs[key] = vector
    return inputs


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(entry, (int, float)) and not isinstance(entry, bool) for entry in value)


def _inline_values(value: Any) -> int:
    """Numbers a recorded parameter holds inline: a list of numbers, or the number lists inside a list."""
    if _is_number_list(value):
        return len(value)
    if isinstance(value, list):
        return sum(len(entry) for entry in value if _is_number_list(entry))
    return 0


def summarize_vector_parameters(parameters: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Recorded ``parameters`` as agent-facing views show them: each inline vector of ``names`` as ``{length, sha256}``.

    A parameter of ``names`` holds a vector or a list of vectors; a vector
    recorded by reference (``{reference, length, sha256}``) stays as it is.
    The stored documents keep the values.
    """
    shown = dict(parameters)
    for name in names:
        value = shown.get(name)
        if _is_number_list(value):
            shown[name] = vector_summary(value)
        elif isinstance(value, list):
            shown[name] = [vector_summary(entry) if _is_number_list(entry) else entry for entry in value]
    return shown


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
    references: Mapping[str, np.ndarray] | None = None,
    managed: bool = False,
) -> tuple[ModelHandle, PatchStep, dict[str, bytes], int]:
    """One edit; returns the candidate, its step, its blocks, and the inline elements it counts."""
    if not isinstance(parameters, Mapping):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "edit parameters must be an object with tensor and selection")
    raw = dict(parameters)
    missing = [name for name in ("tensor", "selection") if name not in raw]
    if missing:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, f"edit parameters are missing {missing}")
    tensor, selection = raw.pop("tensor"), raw.pop("selection")
    label, ref, operator = _resolve_operator(operator_id, registry, policy, managed=managed)
    capabilities = model.adapter.capabilities()
    require_capabilities(capabilities, operator.spec, adapter=str(model.adapter_ref), always=_EDIT_CAPABILITIES)
    info = _info(model, tensor)
    # Protected roles of the model's family descriptor are refused by every policy, before the
    # operator sees its parameters (M2 plan Section 5.1).
    require_unprotected(info.name, model_family(model))
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
    require_partial_scaling(policy, ref, info.name, info.shape, entries)
    # Vectors given by reference were read before any weights (resolve_references); direct use has none.
    inputs = _reference_inputs(operator, raw, references)
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
        inputs=MappingProxyType(inputs),
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
    return candidate, step, blocks, _inline_elements(step, operator)


def _result(base: ModelHandle, candidate: ModelHandle, steps: list[PatchStep],
            blocks: Mapping[str, bytes]) -> EditResult:
    from wse.api import EditResult

    patch = MaterializedPatch(base=base.identity, result_weights_hash=candidate.weights_hash, steps=steps)
    return EditResult(candidate=candidate, patch=patch, diff=tensor_diffs(base, candidate),
                      blocks=MappingProxyType(dict(blocks)))


def _inline_elements(step: PatchStep, operator: Any) -> int:
    """Elements that count toward M1's per-patch limit.

    Inline values (M1 steps' values, or else ``set_slice`` input) plus the
    numbers that the operator's ``inline_parameters`` hold inline
    (``project_out``'s direction and columns, ``add_low_rank``'s inline factor
    columns); vectors given by reference count nothing.
    """
    if step.values is not None:
        count = len(step.values)
    elif "values" in step.parameters:
        count = math.prod(entry.stop - entry.start for entry in step.selection)
    else:
        count = 0
    return count + sum(_inline_values(step.parameters.get(name)) for name in _declared(operator, "inline_parameters"))


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

    candidate, step, blocks, _ = _edit_step(
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
    references: Mapping[str, np.ndarray] | None = None,
    managed: bool = False,
) -> EditResult:
    """Apply ``operations`` in list order to one snapshot through ``apply_operator``.

    Each operation is ``{"op": alias_or_ref, "tensor": ..., "selection": ...,
    <operator parameters>}``. Shapes and finiteness are checked after every
    step and for the final result. Any failure raises (``INVALID_PATCH`` for
    invalid content, with ``details["operation_index"]``) and produces no
    candidate. Patch size limits come from ``policy.limits``.

    Managed previews pass ``managed=True``, which accepts an exact operator
    reference only when the policy lists it, and ``references``, the vectors
    that ``resolve_references`` read for the patch's vector references. Direct
    use passes neither, so any registered edit operator may be named exactly
    and references are refused.
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
            current, step, step_blocks, inline = _edit_step(current, op, request, registry=registry, policy=policy,
                                                            references=references, managed=managed)
        except WSEError as exc:
            raise _wrap_step_error(index, op, exc) from exc
        blocks.update(step_blocks)
        written += inline
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
