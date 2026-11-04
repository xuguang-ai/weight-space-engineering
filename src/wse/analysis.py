"""Bounded tensor reads and registered analyses.

The same functions serve direct library use (``read_tensor_slice`` and
``analyze_tensor`` on an in-memory ``ModelHandle``, labeled ``unmanaged``) and
the managed ``Engine`` (labeled ``managed``). Neither path ever calls the model
adapter's ``forward``.

Response policy:

* A read returns at most ``max_read_elements`` (2,048) elements, and the
  canonical compact JSON of a read must not exceed ``max_inline_json_bytes``
  (32 KiB). Oversized requests fail with ``INVALID_ARGUMENT`` whose message and
  ``details["largest_permitted_selection"]`` state the largest selection with
  the same starts that fits, shrinking the outermost axis first.
* An analysis may cover up to ``max_analysis_elements``. When its result would
  exceed the inline limit, a managed call stores the result as a
  content-addressed artifact and returns its ``result_uri`` with
  ``result=None``. Direct calls have no transport and keep results inline.
* Regions are read without implicit copies: direct use passes a read-only view
  of the handle's tensor, and managed use reads only the selected region from
  the safetensors file.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pydantic import BaseModel

from wse.operators.base import TensorRegion
from wse.operators.tensor import shortest_floats
from wse.policy import OperationPolicy, default_operation_policy, json_size, require_capabilities
from wse.schemas import (
    AnalysisResult,
    AxisSlice,
    ErrorCode,
    ExtensionRef,
    Management,
    ModelIdentity,
    TensorInfo,
    TensorSlice,
    Usage,
    WSEError,
    canonical_json,
    json_hash,
)
from wse.tensors import resolve_selection

if TYPE_CHECKING:
    from wse.adapters.base import AdapterCapabilities
    from wse.api import ModelHandle
    from wse.registry import Registry

SUPPORTED_DTYPES = {"float16": np.float16, "float32": np.float32, "float64": np.float64}

# Timing fields are measured with a fixed-width stand-in so that whether a
# response fits never depends on how long the call took.
_TIMING_PLACEHOLDER = 99999.999999

ReadRegion = Callable[[tuple[slice, ...]], torch.Tensor]
StoreLarge = Callable[[bytes], str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entries(slices: Sequence[slice]) -> list[AxisSlice]:
    return [AxisSlice(start=s.start, stop=s.stop, step=1) for s in slices]


def _slices(entries: Sequence[AxisSlice]) -> tuple[slice, ...]:
    return tuple(slice(entry.start, entry.stop, 1) for entry in entries)


def _extent(entries: Sequence[AxisSlice]) -> list[int]:
    return [entry.stop - entry.start for entry in entries]


def _count(entries: Sequence[AxisSlice]) -> int:
    return math.prod(_extent(entries))


def _selection_text(entries: Sequence[AxisSlice]) -> str:
    """Compact JSON in start/stop/step order, ready to paste into a request."""
    return json.dumps([entry.model_dump() for entry in entries], separators=(",", ":"))


def _require_supported_dtype(info: TensorInfo) -> None:
    if info.dtype not in SUPPORTED_DTYPES:
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"tensor {info.name!r} has dtype {info.dtype}; reads and analyses support "
            f"{sorted(SUPPORTED_DTYPES)}",
            details={"tensor": info.name, "dtype": info.dtype},
        )


def _readonly(tensor: torch.Tensor, dtype: str) -> np.ndarray:
    """A non-writeable NumPy view of ``tensor`` (no copy)."""
    array = tensor.detach().numpy().view()
    if array.dtype != SUPPORTED_DTYPES[dtype]:
        raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored dtype {array.dtype} does not match manifest {dtype}")
    array.flags.writeable = False
    return array


def _timed(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    if "elapsed_seconds" in payload:
        payload["elapsed_seconds"] = _TIMING_PLACEHOLDER
    usage = dict(payload["usage"])
    usage["compute_seconds"] = _TIMING_PLACEHOLDER
    payload["usage"] = usage
    return payload


def inline_size(result: TensorSlice | AnalysisResult) -> int:
    """Canonical JSON size of a response, with timing fields at a fixed width."""
    return json_size(_timed(result.model_dump(mode="json")))


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 6)


def largest_permitted_selection(
    entries: Sequence[AxisSlice], fits: Callable[[list[AxisSlice]], bool]
) -> list[AxisSlice]:
    """The largest selection with the same starts for which ``fits`` holds.

    Shrinks the outermost axis first; if a single index along that axis still
    does not fit, keeps one index there and shrinks the next axis. ``fits``
    must be monotone in the selection's extent.
    """
    current = list(entries)
    for axis in range(len(current)):
        start, full = current[axis].start, current[axis].stop - current[axis].start

        def with_extent(extent: int, axis: int = axis, start: int = start) -> list[AxisSlice]:
            candidate = list(current)
            candidate[axis] = AxisSlice(start=start, stop=start + extent, step=1)
            return candidate

        if fits(with_extent(full)):
            return with_extent(full)
        low, high = 0, full - 1  # largest fitting extent lies in [low, high]; 0 means none
        while low < high:
            middle = (low + high + 1) // 2
            if fits(with_extent(middle)):
                low = middle
            else:
                high = middle - 1
        if low >= 1:
            return with_extent(low)
        current = with_extent(1)
    return current


def _too_large(
    message: str, *, limit: str, requested: list[AxisSlice], suggestion: list[AxisSlice], extra: dict[str, Any]
) -> WSEError:
    return WSEError(
        ErrorCode.INVALID_ARGUMENT,
        f"{message}; the largest permitted selection with the same start is "
        f"{_selection_text(suggestion)}",
        details={
            "limit": limit,
            "requested_selection": [entry.model_dump() for entry in requested],
            "largest_permitted_selection": [entry.model_dump() for entry in suggestion],
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# Bounded reads
# ---------------------------------------------------------------------------


def bounded_read(
    *,
    info: TensorInfo,
    selection: Any,
    policy: OperationPolicy,
    capabilities: AdapterCapabilities,
    adapter: ExtensionRef,
    read_region: ReadRegion,
    checkpoint_id: str | None,
    weights_hash: str,
    management: Management,
) -> tuple[TensorSlice, torch.Tensor]:
    """Read a bounded region; return its JSON form and the region tensor."""
    from wse.adapters.base import CAP_WEIGHTS_READ

    start = time.perf_counter()
    if not capabilities.supports(CAP_WEIGHTS_READ):
        raise WSEError(
            ErrorCode.UNSUPPORTED_CAPABILITY,
            f"adapter {adapter} does not support {CAP_WEIGHTS_READ}",
            details={"adapter": str(adapter), "missing_capabilities": [CAP_WEIGHTS_READ]},
        )
    entries = _entries(resolve_selection(selection, info.shape, tensor=info.name))
    _require_supported_dtype(info)
    limits = policy.limits

    def build(candidate: list[AxisSlice], data: torch.Tensor, elapsed: float) -> TensorSlice:
        return TensorSlice(
            checkpoint_id=checkpoint_id,
            weights_hash=weights_hash,
            tensor=info.name,
            dtype=info.dtype,
            axes=list(info.axes),
            tensor_shape=list(info.shape),
            selection=candidate,
            shape=_extent(candidate),
            element_count=_count(candidate),
            values=shortest_floats(_readonly(data, info.dtype)),
            usage=Usage(tensor_read_calls=1, model_forward_calls=0, compute_seconds=elapsed),
            management=management,
        )

    def json_fit(base: list[AxisSlice], data: torch.Tensor) -> list[AxisSlice]:
        def fits(candidate: list[AxisSlice]) -> bool:
            local = tuple(slice(0, extent) for extent in _extent(candidate))
            return inline_size(build(candidate, data[local], 0.0)) <= limits.max_inline_json_bytes

        return largest_permitted_selection(base, fits)

    requested = _count(entries)
    if requested > limits.max_read_elements:
        by_count = largest_permitted_selection(entries, lambda c: _count(c) <= limits.max_read_elements)
        suggestion = json_fit(by_count, read_region(_slices(by_count)))
        raise _too_large(
            f"a read returns at most {limits.max_read_elements} elements, but the selection of tensor "
            f"{info.name!r} has {requested}",
            limit="max_read_elements",
            requested=entries,
            suggestion=suggestion,
            extra={"max_elements": limits.max_read_elements, "requested_elements": requested},
        )
    data = read_region(_slices(entries))
    result = build(entries, data, _elapsed(start))
    size = inline_size(result)
    if size > limits.max_inline_json_bytes:
        raise _too_large(
            f"the read of tensor {info.name!r} would serialize to {size} bytes of JSON, above the "
            f"{limits.max_inline_json_bytes}-byte response limit",
            limit="max_inline_json_bytes",
            requested=entries,
            suggestion=json_fit(entries, data),
            extra={"max_inline_json_bytes": limits.max_inline_json_bytes, "json_bytes": size},
        )
    return result, data


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def run_analysis(
    *,
    registry: Registry,
    policy: OperationPolicy,
    kind: Any,
    info_for: Callable[[str], TensorInfo],
    tensor: str,
    selection: Any,
    k: Any,
    parameters: Mapping[str, Any] | None,
    capabilities: AdapterCapabilities,
    model: ModelIdentity,
    read_region: Callable[[str, tuple[slice, ...]], torch.Tensor],
    checkpoint_id: str | None,
    management: Management,
    store_large: StoreLarge | None,
) -> AnalysisResult:
    """Dispatch one analysis through the registry; shared by direct and managed use.

    Checks run before any weights are read: permitted kind, exact operator
    version, effect category, adapter capabilities, tensor name, selection,
    size limits, and parameters.
    """
    start = time.perf_counter()
    ref = policy.analysis_operator(kind)
    operator = registry.resolve("operator", ref.extension_id, ref.version)
    spec = operator.spec
    if spec.effect != "analysis":
        raise WSEError(ErrorCode.POLICY_DENIED, f"operator {ref} is not an analysis operator")
    require_capabilities(capabilities, spec, adapter=str(model.adapter))
    info = info_for(tensor)
    entries = _entries(resolve_selection(selection, info.shape, tensor=info.name))
    _require_supported_dtype(info)
    count = _count(entries)
    maximum = policy.limits.max_analysis_elements
    if count > maximum:
        raise _too_large(
            f"an analysis covers at most {maximum} elements, but the selection of tensor "
            f"{info.name!r} has {count}",
            limit="max_analysis_elements",
            requested=entries,
            suggestion=largest_permitted_selection(entries, lambda c: _count(c) <= maximum),
            extra={"max_elements": maximum, "requested_elements": count},
        )
    raw = dict(parameters or {})
    if k is not None:
        if "k" in raw:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "pass k either as an argument or in parameters, not both")
        raw["k"] = k
    resolved = operator.resolve_parameters(raw, count)
    region = TensorRegion(
        tensor=info.name,
        dtype=info.dtype,
        axes=tuple(info.axes),
        tensor_shape=tuple(info.shape),
        selection=tuple(entries),
        values=_readonly(read_region(info.name, _slices(entries)), info.dtype),
    )
    output = operator.analyze(region, resolved)
    if not isinstance(output, operator.result_model):
        raise WSEError(ErrorCode.WORKER_FAILED, f"operator {ref} returned {type(output).__name__}, "
                       f"not {operator.result_model.__name__}")
    payload = output.model_dump(mode="json")
    try:
        payload_text = canonical_json(payload)
    except WSEError as exc:
        raise WSEError(ErrorCode.WORKER_FAILED, f"operator {ref} produced a non-JSON result: {exc.message}") from exc
    resolved_parameters = resolved.model_dump(mode="json") if isinstance(resolved, BaseModel) else {}
    input_hash = json_hash(
        {
            "operator": str(ref),
            "model": model.model_dump(mode="json"),
            "tensor": info.name,
            "selection": [entry.model_dump() for entry in entries],
            "parameters": resolved_parameters,
        }
    )
    elapsed = _elapsed(start)
    result = AnalysisResult(
        kind=kind,
        operator=ref,
        checkpoint_id=checkpoint_id,
        model=model,
        tensor=info.name,
        selection=entries,
        element_count=count,
        parameters=resolved_parameters,
        input_hash=input_hash,
        result=payload,
        result_hash=json_hash(payload),
        result_bytes=len(payload_text.encode("utf-8")),
        elapsed_seconds=elapsed,
        usage=Usage(analysis_calls=spec.usage.units_per_call, model_forward_calls=0, compute_seconds=elapsed),
        management=management,
    )
    if store_large is not None and inline_size(result) > policy.limits.max_inline_json_bytes:
        uri = store_large(payload_text.encode("utf-8"))
        result = result.model_copy(update={"result": None, "result_uri": uri})
    return result


# ---------------------------------------------------------------------------
# Direct (unmanaged) use
# ---------------------------------------------------------------------------


def _handle_info(model: ModelHandle, tensor: Any) -> TensorInfo:
    for info in model.architecture.tensors:
        if info.name == tensor:
            return info
    raise WSEError(
        ErrorCode.NOT_FOUND,
        f"tensor {tensor!r} does not exist; tensor names must match exactly",
        details={"available_tensors": list(model.tensor_names())},
    )


def read_tensor_slice(
    model: ModelHandle, tensor: str, selection: Any, *, policy: OperationPolicy | None = None
) -> TensorSlice:
    """Bounded read of an in-memory model (direct use; same limits as managed reads)."""
    info = _handle_info(model, tensor)
    result, _ = bounded_read(
        info=info,
        selection=selection,
        policy=policy or default_operation_policy(),
        capabilities=model.adapter.capabilities(),
        adapter=model.adapter_ref,
        read_region=lambda slices: model._tensors[info.name][slices],
        checkpoint_id=None,
        weights_hash=model.weights_hash,
        management="unmanaged",
    )
    return result


def analyze_tensor(
    model: ModelHandle,
    kind: str,
    tensor: str,
    selection: Any,
    k: int | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
    registry: Registry | None = None,
    policy: OperationPolicy | None = None,
) -> AnalysisResult:
    """Run a registered analysis on an in-memory model (direct use, unmanaged)."""
    from wse.registry import builtin_registry

    return run_analysis(
        registry=registry or builtin_registry(),
        policy=policy or default_operation_policy(),
        kind=kind,
        info_for=lambda name: _handle_info(model, name),
        tensor=tensor,
        selection=selection,
        k=k,
        parameters=parameters,
        capabilities=model.adapter.capabilities(),
        model=model.identity,
        read_region=lambda name, slices: model._tensors[name][slices],
        checkpoint_id=None,
        management="unmanaged",
        store_large=None,
    )
