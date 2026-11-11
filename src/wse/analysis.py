"""Bounded tensor reads and registered analyses.

The same functions serve direct library use (``read_tensor_slice`` and
``analyze_tensor`` on an in-memory ``ModelHandle``, labeled ``unmanaged``) and
the managed ``Engine`` (labeled ``managed``). Neither path ever calls the model
adapter's ``forward``.

Response policy:

* A read returns at most ``max_read_elements`` (2,048) elements, and the
  canonical compact JSON of a read must not exceed ``max_result_json_bytes``
  (``max_inline_json_bytes`` of 32 KiB less a 1 KiB allowance for the tool
  response envelope, so that an MCP response carrying the result twice stays
  within 64 KiB). Oversized requests fail with ``INVALID_ARGUMENT`` whose message and
  ``details["largest_permitted_selection"]`` state the largest selection with
  the same starts that fits, shrinking the outermost axis first.
* An analysis may cover up to ``max_analysis_elements``. When its result would
  exceed the inline limit, a managed call stores the result as a
  content-addressed artifact and returns its ``result_uri`` with
  ``result=None``. Direct calls have no transport and keep results inline.
* An analysis whose operator declares ``artifact_fields`` (vector results, M2
  plan Section 5.2) always stores its complete result in a managed call and
  returns the result without those fields inline, next to ``result_uri``:
  a summary, the top entries, and the URI. ``result_hash`` and
  ``result_bytes`` describe the complete, stored result. Should even that
  inline part exceed the limit, the result is returned as ``None``, as above.
* A parameter in an operator's ``reference_parameters`` may name a vector of a
  stored result (``wse.operators.base.VectorReference``). A managed call reads
  it through the engine's run-scoped reader before any weights are read;
  direct use has no stored results and refuses references.
* Regions are read without implicit copies: direct use passes a read-only view
  of the handle's tensor, and managed use reads only the selected region from
  the safetensors file. A BF16 region, which NumPy cannot hold, is the one
  exception: it arrives as its exact float32 widening (M2 plan Section 3.2),
  and a read reports each value as the shortest decimal that round-trips that
  widening.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pydantic import BaseModel

from wse.operators.base import MAX_VECTOR_REFERENCE_LENGTH, VECTOR_REFERENCE_PATTERN, TensorRegion
from wse.operators.tensor import FLOAT32_OVERFLOW, shortest_floats
from wse.adapters.base import CAP_WEIGHTS_READ
from wse.policy import (
    OperationPolicy,
    contract_usage,
    default_operation_policy,
    json_size,
    require_capabilities,
)
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

# Stored dtypes that reads and analyses accept, with the NumPy dtype in which a region reaches them.
SUPPORTED_DTYPES = {"bfloat16": np.float32, "float16": np.float16, "float32": np.float32, "float64": np.float64}

# Timing fields are measured with a fixed-width stand-in so that whether a
# response fits never depends on how long the call took.
_TIMING_PLACEHOLDER = 99999.999999
# Analysis result URIs have a fixed length; this stand-in measures a response
# before the result is stored.
_URI_PLACEHOLDER = "wse://analyses/" + "0" * 64

ReadRegion = Callable[[tuple[slice, ...]], torch.Tensor]
StoreLarge = Callable[[bytes], str]
# Returns a stored analysis result (parsed JSON) that the caller may read, or raises NOT_FOUND.
ReadResult = Callable[[str], Any]


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
    """A non-writeable NumPy view of ``tensor`` (no copy); a BF16 tensor becomes its exact float32 widening.

    The widening puts each BF16 bit pattern in the high half of a float32 (as
    ``wse.tensors.bf16_to_f32`` does), shifting in place so that the region
    costs four bytes per element beyond the stored copy.
    """
    if dtype == "bfloat16":
        if tensor.dtype != torch.bfloat16:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored dtype {tensor.dtype} does not match manifest {dtype}")
        widened = tensor.detach().view(torch.int16).numpy().view(np.uint16).astype(np.uint32)
        widened <<= 16
        array = widened.view(np.float32)
    else:
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
            return inline_size(build(candidate, data[local], 0.0)) <= limits.max_result_json_bytes

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
    if size > limits.max_result_json_bytes:
        raise _too_large(
            f"the read of tensor {info.name!r} would serialize to {size} bytes of JSON, above the "
            f"{limits.max_result_json_bytes}-byte result limit (the {limits.max_inline_json_bytes}-byte inline "
            "limit less the tool-response envelope)",
            limit="max_inline_json_bytes",
            requested=entries,
            suggestion=json_fit(entries, data),
            extra={"max_inline_json_bytes": limits.max_inline_json_bytes,
                   "max_result_json_bytes": limits.max_result_json_bytes, "json_bytes": size},
        )
    return result, data


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def resolve_vector_reference(reference: str, read_result: ReadResult, *, parameter: str,
                             max_length: int) -> np.ndarray:
    """The vector that ``reference`` (``<result URI>#<JSON pointer>``) names, as a read-only float32 array.

    ``read_result`` returns a stored result the caller may read; a run reads
    only results its own experiment produced, and any other URI is reported
    exactly like one that does not exist. The pointer must name a non-empty
    list of finite numbers that round to finite float32 values (so
    ``3.4028235e+38``, the float32 maximum in its shortest digits, is one), of
    at most ``max_length`` values.
    """
    if isinstance(reference, str) and len(reference) > MAX_VECTOR_REFERENCE_LENGTH:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"{parameter} reference is {len(reference)} characters long; a reference has at most "
                       f"{MAX_VECTOR_REFERENCE_LENGTH} (wse://analyses/<sha256>#/<field>[/<index>])",
                       details={"parameter": parameter, "length": len(reference),
                                "max_length": MAX_VECTOR_REFERENCE_LENGTH})
    if not isinstance(reference, str) or not re.fullmatch(VECTOR_REFERENCE_PATTERN, reference):
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"{parameter} reference {reference!r} must have the form "
                       "wse://analyses/<sha256>#/<field>[/<index>]",
                       details={"parameter": parameter})
    uri, _, pointer = reference.partition("#")
    try:
        document = read_result(uri)
    except WSEError as exc:
        if exc.code != ErrorCode.NOT_FOUND:
            raise
        raise WSEError(ErrorCode.NOT_FOUND, f"{parameter} reference {reference!r}: {exc.message}",
                       details={"parameter": parameter, "uri": uri}) from None
    value: Any = document
    walked = ""
    for segment in pointer.split("/")[1:]:
        if isinstance(value, dict) and segment in value:
            value = value[segment]
        elif isinstance(value, list) and segment.isdigit() and int(segment) < len(value):
            value = value[int(segment)]
        else:
            if isinstance(value, dict):
                available = f"its fields are {sorted(value)}"
            elif isinstance(value, list):
                available = f"it is a list of {len(value)} entries"
            else:
                available = "it is a single value"
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"{parameter} reference {reference!r} names no value of the stored result: "
                f"{walked + '/' + segment!r} does not exist ({walked or 'the result'}: {available})",
                details={"parameter": parameter, "uri": uri, "missing": walked + "/" + segment},
            )
        walked += "/" + segment
    if not (isinstance(value, list) and value
            and all(isinstance(entry, (int, float)) and not isinstance(entry, bool) for entry in value)):
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"{parameter} reference {reference!r} must name a non-empty list of numbers (a vector); "
                       f"it names {type(value).__name__ if not isinstance(value, list) else 'a list of other values'}",
                       details={"parameter": parameter, "uri": uri})
    if len(value) > max_length:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"{parameter} reference {reference!r} names {len(value)} values; at most {max_length} are "
                       "accepted",
                       details={"parameter": parameter, "uri": uri, "length": len(value), "max_length": max_length})
    wide = np.asarray(value, dtype=np.float64)
    bad = int(np.count_nonzero(~np.isfinite(wide) | (np.abs(wide) >= FLOAT32_OVERFLOW)))
    if bad:
        raise WSEError(ErrorCode.INVALID_ARGUMENT,
                       f"{parameter} reference {reference!r} names {bad} value(s) that are not finite float32 numbers",
                       details={"parameter": parameter, "uri": uri, "nonfinite_count": bad})
    vector = wide.astype(np.float32)
    vector.flags.writeable = False
    return vector


def _resolve_references(operator: Any, resolved: Any, read_result: ReadResult | None,
                        max_length: int) -> dict[str, np.ndarray]:
    """Vectors of the operator's reference parameters that were given as references, keyed by parameter."""
    inputs: dict[str, np.ndarray] = {}
    for name in tuple(getattr(operator, "reference_parameters", ()) or ()):
        value = getattr(resolved, name, None)
        if not isinstance(value, str):
            continue
        if read_result is None:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"{name} names a stored analysis result, and direct use has no stored results; pass the "
                           f"{name} inline",
                           details={"parameter": name})
        inputs[name] = resolve_vector_reference(value, read_result, parameter=name, max_length=max_length)
    return inputs


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
    read_result: ReadResult | None = None,
) -> AnalysisResult:
    """Dispatch one analysis through the registry; shared by direct and managed use.

    Checks run before any weights are read: permitted kind, exact operator
    version, effect category, adapter capabilities, tensor name, selection,
    size limits, and parameters, including the vectors of reference
    parameters, which ``read_result`` reads (``None`` in direct use).
    """
    start = time.perf_counter()
    ref = policy.analysis_operator(kind)
    operator = registry.resolve("operator", ref.extension_id, ref.version)
    spec = operator.spec
    if spec.effect != "analysis":
        raise WSEError(ErrorCode.POLICY_DENIED, f"operator {ref} is not an analysis operator")
    require_capabilities(capabilities, spec, adapter=str(model.adapter), always=(CAP_WEIGHTS_READ,))
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
    inputs = _resolve_references(operator, resolved, read_result, maximum)
    region = TensorRegion(
        tensor=info.name,
        dtype=info.dtype,
        axes=tuple(info.axes),
        tensor_shape=tuple(info.shape),
        selection=tuple(entries),
        values=_readonly(read_region(info.name, _slices(entries)), info.dtype),
        inputs=MappingProxyType(inputs),
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
        usage=contract_usage(spec, compute_seconds=elapsed),
        management=management,
    )
    limit = policy.limits.max_result_json_bytes
    artifact_fields = tuple(getattr(operator, "artifact_fields", ()) or ())
    if store_large is not None and (artifact_fields or inline_size(result) > limit):
        data = payload_text.encode("utf-8")
        # Keep inline what fits: the result without its vector fields, else nothing.
        inline_forms: list[dict[str, Any] | None] = [None]
        if artifact_fields:
            inline_forms.insert(0, {key: value for key, value in payload.items() if key not in artifact_fields})
        usage = result.usage.model_copy(update={"artifact_bytes": len(data)})
        size = 0
        for inline in inline_forms:
            by_reference = result.model_copy(update={"result": inline, "result_uri": _URI_PLACEHOLDER, "usage": usage})
            size = inline_size(by_reference)
            if size <= limit:
                return by_reference.model_copy(update={"result_uri": store_large(data)})
        # Moving the result behind a URI is not enough: the rest of the
        # response (for example large resolved parameters) is too big.
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"the analysis response would serialize to {size} bytes of JSON even with its result "
            f"stored by reference, above the {limit}-byte response limit; pass smaller parameters",
            details={"limit": "max_inline_json_bytes", "max_inline_json_bytes": policy.limits.max_inline_json_bytes,
                     "max_result_json_bytes": limit, "json_bytes": size},
        )
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
