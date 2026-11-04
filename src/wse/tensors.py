"""Tensor normalization, validation, canonical content hashing, and selections.

The weights hash is computed over tensor names (in sorted order), dtypes,
shapes, and raw bytes, so it is independent of dictionary insertion order and
of the safetensors file layout. It identifies weights only; model identity
also includes the adapter and configuration hash.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from wse.schemas import AxisSlice, ErrorCode, WSEError

_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_WEIGHTS_HASH_DOMAIN = b"wse-weights-v1\x00"
_TENSOR_HASH_DOMAIN = b"wse-tensor-v1\x00"

if sys.byteorder != "little":  # pragma: no cover - all supported platforms are little-endian
    raise ImportError("wse requires a little-endian platform for byte-exact hashing")


def dtype_name(tensor: torch.Tensor) -> str:
    return _DTYPE_NAMES.get(tensor.dtype, str(tensor.dtype).removeprefix("torch."))


def normalize_tensors(tensors: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return private, contiguous CPU copies of ``tensors`` keyed by sorted name.

    Accepts ``torch.Tensor`` and ``numpy.ndarray`` values. Rejects non-string
    names, non-tensor values, non-strided layouts, non-CPU devices, and tensors
    that share memory with each other (shared parameters are unsupported in
    the MVP).
    """
    if not isinstance(tensors, Mapping):
        raise WSEError(ErrorCode.INVALID_MODEL, "tensors must be a mapping of name to tensor")
    if not tensors:
        raise WSEError(ErrorCode.INVALID_MODEL, "a model must contain at least one tensor")
    converted: dict[str, torch.Tensor] = {}
    originals: list[tuple[str, Any]] = []
    for name in sorted(tensors, key=_name_key):
        value = tensors[name]
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"tensor {name!r} must be a torch.Tensor or numpy.ndarray, not {type(value).__name__}",
            )
        if tensor.layout != torch.strided:
            raise WSEError(ErrorCode.INVALID_MODEL, f"tensor {name!r} must use a dense strided layout")
        if tensor.device.type != "cpu":
            raise WSEError(ErrorCode.INVALID_MODEL, f"tensor {name!r} must be on the CPU")
        originals.append((name, tensor))
        converted[name] = tensor.detach().clone(memory_format=torch.contiguous_format)
    _reject_shared_memory(originals)
    return converted


def _name_key(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise WSEError(ErrorCode.INVALID_MODEL, f"tensor names must be nonempty strings, got {name!r}")
    return name


def _reject_shared_memory(tensors: Sequence[tuple[str, torch.Tensor]]) -> None:
    spans: list[tuple[int, int, str]] = []
    for name, tensor in tensors:
        if tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        start = storage.data_ptr()
        spans.append((start, start + storage.nbytes(), name))
    spans.sort()
    for (_, end_a, name_a), (start_b, _, name_b) in zip(spans, spans[1:]):
        if start_b < end_a:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"tensors {name_a!r} and {name_b!r} share memory; shared parameters are not supported",
                details={"tensors": [name_a, name_b]},
            )


def nonfinite_count(tensor: torch.Tensor) -> int:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return 0
    return int((~torch.isfinite(tensor)).sum().item())


def require_finite(tensors: Mapping[str, torch.Tensor]) -> None:
    for name in sorted(tensors):
        bad = nonfinite_count(tensors[name])
        if bad:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"tensor {name!r} contains {bad} nonfinite value(s); all weights must be finite",
                details={"tensor": name, "nonfinite_count": bad},
            )


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _update_tensor(digest: Any, name: str, tensor: torch.Tensor) -> None:
    encoded_name = name.encode("utf-8")
    encoded_dtype = dtype_name(tensor).encode("ascii")
    digest.update(struct.pack("<Q", len(encoded_name)) + encoded_name)
    digest.update(struct.pack("<Q", len(encoded_dtype)) + encoded_dtype)
    digest.update(struct.pack("<Q", tensor.dim()))
    for size in tensor.shape:
        digest.update(struct.pack("<Q", size))
    data = _raw_bytes(tensor)
    digest.update(struct.pack("<Q", len(data)))
    digest.update(data)


def tensor_hash(name: str, tensor: torch.Tensor) -> str:
    """Content hash of one named tensor (name, dtype, shape, and bytes)."""
    digest = hashlib.sha256(_TENSOR_HASH_DOMAIN)
    _update_tensor(digest, name, tensor)
    return "sha256:" + digest.hexdigest()


def weights_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    """Canonical content hash of a complete set of named tensors."""
    digest = hashlib.sha256(_WEIGHTS_HASH_DOMAIN)
    names = sorted(tensors)
    digest.update(struct.pack("<Q", len(names)))
    for name in names:
        _update_tensor(digest, name, tensors[name])
    return "sha256:" + digest.hexdigest()


def resolve_selection(selection: Any, shape: Sequence[int], *, tensor: str) -> tuple[slice, ...]:
    """Validate a ``start/stop/step`` selection and convert it to Python slices.

    The MVP accepts one entry per axis; each must be a nonnegative, nonempty,
    in-bounds contiguous slice with step 1. Strings and other expressions are
    rejected.
    """
    if isinstance(selection, (str, bytes)) or not isinstance(selection, Sequence):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            "selection must be a list of {start, stop, step} objects, one per axis; "
            "expressions are not accepted",
            details={"tensor": tensor, "shape": list(shape)},
        )
    if len(selection) != len(shape):
        raise WSEError(
            ErrorCode.INVALID_ARGUMENT,
            f"selection has {len(selection)} axis entries but tensor {tensor!r} has {len(shape)} axes",
            details={"tensor": tensor, "shape": list(shape)},
        )
    slices: list[slice] = []
    for axis, (entry, size) in enumerate(zip(selection, shape)):
        try:
            parsed = entry if isinstance(entry, AxisSlice) else AxisSlice.model_validate(entry)
        except ValueError as exc:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"axis {axis} of the selection must be an object with integer start, stop, "
                "and step fields",
                details={"tensor": tensor, "axis": axis, "shape": list(shape)},
            ) from exc
        if parsed.step != 1:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"axis {axis}: step must be 1 (got {parsed.step})",
                details={"tensor": tensor, "axis": axis},
            )
        if not 0 <= parsed.start < parsed.stop <= size:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"axis {axis}: require 0 <= start < stop <= {size} (got start={parsed.start}, "
                f"stop={parsed.stop})",
                details={"tensor": tensor, "axis": axis, "axis_size": size, "shape": list(shape)},
            )
        slices.append(slice(parsed.start, parsed.stop, 1))
    return tuple(slices)
