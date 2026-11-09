"""Tensor normalization, validation, canonical content hashing, and selections.

The weights hash is computed over tensor names (in sorted order), dtypes,
shapes, and raw bytes, so it is independent of dictionary insertion order and
of the safetensors file layout. It identifies weights only; model identity
also includes the adapter and configuration hash.

A blob ID (``blob_id``) identifies one tensor's raw bytes alone: the SHA-256
of its little-endian row-major bytes in the stored dtype.

Two weights-hash schemes exist. M1 checkpoints keep ``weights_hash`` (names,
dtypes, shapes, and raw bytes), so their identities never change. Imported
open-weight models use ``blob_weights_hash`` (M2 plan Section 2.1): names,
dtypes, shapes, and blob IDs in canonical name order, which a blob manifest
alone determines. An adapter selects its scheme with the attribute
``weights_hash_scheme``; storage accepts a checkpoint whose recorded hash
matches either scheme (``weights_hash_matches``), because the two are
domain-separated.
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
_BLOB_WEIGHTS_HASH_DOMAIN = b"wse-weights-blobs-v1\x00"
WEIGHTS_HASH_RAW = "raw_bytes_v1"
WEIGHTS_HASH_BLOBS = "blob_ids_v1"
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
            tensor = _from_numpy(name, value)
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


def _from_numpy(name: str, value: np.ndarray) -> torch.Tensor:
    """Convert an array, normalizing negative strides and non-native byte order."""
    array = value
    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="))
    if any(stride < 0 for stride in array.strides):
        array = np.ascontiguousarray(array)
    try:
        return torch.from_numpy(array)
    except (TypeError, ValueError) as exc:
        raise WSEError(
            ErrorCode.INVALID_MODEL,
            f"tensor {name!r} has an unsupported numpy array ({array.dtype}): {exc}",
            details={"tensor": name},
        ) from exc


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


_FINITE_CHUNK = 1 << 22  # elements per chunk of the finiteness check


def nonfinite_count(tensor: torch.Tensor) -> int:
    """The number of NaN and infinite elements, counted in chunks.

    Chunks bound the temporary mask to ``_FINITE_CHUNK`` bytes, where one
    mask over a whole embedding of a language model would take hundreds of MB.
    """
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return 0
    flat = tensor.reshape(-1)
    return sum(int((~torch.isfinite(flat[start:start + _FINITE_CHUNK])).sum().item())
               for start in range(0, flat.numel(), _FINITE_CHUNK))


def require_finite(tensors: Mapping[str, torch.Tensor]) -> None:
    for name in sorted(tensors):
        bad = nonfinite_count(tensors[name])
        if bad:
            raise WSEError(
                ErrorCode.INVALID_MODEL,
                f"tensor {name!r} contains {bad} nonfinite value(s); all weights must be finite",
                details={"tensor": name, "nonfinite_count": bad},
            )


def dtype_from_name(name: str) -> torch.dtype:
    """Inverse of :func:`dtype_name`; raise ``ARTIFACT_CORRUPT`` for an unknown name."""
    for dtype, known in _DTYPE_NAMES.items():
        if known == name:
            return dtype
    candidate = getattr(torch, name, None) if name.isidentifier() else None
    if isinstance(candidate, torch.dtype):
        return candidate
    raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored dtype {name!r} is not a torch dtype")


def raw_bytes_view(tensor: torch.Tensor) -> np.ndarray:
    """The tensor's raw bytes as a uint8 array: little-endian, row-major, stored dtype.

    The array may share memory with ``tensor`` when it is already contiguous;
    callers must not modify it.
    """
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy()


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return raw_bytes_view(tensor).tobytes()


def blob_id(tensor: torch.Tensor) -> str:
    """The SHA-256 hex digest of the tensor's raw bytes in its stored dtype (M2 plan Section 3.1).

    This is the convention of the published ``apply_edit.py`` hashes: the
    bytes are little-endian (every supported platform is) and row-major, so a
    non-contiguous view hashes its logical element order. Shape and dtype are
    not part of the digest; manifests record them next to the blob ID.
    """
    return hashlib.sha256(raw_bytes_view(tensor)).hexdigest()


def tensor_from_buffer(buffer: bytearray, dtype: torch.dtype, shape: Sequence[int]) -> torch.Tensor:
    """Interpret raw little-endian row-major bytes as a tensor without copying.

    BF16 and F16 (and every other dtype) load through ``torch.frombuffer`` on
    the raw bytes, so stored bit patterns, signed zeros, and subnormals are
    preserved exactly. The tensor shares memory with ``buffer``.
    """
    count = 1
    for size in shape:
        count *= int(size)
    element_size = torch.empty((), dtype=dtype).element_size()
    if len(buffer) != count * element_size:
        raise WSEError(
            ErrorCode.ARTIFACT_CORRUPT,
            f"stored tensor has {len(buffer)} bytes; shape {list(shape)} of {dtype} needs {count * element_size}",
        )
    if count == 0:
        return torch.empty(tuple(shape), dtype=dtype)
    return torch.frombuffer(buffer, dtype=dtype, count=count).reshape(tuple(shape))


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


def blob_weights_hash(tensors: Mapping[str, torch.Tensor], blob_ids: Mapping[str, str] | None = None) -> str:
    """Canonical hash of a set of named tensors over their blob IDs (M2 plan Section 2.1).

    Each tensor contributes its name, dtype, shape, and blob ID, in canonical
    (sorted) name order. ``blob_ids`` supplies already verified blob IDs, so
    the hash of a stored checkpoint needs no second pass over its bytes.
    """
    digest = hashlib.sha256(_BLOB_WEIGHTS_HASH_DOMAIN)
    names = sorted(tensors)
    digest.update(struct.pack("<Q", len(names)))
    for name in names:
        tensor = tensors[name]
        encoded_name = name.encode("utf-8")
        encoded_dtype = dtype_name(tensor).encode("ascii")
        digest.update(struct.pack("<Q", len(encoded_name)) + encoded_name)
        digest.update(struct.pack("<Q", len(encoded_dtype)) + encoded_dtype)
        digest.update(struct.pack("<Q", tensor.dim()))
        for size in tensor.shape:
            digest.update(struct.pack("<Q", size))
        blob = blob_ids[name] if blob_ids is not None else blob_id(tensor)
        digest.update(bytes.fromhex(blob))
    return "sha256:" + digest.hexdigest()


def hash_weights(tensors: Mapping[str, torch.Tensor], scheme: str = WEIGHTS_HASH_RAW) -> str:
    """The weights hash of ``tensors`` under ``scheme`` (an adapter's ``weights_hash_scheme``)."""
    if scheme == WEIGHTS_HASH_RAW:
        return weights_hash(tensors)
    if scheme == WEIGHTS_HASH_BLOBS:
        return blob_weights_hash(tensors)
    raise WSEError(ErrorCode.INVALID_MODEL, f"unknown weights-hash scheme {scheme!r}")


def weights_hash_scheme(adapter: Any) -> str:
    """The weights-hash scheme an adapter declares; M1 adapters use the raw-bytes scheme."""
    return getattr(adapter, "weights_hash_scheme", WEIGHTS_HASH_RAW)


def weights_hash_matches(
    expected: str, tensors: Mapping[str, torch.Tensor], blob_ids: Mapping[str, str] | None = None
) -> bool:
    """True when ``expected`` is the weights hash of ``tensors`` under either scheme."""
    return blob_weights_hash(tensors, blob_ids) == expected or weights_hash(tensors) == expected


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
