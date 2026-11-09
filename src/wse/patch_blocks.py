"""Binary patch blocks and binary patch files (M2 plan Section 3.3).

A block records the exact stored bits that one materialized step writes into
one tensor, so replay writes them back without running any operator:

* **sparse**: ``count`` flat row-major indices into the tensor, little-endian
  ``uint64``, followed by ``count`` bit patterns of the stored dtype,
  little-endian;
* **dense**: the bit patterns of one contiguous selection, row-major.

The block's ID is the SHA-256 of its bytes. Its ``base_blob_id`` and
``result_blob_id`` bind it to the tensor's bytes before and after the step.
``make_block`` records only the elements whose bits changed, unless a dense
block of the step's selection is smaller.

A binary patch file stores a JSON document (a patch artifact) together with
the blocks its steps reference::

    b"WSEPATCH" | version u32 | header length u64 | header JSON | block bytes ...

The header is canonical JSON ``{"document": ..., "blocks": [{"block_id",
"offset", "size"}, ...]}`` with blocks in ``block_id`` order; offsets count
from the end of the header. A file that does not start with the magic bytes is
an M1 JSON patch artifact with no blocks.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from wse.schemas import AxisSlice, ErrorCode, PatchBlock, WSEError, canonical_json
from wse.tensors import BIT_PATTERN_DTYPES, blob_id, dtype_name, resolve_selection, tensor_bits

MAGIC = b"WSEPATCH"
FILE_VERSION = 1
_PREFIX = struct.Struct("<8sIQ")
_INDEX_BYTES = 8
_MAX_HEADER_BYTES = 64 * 1024 * 1024


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_PATCH, message, details=details)


def _bit_dtype(dtype: str) -> np.dtype:
    if dtype not in BIT_PATTERN_DTYPES:
        raise _invalid(f"blocks support the stored dtypes {sorted(BIT_PATTERN_DTYPES)}, not {dtype}", dtype=dtype)
    return np.dtype(BIT_PATTERN_DTYPES[dtype][1]).newbyteorder("<")


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def make_block(
    base: torch.Tensor, result: torch.Tensor, *, selection: Sequence[AxisSlice | Mapping[str, int]] | None = None
) -> tuple[PatchBlock, bytes]:
    """The block that turns ``base`` into ``result`` (same shape and dtype) and its bytes.

    Without ``selection`` the block is sparse over the changed elements. With
    one, the bits outside it must be unchanged, and the block is dense over the
    selection when that encoding is no larger than the sparse one.
    """
    dtype = dtype_name(base)
    if dtype_name(result) != dtype or tuple(base.shape) != tuple(result.shape):
        raise _invalid("a block joins two tensors of one shape and dtype")
    bit_dtype = _bit_dtype(dtype)
    before = tensor_bits(base).reshape(-1)
    after = tensor_bits(result).reshape(-1)
    changed = np.flatnonzero(before != after).astype(np.uint64)
    entries: list[AxisSlice] | None = None
    dense_count = None
    if selection is not None:
        slices = resolve_selection(list(selection), tuple(base.shape), tensor="block")
        entries = [AxisSlice(start=s.start, stop=s.stop, step=1) for s in slices]
        inside = np.zeros(tuple(base.shape), dtype=bool)
        inside[slices] = True
        if changed.size and not inside.reshape(-1)[changed.astype(np.intp)].all():
            raise _invalid("a step changed elements outside its selection")
        dense_count = math.prod(s.stop - s.start for s in slices)
    width = bit_dtype.itemsize
    if dense_count is not None and dense_count * width <= changed.size * (_INDEX_BYTES + width):
        region = tensor_bits(result)[tuple(slice(e.start, e.stop) for e in entries)]
        data = np.ascontiguousarray(region, dtype=bit_dtype).tobytes()
        kind, count, recorded = "dense", dense_count, entries
    else:
        data = changed.astype("<u8").tobytes() + np.ascontiguousarray(after[changed.astype(np.intp)],
                                                                        dtype=bit_dtype).tobytes()
        kind, count, recorded = "sparse", int(changed.size), None
    block = PatchBlock(kind=kind, dtype=dtype, tensor_shape=list(base.shape), selection=recorded, count=count,
                       block_id=hashlib.sha256(data).hexdigest(), base_blob_id=blob_id(base),
                       result_blob_id=blob_id(result))
    return block, data


def decode_block(block: PatchBlock, data: bytes) -> tuple[np.ndarray | None, np.ndarray]:
    """``(indices, bits)`` of a sparse block, or ``(None, bits)`` of a dense one, after checking its bytes."""
    if hashlib.sha256(data).hexdigest() != block.block_id:
        raise _invalid(f"the bytes of block {block.block_id[:16]}... do not match its ID", block_id=block.block_id)
    bit_dtype = _bit_dtype(block.dtype)
    width = bit_dtype.itemsize
    total = math.prod(block.tensor_shape)
    if block.kind == "sparse":
        if len(data) != block.count * (_INDEX_BYTES + width):
            raise _invalid("a sparse block's size does not match its count", block_id=block.block_id)
        split = block.count * _INDEX_BYTES
        indices = np.frombuffer(data[:split], dtype="<u8").astype(np.uint64)
        if indices.size and (int(indices.max()) >= total or np.any(np.diff(indices.astype(np.int64)) <= 0)):
            raise _invalid("a sparse block's indices must increase and lie inside the tensor",
                           block_id=block.block_id)
        return indices, np.frombuffer(data[split:], dtype=bit_dtype)
    assert block.selection is not None
    slices = resolve_selection(block.selection, block.tensor_shape, tensor="block")
    shape = tuple(s.stop - s.start for s in slices)
    if block.count != math.prod(shape) or len(data) != block.count * width:
        raise _invalid("a dense block's size does not match its selection", block_id=block.block_id)
    return None, np.frombuffer(data, dtype=bit_dtype).reshape(shape)


def apply_block(base: torch.Tensor, block: PatchBlock, data: bytes) -> torch.Tensor:
    """A new tensor: ``base`` with the block's bits written; ``base`` itself is never modified.

    Refuses a base whose blob ID differs from the block's (``VERSION_CONFLICT``)
    and a result whose blob ID differs (``INVALID_PATCH``).
    """
    if dtype_name(base) != block.dtype or list(base.shape) != block.tensor_shape:
        raise _invalid(f"block for a {block.dtype} tensor of shape {block.tensor_shape} applied to a "
                       f"{dtype_name(base)} tensor of shape {list(base.shape)}")
    if blob_id(base) != block.base_blob_id:
        raise WSEError(ErrorCode.VERSION_CONFLICT, "the tensor is not the one this block was materialized against; "
                       "an exact patch applies only to its own base", details={"block_id": block.block_id})
    indices, bits = decode_block(block, data)
    result = base.detach().clone(memory_format=torch.contiguous_format)
    target = tensor_bits(result)
    if indices is None:
        assert block.selection is not None
        target[tuple(slice(e.start, e.stop) for e in block.selection)] = bits
    else:
        target.reshape(-1)[indices.astype(np.intp)] = bits
    if blob_id(result) != block.result_blob_id:
        raise _invalid("the tensor rebuilt from the block does not match its recorded result blob ID",
                       block_id=block.block_id)
    return result


# ---------------------------------------------------------------------------
# Binary patch files
# ---------------------------------------------------------------------------


def write_patch_file(document: Mapping[str, Any], blocks: Mapping[str, bytes]) -> bytes:
    """A binary patch file holding ``document`` (plain JSON) and ``blocks`` (block ID -> bytes)."""
    entries, offset = [], 0
    for block_id in sorted(blocks):
        data = blocks[block_id]
        if hashlib.sha256(data).hexdigest() != block_id:
            raise _invalid("a block's bytes do not match its ID", block_id=block_id)
        entries.append({"block_id": block_id, "offset": offset, "size": len(data)})
        offset += len(data)
    header = canonical_json({"document": dict(document), "blocks": entries}).encode("utf-8")
    return b"".join([_PREFIX.pack(MAGIC, FILE_VERSION, len(header)), header,
                     *(blocks[entry["block_id"]] for entry in entries)])


def is_patch_file(data: bytes) -> bool:
    return data[:len(MAGIC)] == MAGIC


def read_patch_file(data: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """``(document, blocks)`` of a binary patch file, or of an M1 JSON patch (no blocks); checks every block."""
    if not is_patch_file(data):
        try:
            return json.loads(data), {}
        except ValueError as exc:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"patch artifact is neither JSON nor a binary patch: {exc}") \
                from exc

    def corrupt(reason: str) -> WSEError:
        return WSEError(ErrorCode.ARTIFACT_CORRUPT, f"binary patch file is corrupt: {reason}")

    if len(data) < _PREFIX.size:
        raise corrupt("it is truncated")
    _, version, header_size = _PREFIX.unpack_from(data)
    if version != FILE_VERSION:
        raise corrupt(f"unknown version {version}")
    if header_size > min(_MAX_HEADER_BYTES, len(data) - _PREFIX.size):
        raise corrupt("its header is larger than the file")
    start = _PREFIX.size + header_size
    try:
        header = json.loads(data[_PREFIX.size:start])
        document, entries = header["document"], header["blocks"]
        spans = [(str(e["block_id"]), int(e["offset"]), int(e["size"])) for e in entries]
    except (ValueError, KeyError, TypeError) as exc:
        raise corrupt(f"unreadable header ({exc})") from exc
    blocks: dict[str, bytes] = {}
    position = 0
    for block_id, offset, size in spans:
        if offset != position or size < 0 or start + offset + size > len(data):
            raise corrupt("its blocks do not tile the file")
        block = data[start + offset:start + offset + size]
        if hashlib.sha256(block).hexdigest() != block_id:
            raise corrupt(f"block {block_id[:16]}... does not match its ID")
        blocks[block_id] = block
        position = offset + size
    if start + position != len(data):
        raise corrupt("it has trailing bytes")
    return document, blocks
