"""Importing published edits as exact WSE patches (M2 plan Section 3.4).

The ``lm-understand-v1`` format is the edit directory of the article's logs
(``exp8_math/runs/<run>/edit``):

* ``manifest.json``: ``{"tensors": [{"name", "shape", "dtype", "form",
  "base_sha256", "edited_sha256", "changed_entries", ...}, ...]}``, where
  ``form`` is ``sparse`` or ``lowrank`` and the hashes are SHA-256 digests of
  the BF16 tensor bytes (the convention of WSE blob IDs);
* ``edit.npz``: for tensor ``i``, ``t{i}_idx`` (flat indices) and ``t{i}_val``
  (BF16 bit patterns, uint16) of exact overwrites, plus ``t{i}_a`` (rows x
  rank) and ``t{i}_b`` (rank x columns) for the low-rank form.

Import verifies every base hash before rebuilding anything, rebuilds each
tensor exactly as the article's ``apply_edit.py`` does (``rebuild``), and
verifies every edited hash and changed-entry count. Only then does the engine
record the patch and its candidate (``Engine.import_patch``). The published
hash decides exactness: if the low-rank path rounded differently from the
article, the edited hash would not match and the patch would be refused.

The edit files are untrusted input: every member of ``edit.npz`` is checked
for its expected name, then for its dtype and shape from its ``.npy`` header,
before any array data is decompressed (so a small compressed archive cannot
force a large allocation); arrays load with ``allow_pickle=False``, and
indices are range-checked before use.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from wse.patch_blocks import make_block
from wse.schemas import (
    AxisSlice,
    ErrorCode,
    ExtensionRef,
    ExternalPatchSource,
    ImportedPatchTensor,
    MaterializedPatch,
    PatchStep,
    WSEError,
)
from wse.tensors import bf16_to_f32, blob_id, f32_to_bf16_bits_rne, tensor_bits, tensor_hash

if TYPE_CHECKING:
    from wse.api import ModelHandle

FORMAT = "lm-understand-v1"
FORMATS = (FORMAT,)
MANIFEST_FILE = "manifest.json"
ARRAYS_FILE = "edit.npz"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARRAYS_BYTES = 4 * 1024 * 1024 * 1024
FORMS = ("sparse", "lowrank")
# The recorded operator of an imported step: no WSE operator ran; the step's block is the exact result.
IMPORT_OPERATOR = ExtensionRef(extension_id="lm_understand_import", version="1")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_PATCH, message, details=details)


@dataclass(frozen=True)
class LMUnderstandEdit:
    """A parsed and checked ``lm-understand-v1`` edit directory."""

    run: str
    entries: list[dict[str, Any]]
    arrays: Mapping[str, np.ndarray]
    files: dict[str, str]  # file name -> SHA-256 hex digest of the bytes read


def _read(path: Path, limit: int) -> bytes:
    try:
        if path.stat().st_size > limit:
            raise _invalid(f"{path.name} is larger than {limit} bytes")
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise _invalid(f"the edit directory has no {path.name}") from exc
    except OSError as exc:
        raise _invalid(f"cannot read {path.name}: {exc}") from exc


def _entry(index: int, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _invalid(f"manifest tensor {index} must be an object")
    for key in ("name", "shape", "dtype", "form", "base_sha256", "edited_sha256"):
        if key not in raw:
            raise _invalid(f"manifest tensor {index} has no {key!r}", index=index)
    name, shape, form = raw["name"], raw["shape"], raw["form"]
    if not isinstance(name, str) or not name:
        raise _invalid(f"manifest tensor {index} needs a nonempty name", index=index)
    if not isinstance(shape, list) or not shape or not all(isinstance(s, int) and not isinstance(s, bool) and s > 0
                                                          for s in shape):
        raise _invalid(f"tensor {name!r}: shape must be a nonempty list of positive integers", tensor=name)
    if raw["dtype"] != "bfloat16":
        raise _invalid(f"tensor {name!r}: {FORMAT} edits are bfloat16, not {raw['dtype']!r}", tensor=name)
    if form not in FORMS:
        raise _invalid(f"tensor {name!r}: form must be one of {list(FORMS)}, not {form!r}", tensor=name)
    if form == "lowrank" and len(shape) != 2:
        raise _invalid(f"tensor {name!r}: a low-rank edit needs a two-dimensional tensor", tensor=name)
    for key in ("base_sha256", "edited_sha256"):
        if not isinstance(raw[key], str) or not _HEX64.fullmatch(raw[key]):
            raise _invalid(f"tensor {name!r}: {key} must be 64 lowercase hex digits", tensor=name)
    changed = raw.get("changed_entries")
    if changed is not None and (not isinstance(changed, int) or isinstance(changed, bool) or changed < 0):
        raise _invalid(f"tensor {name!r}: changed_entries must be a nonnegative integer", tensor=name)
    return raw


def _expected_keys(i: int, entry: Mapping[str, Any]) -> list[str]:
    return [f"t{i}_idx", f"t{i}_val"] + ([f"t{i}_a", f"t{i}_b"] if entry["form"] == "lowrank" else [])


def _read_header(archive: zipfile.ZipFile, member: str) -> tuple[tuple[int, ...], np.dtype]:
    """The shape and dtype of an archive member, read from its .npy header without decompressing the data."""
    try:
        with archive.open(member) as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f"unsupported .npy format version {version}")
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise _invalid(f"{ARRAYS_FILE} member {member!r} is not a plain .npy array: {exc}") from exc
    if dtype.hasobject or dtype.fields is not None:
        raise _invalid(f"{ARRAYS_FILE} member {member!r} holds Python objects or records, which are not accepted")
    return tuple(int(n) for n in shape), dtype


def _read_member(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    """Decompress one member whose header has already been checked."""
    try:
        with archive.open(member) as f:
            return np.asarray(np.lib.format.read_array(f, allow_pickle=False))
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise _invalid(f"{ARRAYS_FILE} member {member!r} cannot be read: {exc}") from exc


def _check_headers(entries: list[dict[str, Any]], headers: Mapping[str, tuple[tuple[int, ...], np.dtype]]) -> None:
    """Names, dtypes, and shapes of every array, from the headers alone.

    Every size is bounded by the manifest's tensor shapes (indices by the
    tensor's element count, low-rank factors by the smaller side), so the
    arrays loaded afterwards cannot be larger than a few copies of the tensors
    the edit names.
    """
    expected = {key for i, entry in enumerate(entries) for key in _expected_keys(i, entry)}
    unexpected = sorted(set(headers) - expected)
    if unexpected:
        raise _invalid(f"{ARRAYS_FILE} holds arrays no manifest tensor uses: {unexpected[:20]}")
    for i, entry in enumerate(entries):
        name, shape = entry["name"], tuple(entry["shape"])
        missing = [key for key in _expected_keys(i, entry) if key not in headers]
        if missing:
            raise _invalid(f"tensor {name!r}: {ARRAYS_FILE} has no {missing}", tensor=name)
        (idx_shape, idx_dtype), (val_shape, val_dtype) = headers[f"t{i}_idx"], headers[f"t{i}_val"]
        if len(idx_shape) != 1 or idx_dtype.kind not in "iu" or val_shape != idx_shape or val_dtype != np.uint16 \
                or idx_shape[0] > math.prod(shape):
            raise _invalid(f"tensor {name!r}: t{i}_idx must be integer and t{i}_val uint16 bit patterns, "
                           f"one-dimensional, of one length, and at most {math.prod(shape)} long", tensor=name)
        if entry["form"] == "lowrank":
            (a_shape, a_dtype), (b_shape, b_dtype) = headers[f"t{i}_a"], headers[f"t{i}_b"]
            rank = a_shape[1] if len(a_shape) == 2 else 0
            if a_dtype.kind != "f" or b_dtype.kind != "f" or len(a_shape) != 2 or not 1 <= rank <= min(shape) \
                    or a_shape != (shape[0], rank) or b_shape != (rank, shape[1]):
                raise _invalid(f"tensor {name!r}: t{i}_a must be (rows, rank) and t{i}_b (rank, columns), floating "
                               f"point, with 1 <= rank <= {min(shape)}, for shape {list(shape)}", tensor=name)


def _check_values(entries: list[dict[str, Any]], arrays: Mapping[str, np.ndarray]) -> None:
    for i, entry in enumerate(entries):
        idx = arrays[f"t{i}_idx"]
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= math.prod(entry["shape"])):
            raise _invalid(f"tensor {entry['name']!r}: t{i}_idx holds indices outside the tensor",
                           tensor=entry["name"])


def _load_arrays(arrays_bytes: bytes, entries: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Check every member's name, dtype, and shape from its header, then decompress the members."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(arrays_bytes))
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        raise _invalid(f"{ARRAYS_FILE} is not a NumPy archive: {exc}") from exc
    with archive:
        members = archive.namelist()
        if len(set(members)) != len(members):
            raise _invalid(f"{ARRAYS_FILE} names a member more than once")
        not_npy = [m for m in members if not m.endswith(".npy")]
        if not_npy:
            raise _invalid(f"{ARRAYS_FILE} holds members that are not .npy arrays: {sorted(not_npy)[:20]}")
        keys = {m.removesuffix(".npy"): m for m in members}
        expected = {key for i, entry in enumerate(entries) for key in _expected_keys(i, entry)}
        unexpected = sorted(set(keys) - expected)
        if unexpected:  # refused by name before any header is read
            raise _invalid(f"{ARRAYS_FILE} holds arrays no manifest tensor uses: {unexpected[:20]}")
        headers = {key: _read_header(archive, member) for key, member in keys.items()}
        _check_headers(entries, headers)
        arrays = {}
        for key, member in keys.items():
            array = _read_member(archive, member)
            if array.shape != headers[key][0] or array.dtype != headers[key][1]:
                raise _invalid(f"{ARRAYS_FILE} member {member!r} does not match its header")
            arrays[key] = array
    _check_values(entries, arrays)
    return arrays


def read_lm_understand(edit_dir: str | Path) -> LMUnderstandEdit:
    """Read and check an edit directory; record the SHA-256 of both files it reads."""
    directory = Path(edit_dir)
    if not directory.is_dir():
        raise _invalid(f"edit directory {directory} does not exist")
    manifest_bytes = _read(directory / MANIFEST_FILE, MAX_MANIFEST_BYTES)
    arrays_bytes = _read(directory / ARRAYS_FILE, MAX_ARRAYS_BYTES)
    try:
        manifest = json.loads(manifest_bytes)
    except ValueError as exc:
        raise _invalid(f"{MANIFEST_FILE} is not JSON: {exc}") from exc
    tensors = manifest.get("tensors") if isinstance(manifest, dict) else None
    if not isinstance(tensors, list) or not tensors:
        raise _invalid(f"{MANIFEST_FILE} must hold a nonempty 'tensors' list")
    entries = [_entry(i, raw) for i, raw in enumerate(tensors)]
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names):
        raise _invalid(f"{MANIFEST_FILE} lists a tensor more than once")
    arrays = _load_arrays(arrays_bytes, entries)
    run = directory.parent.name if directory.name == "edit" else directory.name
    files = {MANIFEST_FILE: hashlib.sha256(manifest_bytes).hexdigest(),
             ARRAYS_FILE: hashlib.sha256(arrays_bytes).hexdigest()}
    return LMUnderstandEdit(run=run, entries=entries, arrays=arrays, files=files)


def rebuild(base_bits: np.ndarray, entry: Mapping[str, Any], z: Mapping[str, np.ndarray], i: int) -> np.ndarray:
    """base_bits: uint16 bit patterns of a BF16 tensor; returns the edited bit patterns.

    This is the article's ``apply_edit.rebuild`` (M2 plan Section 3.4). The
    low-rank form adds each rank-one term in FP64 to the widened base, rounds
    to FP32, then to BF16 with round-to-nearest-even on the FP32 bit pattern;
    the exact overwrites of ``t{i}_idx`` and ``t{i}_val`` follow in both forms.
    """
    if entry["form"] == "lowrank":
        w = bf16_to_f32(base_bits).astype(np.float64)
        a, b = z[f"t{i}_a"], z[f"t{i}_b"]  # a: (rows, rank); b: (rank, cols)
        for k in range(a.shape[1]):
            w += np.outer(a[:, k], b[k])
        e = f32_to_bf16_bits_rne(w.astype(np.float32)).reshape(-1)
    else:
        e = base_bits.copy().reshape(-1)
    e[z[f"t{i}_idx"]] = z[f"t{i}_val"]  # exact overwrite with stored bit patterns
    return e.reshape(base_bits.shape)


@dataclass(frozen=True)
class ExternalPatchResult:
    candidate: ModelHandle
    patch: MaterializedPatch
    blocks: dict[str, bytes]
    tensors: list[ImportedPatchTensor]


def materialize_lm_understand(model: ModelHandle, edit: LMUnderstandEdit,
                              source: ExternalPatchSource) -> ExternalPatchResult:
    """Verify and rebuild ``edit`` on ``model``; nothing is stored.

    Every base hash is checked before any tensor is rebuilt
    (``VERSION_CONFLICT`` names the first mismatch). A rebuilt tensor whose
    hash or changed-entry count differs from the manifest is refused
    (``INVALID_PATCH``).
    """
    from wse.patches import _derive

    infos = {info.name: info for info in model.architecture.tensors}
    for entry in edit.entries:
        name = entry["name"]
        info = infos.get(name)
        if info is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"the model has no tensor {name!r}; tensor names must match exactly",
                           details={"tensor": name})
        if info.dtype != "bfloat16" or list(info.shape) != list(entry["shape"]):
            raise _invalid(f"tensor {name!r} is {info.dtype} {list(info.shape)} in the model, but the edit is for "
                           f"bfloat16 {entry['shape']}", tensor=name)
        if any(name in group for group in model.architecture.alias_groups):
            raise WSEError(ErrorCode.UNSUPPORTED_CAPABILITY, f"tensor {name!r} belongs to an alias group; edits "
                           "of shared parameters are not supported", details={"tensor": name})
        if blob_id(model._tensors[name]) != entry["base_sha256"]:
            raise WSEError(ErrorCode.VERSION_CONFLICT, f"tensor {name!r} does not match the edit's base_sha256: this "
                           "edit was made on another base model", details={"tensor": name})
    tensors = dict(model._tensors)
    steps, blocks, records = [], {}, []
    for i, entry in enumerate(edit.entries):
        name = entry["name"]
        base = model._tensors[name]
        base_bits = tensor_bits(base)
        edited = rebuild(base_bits, entry, edit.arrays, i)
        digest = hashlib.sha256(np.ascontiguousarray(edited).tobytes()).hexdigest()
        if digest != entry["edited_sha256"]:
            raise _invalid(f"tensor {name!r} rebuilt to SHA-256 {digest}, not the edited_sha256 "
                           f"{entry['edited_sha256']}; the patch is refused", tensor=name, rebuilt_sha256=digest)
        changed = int(np.count_nonzero(edited != base_bits))
        if entry.get("changed_entries") is not None and changed != entry["changed_entries"]:
            raise _invalid(f"tensor {name!r} changes {changed} entries, not the manifest's {entry['changed_entries']}",
                           tensor=name)
        result = torch.from_numpy(np.ascontiguousarray(edited).view(np.int16)).view(torch.bfloat16)
        full = [AxisSlice(start=0, stop=size, step=1) for size in base.shape]
        block, data = make_block(base, result, selection=full)
        blocks[block.block_id] = data
        tensors[name] = result
        parameters = {"format": source.format, "form": entry["form"], "base_sha256": entry["base_sha256"],
                      "edited_sha256": entry["edited_sha256"]}
        if entry["form"] == "lowrank":
            parameters["rank"] = int(edit.arrays[f"t{i}_a"].shape[1])
        steps.append(PatchStep(op=source.format, operator=IMPORT_OPERATOR, tensor=name, selection=full,
                               parameters=parameters, dtype="bfloat16", before_hash=tensor_hash(name, base),
                               after_hash=tensor_hash(name, result), block=block, changed_elements=changed))
        records.append(ImportedPatchTensor(tensor=name, form=entry["form"], base_sha256=entry["base_sha256"],
                                           edited_sha256=digest, changed_elements=changed, block=block.kind))
    candidate = _derive(model, tensors)
    patch = MaterializedPatch(base=model.identity, result_weights_hash=candidate.weights_hash, steps=steps,
                              source=source)
    return ExternalPatchResult(candidate=candidate, patch=patch, blocks=blocks, tensors=records)
