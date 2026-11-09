"""Allow-listed reading of Hugging Face model snapshots (M2 plan Global Constraints, safe import).

A snapshot is a local directory as ``hf download --local-dir`` writes it. The
reader opens only allow-listed files: ``config.json``,
``generation_config.json``, safetensors weights and their index, tokenizer
files, and the chat template. It never unpickles a file and never executes
repository code; other files (pickled checkpoints, ONNX exports, training
arguments, custom modeling code) are never opened. Every file read is
recorded with its SHA-256, in reading order.

Weights are read through safetensors headers (validated here) and
memory-mapped tensors. This module never imports ``transformers``.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from wse.schemas import ErrorCode, WSEError, content_hash, json_hash

CONFIG_FILE = "config.json"
GENERATION_CONFIG_FILE = "generation_config.json"
INDEX_FILE = "model.safetensors.index.json"
SINGLE_WEIGHTS_FILE = "model.safetensors"
# The tokenizer identity covers these files when present (M2 plan Section 2.1);
# a SentencePiece model (tokenizer.model, a protobuf, not a pickle) counts too.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "vocab.json", "merges.txt", "tokenizer.model")
CHAT_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json")
ALLOWED_FILES = frozenset({CONFIG_FILE, GENERATION_CONFIG_FILE, INDEX_FILE, *TOKENIZER_FILES,
                           *CHAT_TEMPLATE_FILES})
# Files whose contents an import keeps (in the store's model_files/), so generation can tokenize and render
# prompts exactly as the identity hashes describe, without the snapshot directory.
MODEL_TEXT_FILES = (*TOKENIZER_FILES, *CHAT_TEMPLATE_FILES, GENERATION_CONFIG_FILE)
_SAFETENSORS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.safetensors$")
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 100 * 1024 * 1024
# safetensors dtype codes and their element sizes; M2 stores BF16, F16, and F32 only.
SAFETENSORS_DTYPE_SIZES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2,
                           "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
STORED_DTYPES = ("BF16", "F16", "F32")


def is_allowed(relpath: str) -> bool:
    """True for the snapshot files the importer may read: allow-listed names at the snapshot root."""
    return relpath in ALLOWED_FILES or bool(_SAFETENSORS_NAME.fullmatch(relpath))


class TensorHeader(NamedTuple):
    """One tensor of a safetensors file, from its header."""

    file: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def size_bytes(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count * SAFETENSORS_DTYPE_SIZES[self.dtype]


def _invalid(message: str, **details: Any) -> WSEError:
    return WSEError(ErrorCode.INVALID_MODEL, message, details=details)


class SnapshotReader:
    """Reads allow-listed files of one snapshot directory and records each file's SHA-256."""

    def __init__(self, snapshot: str | Path) -> None:
        path = Path(snapshot).expanduser()
        if not path.is_dir():
            raise WSEError(ErrorCode.NOT_FOUND, f"snapshot directory {path} does not exist")
        self.root = path.resolve()
        self.files: dict[str, str] = {}
        self.contents: dict[str, bytes] = {}  # the bytes read of MODEL_TEXT_FILES

    @property
    def read_log(self) -> list[str]:
        """The files read so far, in reading order."""
        return list(self.files)

    def _path(self, name: str) -> Path:
        if not is_allowed(name):
            raise _invalid(f"snapshot file {name!r} is not on the import allow-list", file=name)
        return self.root / name

    def has(self, name: str) -> bool:
        """Whether an allow-listed file exists (a stat; nothing is read)."""
        return is_allowed(name) and (self.root / name).is_file()

    def _record(self, name: str, digest: str) -> None:
        self.files[name] = "sha256:" + digest

    def read_bytes(self, name: str, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
        path = self._path(name)
        size = path.stat().st_size
        if size > max_bytes:
            raise _invalid(f"snapshot file {name} is {size} bytes; the limit is {max_bytes}", file=name)
        data = path.read_bytes()
        self._record(name, hashlib.sha256(data).hexdigest())
        if name in MODEL_TEXT_FILES:
            self.contents[name] = data
        return data

    def read_json(self, name: str) -> Any:
        try:
            return json.loads(self.read_bytes(name).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _invalid(f"snapshot file {name} is not valid JSON: {exc}", file=name) from exc

    def _hash_file(self, name: str) -> Path:
        path = self._path(name)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        self._record(name, digest.hexdigest())
        return path

    # -- weights --------------------------------------------------------------

    def _weight_files(self) -> dict[str, str] | None:
        """Tensor name to file from the index, or ``None`` for a single ``model.safetensors``."""
        if self.has(INDEX_FILE):
            index = self.read_json(INDEX_FILE)
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise _invalid(f"{INDEX_FILE} has no weight_map")
            for name, file in weight_map.items():
                if not isinstance(file, str) or not _SAFETENSORS_NAME.fullmatch(file):
                    raise _invalid(f"{INDEX_FILE} maps {name!r} to {file!r}, which is not a safetensors file "
                                   "name at the snapshot root", tensor=name, file=str(file))
            return dict(weight_map)
        if self.has(SINGLE_WEIGHTS_FILE):
            return None
        pickled = sorted(path.name for path in self.root.iterdir() if path.name.endswith(PICKLE_SUFFIXES))
        raise _invalid(
            f"snapshot has no safetensors weights ({SINGLE_WEIGHTS_FILE} or {INDEX_FILE}); pickled checkpoints"
            + (f" such as {pickled}" if pickled else "") + " are never unpickled, so convert the model to "
            "safetensors first",
            pickled_files=pickled, read_files=self.read_log,
        )

    def _header(self, name: str) -> dict[str, TensorHeader]:
        path = self._hash_file(name)
        size = path.stat().st_size
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise _invalid(f"{name} is not a safetensors file", file=name)
            (length,) = struct.unpack("<Q", prefix)
            if length > min(MAX_HEADER_BYTES, size - 8):
                raise _invalid(f"{name} has an invalid safetensors header length", file=name)
            raw = handle.read(length)
        try:
            header = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _invalid(f"{name} has an invalid safetensors header: {exc}", file=name) from exc
        if not isinstance(header, dict):
            raise _invalid(f"{name} has an invalid safetensors header", file=name)
        data_size = size - 8 - length
        tensors: dict[str, TensorHeader] = {}
        spans: list[tuple[int, int, str]] = []
        for tensor, entry in header.items():
            if tensor == "__metadata__":
                continue
            try:
                dtype, shape, (begin, end) = entry["dtype"], tuple(entry["shape"]), entry["data_offsets"]
                valid = (dtype in SAFETENSORS_DTYPE_SIZES and all(isinstance(s, int) and s >= 0 for s in shape)
                         and isinstance(begin, int) and isinstance(end, int) and 0 <= begin <= end <= data_size)
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                raise _invalid(f"{name}: the header entry of tensor {tensor!r} is invalid", file=name, tensor=tensor)
            info = TensorHeader(name, dtype, shape)
            if end - begin != info.size_bytes:
                raise _invalid(f"{name}: tensor {tensor!r} has {end - begin} bytes; its dtype and shape need "
                               f"{info.size_bytes}", file=name, tensor=tensor)
            tensors[tensor] = info
            spans.append((begin, end, tensor))
        # As safetensors requires: the tensors cover the data section contiguously, without overlaps or gaps.
        position = 0
        for begin, end, tensor in sorted(spans):
            if begin != position:
                raise _invalid(f"{name}: tensor {tensor!r} starts at byte {begin} of the data section; the previous "
                               f"tensor ends at {position} (tensors must not overlap or leave gaps)",
                               file=name, tensor=tensor)
            position = end
        if position != data_size:
            raise _invalid(f"{name}: the tensors cover {position} of the {data_size} bytes of the data section",
                           file=name)
        return tensors

    def tensor_table(self) -> dict[str, TensorHeader]:
        """Every tensor of the snapshot's weights, from the safetensors headers (each file is hashed)."""
        weight_map = self._weight_files()
        files = sorted(set(weight_map.values())) if weight_map is not None else [SINGLE_WEIGHTS_FILE]
        table: dict[str, TensorHeader] = {}
        for file in files:
            if not self.has(file):
                raise _invalid(f"weights file {file} listed in {INDEX_FILE} does not exist", file=file)
            for tensor, info in self._header(file).items():
                if tensor in table:
                    raise _invalid(f"tensor {tensor!r} appears in both {table[tensor].file} and {file}", tensor=tensor)
                table[tensor] = info
        if weight_map is not None:
            listed = {tensor: file for tensor, file in weight_map.items()}
            stored = {tensor: info.file for tensor, info in table.items()}
            if listed != stored:
                differing = sorted(set(listed.items()) ^ set(stored.items()))[:20]
                raise _invalid(f"{INDEX_FILE} and the safetensors headers disagree; first differences: "
                               f"{differing}", differences=[list(item) for item in differing])
        return table

    @contextmanager
    def open_weights(self, file: str) -> Iterator[Any]:
        """A memory-mapped safetensors handle of a weights file that ``tensor_table`` read."""
        from safetensors import safe_open

        if file not in self.files:
            raise _invalid(f"weights file {file} was not read through the snapshot's tensor table", file=file)
        try:
            handle = safe_open(str(self._path(file)), framework="pt")
        except Exception as exc:  # safetensors raises its own SafetensorError for malformed files
            raise _invalid(f"weights file {file} cannot be opened as safetensors: {exc}", file=file) from exc
        with handle:
            yield handle

    # -- identity -------------------------------------------------------------

    def tokenizer_hash(self) -> str | None:
        """Hash over the SHA-256 of each tokenizer file present, by name; ``None`` without any."""
        present = {}
        for name in TOKENIZER_FILES:
            if self.has(name):
                self.read_bytes(name)
                present[name] = self.files[name]
        return json_hash(present) if present else None

    def chat_template_hash(self) -> str | None:
        """Hash of the chat template: ``chat_template.jinja``, ``chat_template.json``, or ``tokenizer_config.json``."""
        if self.has("chat_template.jinja"):
            return content_hash(self.read_bytes("chat_template.jinja"))
        template: Any = None
        if self.has("chat_template.json"):
            data = self.read_json("chat_template.json")
            template = data.get("chat_template") if isinstance(data, dict) else None
        elif self.has("tokenizer_config.json"):
            data = self.read_json("tokenizer_config.json")
            template = data.get("chat_template") if isinstance(data, dict) else None
        if template is None:
            return None
        return content_hash(template.encode("utf-8")) if isinstance(template, str) else json_hash(template)

    def generation_config_hash(self) -> str | None:
        if not self.has(GENERATION_CONFIG_FILE):
            return None
        return json_hash(self.read_json(GENERATION_CONFIG_FILE))
