"""Evaluator storage: datasets kept apart from the artifact store and every agent-facing read.

Datasets live under ``<workspace>/evaluator/datasets/``, outside
``artifacts/``, as read-only canonical JSON named by the hex digest of their
``data_hash``: tensor examples (``SuiteData``) or the problems of a
generation suite (``TextSuiteData``, M2). An optional generation record (for example the generator and
seed of a pregenerated final set) is stored next to the data and is read
only by operator code. Suites, reports, and comparisons refer to a dataset
only through its ``DatasetManifest``.

A store opened for a run-bound engine is sealed: every read and write fails
with ``POLICY_DENIED``, so no agent-facing service can reach the examples.
Only the worker and the operator's final evaluation open readable stores.
"""

from __future__ import annotations

import json
import os
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wse.schemas import (
    HASH_PATTERN,
    DatasetManifest,
    ErrorCode,
    SuiteData,
    TextDatasetManifest,
    TextSuiteData,
    WSEError,
    canonical_json,
    dataset_manifest_matches,
)

EVALUATOR_DIR = "evaluator"
DATASETS_DIR = "datasets"
_READ_ONLY = stat.S_IRUSR


class DatasetStore:
    """Content-addressed datasets under ``<workspace>/evaluator/datasets``."""

    def __init__(self, workspace: str | Path, *, sealed: bool = False) -> None:
        self.root = Path(workspace) / EVALUATOR_DIR / DATASETS_DIR
        self.sealed = sealed

    def _require_open(self, action: str) -> None:
        if self.sealed:
            raise WSEError(ErrorCode.POLICY_DENIED,
                           f"{action} is reserved for the evaluator and the operator and is not available to a run")

    def _path(self, data_hash: str, suffix: str = ".json") -> Path:
        if not isinstance(data_hash, str) or not HASH_PATTERN.fullmatch(data_hash):
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "data_hash must be 'sha256:<64 hex digits>'")
        return self.root / f"{data_hash.removeprefix('sha256:')}{suffix}"

    @staticmethod
    def _write(path: Path, text: str) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, _READ_ONLY)
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def put(self, data: SuiteData | TextSuiteData | Mapping[str, Any], *,
            generation: Mapping[str, Any] | None = None) -> DatasetManifest | TextDatasetManifest:
        """Store a dataset (operator only) and return its manifest; storing it again is a no-op."""
        self._require_open("storing evaluation data")
        if isinstance(data, (SuiteData, TextSuiteData)):
            parsed = data
        elif dict(data).get("kind") == "text":
            parsed = TextSuiteData.model_validate(dict(data))
        else:
            parsed = SuiteData.model_validate(dict(data))
        manifest = TextDatasetManifest.from_data(parsed) if isinstance(parsed, TextSuiteData) \
            else DatasetManifest.from_data(parsed)
        self._write(self._path(manifest.data_hash), parsed.canonical_json())
        if generation is not None:
            self._write(self._path(manifest.data_hash, ".generation.json"), canonical_json(dict(generation)))
        return manifest

    def exists(self, data_hash: str) -> bool:
        self._require_open("reading evaluation data")
        return self._path(data_hash).is_file()

    def load(self, dataset: DatasetManifest | TextDatasetManifest | str) -> SuiteData | TextSuiteData:
        """The verified examples of a stored dataset (evaluator and operator use only)."""
        self._require_open("reading evaluation data")
        manifest = dataset if isinstance(dataset, (DatasetManifest, TextDatasetManifest)) else None
        data_hash = manifest.data_hash if manifest is not None else dataset
        path = self._path(data_hash)
        if not path.is_file():
            raise WSEError(ErrorCode.NOT_FOUND, f"dataset {data_hash} is not in evaluator storage",
                           details={"data_hash": data_hash})
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            text_data = isinstance(raw, dict) and raw.get("kind") == "text"
            data = TextSuiteData.model_validate(raw) if text_data else SuiteData.model_validate(raw)
        except ValueError as exc:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored dataset {data_hash} is unreadable") from exc
        if data.data_hash != data_hash:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored dataset {data_hash} does not match its hash")
        if manifest is not None and not dataset_manifest_matches(data, manifest):
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"stored dataset {data_hash} does not match its manifest")
        return data

    def generation(self, data_hash: str) -> dict[str, Any] | None:
        """The generation record stored with a dataset (operator use only), or ``None``."""
        self._require_open("reading evaluation data")
        path = self._path(data_hash, ".generation.json")
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
