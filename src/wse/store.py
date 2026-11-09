"""Content-addressed artifact storage with READY visibility and recovery.

Commit protocol (the filesystem and SQLite share no transaction):

1. Write each content file to ``artifacts/tmp/``, sync it, and verify it.
2. Publish it to its immutable content-addressed path with an atomic rename,
   then sync the directory. A published file without database metadata is an
   invisible orphan.
3. In one SQLite transaction, write artifact metadata, the READY checkpoint
   row, and an audit event. Callers add their own rows (ownership, budget
   charges, idempotent results) to the same transaction through
   ``commit_checkpoint(on_commit=...)``.

Patch previews follow the same protocol: their content-addressed JSON file
is published first and becomes visible only through a ``patches`` row.

Tensors are stored per tensor as content-addressed blobs (M2 plan Section
3.1): ``artifacts/blobs/<sha256>`` holds one tensor's raw little-endian
row-major bytes in its stored dtype, and its name is the SHA-256 of those
bytes (``wse.tensors.blob_id``). Each weights artifact has a blob manifest in
the ``weight_tensors`` table that maps canonical tensor names to blob IDs,
dtypes, and shapes, so a candidate stores only the blobs that differ from its
parent and tensors with identical bytes share one blob. A blob's reference
count is its number of ``weight_tensors`` rows. Blobs are published before the
weights artifact, by the same write-sync-rename sequence, and become visible
with the commit transaction that references them.

The weights artifact itself is M1's single safetensors file for checkpoints of
at most ``single_file_max_bytes`` (so M1 layouts and exports are unchanged), and
a canonical JSON blob index (``weights/<hash>.json``) above that limit, so a
large candidate never copies its unchanged tensors.

Only READY checkpoints can be listed or loaded. Recovery removes expired
temporary files, unreferenced artifacts, and unreferenced blobs, and marks
checkpoints whose files are missing or fail their hash check as CORRUPT, which
prohibits loading. A tampered blob marks every checkpoint that uses it CORRUPT.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from wse.database import Database, insert_audit_event, utc_now
from wse.schemas import (
    CHECKPOINT_ID_PATTERN,
    SCHEMA_VERSION,
    ErrorCode,
    ModelManifest,
    WSEError,
    canonical_json,
    content_hash,
)
from wse.tensors import (
    blob_id,
    dtype_from_name,
    dtype_name,
    raw_bytes_view,
    tensor_from_buffer,
    weights_hash_matches,
)

WEIGHTS_DIR = "weights"
MANIFESTS_DIR = "manifests"
ANALYSES_DIR = "analyses"
PATCHES_DIR = "patches"
BLOBS_DIR = "blobs"
ANALYSIS_URI_PREFIX = "wse://analyses/"
TMP_DIR = "tmp"
DEFAULT_RECOVERY_GRACE_SECONDS = 600.0
# Checkpoints up to this size keep M1's single safetensors weights file next to
# their blobs; larger ones get a JSON blob index as their weights artifact.
DEFAULT_SINGLE_FILE_MAX_BYTES = 64 * 1024 * 1024
WEIGHTS_INDEX_FORMAT = "wse-weights-index-v1"
_RELPATH_PATTERN = re.compile(
    r"^(?:weights/[0-9a-f]{64}\.(?:safetensors|json)|manifests/[0-9a-f]{64}\.json"
    r"|analyses/[0-9a-f]{64}\.json|patches/[0-9a-f]{64}\.json|blobs/[0-9a-f]{64})$"
)
# Integer dtypes whose bit patterns carry a stored dtype of the same width.
_BIT_DTYPES: dict[int, tuple[type, torch.dtype]] = {
    1: (np.uint8, torch.uint8),
    2: (np.int16, torch.int16),
    4: (np.int32, torch.int32),
    8: (np.int64, torch.int64),
}
_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


@dataclass
class RecoveryReport:
    removed_temp_files: list[str] = field(default_factory=list)
    removed_orphans: list[str] = field(default_factory=list)
    corrupted_checkpoints: list[str] = field(default_factory=list)
    corrupted_patches: list[str] = field(default_factory=list)
    removed_blobs: list[str] = field(default_factory=list)


class TensorEntry(NamedTuple):
    """One row of a blob manifest: a canonical tensor name and its stored blob."""

    name: str
    blob_id: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def size_bytes(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count * torch.empty((), dtype=dtype_from_name(self.dtype)).element_size()


OnCommit = Callable[[sqlite3.Connection, ModelManifest], None]


def checkpoint_not_found(checkpoint_id: object) -> WSEError:
    """The error for a checkpoint ID that does not exist or is not visible to the caller.

    Hidden and nonexistent checkpoints share this exact error, so a caller
    cannot learn whether an ID it may not see exists.
    """
    return WSEError(
        ErrorCode.NOT_FOUND,
        f"checkpoint {checkpoint_id!r} does not exist",
        details={"checkpoint_id": str(checkpoint_id)},
    )


def _hex(hash_text: str) -> str:
    return hash_text.removeprefix("sha256:")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blob_relpath(blob: str) -> str:
    return f"{BLOBS_DIR}/{blob}"


def _tensor_entries(tensors: Mapping[str, torch.Tensor]) -> list[TensorEntry]:
    return [
        TensorEntry(name, blob_id(t), dtype_name(t), tuple(int(size) for size in t.shape))
        for name, t in sorted(tensors.items())
    ]


def _write_array(path: Path, data: np.ndarray) -> None:
    with open(path, "wb") as handle:
        handle.write(memoryview(data))


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ArtifactStore:
    """Immutable artifacts under ``<workspace>/artifacts`` plus SQLite metadata."""

    def __init__(self, workspace: Path | str, *, single_file_max_bytes: int = DEFAULT_SINGLE_FILE_MAX_BYTES) -> None:
        self.workspace = Path(workspace)
        self.artifacts_dir = self.workspace / "artifacts"
        self.tmp_dir = self.artifacts_dir / TMP_DIR
        self.single_file_max_bytes = int(single_file_max_bytes)
        for directory in (
            self.tmp_dir,
            self.artifacts_dir / WEIGHTS_DIR,
            self.artifacts_dir / MANIFESTS_DIR,
            self.artifacts_dir / ANALYSES_DIR,
            self.artifacts_dir / PATCHES_DIR,
            self.artifacts_dir / BLOBS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.workspace / "metadata.sqlite")
        self._verified: dict[str, tuple[int, int, int]] = {}
        try:
            self._index_unindexed_weights()
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> ArtifactStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- identifiers and paths ------------------------------------------------

    @staticmethod
    def new_checkpoint_id() -> str:
        return "ckpt_" + secrets.token_hex(16)

    def _artifact_path(self, relpath: str) -> Path:
        if not _RELPATH_PATTERN.fullmatch(relpath):
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"artifact path {relpath!r} is not a store path")
        return self.artifacts_dir / relpath

    # -- publishing -----------------------------------------------------------

    def _publish(
        self, relpath: str, write: Callable[[Path], None], *, expected_sha256: str | None = None
    ) -> tuple[str, int]:
        """Write a file through ``tmp/`` and publish it at ``relpath``.

        Returns the file's SHA-256 hex digest and size. When ``expected_sha256``
        is given, a written file with another digest is never published. The
        fresh file always replaces any existing file at ``relpath`` (same
        content, by construction), so a published file is never older than the
        recovery grace period when its database commit runs, even if an earlier
        crashed operation left an identical orphan behind.
        """
        final = self._artifact_path(relpath)
        temp = self.tmp_dir / f"{uuid.uuid4().hex}.part"
        try:
            write(temp)
            _fsync_path(temp)
            os.chmod(temp, _READ_ONLY)
            digest, size = _file_sha256(temp), temp.stat().st_size
            if expected_sha256 is not None and digest != expected_sha256:
                raise WSEError(
                    ErrorCode.ARTIFACT_CORRUPT,
                    f"written artifact {relpath} did not read back with its expected SHA-256",
                    retryable=True,
                )
            os.replace(temp, final)
            _fsync_path(final.parent)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return digest, size

    def _publish_weights(self, relpath: str, tensors: Mapping[str, torch.Tensor], expected: str) -> tuple[str, int]:
        def write(path: Path) -> None:
            # safetensors refuses tensors that share memory (an alias group's members
            # can be one tensor), so every member after the first is written from a copy.
            ordered: dict[str, torch.Tensor] = {}
            storages: set[int] = set()
            for name in sorted(tensors):
                tensor = tensors[name]
                pointer = tensor.untyped_storage().data_ptr() if tensor.numel() else None
                if pointer is not None and pointer in storages:
                    tensor = tensor.clone()
                elif pointer is not None:
                    storages.add(pointer)
                ordered[name] = tensor
            save_file(ordered, str(path))
            if not weights_hash_matches(expected, load_file(str(path))):
                raise WSEError(
                    ErrorCode.ARTIFACT_CORRUPT,
                    "written weights did not read back with the expected content hash",
                    retryable=True,
                )

        return self._publish(relpath, write)

    def _publish_bytes(self, relpath: str, data: bytes) -> tuple[str, int]:
        return self._publish(relpath, lambda path: path.write_bytes(data))

    def _publish_missing_blobs(self, entries: Iterable[TensorEntry], tensors: Mapping[str, torch.Tensor]) -> None:
        """Publish the blob of every entry whose blob is not registered yet.

        A registered blob commits together with the blob manifest that
        references it, and recovery never removes a referenced blob, so it is
        reused after a (cached) hash check instead of being rewritten; a failed
        check marks its checkpoints CORRUPT and refuses the commit, as for an
        M1 weights file. An unregistered file (an orphan of a crashed commit)
        is always rewritten, which refreshes it so a concurrent recovery scan
        cannot delete it before this commit.
        """
        published: set[str] = set()
        for entry in entries:
            if entry.blob_id in published:
                continue
            published.add(entry.blob_id)
            if self.db.conn.execute("SELECT 1 FROM blobs WHERE blob_id = ?", (entry.blob_id,)).fetchone():
                self._verified_blob_path(entry.blob_id)
                continue
            data = raw_bytes_view(tensors[entry.name])
            self._publish(_blob_relpath(entry.blob_id), lambda path, data=data: _write_array(path, data),
                          expected_sha256=entry.blob_id)

    def _insert_weight_tensors(
        self, conn: sqlite3.Connection, weights_artifact_id: str, entries: list[TensorEntry], now: str
    ) -> None:
        """Inside a write transaction: register blobs and the blob manifest of a weights artifact."""
        sizes: dict[str, int] = {}
        for entry in entries:
            sizes.setdefault(entry.blob_id, entry.size_bytes)
        for blob, size in sorted(sizes.items()):
            # Recovery deletes unreferenced blob files inside its own write
            # transaction, so a file that exists here survives until this commit.
            self.require_published(_blob_relpath(blob))
            conn.execute(
                "INSERT OR IGNORE INTO blobs (blob_id, relpath, size_bytes, created_at) VALUES (?, ?, ?, ?)",
                (blob, _blob_relpath(blob), size, now),
            )
        conn.executemany(
            "INSERT OR IGNORE INTO weight_tensors (weights_artifact_id, name, blob_id, dtype, shape) "
            "VALUES (?, ?, ?, ?, ?)",
            [(weights_artifact_id, e.name, e.blob_id, e.dtype, canonical_json(list(e.shape))) for e in entries],
        )

    def _publish_weights_index(self, relpath: str, weights: str, entries: list[TensorEntry]) -> tuple[str, int]:
        data = canonical_json({
            "format": WEIGHTS_INDEX_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "weights_hash": weights,
            "tensors": [
                {"name": e.name, "blob_id": e.blob_id, "dtype": e.dtype, "shape": list(e.shape)} for e in entries
            ],
        }).encode("utf-8")
        return self._publish(relpath, lambda path: path.write_bytes(data),
                             expected_sha256=hashlib.sha256(data).hexdigest())

    def commit_checkpoint(
        self,
        manifest: ModelManifest,
        tensors: Mapping[str, torch.Tensor],
        *,
        dedup_key: str | None,
        event_type: str,
        on_commit: OnCommit | None = None,
        experiment_id: str | None = None,
    ) -> ModelManifest:
        """Store ``tensors`` and ``manifest`` and make the checkpoint READY.

        When ``dedup_key`` matches an existing checkpoint, that checkpoint is
        returned and nothing is written. Identical weights are stored once.
        ``on_commit(conn, manifest)`` runs inside the commit transaction after
        the checkpoint row is written; if it raises, nothing is committed and
        the published files remain invisible orphans.
        """
        if dedup_key is not None:
            existing = self._checkpoint_for_dedup_key(dedup_key)
            if existing is not None:
                return self.load_manifest(existing)

        entries = _tensor_entries(tensors)
        self._publish_missing_blobs(entries, tensors)
        row = self.db.conn.execute(
            "SELECT * FROM artifacts WHERE kind = 'weights' AND content_hash = ?",
            (manifest.weights_hash,),
        ).fetchone()
        if row is not None:
            self._verified_path(row)
            weights_rel, weights_sha, weights_size = row["relpath"], row["file_sha256"], row["size_bytes"]
        elif sum(entry.size_bytes for entry in entries) <= self.single_file_max_bytes:
            weights_rel = f"{WEIGHTS_DIR}/{_hex(manifest.weights_hash)}.safetensors"
            weights_sha, weights_size = self._publish_weights(weights_rel, tensors, manifest.weights_hash)
        else:
            # The single-file path checks this on read-back; an index must not
            # register tensors under another weights hash either.
            if not weights_hash_matches(manifest.weights_hash, tensors, {e.name: e.blob_id for e in entries}):
                raise WSEError(ErrorCode.ARTIFACT_CORRUPT, "tensors do not match the manifest's weights hash")
            weights_rel = f"{WEIGHTS_DIR}/{_hex(manifest.weights_hash)}.json"
            weights_sha, weights_size = self._publish_weights_index(weights_rel, manifest.weights_hash, entries)

        manifest_bytes = manifest.canonical_json().encode("utf-8")
        manifest_hash = content_hash(manifest_bytes)
        manifest_rel = f"{MANIFESTS_DIR}/{_hex(manifest_hash)}.json"
        manifest_sha, manifest_size = self._publish_bytes(manifest_rel, manifest_bytes)

        weights_artifact_id = f"weights_{_hex(manifest.weights_hash)}"
        manifest_artifact_id = f"manifest_{_hex(manifest_hash)}"
        now = utc_now()
        winner: str | None = None
        with self.db.transaction() as conn:
            if dedup_key is not None:
                found = conn.execute(
                    "SELECT checkpoint_id FROM checkpoints WHERE dedup_key = ?", (dedup_key,)
                ).fetchone()
                winner = found["checkpoint_id"] if found else None
            if winner is None:
                # Recovery deletes orphans inside its own write transaction, so a
                # file that exists here cannot be removed before this commit.
                for relpath in (weights_rel, manifest_rel):
                    if not self._artifact_path(relpath).is_file():
                        raise WSEError(
                            ErrorCode.ARTIFACT_CORRUPT,
                            f"published artifact {relpath} disappeared before commit; retry the operation",
                            retryable=True,
                        )
                conn.execute(
                    "INSERT OR IGNORE INTO artifacts (artifact_id, kind, content_hash, file_sha256, "
                    "relpath, size_bytes, created_at) VALUES (?, 'weights', ?, ?, ?, ?, ?)",
                    (weights_artifact_id, manifest.weights_hash, weights_sha, weights_rel, weights_size, now),
                )
                self._insert_weight_tensors(conn, weights_artifact_id, entries, now)
                conn.execute(
                    "INSERT INTO artifacts (artifact_id, kind, content_hash, file_sha256, relpath, "
                    "size_bytes, created_at) VALUES (?, 'manifest', ?, ?, ?, ?, ?)",
                    (manifest_artifact_id, manifest_hash, manifest_sha, manifest_rel, manifest_size, now),
                )
                conn.execute(
                    "INSERT INTO checkpoints (checkpoint_id, state, schema_version, adapter_id, "
                    "adapter_version, config_hash, weights_hash, weights_artifact_id, "
                    "manifest_artifact_id, parent_checkpoint_id, dedup_key, created_at) "
                    "VALUES (?, 'READY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        manifest.checkpoint_id,
                        manifest.schema_version,
                        manifest.adapter.extension_id,
                        manifest.adapter.version,
                        manifest.config_hash,
                        manifest.weights_hash,
                        weights_artifact_id,
                        manifest_artifact_id,
                        manifest.provenance.parent_checkpoint_id,
                        dedup_key,
                        now,
                    ),
                )
                insert_audit_event(
                    conn,
                    event_type,
                    manifest.checkpoint_id,
                    {
                        "weights_hash": manifest.weights_hash,
                        "config_hash": manifest.config_hash,
                        "adapter": str(manifest.adapter),
                        "manifest_hash": manifest_hash,
                        "parent_checkpoint_id": manifest.provenance.parent_checkpoint_id,
                    },
                    experiment_id=experiment_id,
                )
                if on_commit is not None:
                    on_commit(conn, manifest)
        if winner is not None:
            # A concurrent identical import won; this call's manifest file is an orphan.
            return self.load_manifest(winner)
        return manifest

    # -- migration 8: blob manifests for M1 weights ---------------------------

    def _index_unindexed_weights(self) -> None:
        """Give every weights artifact without a blob manifest its blobs, in place.

        This is the data step of migration 8: M1 stored each checkpoint's
        weights as one safetensors file. Each such file is hash-checked, its
        tensors are published as blobs, and its blob manifest is committed in
        one transaction. The weights artifact, the manifests, and the
        checkpoints are not modified, so every M1 identity is unchanged. The
        step is idempotent and safe to run in concurrent processes; a crash
        leaves only orphan blobs, which recovery removes.
        """
        rows = self.db.conn.execute(
            "SELECT * FROM artifacts a WHERE a.kind = 'weights' AND NOT EXISTS "
            "(SELECT 1 FROM weight_tensors w WHERE w.weights_artifact_id = a.artifact_id) ORDER BY a.artifact_id"
        ).fetchall()
        for row in rows:
            if not row["relpath"].endswith(".safetensors"):  # pragma: no cover - index artifacts commit their rows
                continue
            try:
                path = self._verified_path(row)
            except WSEError:
                continue  # every checkpoint that uses it is now CORRUPT
            tensors = load_file(str(path))
            if not weights_hash_matches(row["content_hash"], tensors):
                self._mark_corrupt_for_artifact(row["artifact_id"], "weights content hash mismatch")
                continue
            entries = _tensor_entries(tensors)
            try:
                self._publish_missing_blobs(entries, tensors)
            except WSEError:
                continue  # a shared blob failed its check; its checkpoints are now CORRUPT
            with self.db.transaction() as conn:
                if conn.execute("SELECT 1 FROM weight_tensors WHERE weights_artifact_id = ? LIMIT 1",
                                (row["artifact_id"],)).fetchone():
                    continue  # a concurrent process indexed it first
                self._insert_weight_tensors(conn, row["artifact_id"], entries, utc_now())
                insert_audit_event(conn, "weights.blobs_indexed", row["artifact_id"], {
                    "weights_hash": row["content_hash"],
                    "tensor_count": len(entries),
                    "blob_ids": sorted({entry.blob_id for entry in entries}),
                })

    # -- large analysis results -----------------------------------------------

    def put_analysis_result(self, data: bytes) -> str:
        """Store a large analysis result; return its ``wse://analyses/<sha256>`` URI.

        These files are content-addressed and immutable. They are derived data
        (recomputable from a READY checkpoint), carry no checkpoint metadata, and
        are not part of the orphan scan. An existing file is reused only when
        its content still matches its digest; a damaged file is replaced.
        """
        digest = hashlib.sha256(data).hexdigest()
        relpath = f"{ANALYSES_DIR}/{digest}.json"
        path = self._artifact_path(relpath)
        if not (path.is_file() and _file_sha256(path) == digest):
            self._publish_bytes(relpath, data)
        return ANALYSIS_URI_PREFIX + digest

    def read_analysis_result(self, uri: str) -> bytes:
        """Return the stored bytes of an analysis result URI after a hash check."""
        digest = uri.removeprefix(ANALYSIS_URI_PREFIX) if isinstance(uri, str) else ""
        if not (isinstance(uri, str) and uri.startswith(ANALYSIS_URI_PREFIX)
                and re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"{uri!r} is not an analysis result URI ({ANALYSIS_URI_PREFIX}<sha256 hex>)",
            )
        path = self._artifact_path(f"{ANALYSES_DIR}/{digest}.json")
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise WSEError(ErrorCode.NOT_FOUND, f"analysis result {uri} does not exist") from None
        if hashlib.sha256(data).hexdigest() != digest:
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"analysis result {uri} failed its hash check")
        return data

    # -- patch artifacts ------------------------------------------------------

    def publish_patch(self, data: bytes) -> tuple[str, str, int]:
        """Publish a patch artifact; return its relpath, SHA-256 hex, and size.

        The file is invisible until a ``patches`` row references it; callers
        must check ``require_published`` inside their commit transaction.
        """
        digest = hashlib.sha256(data).hexdigest()
        relpath = f"{PATCHES_DIR}/{digest}.json"
        file_sha, size = self._publish_bytes(relpath, data)
        return relpath, file_sha, size

    def require_published(self, relpath: str) -> None:
        """Inside a write transaction: fail if a published file vanished."""
        if not self._artifact_path(relpath).is_file():
            raise WSEError(
                ErrorCode.ARTIFACT_CORRUPT,
                f"published artifact {relpath} disappeared before commit; retry the operation",
                retryable=True,
            )

    def read_patch(self, patch: sqlite3.Row) -> bytes:
        """Return the bytes of a READY patch row's artifact after a hash check.

        A missing or mismatching file marks the patch CORRUPT.
        """
        if patch["state"] != "READY":
            raise WSEError(
                ErrorCode.ARTIFACT_CORRUPT,
                f"patch {patch['patch_id']} is {patch['state']} and cannot be used",
                details={"patch_id": patch["patch_id"], "reason": patch["corrupt_reason"]},
            )
        path = self._artifact_path(patch["relpath"])
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = None
        if data is None or len(data) != patch["size_bytes"] or hashlib.sha256(data).hexdigest() != patch["file_sha256"]:
            reason = "patch artifact is missing" if data is None else "patch artifact hash mismatch"
            self._mark_patches_corrupt(patch["relpath"], reason)
            raise WSEError(
                ErrorCode.ARTIFACT_CORRUPT,
                f"patch {patch['patch_id']}: {reason}; it can no longer be applied",
                details={"patch_id": patch["patch_id"]},
            )
        return data

    def _mark_patches_corrupt(self, relpath: str, reason: str) -> list[str]:
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT patch_id, experiment_id FROM patches WHERE relpath = ? AND state = 'READY'", (relpath,)
            ).fetchall()
            conn.execute(
                "UPDATE patches SET state = 'CORRUPT', corrupt_reason = ? WHERE relpath = ? AND state = 'READY'",
                (reason, relpath),
            )
            for row in rows:
                insert_audit_event(conn, "patch.corrupt", row["patch_id"], {"reason": reason},
                                   experiment_id=row["experiment_id"])
        return [row["patch_id"] for row in rows]

    def _checkpoint_for_dedup_key(self, dedup_key: str) -> str | None:
        row = self.db.conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return row["checkpoint_id"] if row else None

    # -- reading --------------------------------------------------------------

    def _checkpoint_row(self, checkpoint_id: str) -> sqlite3.Row:
        row = None
        if isinstance(checkpoint_id, str) and CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id):
            row = self.db.conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            ).fetchone()
        if row is None:
            raise checkpoint_not_found(checkpoint_id)
        if row["state"] != "READY":
            raise WSEError(
                ErrorCode.ARTIFACT_CORRUPT,
                f"checkpoint {checkpoint_id} is {row['state']} and cannot be loaded",
                details={"checkpoint_id": checkpoint_id, "reason": row["corrupt_reason"]},
            )
        return row

    def _artifact_row(self, artifact_id: str) -> sqlite3.Row:
        row = self.db.conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:  # pragma: no cover - prevented by foreign keys
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"artifact {artifact_id} has no metadata")
        return row

    def _verified_path(self, artifact: sqlite3.Row) -> Path:
        """Return the artifact's path after checking its file hash.

        A missing or mismatching file marks every checkpoint that references
        it CORRUPT.
        """
        path = self._artifact_path(artifact["relpath"])
        try:
            info = path.stat()
        except FileNotFoundError:
            self._mark_corrupt_for_artifact(artifact["artifact_id"], "artifact file is missing")
            raise WSEError(
                ErrorCode.ARTIFACT_CORRUPT,
                f"artifact {artifact['artifact_id']} is missing; dependent checkpoints are now CORRUPT",
            ) from None
        signature = (info.st_ino, info.st_size, info.st_mtime_ns)
        if self._verified.get(artifact["artifact_id"]) == signature:
            return path
        if info.st_size != artifact["size_bytes"] or _file_sha256(path) != artifact["file_sha256"]:
            self._mark_corrupt_for_artifact(artifact["artifact_id"], "artifact hash mismatch")
            raise WSEError(
                ErrorCode.ARTIFACT_CORRUPT,
                f"artifact {artifact['artifact_id']} failed its hash check; dependent checkpoints are now CORRUPT",
            )
        self._verified[artifact["artifact_id"]] = signature
        return path

    def load_manifest(self, checkpoint_id: str) -> ModelManifest:
        row = self._checkpoint_row(checkpoint_id)
        path = self._verified_path(self._artifact_row(row["manifest_artifact_id"]))
        manifest = ModelManifest.model_validate_json(path.read_bytes())
        if manifest.checkpoint_id != checkpoint_id or manifest.weights_hash != row["weights_hash"]:
            self._mark_corrupt([checkpoint_id], "manifest does not match checkpoint metadata")
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"manifest of {checkpoint_id} is inconsistent")
        return manifest

    def weights_path(self, checkpoint_id: str) -> Path:
        row = self._checkpoint_row(checkpoint_id)
        return self._verified_path(self._artifact_row(row["weights_artifact_id"]))

    def _weight_entries(self, checkpoint_id: str, name: str | None = None) -> list[TensorEntry]:
        """The blob manifest of a READY checkpoint's verified weights artifact, in name order.

        With ``name``, only that tensor's entry (an empty list when it does not exist).
        """
        row = self._checkpoint_row(checkpoint_id)
        artifact_id = row["weights_artifact_id"]
        self._verified_path(self._artifact_row(artifact_id))
        if not self.db.conn.execute(
            "SELECT 1 FROM weight_tensors WHERE weights_artifact_id = ? LIMIT 1", (artifact_id,)
        ).fetchone():
            raise WSEError(
                ErrorCode.ARTIFACT_CORRUPT,
                f"weights of {checkpoint_id} have no blob manifest; reopen the workspace to finish migration 8",
                retryable=True,
            )
        query = "SELECT name, blob_id, dtype, shape FROM weight_tensors WHERE weights_artifact_id = ?"
        if name is None:
            rows = self.db.conn.execute(query + " ORDER BY name", (artifact_id,)).fetchall()
        else:
            rows = self.db.conn.execute(query + " AND name = ?", (artifact_id, name)).fetchall()
        return [TensorEntry(r["name"], r["blob_id"], r["dtype"], tuple(json.loads(r["shape"]))) for r in rows]

    def tensor_blob_ids(self, checkpoint_id: str) -> dict[str, str]:
        """Canonical tensor name to blob ID for a READY checkpoint."""
        return {entry.name: entry.blob_id for entry in self._weight_entries(checkpoint_id)}

    def blob_reference_counts(self) -> dict[str, int]:
        """Every registered blob's reference count: the blob-manifest rows that use it."""
        rows = self.db.conn.execute(
            "SELECT b.blob_id, COUNT(w.blob_id) AS refs FROM blobs b "
            "LEFT JOIN weight_tensors w ON w.blob_id = b.blob_id GROUP BY b.blob_id ORDER BY b.blob_id"
        ).fetchall()
        return {row["blob_id"]: row["refs"] for row in rows}

    def _blob_failed(self, blob: str, reason: str) -> WSEError:
        self._mark_corrupt_for_blob(blob, reason)
        return WSEError(
            ErrorCode.ARTIFACT_CORRUPT,
            f"blob {blob}: {reason}; every checkpoint that uses it is now CORRUPT",
            details={"blob_id": blob},
        )

    def _blob_path(self, blob: str) -> tuple[Path, int]:
        row = self.db.conn.execute("SELECT relpath, size_bytes FROM blobs WHERE blob_id = ?", (blob,)).fetchone()
        if row is None:  # pragma: no cover - prevented by foreign keys
            raise self._blob_failed(blob, "blob has no metadata")
        return self._artifact_path(row["relpath"]), row["size_bytes"]

    def _verified_blob_path(self, blob: str) -> Path:
        """Return a blob's path after checking that its SHA-256 equals its ID."""
        path, size = self._blob_path(blob)
        try:
            info = path.stat()
        except FileNotFoundError:
            raise self._blob_failed(blob, "blob file is missing") from None
        signature = (info.st_ino, info.st_size, info.st_mtime_ns)
        if self._verified.get("blob:" + blob) == signature:
            return path
        if info.st_size != size or _file_sha256(path) != blob:
            raise self._blob_failed(blob, "blob hash mismatch")
        self._verified["blob:" + blob] = signature
        return path

    def _read_blob(self, blob: str) -> bytearray:
        """Read a blob's bytes in one pass and check them against its ID."""
        path, size = self._blob_path(blob)
        try:
            with open(path, "rb") as handle:
                info = os.fstat(handle.fileno())
                buffer = bytearray(info.st_size)
                read = handle.readinto(buffer)
        except FileNotFoundError:
            raise self._blob_failed(blob, "blob file is missing") from None
        signature = (info.st_ino, info.st_size, info.st_mtime_ns)
        if self._verified.get("blob:" + blob) != signature:
            if read != size or info.st_size != size or hashlib.sha256(buffer).hexdigest() != blob:
                raise self._blob_failed(blob, "blob hash mismatch")
            self._verified["blob:" + blob] = signature
        return buffer

    def load_tensors(self, checkpoint_id: str) -> dict[str, torch.Tensor]:
        """Load every tensor of a READY checkpoint from its blobs.

        Each blob is hash-checked and loaded through ``torch.frombuffer`` on its
        raw bytes in the stored dtype. Names that share a blob get separate
        tensors.
        """
        manifest = self.load_manifest(checkpoint_id)
        tensors: dict[str, torch.Tensor] = {}
        buffers: dict[str, bytearray] = {}
        entries = self._weight_entries(checkpoint_id)
        for entry in entries:
            buffer = buffers.get(entry.blob_id)
            if buffer is None:
                buffer = buffers[entry.blob_id] = self._read_blob(entry.blob_id)
            else:
                buffer = bytearray(buffer)
            tensors[entry.name] = tensor_from_buffer(buffer, dtype_from_name(entry.dtype), entry.shape)
        # Every blob was checked against its ID, so the blob-ID scheme needs no second pass.
        if not weights_hash_matches(manifest.weights_hash, tensors, {e.name: e.blob_id for e in entries}):
            self._mark_corrupt([checkpoint_id], "weights content hash mismatch")
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"weights of {checkpoint_id} failed their content hash")
        return tensors

    def read_slice(self, checkpoint_id: str, tensor: str, slices: tuple[slice, ...]) -> torch.Tensor:
        """Read one region of a stored tensor from its memory-mapped blob."""
        entry = next(iter(self._weight_entries(checkpoint_id, tensor)), None)
        if entry is None:
            raise WSEError(
                ErrorCode.NOT_FOUND,
                f"tensor {tensor!r} does not exist in checkpoint {checkpoint_id}",
                details={"tensor": tensor},
            )
        dtype = dtype_from_name(entry.dtype)
        path = self._verified_blob_path(entry.blob_id)
        bits = _BIT_DTYPES.get(torch.empty((), dtype=dtype).element_size())
        if entry.size_bytes == 0 or bits is None:
            whole = tensor_from_buffer(bytearray(path.read_bytes()), dtype, entry.shape)
            return (whole[slices] if slices else whole).clone(memory_format=torch.contiguous_format)
        # Map the blob's bit patterns with an integer dtype of the same width
        # (NumPy has no BF16), copy the region, and view it as the stored dtype.
        mapped = np.memmap(path, dtype=bits[0], mode="r").reshape(entry.shape)
        region = np.array(mapped[slices] if slices else mapped, copy=True)
        del mapped
        return torch.from_numpy(region).view(dtype)

    def export_safetensors(self, checkpoint_id: str, destination: Path | str) -> Path:
        """Write a READY checkpoint's tensors under their canonical names to one safetensors file.

        A checkpoint whose weights artifact is already a single safetensors file
        is copied byte for byte; one stored as a blob index is materialized.
        """
        destination = Path(destination)
        path = self.weights_path(checkpoint_id)
        if path.suffix == ".safetensors":
            shutil.copyfile(path, destination)
        else:
            tensors = self.load_tensors(checkpoint_id)
            save_file({name: tensors[name] for name in sorted(tensors)}, str(destination))
        return destination

    def list_ready(self) -> list[str]:
        rows = self.db.conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE state = 'READY' ORDER BY created_at, checkpoint_id"
        ).fetchall()
        return [row["checkpoint_id"] for row in rows]

    def checkpoint_state(self, checkpoint_id: str) -> str | None:
        row = self.db.conn.execute(
            "SELECT state FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
        ).fetchone()
        return row["state"] if row else None

    # -- corruption and recovery ---------------------------------------------

    def _mark_corrupt(self, checkpoint_ids: list[str], reason: str) -> list[str]:
        marked: list[str] = []
        with self.db.transaction() as conn:
            for checkpoint_id in checkpoint_ids:
                updated = conn.execute(
                    "UPDATE checkpoints SET state = 'CORRUPT', corrupt_reason = ? "
                    "WHERE checkpoint_id = ? AND state = 'READY'",
                    (reason, checkpoint_id),
                ).rowcount
                if updated:
                    marked.append(checkpoint_id)
                    insert_audit_event(conn, "checkpoint.corrupt", checkpoint_id, {"reason": reason})
        return marked

    def _mark_corrupt_for_blob(self, blob: str, reason: str) -> list[str]:
        rows = self.db.conn.execute(
            "SELECT DISTINCT c.checkpoint_id FROM checkpoints c JOIN weight_tensors w "
            "ON w.weights_artifact_id = c.weights_artifact_id WHERE w.blob_id = ? ORDER BY c.checkpoint_id",
            (blob,),
        ).fetchall()
        self._verified.pop("blob:" + blob, None)
        return self._mark_corrupt([row["checkpoint_id"] for row in rows], reason)

    def _mark_corrupt_for_artifact(self, artifact_id: str, reason: str) -> list[str]:
        rows = self.db.conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE weights_artifact_id = ? OR manifest_artifact_id = ?",
            (artifact_id, artifact_id),
        ).fetchall()
        self._verified.pop(artifact_id, None)
        return self._mark_corrupt([row["checkpoint_id"] for row in rows], reason)

    def recover(self, *, grace_seconds: float = DEFAULT_RECOVERY_GRACE_SECONDS) -> RecoveryReport:
        """Startup recovery scan.

        Files younger than ``grace_seconds`` are left alone, because another
        process may be between publishing and committing them.
        """
        report = RecoveryReport()
        cutoff = time.time() - grace_seconds
        for path in sorted(self.tmp_dir.iterdir()):
            if path.is_file() and path.stat().st_mtime <= cutoff:
                path.unlink(missing_ok=True)
                report.removed_temp_files.append(path.name)
        # Hold the write lock while deleting orphans: a concurrent commit checks
        # that its files exist inside its own write transaction, so it either
        # commits first (and its files are referenced here) or sees the deletion.
        with self.db.transaction() as conn:
            referenced = {row["relpath"] for row in conn.execute("SELECT relpath FROM artifacts")}
            referenced |= {row["relpath"] for row in conn.execute("SELECT relpath FROM patches")}
            for directory in (WEIGHTS_DIR, MANIFESTS_DIR, PATCHES_DIR):
                for path in sorted((self.artifacts_dir / directory).iterdir()):
                    relpath = f"{directory}/{path.name}"
                    if relpath in referenced or not path.is_file() or path.stat().st_mtime > cutoff:
                        continue
                    path.unlink(missing_ok=True)
                    report.removed_orphans.append(relpath)
            # Blobs are reference counted: a file without a registered blob, or a
            # registered blob that no blob manifest references, is removed.
            # A blob row whose count is zero is deleted with its file at once; an
            # unregistered file only after the grace period, because a commit
            # may be between publishing it and registering it.
            counts = self.blob_reference_counts()
            unreferenced = {blob for blob, refs in counts.items() if refs == 0}
            conn.executemany("DELETE FROM blobs WHERE blob_id = ?", [(blob,) for blob in sorted(unreferenced)])
            for path in sorted((self.artifacts_dir / BLOBS_DIR).iterdir()):
                if not path.is_file() or (path.name in counts and path.name not in unreferenced):
                    continue
                if path.name not in unreferenced and path.stat().st_mtime > cutoff:
                    continue
                path.unlink(missing_ok=True)
                report.removed_blobs.append(path.name)
        rows = self.db.conn.execute(
            "SELECT DISTINCT a.* FROM artifacts a JOIN checkpoints c "
            "ON a.artifact_id IN (c.weights_artifact_id, c.manifest_artifact_id) WHERE c.state = 'READY'"
        ).fetchall()
        for artifact in rows:
            self._verified.pop(artifact["artifact_id"], None)
            path = self._artifact_path(artifact["relpath"])
            if not path.is_file():
                reason = "artifact file is missing"
            elif path.stat().st_size != artifact["size_bytes"] or _file_sha256(path) != artifact["file_sha256"]:
                reason = "artifact hash mismatch"
            else:
                continue
            report.corrupted_checkpoints.extend(self._mark_corrupt_for_artifact(artifact["artifact_id"], reason))
        blobs = self.db.conn.execute(
            "SELECT DISTINCT b.blob_id, b.relpath, b.size_bytes FROM blobs b "
            "JOIN weight_tensors w ON w.blob_id = b.blob_id "
            "JOIN checkpoints c ON c.weights_artifact_id = w.weights_artifact_id "
            "WHERE c.state = 'READY' ORDER BY b.blob_id"
        ).fetchall()
        for blob in blobs:
            self._verified.pop("blob:" + blob["blob_id"], None)
            path = self._artifact_path(blob["relpath"])
            if not path.is_file():
                reason = "blob file is missing"
            elif path.stat().st_size != blob["size_bytes"] or _file_sha256(path) != blob["blob_id"]:
                reason = "blob hash mismatch"
            else:
                continue
            for checkpoint_id in self._mark_corrupt_for_blob(blob["blob_id"], reason):
                if checkpoint_id not in report.corrupted_checkpoints:
                    report.corrupted_checkpoints.append(checkpoint_id)
        patches = self.db.conn.execute(
            "SELECT DISTINCT relpath, file_sha256, size_bytes FROM patches WHERE state = 'READY'"
        ).fetchall()
        for patch in patches:
            path = self._artifact_path(patch["relpath"])
            if not path.is_file():
                reason = "patch artifact is missing"
            elif path.stat().st_size != patch["size_bytes"] or _file_sha256(path) != patch["file_sha256"]:
                reason = "patch artifact hash mismatch"
            else:
                continue
            report.corrupted_patches.extend(self._mark_patches_corrupt(patch["relpath"], reason))
        return report
