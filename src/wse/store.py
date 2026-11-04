"""Content-addressed artifact storage with READY visibility and recovery.

Commit protocol (the filesystem and SQLite share no transaction):

1. Write each content file to ``artifacts/tmp/``, sync it, and verify it.
2. Publish it to its immutable content-addressed path with an atomic rename,
   then sync the directory. A published file without database metadata is an
   invisible orphan.
3. In one SQLite transaction, write artifact metadata, the READY checkpoint
   row, and an audit event.

Only READY checkpoints can be listed or loaded. Recovery removes expired
temporary files and unreferenced artifacts, and marks checkpoints whose files
are missing or fail their hash check as CORRUPT, which prohibits loading.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from wse.database import Database, insert_audit_event, utc_now
from wse.schemas import (
    CHECKPOINT_ID_PATTERN,
    ErrorCode,
    ModelManifest,
    WSEError,
    content_hash,
)
from wse.tensors import weights_hash

WEIGHTS_DIR = "weights"
MANIFESTS_DIR = "manifests"
TMP_DIR = "tmp"
DEFAULT_RECOVERY_GRACE_SECONDS = 600.0
_RELPATH_PATTERN = re.compile(
    r"^(?:weights/[0-9a-f]{64}\.safetensors|manifests/[0-9a-f]{64}\.json)$"
)
_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


@dataclass
class RecoveryReport:
    removed_temp_files: list[str] = field(default_factory=list)
    removed_orphans: list[str] = field(default_factory=list)
    corrupted_checkpoints: list[str] = field(default_factory=list)


def _hex(hash_text: str) -> str:
    return hash_text.removeprefix("sha256:")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ArtifactStore:
    """Immutable artifacts under ``<workspace>/artifacts`` plus SQLite metadata."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace)
        self.artifacts_dir = self.workspace / "artifacts"
        self.tmp_dir = self.artifacts_dir / TMP_DIR
        for directory in (self.tmp_dir, self.artifacts_dir / WEIGHTS_DIR, self.artifacts_dir / MANIFESTS_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.workspace / "metadata.sqlite")
        self._verified: dict[str, tuple[int, int, int]] = {}

    def close(self) -> None:
        self.db.close()

    # -- identifiers and paths ------------------------------------------------

    @staticmethod
    def new_checkpoint_id() -> str:
        return "ckpt_" + secrets.token_hex(16)

    def _artifact_path(self, relpath: str) -> Path:
        if not _RELPATH_PATTERN.fullmatch(relpath):
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"artifact path {relpath!r} is not a store path")
        return self.artifacts_dir / relpath

    # -- publishing -----------------------------------------------------------

    def _publish(self, relpath: str, write: Callable[[Path], None]) -> tuple[str, int]:
        """Write a file through ``tmp/`` and publish it at ``relpath``.

        Returns the file's SHA-256 hex digest and size. The fresh file always
        replaces any existing file at ``relpath`` (same content, by
        construction), so a published file is never older than the recovery
        grace period when its database commit runs, even if an earlier crashed
        operation left an identical orphan behind.
        """
        final = self._artifact_path(relpath)
        temp = self.tmp_dir / f"{uuid.uuid4().hex}.part"
        try:
            write(temp)
            _fsync_path(temp)
            os.chmod(temp, _READ_ONLY)
            digest, size = _file_sha256(temp), temp.stat().st_size
            os.replace(temp, final)
            _fsync_path(final.parent)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return digest, size

    def _publish_weights(self, relpath: str, tensors: Mapping[str, torch.Tensor], expected: str) -> tuple[str, int]:
        def write(path: Path) -> None:
            save_file({name: tensors[name] for name in sorted(tensors)}, str(path))
            if weights_hash(load_file(str(path))) != expected:
                raise WSEError(
                    ErrorCode.ARTIFACT_CORRUPT,
                    "written weights did not read back with the expected content hash",
                    retryable=True,
                )

        return self._publish(relpath, write)

    def _publish_bytes(self, relpath: str, data: bytes) -> tuple[str, int]:
        return self._publish(relpath, lambda path: path.write_bytes(data))

    def commit_checkpoint(
        self,
        manifest: ModelManifest,
        tensors: Mapping[str, torch.Tensor],
        *,
        dedup_key: str | None,
        event_type: str,
    ) -> ModelManifest:
        """Store ``tensors`` and ``manifest`` and make the checkpoint READY.

        When ``dedup_key`` matches an existing checkpoint, that checkpoint is
        returned and nothing is written. Identical weights are stored once.
        """
        if dedup_key is not None:
            existing = self._checkpoint_for_dedup_key(dedup_key)
            if existing is not None:
                return self.load_manifest(existing)

        weights_rel = f"{WEIGHTS_DIR}/{_hex(manifest.weights_hash)}.safetensors"
        row = self.db.conn.execute(
            "SELECT * FROM artifacts WHERE kind = 'weights' AND content_hash = ?",
            (manifest.weights_hash,),
        ).fetchone()
        if row is not None:
            self._verified_path(row)
            weights_sha, weights_size = row["file_sha256"], row["size_bytes"]
        else:
            weights_sha, weights_size = self._publish_weights(weights_rel, tensors, manifest.weights_hash)

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
                    },
                )
        if winner is not None:
            # A concurrent identical import won; this call's manifest file is an orphan.
            return self.load_manifest(winner)
        return manifest

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
            raise WSEError(
                ErrorCode.NOT_FOUND,
                f"checkpoint {checkpoint_id!r} does not exist",
                details={"checkpoint_id": str(checkpoint_id)},
            )
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

    def load_tensors(self, checkpoint_id: str) -> dict[str, torch.Tensor]:
        manifest = self.load_manifest(checkpoint_id)
        tensors = load_file(str(self.weights_path(checkpoint_id)))
        if weights_hash(tensors) != manifest.weights_hash:
            self._mark_corrupt([checkpoint_id], "weights content hash mismatch")
            raise WSEError(ErrorCode.ARTIFACT_CORRUPT, f"weights of {checkpoint_id} failed their content hash")
        return tensors

    def read_slice(self, checkpoint_id: str, tensor: str, slices: tuple[slice, ...]) -> torch.Tensor:
        """Read one region of a stored tensor without loading the whole file."""
        path = self.weights_path(checkpoint_id)
        with safe_open(str(path), framework="pt") as handle:
            region = handle.get_slice(tensor)[slices] if slices else handle.get_tensor(tensor)
        return region.clone(memory_format=torch.contiguous_format)

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
            for directory in (WEIGHTS_DIR, MANIFESTS_DIR):
                for path in sorted((self.artifacts_dir / directory).iterdir()):
                    relpath = f"{directory}/{path.name}"
                    if relpath in referenced or not path.is_file() or path.stat().st_mtime > cutoff:
                        continue
                    path.unlink(missing_ok=True)
                    report.removed_orphans.append(relpath)
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
        return report
