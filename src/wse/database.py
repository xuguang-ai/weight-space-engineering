"""SQLite schema, migrations, and transactions for managed metadata.

Connections run in autocommit mode; every multi-statement change goes through
:meth:`Database.transaction`, which opens ``BEGIN IMMEDIATE`` so concurrent
processes sharing the database serialize their writes instead of interleaving
a check and an update.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wse.schemas import ErrorCode, WSEError, canonical_json

# Each migration is a list of statements applied in one transaction. Never edit
# a released migration; append a new one.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            """
            CREATE TABLE artifacts (
                artifact_id   TEXT PRIMARY KEY,
                kind          TEXT NOT NULL CHECK (kind IN ('weights', 'manifest')),
                content_hash  TEXT NOT NULL,
                file_sha256   TEXT NOT NULL,
                relpath       TEXT NOT NULL UNIQUE,
                size_bytes    INTEGER NOT NULL CHECK (size_bytes >= 0),
                created_at    TEXT NOT NULL,
                UNIQUE (kind, content_hash)
            )
            """,
            """
            CREATE TABLE checkpoints (
                checkpoint_id         TEXT PRIMARY KEY,
                state                 TEXT NOT NULL CHECK (state IN ('READY', 'CORRUPT')),
                schema_version        TEXT NOT NULL,
                adapter_id            TEXT NOT NULL,
                adapter_version       TEXT NOT NULL,
                config_hash           TEXT NOT NULL,
                weights_hash          TEXT NOT NULL,
                weights_artifact_id   TEXT NOT NULL REFERENCES artifacts (artifact_id),
                manifest_artifact_id  TEXT NOT NULL UNIQUE REFERENCES artifacts (artifact_id),
                parent_checkpoint_id  TEXT REFERENCES checkpoints (checkpoint_id),
                dedup_key             TEXT UNIQUE,
                corrupt_reason        TEXT,
                created_at            TEXT NOT NULL
            )
            """,
            "CREATE INDEX checkpoints_state ON checkpoints (state, created_at)",
            """
            CREATE TABLE audit_events (
                event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                subject_id  TEXT,
                payload     TEXT NOT NULL
            )
            """,
        ],
    ),
]

LATEST_SCHEMA_VERSION = MIGRATIONS[-1][0]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class Database:
    """A connection to ``metadata.sqlite`` with migrations applied."""

    def __init__(self, path: Path, *, busy_timeout_ms: int = 10_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None, timeout=busy_timeout_ms / 1000
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run the body inside ``BEGIN IMMEDIATE`` ... ``COMMIT``; roll back on error."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def schema_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        with self.transaction() as conn:
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current > LATEST_SCHEMA_VERSION:
                raise WSEError(
                    ErrorCode.VERSION_CONFLICT,
                    f"database schema version {current} is newer than this wse supports "
                    f"({LATEST_SCHEMA_VERSION}); upgrade wse",
                )
            for version, statements in MIGRATIONS:
                if version <= current:
                    continue
                for statement in statements:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {int(version)}")


def insert_audit_event(
    conn: sqlite3.Connection, event_type: str, subject_id: str | None, payload: dict[str, Any]
) -> None:
    """Append one structured audit event inside the caller's transaction."""
    conn.execute(
        "INSERT INTO audit_events (created_at, event_type, subject_id, payload) VALUES (?, ?, ?, ?)",
        (utc_now(), event_type, subject_id, canonical_json(payload)),
    )


def audit_events(conn: sqlite3.Connection, subject_id: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT event_id, created_at, event_type, subject_id, payload FROM audit_events"
    args: tuple[Any, ...] = ()
    if subject_id is not None:
        query += " WHERE subject_id = ?"
        args = (subject_id,)
    rows = conn.execute(query + " ORDER BY event_id", args).fetchall()
    return [
        {
            "event_id": row["event_id"],
            "created_at": row["created_at"],
            "event_type": row["event_type"],
            "subject_id": row["subject_id"],
            "payload": json.loads(row["payload"]),
        }
        for row in rows
    ]
