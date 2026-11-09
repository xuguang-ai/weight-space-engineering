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
    (
        2,
        [
            # Immutable experiment policies; policy_json is canonical JSON.
            """
            CREATE TABLE policies (
                policy_id    TEXT PRIMARY KEY,
                mode         TEXT NOT NULL,
                policy_hash  TEXT NOT NULL,
                policy_json  TEXT NOT NULL,
                created_at   TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE run_grants (
                run_id              TEXT PRIMARY KEY,
                task_id             TEXT NOT NULL,
                policy_id           TEXT NOT NULL REFERENCES policies (policy_id),
                base_checkpoint_id  TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                created_at          TEXT NOT NULL
            )
            """,
            # UNIQUE (run_id): at most one experiment per run grant.
            """
            CREATE TABLE experiments (
                experiment_id           TEXT PRIMARY KEY,
                run_id                  TEXT NOT NULL UNIQUE REFERENCES run_grants (run_id),
                task_id                 TEXT NOT NULL,
                policy_id               TEXT NOT NULL REFERENCES policies (policy_id),
                policy_hash             TEXT NOT NULL,
                mode                    TEXT NOT NULL,
                base_checkpoint_id      TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                selected_checkpoint_id  TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                state                   TEXT NOT NULL CHECK (state IN ('OPEN', 'FROZEN', 'CLOSED')),
                created_at              TEXT NOT NULL,
                ended_at                TEXT
            )
            """,
            # Consumption can never exceed the limit, so a conditional UPDATE that
            # would overdraw fails instead of overallocating.
            """
            CREATE TABLE budget_ledger (
                experiment_id  TEXT NOT NULL REFERENCES experiments (experiment_id),
                counter        TEXT NOT NULL,
                limit_value    NUMERIC NOT NULL CHECK (limit_value >= 0),
                consumed       NUMERIC NOT NULL DEFAULT 0 CHECK (consumed >= 0 AND consumed <= limit_value),
                PRIMARY KEY (experiment_id, counter)
            )
            """,
            """
            CREATE TABLE patches (
                patch_id               TEXT PRIMARY KEY,
                experiment_id          TEXT NOT NULL REFERENCES experiments (experiment_id),
                state                  TEXT NOT NULL CHECK (state IN ('READY', 'CORRUPT')),
                spec_hash              TEXT NOT NULL,
                base_checkpoint_id     TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                result_weights_hash    TEXT NOT NULL,
                relpath                TEXT NOT NULL,
                file_sha256            TEXT NOT NULL,
                size_bytes             INTEGER NOT NULL,
                preview_json           TEXT NOT NULL,
                applied_checkpoint_id  TEXT REFERENCES checkpoints (checkpoint_id),
                corrupt_reason         TEXT,
                created_at             TEXT NOT NULL
            )
            """,
            "CREATE INDEX patches_experiment ON patches (experiment_id, created_at)",
            """
            CREATE TABLE experiment_checkpoints (
                experiment_id  TEXT NOT NULL REFERENCES experiments (experiment_id),
                checkpoint_id  TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                role           TEXT NOT NULL CHECK (role IN ('base', 'candidate')),
                patch_id       TEXT REFERENCES patches (patch_id),
                created_at     TEXT NOT NULL,
                PRIMARY KEY (experiment_id, checkpoint_id)
            )
            """,
            # One record per (scope, key); the request hash detects key reuse
            # with different input.
            """
            CREATE TABLE idempotency_records (
                scope            TEXT NOT NULL,
                idempotency_key  TEXT NOT NULL,
                operation        TEXT NOT NULL,
                request_hash     TEXT NOT NULL,
                result_json      TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                PRIMARY KEY (scope, idempotency_key)
            )
            """,
            "ALTER TABLE audit_events ADD COLUMN experiment_id TEXT",
            "CREATE INDEX audit_events_experiment ON audit_events (experiment_id, event_id)",
        ],
    ),
    (
        3,
        [
            # Allowance held by unsettled reservations. Every reservation is a
            # conditional UPDATE (consumed + reserved + amount <= limit_value)
            # inside BEGIN IMMEDIATE, so concurrent requests cannot overallocate.
            "ALTER TABLE budget_ledger ADD COLUMN reserved NUMERIC NOT NULL DEFAULT 0 CHECK (reserved >= 0)",
            """
            CREATE TABLE suites (
                suite_id     TEXT PRIMARY KEY,
                task_id      TEXT NOT NULL,
                split        TEXT NOT NULL CHECK (split IN ('development', 'final')),
                suite_hash   TEXT NOT NULL,
                suite_json   TEXT NOT NULL,
                created_at   TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE reservations (
                reservation_id  TEXT PRIMARY KEY,
                experiment_id   TEXT NOT NULL REFERENCES experiments (experiment_id),
                job_id          TEXT NOT NULL UNIQUE,
                state           TEXT NOT NULL CHECK (state IN ('HELD', 'SETTLED')),
                requested_json  TEXT NOT NULL,
                charged_json    TEXT,
                terminal_state  TEXT CHECK (terminal_state IN ('succeeded', 'failed', 'cancelled')),
                created_at      TEXT NOT NULL,
                settled_at      TEXT
            )
            """,
            "CREATE INDEX reservations_experiment ON reservations (experiment_id, state)",
            # lease_expires_at is a Unix timestamp shared by every process.
            """
            CREATE TABLE jobs (
                job_id               TEXT PRIMARY KEY,
                experiment_id        TEXT NOT NULL REFERENCES experiments (experiment_id),
                kind                 TEXT NOT NULL CHECK (kind IN ('evaluation')),
                checkpoint_id        TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                suite_id             TEXT NOT NULL REFERENCES suites (suite_id),
                config_json          TEXT NOT NULL,
                state                TEXT NOT NULL
                                     CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
                reservation_id       TEXT NOT NULL UNIQUE REFERENCES reservations (reservation_id),
                cancel_requested_at  TEXT,
                progress_json        TEXT NOT NULL,
                usage_json           TEXT,
                result_json          TEXT,
                error_json           TEXT,
                lease_owner          TEXT,
                lease_expires_at     REAL,
                heartbeat_at         TEXT,
                attempts             INTEGER NOT NULL DEFAULT 0,
                created_at           TEXT NOT NULL,
                started_at           TEXT,
                finished_at          TEXT
            )
            """,
            "CREATE INDEX jobs_state ON jobs (state, created_at)",
            "CREATE INDEX jobs_experiment ON jobs (experiment_id, created_at)",
        ],
    ),
    (
        4,
        [
            # The process ID of the local worker holding a job's lease. A restarted
            # engine settles a job whose worker process no longer exists at once,
            # without waiting for the lease to expire.
            "ALTER TABLE jobs ADD COLUMN lease_pid INTEGER",
        ],
    ),
    (
        5,
        [
            # Immutable task definitions (task.json); task_json is canonical JSON.
            """
            CREATE TABLE tasks (
                task_id     TEXT PRIMARY KEY,
                task_hash   TEXT NOT NULL,
                task_json   TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
            """,
            # Evaluation reports. Development reports are written with the job
            # that produced them; final reports with the final evaluation.
            """
            CREATE TABLE reports (
                report_id      TEXT PRIMARY KEY,
                experiment_id  TEXT NOT NULL REFERENCES experiments (experiment_id),
                job_id         TEXT UNIQUE REFERENCES jobs (job_id),
                checkpoint_id  TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                suite_id       TEXT NOT NULL REFERENCES suites (suite_id),
                split          TEXT NOT NULL CHECK (split IN ('development', 'final')),
                report_hash    TEXT NOT NULL,
                report_json    TEXT NOT NULL,
                created_at     TEXT NOT NULL
            )
            """,
            "CREATE INDEX reports_experiment ON reports (experiment_id, created_at)",
            # One sealed final evaluation per experiment, written when it closes.
            """
            CREATE TABLE final_evaluations (
                experiment_id        TEXT PRIMARY KEY REFERENCES experiments (experiment_id),
                candidate_report_id  TEXT NOT NULL REFERENCES reports (report_id),
                control_report_id    TEXT NOT NULL REFERENCES reports (report_id),
                final_hash           TEXT NOT NULL,
                final_json           TEXT NOT NULL,
                created_at           TEXT NOT NULL
            )
            """,
            "ALTER TABLE jobs ADD COLUMN report_id TEXT REFERENCES reports (report_id)",
            "ALTER TABLE experiments ADD COLUMN frozen_at TEXT",
        ],
    ),
    (
        6,
        [
            # Previews are a ledger counter. Experiments created before it existed
            # get the default allowance of 50 (raised to their preview count when
            # larger), with their stored previews counted as consumed.
            """
            INSERT INTO budget_ledger (experiment_id, counter, limit_value, consumed)
            SELECT e.experiment_id, 'patch_previews',
                   MAX(50, (SELECT COUNT(*) FROM patches p WHERE p.experiment_id = e.experiment_id)),
                   (SELECT COUNT(*) FROM patches p WHERE p.experiment_id = e.experiment_id)
            FROM experiments e
            WHERE NOT EXISTS (SELECT 1 FROM budget_ledger b
                              WHERE b.experiment_id = e.experiment_id AND b.counter = 'patch_previews')
            """,
            # Which experiment produced a stored large analysis result. The files
            # are content-addressed, so one URI can belong to several experiments;
            # a run may read a URI only when its own experiment produced it.
            """
            CREATE TABLE analysis_results (
                uri            TEXT NOT NULL,
                experiment_id  TEXT NOT NULL REFERENCES experiments (experiment_id),
                created_at     TEXT NOT NULL,
                PRIMARY KEY (uri, experiment_id)
            )
            """,
        ],
    ),
    (
        7,
        [
            # One controller per experiment: the strategy or agent adapter that
            # drove it, its exact version and configuration, the tracing setting,
            # the constraint status, and (once it ends) its termination record.
            """
            CREATE TABLE controller_runs (
                experiment_id      TEXT PRIMARY KEY REFERENCES experiments (experiment_id),
                kind               TEXT NOT NULL CHECK (kind IN ('strategy', 'agent')),
                controller_id      TEXT NOT NULL,
                controller_version TEXT NOT NULL,
                config             TEXT NOT NULL,
                config_hash        TEXT NOT NULL,
                constraint_status  TEXT NOT NULL CHECK (constraint_status IN ('verified', 'unverified')),
                tracing            TEXT NOT NULL,
                started_at         TEXT NOT NULL,
                ended_at           TEXT,
                result             TEXT
            )
            """,
            # The controller's normalized trace: tool calls and results, model
            # responses with usage, stop conditions, and cancellations.
            """
            CREATE TABLE controller_events (
                experiment_id  TEXT NOT NULL REFERENCES experiments (experiment_id),
                seq            INTEGER NOT NULL,
                created_at     TEXT NOT NULL,
                kind           TEXT NOT NULL,
                payload        TEXT NOT NULL,
                PRIMARY KEY (experiment_id, seq)
            )
            """,
        ],
    ),
    (
        8,
        [
            # Per-tensor content-addressed blobs. blob_id is the SHA-256 hex of
            # the tensor's raw little-endian row-major bytes in its stored dtype,
            # which is also the blob file's SHA-256. Shape and dtype live in
            # weight_tensors, so tensors with identical bytes share one blob.
            """
            CREATE TABLE blobs (
                blob_id     TEXT PRIMARY KEY,
                relpath     TEXT NOT NULL UNIQUE,
                size_bytes  INTEGER NOT NULL CHECK (size_bytes >= 0),
                created_at  TEXT NOT NULL
            )
            """,
            # The blob manifest of each weights artifact: canonical tensor name to
            # blob, dtype, and shape (canonical JSON list). A blob's reference count
            # is its number of rows here. Existing M1 weights artifacts are indexed
            # in place by ArtifactStore when it opens (the data step of this
            # migration); their artifacts, manifests, and checkpoints are unchanged.
            """
            CREATE TABLE weight_tensors (
                weights_artifact_id  TEXT NOT NULL REFERENCES artifacts (artifact_id),
                name                 TEXT NOT NULL,
                blob_id              TEXT NOT NULL REFERENCES blobs (blob_id),
                dtype                TEXT NOT NULL,
                shape                TEXT NOT NULL,
                PRIMARY KEY (weights_artifact_id, name)
            )
            """,
            "CREATE INDEX weight_tensors_blob ON weight_tensors (blob_id)",
        ],
    ),
    (
        9,
        [
            # External patches imported as exact WSE patches (M2 plan Section 3.4), with the
            # candidate each produced. One row per (base, source files); the binary patch
            # file at relpath is referenced here, so recovery keeps it.
            """
            CREATE TABLE imported_patches (
                patch_id              TEXT PRIMARY KEY,
                format                TEXT NOT NULL,
                dedup_key             TEXT NOT NULL UNIQUE,
                base_checkpoint_id    TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                checkpoint_id         TEXT NOT NULL REFERENCES checkpoints (checkpoint_id),
                result_weights_hash   TEXT NOT NULL,
                relpath               TEXT NOT NULL,
                file_sha256           TEXT NOT NULL,
                size_bytes            INTEGER NOT NULL,
                record_json           TEXT NOT NULL,
                created_at            TEXT NOT NULL
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
        # A connection may be handed to another thread (a worker built in one
        # thread and run in another) but is never used by two threads at once;
        # concurrent threads open their own Database.
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None, timeout=busy_timeout_ms / 1000, check_same_thread=False
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
    conn: sqlite3.Connection,
    event_type: str,
    subject_id: str | None,
    payload: dict[str, Any],
    *,
    experiment_id: str | None = None,
) -> None:
    """Append one structured audit event inside the caller's transaction."""
    conn.execute(
        "INSERT INTO audit_events (created_at, event_type, subject_id, payload, experiment_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (utc_now(), event_type, subject_id, canonical_json(payload), experiment_id),
    )


def audit_events(
    conn: sqlite3.Connection, subject_id: str | None = None, *, experiment_id: str | None = None
) -> list[dict[str, Any]]:
    query = "SELECT event_id, created_at, event_type, subject_id, experiment_id, payload FROM audit_events"
    clauses, args = [], []
    if subject_id is not None:
        clauses.append("subject_id = ?")
        args.append(subject_id)
    if experiment_id is not None:
        clauses.append("experiment_id = ?")
        args.append(experiment_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(query + " ORDER BY event_id", tuple(args)).fetchall()
    return [
        {
            "event_id": row["event_id"],
            "created_at": row["created_at"],
            "event_type": row["event_type"],
            "subject_id": row["subject_id"],
            "experiment_id": row["experiment_id"],
            "payload": json.loads(row["payload"]),
        }
        for row in rows
    ]
