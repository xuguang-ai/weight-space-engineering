"""Structured audit events.

Every state change of a managed experiment (creation, previews, candidate
commits, selections) and every rejected mutation is recorded as an event in
the same SQLite database. Events that change state are written in the same
transaction as the change; failures are written in their own transaction
after the failed operation has rolled back, so they remain traceable without
affecting any experiment state.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from wse.database import Database, audit_events, insert_audit_event
from wse.schemas import WSEError, canonical_json

__all__ = ["experiment_events", "record_event", "record_failure"]


def record_event(
    conn: sqlite3.Connection,
    event_type: str,
    subject_id: str | None,
    payload: dict[str, Any],
    *,
    experiment_id: str | None = None,
) -> None:
    """Append one event inside the caller's transaction."""
    insert_audit_event(conn, event_type, subject_id, payload, experiment_id=experiment_id)


def record_failure(
    db: Database,
    operation: str,
    error: WSEError,
    *,
    experiment_id: str | None,
    subject_id: str | None = None,
    request_hash: str | None = None,
) -> None:
    """Record a rejected mutation as an ``<operation>.failed`` event (own transaction)."""
    try:
        details = json.loads(canonical_json(error.details))
    except WSEError:
        details = {"unserializable_details": sorted(error.details)}
    with db.transaction() as conn:
        insert_audit_event(
            conn,
            f"{operation}.failed",
            subject_id,
            {
                "code": error.code.value,
                "message": error.message,
                "details": details,
                "request_hash": request_hash,
            },
            experiment_id=experiment_id,
        )


def experiment_events(db: Database, experiment_id: str) -> list[dict[str, Any]]:
    """All events of one experiment, oldest first."""
    return audit_events(db.conn, experiment_id=experiment_id)
