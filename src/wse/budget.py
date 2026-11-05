"""Atomic budget reservations and settlement (Section 6.2 of the plan).

Every experiment has one ledger row per counter with ``limit_value``,
``consumed``, and ``reserved``. The scheduler, worker, and runner are separate
processes sharing the database, so allowance is never checked in one statement
and taken in another:

* ``reserve`` runs inside ``BEGIN IMMEDIATE`` and takes each counter with a
  conditional ``UPDATE ... WHERE consumed + reserved + amount <= limit_value``.
  If any counter lacks allowance the whole reservation rolls back, so
  concurrent requests cannot overallocate.
* ``settle`` charges the work actually executed (never more than was reserved)
  and releases the rest, in one transaction. Settling twice is a no-op.
* ``charge`` consumes allowance immediately for synchronous operations (reads,
  analyses, commits) with the same conditional ``UPDATE``.
* ``charge_compute`` records the elapsed seconds of a synchronous operation,
  capped at the unreserved remainder so the ledger can never exceed its limit;
  ``require`` rejects an operation up front when no compute time remains.

Consumed allowance is never refunded: cancelling or rolling back changes
nothing that was charged.
"""

from __future__ import annotations

import json
import math
import secrets
import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from wse.audit import record_event
from wse.database import utc_now
from wse.policy import LEDGER_COUNTERS
from wse.schemas import (
    OBJECT_ID_PATTERNS,
    TERMINAL_JOB_STATES,
    ErrorCode,
    Reservation,
    Usage,
    WSEError,
    canonical_json,
)

if TYPE_CHECKING:
    from wse.database import Database

UsageLike = Mapping[str, Any] | Usage


def _number(value: Any) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def usage_amounts(usage: UsageLike, *, allow_empty: bool = True) -> dict[str, int | float]:
    """Ledger counters and amounts from a ``Usage`` or mapping; ``None`` entries are skipped."""
    if isinstance(usage, Usage):
        raw = {name: getattr(usage, name) for name in LEDGER_COUNTERS}
    elif isinstance(usage, Mapping):
        raw = dict(usage)
    else:
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "usage must be a mapping of ledger counters to amounts")
    amounts: dict[str, int | float] = {}
    for counter, amount in raw.items():
        if amount is None:
            continue
        if counter not in LEDGER_COUNTERS:
            raise WSEError(
                ErrorCode.INVALID_ARGUMENT,
                f"{counter!r} is not a ledger counter; counters are {list(LEDGER_COUNTERS)}",
                details={"counter": str(counter)},
            )
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) \
                or amount < 0:
            raise WSEError(ErrorCode.INVALID_ARGUMENT, f"usage amount of {counter} must be a finite number >= 0",
                           details={"counter": counter})
        amounts[counter] = _number(amount)
    if not allow_empty and not any(amounts.values()):
        raise WSEError(ErrorCode.INVALID_ARGUMENT, "a reservation must request a positive amount of some counter")
    return amounts


def budget_error(conn: sqlite3.Connection, experiment_id: str, counter: str, amount: int | float) -> WSEError:
    row = conn.execute(
        "SELECT limit_value, consumed, reserved FROM budget_ledger WHERE experiment_id = ? AND counter = ?",
        (experiment_id, counter),
    ).fetchone()
    limit, consumed, reserved = (
        (_number(row["limit_value"]), _number(row["consumed"]), _number(row["reserved"])) if row else (0, 0, 0)
    )
    held = f", {reserved} reserved by pending jobs" if reserved else ""
    return WSEError(
        ErrorCode.BUDGET_EXCEEDED,
        f"the experiment's {counter} budget is exhausted ({consumed} of {limit} used{held}; {amount} requested); "
        "select among existing checkpoints or stop",
        details={"counter": counter, "limit": limit, "consumed": consumed, "requested": amount,
                 **({"reserved": reserved} if reserved else {})},
    )


class BudgetLedger:
    """Reservations and charges on one workspace database."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # -- reading --------------------------------------------------------------

    @staticmethod
    def remaining(conn: sqlite3.Connection, experiment_id: str, counter: str) -> int | float:
        row = conn.execute(
            "SELECT limit_value - consumed - reserved AS remaining FROM budget_ledger "
            "WHERE experiment_id = ? AND counter = ?",
            (experiment_id, counter),
        ).fetchone()
        return max(_number(row["remaining"]), 0) if row else 0

    def get(self, reservation_id: Any, conn: sqlite3.Connection | None = None) -> Reservation:
        row = None
        if isinstance(reservation_id, str) and OBJECT_ID_PATTERNS["reservation"].fullmatch(reservation_id):
            row = (conn or self.db.conn).execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
        if row is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"reservation {reservation_id!r} does not exist",
                           details={"reservation_id": str(reservation_id)})
        return self._reservation(row)

    @staticmethod
    def _reservation(row: sqlite3.Row) -> Reservation:
        return Reservation(
            reservation_id=row["reservation_id"],
            experiment_id=row["experiment_id"],
            job_id=row["job_id"],
            state=row["state"],
            requested=json.loads(row["requested_json"]),
            charged=json.loads(row["charged_json"]) if row["charged_json"] else None,
            terminal_state=row["terminal_state"],
            created_at=row["created_at"],
            settled_at=row["settled_at"],
        )

    # -- synchronous charges --------------------------------------------------

    def require(self, conn: sqlite3.Connection, experiment_id: str, amounts: Mapping[str, int | float]) -> None:
        """Fail with ``BUDGET_EXCEEDED`` unless every counter has the amount left.

        A ``compute_seconds`` amount of 0 means "some time must remain".
        This is an early check; the charge itself is the authoritative one.
        """
        for counter, amount in amounts.items():
            left = self.remaining(conn, experiment_id, counter)
            if left < amount or (counter == "compute_seconds" and left <= 0):
                raise budget_error(conn, experiment_id, counter, amount)

    @staticmethod
    def charge(conn: sqlite3.Connection, experiment_id: str, counter: str, amount: int | float) -> None:
        """Consume ``amount`` now; one conditional UPDATE that cannot exceed the limit."""
        if amount <= 0:
            return
        updated = conn.execute(
            "UPDATE budget_ledger SET consumed = consumed + ? "
            "WHERE experiment_id = ? AND counter = ? AND consumed + reserved + ? <= limit_value",
            (amount, experiment_id, counter, amount),
        ).rowcount
        if not updated:
            raise budget_error(conn, experiment_id, counter, amount)

    @staticmethod
    def charge_compute(conn: sqlite3.Connection, experiment_id: str, seconds: float) -> float:
        """Record ``seconds`` of synchronous computation, capped at the unreserved remainder."""
        if seconds <= 0:
            return 0.0
        row = conn.execute(
            "SELECT consumed FROM budget_ledger WHERE experiment_id = ? AND counter = 'compute_seconds'",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return 0.0
        conn.execute(
            "UPDATE budget_ledger SET consumed = MAX(consumed, MIN(limit_value - reserved, consumed + ?)) "
            "WHERE experiment_id = ? AND counter = 'compute_seconds'",
            (seconds, experiment_id),
        )
        after = conn.execute(
            "SELECT consumed FROM budget_ledger WHERE experiment_id = ? AND counter = 'compute_seconds'",
            (experiment_id,),
        ).fetchone()
        return float(after["consumed"]) - float(row["consumed"])

    # -- reservations ---------------------------------------------------------

    def reserve(
        self,
        experiment_id: str,
        job_id: str,
        requested_usage: UsageLike,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> Reservation:
        """Atomically reserve ``requested_usage`` for ``job_id`` in an OPEN experiment.

        All counters are reserved or none is (``BUDGET_EXCEEDED``). Reserving
        again for the same job with the same request returns the existing
        reservation; a different request is a ``VERSION_CONFLICT``. Runs in the
        caller's transaction when ``conn`` is given, else in its own
        ``BEGIN IMMEDIATE``.
        """
        amounts = usage_amounts(requested_usage, allow_empty=False)
        if not isinstance(job_id, str) or not OBJECT_ID_PATTERNS["job"].fullmatch(job_id):
            raise WSEError(ErrorCode.INVALID_ARGUMENT, "job_id must match job_<32 hex digits>")
        if conn is None:
            with self.db.transaction() as own:
                return self._reserve(own, experiment_id, job_id, amounts)
        return self._reserve(conn, experiment_id, job_id, amounts)

    def _reserve(self, conn: sqlite3.Connection, experiment_id: str, job_id: str,
                 amounts: dict[str, int | float]) -> Reservation:
        experiment = None
        if isinstance(experiment_id, str) and OBJECT_ID_PATTERNS["experiment"].fullmatch(experiment_id):
            experiment = conn.execute("SELECT state FROM experiments WHERE experiment_id = ?",
                                      (experiment_id,)).fetchone()
        if experiment is None:
            raise WSEError(ErrorCode.NOT_FOUND, f"experiment {experiment_id!r} does not exist",
                           details={"experiment_id": str(experiment_id)})
        existing = conn.execute("SELECT * FROM reservations WHERE job_id = ?", (job_id,)).fetchone()
        if existing is not None:
            if existing["experiment_id"] == experiment_id and json.loads(existing["requested_json"]) == amounts:
                return self._reservation(existing)
            raise WSEError(ErrorCode.VERSION_CONFLICT, f"job {job_id} already holds a different reservation",
                           details={"job_id": job_id, "reservation_id": existing["reservation_id"]})
        if experiment["state"] != "OPEN":
            raise WSEError(
                ErrorCode.VERSION_CONFLICT,
                f"experiment {experiment_id} is {experiment['state']}; only OPEN experiments accept new work",
                details={"experiment_id": experiment_id, "state": experiment["state"]},
            )
        for counter in LEDGER_COUNTERS:
            amount = amounts.get(counter, 0)
            if amount <= 0:
                continue
            updated = conn.execute(
                "UPDATE budget_ledger SET reserved = reserved + ? "
                "WHERE experiment_id = ? AND counter = ? AND consumed + reserved + ? <= limit_value",
                (amount, experiment_id, counter, amount),
            ).rowcount
            if not updated:
                raise budget_error(conn, experiment_id, counter, amount)
        reservation = Reservation(
            reservation_id=f"resv_{secrets.token_hex(16)}",
            experiment_id=experiment_id,
            job_id=job_id,
            state="HELD",
            requested={counter: amounts[counter] for counter in LEDGER_COUNTERS if amounts.get(counter, 0) > 0},
            created_at=utc_now(),
        )
        conn.execute(
            "INSERT INTO reservations (reservation_id, experiment_id, job_id, state, requested_json, created_at) "
            "VALUES (?, ?, ?, 'HELD', ?, ?)",
            (reservation.reservation_id, experiment_id, job_id, canonical_json(reservation.requested),
             reservation.created_at),
        )
        record_event(conn, "budget.reserved", reservation.reservation_id,
                     {"job_id": job_id, "requested": reservation.requested}, experiment_id=experiment_id)
        return reservation

    def settle(
        self,
        reservation_id: str,
        actual_usage: UsageLike,
        terminal_state: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> Reservation:
        """Charge the executed part of a reservation and release the rest.

        Each counter is charged ``min(actual, reserved)``. A reservation that
        is already settled is returned unchanged, so settlement never charges
        twice.
        """
        if terminal_state not in TERMINAL_JOB_STATES:
            raise WSEError(ErrorCode.INVALID_ARGUMENT,
                           f"terminal_state must be one of {list(TERMINAL_JOB_STATES)}")
        actual = usage_amounts(actual_usage)
        if conn is None:
            with self.db.transaction() as own:
                return self._settle(own, reservation_id, actual, terminal_state)
        return self._settle(conn, reservation_id, actual, terminal_state)

    def _settle(self, conn: sqlite3.Connection, reservation_id: str, actual: dict[str, int | float],
                terminal_state: str) -> Reservation:
        reservation = self.get(reservation_id, conn)
        if reservation.state == "SETTLED":
            return reservation
        charged = {counter: _number(min(actual.get(counter, 0), amount))
                   for counter, amount in reservation.requested.items()}
        for counter, amount in reservation.requested.items():
            conn.execute(
                "UPDATE budget_ledger SET reserved = MAX(0, reserved - ?), "
                "consumed = MIN(limit_value, consumed + ?) WHERE experiment_id = ? AND counter = ?",
                (amount, charged[counter], reservation.experiment_id, counter),
            )
        settled_at = utc_now()
        conn.execute(
            "UPDATE reservations SET state = 'SETTLED', charged_json = ?, terminal_state = ?, settled_at = ? "
            "WHERE reservation_id = ? AND state = 'HELD'",
            (canonical_json(charged), terminal_state, settled_at, reservation_id),
        )
        record_event(conn, "budget.settled", reservation_id,
                     {"job_id": reservation.job_id, "terminal_state": terminal_state, "charged": charged,
                      "released": {c: _number(a - charged[c]) for c, a in reservation.requested.items()}},
                     experiment_id=reservation.experiment_id)
        return reservation.model_copy(update={"state": "SETTLED", "charged": charged,
                                              "terminal_state": terminal_state, "settled_at": settled_at})
