from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from .models import (
    Belief,
    BeliefCreate,
    BeliefStatus,
    Event,
    EventCreate,
    Generation,
    TrustEvent,
    TrustSource,
    utc_now,
)


DB_PATH = Path(
    "data/igl_memory.sqlite3"
)


def new_id(prefix: str) -> str:
    return (
        f"{prefix}_"
        f"{uuid.uuid4().hex}"
    )


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = (
        sqlite3.Row
    )

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,

                event_type TEXT NOT NULL,

                content_json TEXT NOT NULL,

                sensitivity TEXT NOT NULL,

                deleted_at TEXT
            );

            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,

                created_at TEXT NOT NULL,

                parent_id TEXT,

                reason TEXT NOT NULL,

                FOREIGN KEY(parent_id)
                    REFERENCES generations(id)
            );

            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,

                generation_id TEXT NOT NULL,

                belief_type TEXT NOT NULL,

                value_json TEXT NOT NULL,

                status TEXT NOT NULL,

                confidence REAL NOT NULL,
                uncertainty REAL,

                evidence_count INTEGER NOT NULL,

                applicability_json TEXT NOT NULL,

                created_at TEXT NOT NULL,

                last_validated_at TEXT,

                FOREIGN KEY(generation_id)
                    REFERENCES generations(id)
            );

            CREATE TABLE IF NOT EXISTS provenance (
                belief_id TEXT NOT NULL,
                event_id TEXT NOT NULL,

                relation TEXT NOT NULL,

                PRIMARY KEY(
                    belief_id,
                    event_id
                ),

                FOREIGN KEY(belief_id)
                    REFERENCES beliefs(id),

                FOREIGN KEY(event_id)
                    REFERENCES events(id)
            );

            CREATE TABLE IF NOT EXISTS trust_events (
                id TEXT PRIMARY KEY,

                belief_id TEXT NOT NULL,

                created_at TEXT NOT NULL,

                action TEXT NOT NULL,
                source TEXT NOT NULL,

                score_before REAL NOT NULL,
                score_after REAL NOT NULL,

                reason_json TEXT NOT NULL,

                FOREIGN KEY(belief_id)
                    REFERENCES beliefs(id)
            );

            CREATE INDEX IF NOT EXISTS
                idx_events_user
            ON events(user_id);

            CREATE INDEX IF NOT EXISTS
                idx_beliefs_user
            ON beliefs(user_id);

            CREATE INDEX IF NOT EXISTS
                idx_beliefs_status
            ON beliefs(status);

            CREATE INDEX IF NOT EXISTS
                idx_trust_belief
            ON trust_events(belief_id);
            """
        )


# ============================================================
# Events
# ============================================================

def create_event(
    data: EventCreate,
) -> Event:

    item = Event(
        id=new_id("m"),
        user_id=data.user_id,
        created_at=utc_now(),

        event_type=data.event_type,
        content=data.content,

        sensitivity=data.sensitivity,

        deleted_at=None,
    )

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO events (
                id,
                user_id,
                created_at,
                event_type,
                content_json,
                sensitivity,
                deleted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.user_id,
                item.created_at,
                item.event_type,
                json.dumps(
                    item.content,
                    ensure_ascii=False,
                ),
                item.sensitivity,
                None,
            ),
        )

    return item


def get_event(
    event_id: str,
) -> Event | None:

    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM events
            WHERE id = ?
            """,
            (event_id,),
        ).fetchone()

    if row is None:
        return None

    return Event(
        id=row["id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        event_type=row["event_type"],
        content=json.loads(
            row["content_json"]
        ),
        sensitivity=row["sensitivity"],
        deleted_at=row["deleted_at"],
    )


# ============================================================
# Generations
# ============================================================

def latest_generation(
    user_id: str,
) -> Generation | None:

    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM generations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    return Generation(
        id=row["id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        parent_id=row["parent_id"],
        reason=row["reason"],
    )


def create_generation(
    user_id: str,
    reason: str,
    parent_id: str | None = None,
) -> Generation:

    generation = Generation(
        id=new_id("g"),
        user_id=user_id,
        created_at=utc_now(),
        parent_id=parent_id,
        reason=reason,
    )

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO generations (
                id,
                user_id,
                created_at,
                parent_id,
                reason
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                generation.id,
                generation.user_id,
                generation.created_at,
                generation.parent_id,
                generation.reason,
            ),
        )

    return generation


def ensure_generation(
    user_id: str,
) -> Generation:

    generation = latest_generation(
        user_id
    )

    if generation is not None:
        return generation

    return create_generation(
        user_id=user_id,
        reason="initial",
    )


# ============================================================
# Beliefs
# ============================================================

def _belief_from_row(
    row: sqlite3.Row,
) -> Belief:

    return Belief(
        id=row["id"],
        user_id=row["user_id"],

        generation_id=row[
            "generation_id"
        ],

        belief_type=row[
            "belief_type"
        ],

        value=json.loads(
            row["value_json"]
        ),

        status=BeliefStatus(
            row["status"]
        ),

        confidence=float(
            row["confidence"]
        ),

        uncertainty=(
            None
            if row["uncertainty"]
            is None
            else float(
                row["uncertainty"]
            )
        ),

        evidence_count=int(
            row["evidence_count"]
        ),

        applicability=json.loads(
            row[
                "applicability_json"
            ]
        ),

        created_at=row[
            "created_at"
        ],

        last_validated_at=row[
            "last_validated_at"
        ],
    )


def create_belief(
    data: BeliefCreate,
) -> Belief:

    generation = ensure_generation(
        data.user_id
    )

    # Verify provenance before creating D.
    for event_id in data.source_event_ids:
        event = get_event(
            event_id
        )

        if event is None:
            raise ValueError(
                f"Unknown event: "
                f"{event_id}"
            )

        if event.user_id != data.user_id:
            raise ValueError(
                "Cross-user provenance "
                "is not allowed"
            )

    belief = Belief(
        id=new_id("d"),

        user_id=data.user_id,

        generation_id=(
            generation.id
        ),

        belief_type=(
            data.belief_type
        ),

        value=data.value,

        status=data.status,

        confidence=data.confidence,

        uncertainty=data.uncertainty,

        evidence_count=len(
            data.source_event_ids
        ),

        applicability=(
            data.applicability
        ),

        created_at=utc_now(),

        last_validated_at=None,
    )

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO beliefs (
                id,
                user_id,
                generation_id,
                belief_type,
                value_json,
                status,
                confidence,
                uncertainty,
                evidence_count,
                applicability_json,
                created_at,
                last_validated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                belief.id,
                belief.user_id,
                belief.generation_id,
                belief.belief_type,

                json.dumps(
                    belief.value,
                    ensure_ascii=False,
                ),

                belief.status.value,

                belief.confidence,
                belief.uncertainty,

                belief.evidence_count,

                json.dumps(
                    belief.applicability,
                    ensure_ascii=False,
                ),

                belief.created_at,
                belief.last_validated_at,
            ),
        )

        for event_id in (
            data.source_event_ids
        ):
            conn.execute(
                """
                INSERT INTO provenance (
                    belief_id,
                    event_id,
                    relation
                )
                VALUES (?, ?, ?)
                """,
                (
                    belief.id,
                    event_id,
                    "SUPPORTS",
                ),
            )

    return belief


def get_belief(
    belief_id: str,
) -> Belief | None:

    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM beliefs
            WHERE id = ?
            """,
            (belief_id,),
        ).fetchone()

    if row is None:
        return None

    return _belief_from_row(
        row
    )


def list_beliefs(
    user_id: str,
) -> list[Belief]:

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM beliefs
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

    return [
        _belief_from_row(row)
        for row in rows
    ]


def get_provenance(
    belief_id: str,
) -> list[Event]:

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT e.*
            FROM provenance p
            JOIN events e
              ON e.id = p.event_id
            WHERE p.belief_id = ?
            ORDER BY e.created_at ASC
            """,
            (belief_id,),
        ).fetchall()

    return [
        Event(
            id=row["id"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            event_type=row["event_type"],
            content=json.loads(
                row["content_json"]
            ),
            sensitivity=row[
                "sensitivity"
            ],
            deleted_at=row[
                "deleted_at"
            ],
        )
        for row in rows
    ]


# ============================================================
# Mutable derived state
#
# M remains append-only.
# D trust/status may change, while trust_events
# preserve the lineage of those changes.
# ============================================================

def update_belief_trust(
    belief_id: str,

    status: BeliefStatus,

    confidence: float,

    action: str,

    source: TrustSource,

    reason: dict,
) -> Belief:

    belief = get_belief(
        belief_id
    )

    if belief is None:
        raise ValueError(
            "Belief not found"
        )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    trust_event = TrustEvent(
        id=new_id("t"),

        belief_id=belief_id,

        created_at=utc_now(),

        action=action,
        source=source,

        score_before=(
            belief.confidence
        ),

        score_after=confidence,

        reason=reason,
    )

    validated_at = (
        utc_now()
        if action == "VALIDATE"
        else belief.last_validated_at
    )

    with connect() as conn:
        conn.execute(
            """
            UPDATE beliefs
            SET status = ?,
                confidence = ?,
                last_validated_at = ?
            WHERE id = ?
            """,
            (
                status.value,
                confidence,
                validated_at,
                belief_id,
            ),
        )

        conn.execute(
            """
            INSERT INTO trust_events (
                id,
                belief_id,
                created_at,
                action,
                source,
                score_before,
                score_after,
                reason_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trust_event.id,
                trust_event.belief_id,
                trust_event.created_at,
                trust_event.action,
                trust_event.source.value,
                trust_event.score_before,
                trust_event.score_after,

                json.dumps(
                    trust_event.reason,
                    ensure_ascii=False,
                ),
            ),
        )

    updated = get_belief(
        belief_id
    )

    assert updated is not None

    return updated