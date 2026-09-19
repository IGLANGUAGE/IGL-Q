from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


class BeliefStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    SUSPECT = "SUSPECT"
    INVALID = "INVALID"
    USER_REJECTED = "USER_REJECTED"


class TrustSource(str, Enum):
    SYSTEM = "SYSTEM"
    EVIDENCE = "EVIDENCE"
    USER = "USER"
    ADMIN = "ADMIN"


# ============================================================
# Events = M
# ============================================================

class EventCreate(BaseModel):
    user_id: str
    event_type: str

    content: dict[str, Any]

    sensitivity: str = "private"


class Event(BaseModel):
    id: str
    user_id: str

    created_at: str

    event_type: str
    content: dict[str, Any]

    sensitivity: str

    deleted_at: str | None = None


# ============================================================
# Beliefs = D
# ============================================================

class BeliefCreate(BaseModel):
    user_id: str

    belief_type: str

    value: dict[str, Any]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    uncertainty: float | None = Field(
        default=None,
        ge=0.0,
    )

    applicability: dict[str, Any] = (
        Field(default_factory=dict)
    )

    source_event_ids: list[str] = (
        Field(default_factory=list)
    )

    status: BeliefStatus = (
        BeliefStatus.CANDIDATE
    )


class Belief(BaseModel):
    id: str
    user_id: str

    generation_id: str

    belief_type: str
    value: dict[str, Any]

    status: BeliefStatus

    confidence: float
    uncertainty: float | None

    evidence_count: int

    applicability: dict[str, Any]

    created_at: str
    last_validated_at: str | None


# ============================================================
# Trust
# ============================================================

class TrustAction(BaseModel):
    action: str

    source: TrustSource

    reason: dict[str, Any] = Field(
        default_factory=dict
    )

    confidence_delta: float = 0.0


class TrustEvent(BaseModel):
    id: str

    belief_id: str

    created_at: str

    action: str
    source: TrustSource

    score_before: float
    score_after: float

    reason: dict[str, Any]


# ============================================================
# Gate
# ============================================================

class GateQuery(BaseModel):
    user_id: str

    purpose: str

    context: dict[str, Any] = Field(
        default_factory=dict
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )


class GateAccepted(BaseModel):
    belief: Belief

    trusted: bool
    applicable: bool

    reason: str


class GateWithheld(BaseModel):
    belief_id: str

    reason: str


class GateResult(BaseModel):
    accepted: list[GateAccepted]

    withheld: list[GateWithheld]


# ============================================================
# Generation
# ============================================================

class Generation(BaseModel):
    id: str
    user_id: str

    created_at: str

    parent_id: str | None

    reason: str