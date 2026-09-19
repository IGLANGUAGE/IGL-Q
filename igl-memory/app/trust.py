from __future__ import annotations

from . import db

from .models import (
    Belief,
    BeliefStatus,
    TrustSource,
)


ACTIVE_THRESHOLD = 0.70
SUSPECT_THRESHOLD = 0.40


def is_trusted(
    belief: Belief,
) -> bool:

    return (
        belief.status
        == BeliefStatus.ACTIVE
        and belief.confidence
        >= ACTIVE_THRESHOLD
    )


def validate(
    belief_id: str,

    confidence_delta: float = 0.10,

    source: TrustSource = (
        TrustSource.EVIDENCE
    ),

    reason: dict | None = None,
):
    belief = db.get_belief(
        belief_id
    )

    if belief is None:
        raise ValueError(
            "Belief not found"
        )

    new_confidence = min(
        1.0,
        belief.confidence
        + confidence_delta,
    )

    status = (
        BeliefStatus.ACTIVE
        if new_confidence
        >= ACTIVE_THRESHOLD
        else belief.status
    )

    return db.update_belief_trust(
        belief_id=belief_id,

        status=status,

        confidence=new_confidence,

        action="VALIDATE",

        source=source,

        reason=reason or {},
    )


def challenge(
    belief_id: str,

    confidence_delta: float = 0.25,

    source: TrustSource = (
        TrustSource.EVIDENCE
    ),

    reason: dict | None = None,
):
    belief = db.get_belief(
        belief_id
    )

    if belief is None:
        raise ValueError(
            "Belief not found"
        )

    new_confidence = max(
        0.0,
        belief.confidence
        - confidence_delta,
    )

    if (
        new_confidence
        < SUSPECT_THRESHOLD
    ):
        status = (
            BeliefStatus.SUSPECT
        )
    else:
        status = belief.status

    return db.update_belief_trust(
        belief_id=belief_id,

        status=status,

        confidence=new_confidence,

        action="CHALLENGE",

        source=source,

        reason=reason or {},
    )


def invalidate(
    belief_id: str,

    source: TrustSource,

    reason: dict | None = None,
):
    status = (
        BeliefStatus.USER_REJECTED
        if source
        == TrustSource.USER
        else BeliefStatus.INVALID
    )

    return db.update_belief_trust(
        belief_id=belief_id,

        status=status,

        confidence=0.0,

        action="INVALIDATE",

        source=source,

        reason=reason or {},
    )


def restore(
    belief_id: str,

    confidence: float = 0.75,

    source: TrustSource = (
        TrustSource.USER
    ),

    reason: dict | None = None,
):
    return db.update_belief_trust(
        belief_id=belief_id,

        status=BeliefStatus.ACTIVE,

        confidence=confidence,

        action="RESTORE",

        source=source,

        reason=reason or {},
    )