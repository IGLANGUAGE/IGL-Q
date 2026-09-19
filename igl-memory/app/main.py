from __future__ import annotations

from fastapi import (
    FastAPI,
    HTTPException,
)

from . import db
from .gate import query_gate
from .models import (
    Belief,
    BeliefCreate,
    Event,
    EventCreate,
    GateQuery,
    GateResult,
    TrustAction,
    TrustSource,
)
from . import trust


app = FastAPI(
    title="IGL Memory",
    version="0.1.0",
    description=(
        "Provenance-first memory layer "
        "separating events, beliefs, "
        "trust and applicability."
    ),
)


@app.on_event("startup")
def startup():
    db.init_db()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "architecture":
            "M -> D -> Trust -> Gate -> K",
    }


# ============================================================
# M / Events
# ============================================================

@app.post(
    "/events",
    response_model=Event,
)
def create_event(
    payload: EventCreate,
):
    return db.create_event(
        payload
    )


@app.get(
    "/events/{event_id}",
    response_model=Event,
)
def get_event(
    event_id: str,
):
    item = db.get_event(
        event_id
    )

    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found",
        )

    return item


# ============================================================
# D / Beliefs
# ============================================================

@app.post(
    "/beliefs",
    response_model=Belief,
)
def create_belief(
    payload: BeliefCreate,
):
    try:
        return db.create_belief(
            payload
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get(
    "/beliefs",
    response_model=list[Belief],
)
def list_beliefs(
    user_id: str,
):
    return db.list_beliefs(
        user_id
    )


@app.get(
    "/beliefs/{belief_id}",
    response_model=Belief,
)
def get_belief(
    belief_id: str,
):
    item = db.get_belief(
        belief_id
    )

    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Belief not found",
        )

    return item


@app.get(
    "/beliefs/{belief_id}/provenance",
    response_model=list[Event],
)
def provenance(
    belief_id: str,
):
    belief = db.get_belief(
        belief_id
    )

    if belief is None:
        raise HTTPException(
            status_code=404,
            detail="Belief not found",
        )

    return db.get_provenance(
        belief_id
    )


# ============================================================
# Trust
# ============================================================

@app.post(
    "/beliefs/{belief_id}/validate",
    response_model=Belief,
)
def validate_belief(
    belief_id: str,
    action: TrustAction,
):
    try:
        return trust.validate(
            belief_id,

            confidence_delta=abs(
                action.confidence_delta
            ),

            source=action.source,

            reason=action.reason,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@app.post(
    "/beliefs/{belief_id}/challenge",
    response_model=Belief,
)
def challenge_belief(
    belief_id: str,
    action: TrustAction,
):
    try:
        return trust.challenge(
            belief_id,

            confidence_delta=abs(
                action.confidence_delta
            ),

            source=action.source,

            reason=action.reason,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@app.post(
    "/beliefs/{belief_id}/invalidate",
    response_model=Belief,
)
def invalidate_belief(
    belief_id: str,
    action: TrustAction,
):
    try:
        return trust.invalidate(
            belief_id,

            source=action.source,

            reason=action.reason,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@app.post(
    "/beliefs/{belief_id}/restore",
    response_model=Belief,
)
def restore_belief(
    belief_id: str,
    action: TrustAction,
):
    confidence = (
        action.confidence_delta
        if action.confidence_delta > 0
        else 0.75
    )

    try:
        return trust.restore(
            belief_id,

            confidence=confidence,

            source=action.source,

            reason=action.reason,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


# ============================================================
# Gate
# ============================================================

@app.post(
    "/gate/query",
    response_model=GateResult,
)
def gate(
    payload: GateQuery,
):
    return query_gate(
        payload
    )