from __future__ import annotations

from typing import Any

from . import db
from .models import (
    Belief,
    GateAccepted,
    GateQuery,
    GateResult,
    GateWithheld,
)
from .trust import is_trusted


def value_matches(
    required: Any,
    actual: Any,
) -> bool:
    """
    v0.1 applicability semantics:

    Each key in belief.applicability must match
    the current context exactly.

    Later versions can add predicates/ranges.
    """

    return required == actual


def is_applicable(
    belief: Belief,
    context: dict[str, Any],
) -> tuple[bool, str]:

    rules = belief.applicability

    for key, required in rules.items():
        if key not in context:
            return (
                False,
                f"context_missing:{key}",
            )

        if not value_matches(
            required,
            context[key],
        ):
            return (
                False,
                f"context_mismatch:{key}",
            )

    return (
        True,
        "applicable",
    )


def query_gate(
    query: GateQuery,
) -> GateResult:

    beliefs = db.list_beliefs(
        query.user_id
    )

    accepted: list[
        GateAccepted
    ] = []

    withheld: list[
        GateWithheld
    ] = []

    for belief in beliefs:
        # ------------------------------------
        # Exist(D) != Trust(D)
        # ------------------------------------

        if not is_trusted(
            belief
        ):
            withheld.append(
                GateWithheld(
                    belief_id=belief.id,
                    reason=(
                        "not_trusted:"
                        f"{belief.status.value}"
                    ),
                )
            )

            continue

        # ------------------------------------
        # Trust(D) != Apply(D)
        # ------------------------------------

        applicable, reason = (
            is_applicable(
                belief,
                query.context,
            )
        )

        if not applicable:
            withheld.append(
                GateWithheld(
                    belief_id=belief.id,
                    reason=reason,
                )
            )

            continue

        accepted.append(
            GateAccepted(
                belief=belief,

                trusted=True,
                applicable=True,

                reason=(
                    "trusted_and_applicable"
                ),
            )
        )

    # High-confidence items first.
    accepted.sort(
        key=lambda x:
            x.belief.confidence,
        reverse=True,
    )

    accepted = accepted[
        :query.limit
    ]

    return GateResult(
        accepted=accepted,
        withheld=withheld,
    )