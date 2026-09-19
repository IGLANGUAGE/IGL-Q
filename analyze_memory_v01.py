"""
analyze_memory_v01.py

Frozen forensic analysis of Q-IGL Memory v0.1.

NO simulation.
NO retraining.
NO modification of D.

Tests the exploitation-trap hypothesis:

    low exploration
      -> structured policies lack coverage
      -> confidence gate rejects them
      -> B1 fallback dominates
      -> even more B1 evidence

Reads:
    results_memory_v01/D_trained.json
"""

import json
import hashlib
from pathlib import Path

import numpy as np


D_PATH = Path(
    "results_memory_v01/D_trained.json"
)

MIN_SAMPLES = 4

POLICIES = (
    "B1",
    "GLOBAL",
    "LOCAL",
    "HYBRID",
)


def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def main():
    print(
        "Q-IGL Memory v0.1 Forensic"
    )
    print(
        "==========================="
    )

    print(
        "D_SHA256 =",
        sha256_file(D_PATH)
    )

    D = json.loads(
        D_PATH.read_text(
            encoding="utf-8"
        )
    )

    prototypes = D["prototypes"]

    print(
        "prototypes =",
        len(prototypes)
    )

    total_counts = {
        p: 0 for p in POLICIES
    }

    qualified = {
        p: 0 for p in POLICIES
    }

    best_by_mean = {
        p: 0 for p in POLICIES
    }

    rows = []

    for proto in prototypes:
        pid = proto[
            "prototype_id"
        ]

        stats = proto[
            "stats"
        ]

        counts = {}

        means = {}

        for policy in POLICIES:
            s = stats[policy]

            count = int(
                s["count"]
            )

            counts[policy] = count

            total_counts[
                policy
            ] += count

            if count >= MIN_SAMPLES:
                qualified[
                    policy
                ] += 1

            value = s["mean"]

            means[policy] = (
                float(value)
                if value is not None
                else np.nan
            )

        eligible = [
            p for p in POLICIES
            if counts[p] >= MIN_SAMPLES
            and np.isfinite(
                means[p]
            )
        ]

        if eligible:
            best = max(
                eligible,
                key=lambda p:
                    means[p]
            )

            best_by_mean[
                best
            ] += 1
        else:
            best = "NONE"

        row = {
            "id": pid,
            "visits": proto["visits"],
            "counts": counts,
            "means": means,
            "eligible": eligible,
            "best": best,
        }

        rows.append(row)

    print(
        "\nTOTAL POLICY OBSERVATIONS"
    )
    print(
        "-------------------------"
    )

    total = sum(
        total_counts.values()
    )

    for policy in POLICIES:
        n = total_counts[
            policy
        ]

        frac = (
            n / total
            if total else 0.0
        )

        print(
            f"{policy:8s}: "
            f"{n:4d} "
            f"({frac:.3f})"
        )

    print(
        "\nPROTOTYPES WITH count >= "
        f"{MIN_SAMPLES}"
    )
    print(
        "--------------------------"
    )

    for policy in POLICIES:
        n = qualified[
            policy
        ]

        frac = (
            n / len(prototypes)
            if prototypes
            else 0.0
        )

        print(
            f"{policy:8s}: "
            f"{n:2d}/"
            f"{len(prototypes)} "
            f"({frac:.3f})"
        )

    print(
        "\nBEST ELIGIBLE POLICY "
        "BY STORED MEAN"
    )
    print(
        "--------------------------"
    )

    for policy in POLICIES:
        print(
            f"{policy:8s}: "
            f"{best_by_mean[policy]}"
        )

    print(
        "\nPER-PROTOTYPE"
    )
    print(
        "-------------"
    )

    for row in rows:
        print(
            f"\nPrototype "
            f"{row['id']} "
            f"(visits={row['visits']})"
        )

        for policy in POLICIES:
            c = row[
                "counts"
            ][policy]

            m = row[
                "means"
            ][policy]

            m_text = (
                f"{m:.4f}"
                if np.isfinite(m)
                else "NA"
            )

            flag = (
                "*"
                if c >= MIN_SAMPLES
                else " "
            )

            print(
                f" {flag} "
                f"{policy:8s} "
                f"n={c:3d} "
                f"mean={m_text}"
            )

        print(
            " eligible =",
            row["eligible"]
        )

        print(
            " best     =",
            row["best"]
        )

    # --------------------------------------------
    # Coverage imbalance
    # --------------------------------------------

    b1_counts = np.asarray(
        [
            r["counts"]["B1"]
            for r in rows
        ],
        dtype=float,
    )

    structured_counts = np.asarray(
        [
            sum(
                r["counts"][p]
                for p in (
                    "GLOBAL",
                    "LOCAL",
                    "HYBRID",
                )
            )
            for r in rows
        ],
        dtype=float,
    )

    print(
        "\nCOVERAGE IMBALANCE"
    )
    print(
        "------------------"
    )

    print(
        "mean B1/prototype        =",
        f"{np.mean(b1_counts):.3f}"
    )

    print(
        "mean structured/prototype=",
        f"{np.mean(structured_counts):.3f}"
    )

    ratio = (
        np.sum(b1_counts)
        / max(
            np.sum(
                structured_counts
            ),
            1.0,
        )
    )

    print(
        "B1 / all-structured ratio=",
        f"{ratio:.3f}"
    )

    all_covered = sum(
        all(
            r["counts"][p]
            >= MIN_SAMPLES
            for p in POLICIES
        )
        for r in rows
    )

    print(
        "prototypes where ALL "
        "policies eligible =",
        f"{all_covered}/"
        f"{len(rows)}"
    )


if __name__ == "__main__":
    main()