"""
analyze_memory_v02_recovery.py

Frozen forensic analysis of Q-IGL Memory v0.2 recovery.

NO simulation.
NO retraining.
NO modification of D.

Reads:
    results_memory_v02/corruption.csv
    results_memory_v02/regime_shift.csv

Questions:

CORRUPTION
----------
1. Were there false generation changes before event t=300?
2. Was corruption detected after the event?
3. Did regret recover?
4. How long did recovery take?
5. Was fallback/learning actually used?

REGIME SHIFT
------------
1. Was there a false alarm?
2. If no detection occurred, was old D still adequate?
3. Did regret increase relative to pre-shift?
4. Did v0.2 remain better/worse than B1?

Important:
This script does not decide whether v0.3 is needed.
It reports forensic facts.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np


# ============================================================
# Frozen forensic parameters
# ============================================================

RESULT_DIR = Path(
    "results_memory_v02"
)

CORRUPTION_PATH = (
    RESULT_DIR / "corruption.csv"
)

SHIFT_PATH = (
    RESULT_DIR / "regime_shift.csv"
)

EVENT_STEP = 300

# Windows fixed for forensic interpretation.
WINDOWS = {
    "PRE": (200, 300),
    "ACUTE": (300, 350),
    "RECOVERY": (350, 500),
    "LATE": (500, 800),
}

# Rolling recovery definition.
ROLLING_WINDOW = 40

# Recovery means:
#
# rolling regret <= pre-event mean regret + tolerance
#
# for SUSTAIN_STEPS consecutive evaluated points.
RECOVERY_TOLERANCE = 0.005
SUSTAIN_STEPS = 20

BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260927


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    parsed = []

    for row in rows:
        parsed.append({
            "t":
                int(row["t"]),

            "shifted":
                int(row["shifted"]),

            "generation":
                int(row["generation"]),

            "policy":
                row["policy"],

            "mode":
                row["mode"],

            "trusted":
                int(row["trusted"]),

            "reward":
                float(row["reward"]),

            "regret_v02":
                float(row["regret_v02"]),

            "regret_b1":
                float(row["regret_b1"]),

            "adv_v02_vs_b1":
                float(
                    row[
                        "adv_v02_vs_b1"
                    ]
                ),
        })

    return parsed


def arr(rows, key):
    return np.asarray(
        [r[key] for r in rows]
    )


def select_window(
    rows,
    start,
    end,
):
    return [
        r for r in rows
        if start <= r["t"] < end
    ]


def bootstrap_ci(
    values,
    seed,
    n_boot=BOOTSTRAPS,
    alpha=0.05,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    boot = np.empty(
        n_boot
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = np.mean(
            values[idx]
        )

    return (
        float(
            np.quantile(
                boot,
                alpha / 2,
            )
        ),
        float(
            np.quantile(
                boot,
                1 - alpha / 2,
            )
        ),
    )


# ============================================================
# Generation forensic
# ============================================================

def generation_changes(rows):
    changes = []

    if not rows:
        return changes

    previous = rows[0][
        "generation"
    ]

    for row in rows[1:]:
        current = row[
            "generation"
        ]

        if current != previous:
            changes.append({
                "t": row["t"],
                "from": previous,
                "to": current,
                "mode": row["mode"],
            })

            previous = current

    return changes


# ============================================================
# Modes / policies
# ============================================================

def distribution(
    rows,
    key,
):
    counts = {}

    for row in rows:
        value = row[key]

        counts[value] = (
            counts.get(
                value,
                0,
            )
            + 1
        )

    total = len(rows)

    return {
        key: {
            value: {
                "count": count,
                "fraction": (
                    count / total
                    if total
                    else 0.0
                ),
            }
            for value, count
            in sorted(
                counts.items()
            )
        }
    }


# ============================================================
# Window statistics
# ============================================================

def window_stats(
    rows,
    name,
    start,
    end,
    seed,
):
    w = select_window(
        rows,
        start,
        end,
    )

    rv = arr(
        w,
        "regret_v02",
    )

    rb = arr(
        w,
        "regret_b1",
    )

    advantage = (
        rb - rv
    )

    ci = bootstrap_ci(
        advantage,
        seed,
    )

    return {
        "name": name,
        "range": (
            start,
            end,
        ),

        "n":
            len(w),

        "mean_regret_v02":
            float(
                np.mean(rv)
            )
            if len(rv)
            else float("nan"),

        "mean_regret_b1":
            float(
                np.mean(rb)
            )
            if len(rb)
            else float("nan"),

        "v02_advantage_vs_b1":
            float(
                np.mean(
                    advantage
                )
            )
            if len(advantage)
            else float("nan"),

        "advantage_ci95":
            ci,

        "mean_reward":
            float(
                np.mean(
                    arr(
                        w,
                        "reward",
                    )
                )
            )
            if w
            else float("nan"),

        "trusted_fraction":
            float(
                np.mean(
                    arr(
                        w,
                        "trusted",
                    )
                )
            )
            if w
            else float("nan"),

        "modes":
            distribution(
                w,
                "mode",
            )["mode"],

        "policies":
            distribution(
                w,
                "policy",
            )["policy"],
    }


# ============================================================
# Rolling recovery
# ============================================================

def rolling_mean(
    values,
    window,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < window:
        return np.asarray([])

    kernel = (
        np.ones(window)
        / window
    )

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


def find_recovery(
    rows,
    event_step,
    pre_mean,
):
    """
    First point after event where rolling regret
    stays <= pre_mean + tolerance for SUSTAIN_STEPS
    consecutive rolling estimates.

    Returned t corresponds to the END of the
    first qualifying rolling window.
    """

    post = [
        r for r in rows
        if r["t"] >= event_step
    ]

    values = [
        r["regret_v02"]
        for r in post
    ]

    times = [
        r["t"]
        for r in post
    ]

    roll = rolling_mean(
        values,
        ROLLING_WINDOW,
    )

    if len(roll) == 0:
        return None, None

    threshold = (
        pre_mean
        + RECOVERY_TOLERANCE
    )

    good = (
        roll <= threshold
    )

    for i in range(
        0,
        len(good)
        - SUSTAIN_STEPS
        + 1
    ):
        if np.all(
            good[
                i:
                i + SUSTAIN_STEPS
            ]
        ):
            # Rolling value i ends at
            # post index i+window-1.
            end_index = (
                i
                + ROLLING_WINDOW
                - 1
            )

            return (
                times[end_index],
                float(
                    roll[i]
                ),
            )

    return None, None


# ============================================================
# Cumulative excess regret
# ============================================================

def cumulative_excess_after_event(
    rows,
    event_step,
):
    post = [
        r for r in rows
        if r["t"] >= event_step
    ]

    # Positive means v02 accumulated MORE regret
    # than always-B1.
    excess = np.asarray([
        r["regret_v02"]
        - r["regret_b1"]
        for r in post
    ])

    return {
        "total":
            float(
                np.sum(excess)
            ),

        "mean":
            float(
                np.mean(excess)
            )
            if len(excess)
            else float("nan"),

        "fraction_v02_worse":
            float(
                np.mean(
                    excess > 0
                )
            )
            if len(excess)
            else float("nan"),
    }


# ============================================================
# Report one scenario
# ============================================================

def analyze_scenario(
    name,
    path,
    seed_offset,
):
    rows = load_csv(
        path
    )

    print(
        f"\n{name}"
    )
    print(
        "=" * len(name)
    )

    print(
        "FILE_SHA256 =",
        sha256_file(path)
    )

    print(
        "rows =",
        len(rows)
    )

    # --------------------------------------------------------
    # Generation changes
    # --------------------------------------------------------

    changes = generation_changes(
        rows
    )

    print(
        "\nGENERATION CHANGES"
    )
    print(
        "------------------"
    )

    if not changes:
        print(
            "none"
        )
    else:
        for c in changes:
            print(
                f"t={c['t']:3d} "
                f"{c['from']} -> "
                f"{c['to']} "
                f"mode={c['mode']}"
            )

    pre_changes = [
        c for c in changes
        if c["t"] < EVENT_STEP
    ]

    post_changes = [
        c for c in changes
        if c["t"] >= EVENT_STEP
    ]

    print(
        "changes before event =",
        len(pre_changes)
    )

    print(
        "changes after event  =",
        len(post_changes)
    )

    # --------------------------------------------------------
    # Window statistics
    # --------------------------------------------------------

    print(
        "\nWINDOWS"
    )
    print(
        "-------"
    )

    stats = {}

    for i, (
        window_name,
        (start, end),
    ) in enumerate(
        WINDOWS.items()
    ):
        s = window_stats(
            rows,
            window_name,
            start,
            end,
            BOOTSTRAP_SEED
            + seed_offset
            + i,
        )

        stats[
            window_name
        ] = s

        lo, hi = (
            s[
                "advantage_ci95"
            ]
        )

        print(
            f"\n{window_name} "
            f"[{start},{end}) "
            f"n={s['n']}"
        )

        print(
            "  regret v02 :",
            f"{s['mean_regret_v02']:.6f}"
        )

        print(
            "  regret B1  :",
            f"{s['mean_regret_b1']:.6f}"
        )

        print(
            "  advantage  :",
            f"{s['v02_advantage_vs_b1']:+.6f}",
            f"CI [{lo:+.6f}, "
            f"{hi:+.6f}]"
        )

        print(
            "  trusted    :",
            f"{s['trusted_fraction']:.3f}"
        )

        print(
            "  modes      :",
            s["modes"]
        )

        print(
            "  policies   :",
            s["policies"]
        )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    pre_mean = stats[
        "PRE"
    ][
        "mean_regret_v02"
    ]

    recovery_t, recovery_roll = (
        find_recovery(
            rows,
            EVENT_STEP,
            pre_mean,
        )
    )

    print(
        "\nRECOVERY TEST"
    )
    print(
        "-------------"
    )

    print(
        "pre mean regret =",
        f"{pre_mean:.6f}"
    )

    print(
        "threshold =",
        f"{pre_mean + RECOVERY_TOLERANCE:.6f}"
    )

    print(
        "rolling window =",
        ROLLING_WINDOW
    )

    print(
        "sustain steps =",
        SUSTAIN_STEPS
    )

    print(
        "recovered at t =",
        recovery_t
    )

    if recovery_t is not None:
        print(
            "recovery delay =",
            recovery_t
            - EVENT_STEP
        )

        print(
            "rolling regret =",
            f"{recovery_roll:.6f}"
        )

    # --------------------------------------------------------
    # Cumulative excess vs B1
    # --------------------------------------------------------

    excess = (
        cumulative_excess_after_event(
            rows,
            EVENT_STEP,
        )
    )

    print(
        "\nPOST-EVENT EXCESS REGRET "
        "vs B1"
    )
    print(
        "-------------------------"
    )

    print(
        "total =",
        f"{excess['total']:+.6f}"
    )

    print(
        "mean  =",
        f"{excess['mean']:+.6f}"
    )

    print(
        "fraction v02 worse =",
        f"{excess['fraction_v02_worse']:.3f}"
    )

    # --------------------------------------------------------
    # Overall mode distribution
    # --------------------------------------------------------

    print(
        "\nOVERALL GATE MODES"
    )
    print(
        "------------------"
    )

    print(
        distribution(
            rows,
            "mode",
        )["mode"]
    )

    return {
        "rows":
            rows,

        "changes":
            changes,

        "window_stats":
            stats,

        "recovery_t":
            recovery_t,

        "excess":
            excess,
    }


# ============================================================
# Main
# ============================================================

def main():
    print(
        "Q-IGL Memory v0.2 "
        "Frozen Recovery Forensic"
    )
    print(
        "========================================"
    )

    corruption = (
        analyze_scenario(
            "CORRUPTION",
            CORRUPTION_PATH,
            seed_offset=0,
        )
    )

    shift = (
        analyze_scenario(
            "REGIME SHIFT",
            SHIFT_PATH,
            seed_offset=100,
        )
    )

    print(
        "\nFINAL FORENSIC QUESTIONS"
    )
    print(
        "========================"
    )

    corruption_false_alarms = [
        c
        for c in corruption[
            "changes"
        ]
        if c["t"] < EVENT_STEP
    ]

    shift_false_alarms = [
        c
        for c in shift[
            "changes"
        ]
        if c["t"] < EVENT_STEP
    ]

    print(
        "corruption false "
        "generation changes before event:",
        len(
            corruption_false_alarms
        )
    )

    print(
        "shift false "
        "generation changes before event:",
        len(
            shift_false_alarms
        )
    )

    print(
        "corruption recovery t:",
        corruption[
            "recovery_t"
        ]
    )

    print(
        "shift recovery t:",
        shift[
            "recovery_t"
        ]
    )

    print(
        "\nInterpretation is deliberately "
        "NOT automated."
    )


if __name__ == "__main__":
    main()