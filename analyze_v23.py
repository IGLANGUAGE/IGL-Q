"""
analyze_v23.py

Forensic analysis of frozen Q-IGL v23 results.

NO simulation.
NO new trajectories.
NO controller execution.

Reads:
    results_v23/trajectories.csv

Computes:
    Hybrid - Global
    Hybrid - Local
    B1 - Hybrid
    exact/numerical H == G coincidence
    paired bootstrap CIs
    win/loss fractions
    control-energy deltas
"""

from pathlib import Path
import csv
import hashlib
import numpy as np


CSV_PATH = Path("results_v23/trajectories.csv")

BOOTSTRAPS = 10000
SEED = 20260924
TOL = 1e-12


def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    def col(name):
        return np.asarray(
            [float(r[name]) for r in rows],
            dtype=float
        )

    return rows, col


def bootstrap_mean_ci(
    values,
    rng,
    n_boot=BOOTSTRAPS,
    alpha=0.05
):
    values = np.asarray(values)
    n = len(values)

    out = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(
            0, n, size=n
        )
        out[b] = np.mean(
            values[idx]
        )

    return (
        float(np.quantile(
            out, alpha / 2
        )),
        float(np.quantile(
            out, 1 - alpha / 2
        ))
    )


def paired_effect_size(delta):
    sd = np.std(
        delta,
        ddof=1
    )

    if sd == 0:
        return 0.0

    return float(
        np.mean(delta) / sd
    )


def report_delta(
    name,
    delta,
    rng
):
    ci = bootstrap_mean_ci(
        delta,
        rng
    )

    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"mean       : "
        f"{np.mean(delta):+.12f}"
    )

    print(
        f"median     : "
        f"{np.median(delta):+.12f}"
    )

    print(
        f"95% CI     : "
        f"[{ci[0]:+.12f}, "
        f"{ci[1]:+.12f}]"
    )

    print(
        f"paired d   : "
        f"{paired_effect_size(delta):+.6f}"
    )

    print(
        f"P(delta>0) : "
        f"{np.mean(delta > 0):.6f}"
    )

    print(
        f"P(delta<0) : "
        f"{np.mean(delta < 0):.6f}"
    )

    print(
        f"P(delta=0) : "
        f"{np.mean(delta == 0):.6f}"
    )

    print(
        f"min/max    : "
        f"{np.min(delta):+.12f} / "
        f"{np.max(delta):+.12f}"
    )


def main():
    print(
        "Q-IGL v23 Frozen Forensic Analysis"
    )
    print(
        "=================================="
    )

    print(
        "CSV_SHA256 =",
        sha256_file(CSV_PATH)
    )

    rows, col = load_csv(
        CSV_PATH
    )

    print(
        "TRAJECTORIES =",
        len(rows)
    )

    # ---------------- Fidelity ----------------

    Fb = col("B1_F")
    Fg = col("GLOBAL_F")
    Fl = col("LOCAL_F")
    Fh = col("HYBRID_F")

    d_hg = Fh - Fg
    d_hl = Fh - Fl
    d_bh = Fb - Fh

    rng = np.random.default_rng(
        SEED
    )

    report_delta(
        "HYBRID - GLOBAL",
        d_hg,
        rng
    )

    report_delta(
        "HYBRID - LOCAL",
        d_hl,
        rng
    )

    report_delta(
        "B1 - HYBRID",
        d_bh,
        rng
    )

    # ---------------- H ~= G ----------------

    abs_hg = np.abs(
        d_hg
    )

    print(
        "\nHybrid / Global numerical identity"
    )
    print(
        "----------------------------------"
    )

    print(
        f"exact equality fraction : "
        f"{np.mean(Fh == Fg):.6f}"
    )

    print(
        f"|H-G| <= {TOL:g}       : "
        f"{np.mean(abs_hg <= TOL):.6f}"
    )

    print(
        f"mean |H-G|             : "
        f"{np.mean(abs_hg):.12e}"
    )

    print(
        f"max |H-G|              : "
        f"{np.max(abs_hg):.12e}"
    )

    # ---------------- Energy ----------------

    Eb = col("B1_E")
    Eg = col("GLOBAL_E")
    El = col("LOCAL_E")
    Eh = col("HYBRID_E")

    report_delta(
        "ENERGY: HYBRID - GLOBAL",
        Eh - Eg,
        rng
    )

    report_delta(
        "ENERGY: HYBRID - LOCAL",
        Eh - El,
        rng
    )

    report_delta(
        "ENERGY: B1 - HYBRID",
        Eb - Eh,
        rng
    )

    # ---------------- Efficiency ----------------
    #
    # Descriptive only.
    # Not preregistered as primary.
    #

    gain_hg = Fh - Fg
    extra_e_hg = Eh - Eg

    valid = (
        np.abs(extra_e_hg)
        > 1e-12
    )

    ratio = (
        gain_hg[valid]
        / extra_e_hg[valid]
    )

    print(
        "\nDescriptive marginal payoff "
        "(H-G fidelity / extra energy)"
    )
    print(
        "----------------------------------------"
    )

    if len(ratio):
        print(
            f"mean   : "
            f"{np.mean(ratio):+.12f}"
        )
        print(
            f"median : "
            f"{np.median(ratio):+.12f}"
        )

    # ---------------- Frozen conclusion inputs ----------------

    print(
        "\nForensic facts only"
    )
    print(
        "-------------------"
    )

    print(
        "mean F:",
        {
            "B1": float(np.mean(Fb)),
            "GLOBAL": float(np.mean(Fg)),
            "LOCAL": float(np.mean(Fl)),
            "HYBRID": float(np.mean(Fh)),
        }
    )

    print(
        "mean Energy:",
        {
            "B1": float(np.mean(Eb)),
            "GLOBAL": float(np.mean(Eg)),
            "LOCAL": float(np.mean(El)),
            "HYBRID": float(np.mean(Eh)),
        }
    )


if __name__ == "__main__":
    main()