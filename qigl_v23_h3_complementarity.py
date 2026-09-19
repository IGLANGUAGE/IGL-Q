"""
qigl_v23_h3_complementarity.py

Q-IGL H3 Complementarity Test
-----------------------------
Primary hypothesis:
In mixed global + local structured noise, Hybrid control
outperforms both Global-only and Local-only controls under
the same observations and actuator bounds.

Four paired arms:
    B1      baseline
    GLOBAL  shared latent/common-mode estimator
    LOCAL   local graph-aware controller
    HYBRID  GLOBAL + LOCAL residual control

Primary statistic:
    Delta_H =
        mean(F_HYBRID)
        - max(mean(F_GLOBAL), mean(F_LOCAL))

Primary success criterion:
    lower bound of paired bootstrap 95% CI(Delta_H) > 0

Important:
- No controller receives ground-truth noise.
- Same telemetry for all four arms.
- Same u_max for all four arms.
- Same physical noise trajectory for all four arms.
- Results are written per trajectory.
- No automatic "success" labels are printed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# ============================================================
# Frozen config
# ============================================================

@dataclass(frozen=True)
class Config:
    n_qubits: int = 20
    steps: int = 120
    trajectories: int = 1000
    dt: float = 0.04
    seed: int = 20260923

    # Noise dynamics
    rho_global: float = 0.985
    rho_local: float = 0.94

    sigma_global: float = 0.035
    sigma_local: float = 0.025
    sigma_independent: float = 0.012
    sigma_probe: float = 0.055

    # Heterogeneous sensitivity to global latent mode
    g_min: float = 0.70
    g_max: float = 1.30

    # Local graph/crosstalk
    local_coupling: float = 0.32

    # Controllers
    b1_alpha: float = 0.16
    global_alpha: float = 0.14
    local_alpha: float = 0.18

    # Hybrid mixing
    hybrid_global_gain: float = 1.0
    hybrid_local_gain: float = 1.0

    # Equal actuator bound for ALL arms
    u_max: float = 1.50

    # Statistics
    bootstrap_samples: int = 10000
    ci_alpha: float = 0.05

    output_dir: str = "results_v23"


CFG = Config()


# ============================================================
# Provenance
# ============================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def config_hash(cfg: Config) -> str:
    payload = json.dumps(
        asdict(cfg),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNAVAILABLE"


# ============================================================
# Target metric
#
# Deliberately use a GHZ-like global phase-preservation target.
# Under Z-type phase noise:
#
#       F = cos^2(phi_total / 2)
#
# This is an analytically cheap process-fidelity proxy.
# ============================================================

def fidelity_from_phase(phi: float) -> float:
    return float(np.cos(0.5 * phi) ** 2)


# ============================================================
# Local topology
# ============================================================

def ring_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)

    for i in range(n):
        A[i, (i - 1) % n] = 1.0
        A[i, (i + 1) % n] = 1.0

    # Each row averages its two neighbors.
    A /= A.sum(axis=1, keepdims=True)
    return A


# ============================================================
# Environment
# ============================================================

def generate_environment(
    cfg: Config,
    rng: np.random.Generator,
):
    """
    Mixed noise:
        detuning = G*z + L*xi + eta

    z:
        shared low-rank latent mode

    xi:
        graph-structured local latent process

    eta:
        independent residual noise

    Ground truth is NEVER exposed to controllers.
    """

    T = cfg.steps
    N = cfg.n_qubits

    graph = ring_adjacency(N)
    g = np.linspace(cfg.g_min, cfg.g_max, N)

    z = np.zeros(T)
    xi = np.zeros((T, N))
    eta = np.zeros((T, N))

    for t in range(T):
        if t == 0:
            z[t] = rng.normal(
                0.0,
                cfg.sigma_global
            )

            xi[t] = rng.normal(
                0.0,
                cfg.sigma_local,
                N
            )
        else:
            z[t] = (
                cfg.rho_global * z[t - 1]
                + rng.normal(
                    0.0,
                    cfg.sigma_global
                )
            )

            innovations = rng.normal(
                0.0,
                cfg.sigma_local,
                N
            )

            xi[t] = (
                cfg.rho_local * xi[t - 1]
                + innovations
            )

        eta[t] = rng.normal(
            0.0,
            cfg.sigma_independent,
            N
        )

    # Local graph component.
    local_structured = (
        xi
        + cfg.local_coupling * (xi @ graph.T)
    )

    global_component = z[:, None] * g[None, :]

    detuning = (
        global_component
        + local_structured
        + eta
    )

    truth = {
        "detuning": detuning,
        "z": z,
        "xi": xi,
        "g": g,
        "graph": graph,
    }

    return truth


def generate_probes(
    cfg: Config,
    detuning: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    return detuning + rng.normal(
        0.0,
        cfg.sigma_probe,
        detuning.shape,
    )


# ============================================================
# Controllers
# ============================================================

class Controller:
    def update(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class B1Controller(Controller):
    """
    Independent EMA estimator.
    No global latent structure.
    No local graph structure.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.x = np.zeros(cfg.n_qubits)

    def update(self, observation):
        a = self.cfg.b1_alpha

        self.x = (
            (1.0 - a) * self.x
            + a * observation
        )

        return np.clip(
            -self.x,
            -self.cfg.u_max,
            self.cfg.u_max,
        )


class GlobalController(Controller):
    """
    Estimates only a shared low-rank/global component.

    Uses the known sensitivity shape g but does not see true z.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.g = np.linspace(
            cfg.g_min,
            cfg.g_max,
            cfg.n_qubits
        )
        self.z_hat = 0.0

    def estimate_global(self, observation):
        # Least-squares shared latent estimate.
        z_obs = (
            np.dot(self.g, observation)
            / np.dot(self.g, self.g)
        )

        a = self.cfg.global_alpha

        self.z_hat = (
            (1.0 - a) * self.z_hat
            + a * z_obs
        )

        return self.g * self.z_hat

    def update(self, observation):
        global_hat = self.estimate_global(
            observation
        )

        return np.clip(
            -global_hat,
            -self.cfg.u_max,
            self.cfg.u_max,
        )


class LocalController(Controller):
    """
    Graph-aware local estimator.

    Removes instantaneous common mean before estimating
    graph/local residual structure.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.graph = ring_adjacency(
            cfg.n_qubits
        )

        self.local_hat = np.zeros(
            cfg.n_qubits
        )

    def estimate_local(self, observation):
        residual = (
            observation
            - np.mean(observation)
        )

        graph_signal = (
            residual
            + self.cfg.local_coupling
            * (self.graph @ residual)
        )

        a = self.cfg.local_alpha

        self.local_hat = (
            (1.0 - a) * self.local_hat
            + a * graph_signal
        )

        return self.local_hat.copy()

    def update(self, observation):
        local_hat = self.estimate_local(
            observation
        )

        return np.clip(
            -local_hat,
            -self.cfg.u_max,
            self.cfg.u_max,
        )


class HybridController(Controller):
    """
    Global estimate first, then local controller works
    on the residual.

    Same observation vector and same final actuator bound
    as the other arms.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.global_part = GlobalController(
            cfg
        )

        self.local_part = LocalController(
            cfg
        )

    def update(self, observation):
        global_hat = (
            self.global_part.estimate_global(
                observation
            )
        )

        residual = observation - global_hat

        local_hat = (
            self.local_part.estimate_local(
                residual
            )
        )

        estimate = (
            self.cfg.hybrid_global_gain
            * global_hat
            + self.cfg.hybrid_local_gain
            * local_hat
        )

        return np.clip(
            -estimate,
            -self.cfg.u_max,
            self.cfg.u_max,
        )


# ============================================================
# Paired trajectory
# ============================================================

def run_arm(
    cfg: Config,
    detuning: np.ndarray,
    probes: np.ndarray,
    controller: Controller,
):
    phi = 0.0

    fidelity_trace = np.zeros(
        cfg.steps
    )

    control_energy = 0.0

    for t in range(cfg.steps):
        # Controller receives only observed telemetry.
        u = controller.update(
            probes[t]
        )

        # Hidden true environment.
        residual = detuning[t] + u

        # GHZ global relative phase.
        phi += (
            np.sum(residual)
            * cfg.dt
        )

        fidelity_trace[t] = (
            fidelity_from_phase(phi)
        )

        control_energy += (
            np.sum(u ** 2) * cfg.dt
        )

    return {
        "final_fidelity":
            float(fidelity_trace[-1]),

        "integrated_fidelity":
            float(np.mean(fidelity_trace)),

        "control_energy":
            float(control_energy),
    }


def run_paired(
    cfg: Config,
    seed: int,
):
    seq = np.random.SeedSequence(seed)

    env_seq, probe_seq = seq.spawn(2)

    env_rng = np.random.default_rng(
        env_seq
    )

    probe_rng = np.random.default_rng(
        probe_seq
    )

    truth = generate_environment(
        cfg,
        env_rng
    )

    probes = generate_probes(
        cfg,
        truth["detuning"],
        probe_rng,
    )

    arms = {
        "B1":
            B1Controller(cfg),

        "GLOBAL":
            GlobalController(cfg),

        "LOCAL":
            LocalController(cfg),

        "HYBRID":
            HybridController(cfg),
    }

    results = {}

    for name, controller in arms.items():
        results[name] = run_arm(
            cfg,
            truth["detuning"],
            probes,
            controller,
        )

    return results


# ============================================================
# Statistics
# ============================================================

def primary_statistic(
    Fg: np.ndarray,
    Fl: np.ndarray,
    Fh: np.ndarray,
) -> float:

    return float(
        np.mean(Fh)
        - max(
            np.mean(Fg),
            np.mean(Fl)
        )
    )


def bootstrap_primary_ci(
    Fg,
    Fl,
    Fh,
    seed,
    samples,
    alpha,
):
    """
    Paired bootstrap.

    Each bootstrap resamples trajectory indices.
    The max of Global/Local means is recomputed
    inside every bootstrap sample.
    """

    Fg = np.asarray(Fg)
    Fl = np.asarray(Fl)
    Fh = np.asarray(Fh)

    n = len(Fh)

    rng = np.random.default_rng(seed)

    values = np.empty(samples)

    for b in range(samples):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        values[b] = (
            np.mean(Fh[idx])
            - max(
                np.mean(Fg[idx]),
                np.mean(Fl[idx])
            )
        )

    lo = np.quantile(
        values,
        alpha / 2
    )

    hi = np.quantile(
        values,
        1 - alpha / 2
    )

    return (
        float(lo),
        float(hi)
    )


def secondary_adversarial_metric(
    Fg,
    Fl,
    Fh,
):
    """
    Deliberately stronger secondary metric:
    Hybrid must beat the better arm trajectory-by-trajectory.
    """

    d = (
        np.asarray(Fh)
        - np.maximum(
            np.asarray(Fg),
            np.asarray(Fl)
        )
    )

    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "fraction_positive":
            float(np.mean(d > 0)),
    }


# ============================================================
# Experiment
# ============================================================

def main():
    cfg = CFG

    source_path = Path(__file__).resolve()

    out = Path(cfg.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    source_sha = sha256_file(
        source_path
    )

    cfg_sha = config_hash(
        cfg
    )

    commit = git_commit()

    print("Q-IGL v23 H3 Complementarity Test")
    print("=================================")
    print(
        "SOURCE_SHA256 =",
        source_sha
    )
    print(
        "CONFIG_SHA256 =",
        cfg_sha
    )
    print(
        "GIT_COMMIT    =",
        commit
    )
    print(
        "TRAJECTORIES  =",
        cfg.trajectories
    )

    master = np.random.SeedSequence(
        cfg.seed
    )

    seed_sequences = master.spawn(
        cfg.trajectories
    )

    seeds = [
        int(
            s.generate_state(
                1,
                dtype=np.uint64
            )[0]
        )
        for s in seed_sequences
    ]

    rows = []

    for index, seed in enumerate(seeds):
        result = run_paired(
            cfg,
            seed
        )

        row = {
            "trajectory": index,
            "seed": seed,
        }

        for arm in (
            "B1",
            "GLOBAL",
            "LOCAL",
            "HYBRID",
        ):
            r = result[arm]

            row[f"{arm}_F"] = (
                r["final_fidelity"]
            )

            row[f"{arm}_FI"] = (
                r["integrated_fidelity"]
            )

            row[f"{arm}_E"] = (
                r["control_energy"]
            )

        rows.append(row)

        if (index + 1) % 100 == 0:
            print(
                f"{index + 1}/"
                f"{cfg.trajectories}"
            )

    # ---------------- CSV ----------------

    csv_path = out / "trajectories.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)

    # ---------------- arrays ----------------

    def col(name):
        return np.asarray(
            [r[name] for r in rows],
            dtype=float,
        )

    Fb = col("B1_F")
    Fg = col("GLOBAL_F")
    Fl = col("LOCAL_F")
    Fh = col("HYBRID_F")

    primary = primary_statistic(
        Fg,
        Fl,
        Fh
    )

    ci = bootstrap_primary_ci(
        Fg,
        Fl,
        Fh,
        seed=cfg.seed + 999,
        samples=cfg.bootstrap_samples,
        alpha=cfg.ci_alpha,
    )

    secondary = (
        secondary_adversarial_metric(
            Fg,
            Fl,
            Fh
        )
    )

    means = {
        "B1": float(np.mean(Fb)),
        "GLOBAL": float(np.mean(Fg)),
        "LOCAL": float(np.mean(Fl)),
        "HYBRID": float(np.mean(Fh)),
    }

    integrated = {
        arm: float(
            np.mean(
                col(f"{arm}_FI")
            )
        )
        for arm in (
            "B1",
            "GLOBAL",
            "LOCAL",
            "HYBRID",
        )
    }

    energy = {
        arm: float(
            np.mean(
                col(f"{arm}_E")
            )
        )
        for arm in (
            "B1",
            "GLOBAL",
            "LOCAL",
            "HYBRID",
        )
    }

    summary = {
        "source_sha256": source_sha,
        "config_sha256": cfg_sha,
        "git_commit": commit,
        "config": asdict(cfg),

        "mean_final_fidelity":
            means,

        "mean_integrated_fidelity":
            integrated,

        "mean_control_energy":
            energy,

        "primary_delta_hybrid":
            primary,

        "primary_95_ci":
            list(ci),

        "secondary_per_trajectory":
            secondary,

        # No success/failure verdict here.
        # Interpretation happens after the frozen run.
    }

    summary_path = (
        out / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\nFinal Fidelity")
    print("--------------")
    for arm, value in means.items():
        print(
            f"{arm:8s}: {value:.6f}"
        )

    print(
        "\nPrimary Delta_H = "
        "mean(H) - max(mean(G), mean(L))"
    )
    print(
        f"Delta_H : {primary:+.6f}"
    )
    print(
        "95% CI  : "
        f"[{ci[0]:+.6f}, "
        f"{ci[1]:+.6f}]"
    )

    print(
        "\nSecondary trajectory-wise"
    )
    print(
        json.dumps(
            secondary,
            indent=2
        )
    )

    print("\nMean Control Energy")
    print("-------------------")
    for arm, value in energy.items():
        print(
            f"{arm:8s}: {value:.6f}"
        )

    print(
        "\nCSV_SHA256 =",
        sha256_file(csv_path)
    )
    print(
        "SUMMARY_SHA256 =",
        sha256_file(summary_path)
    )


if __name__ == "__main__":
    main()