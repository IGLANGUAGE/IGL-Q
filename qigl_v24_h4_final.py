"""
qigl_v24_h4_final.py

FINAL Q-IGL validation experiment.

H4 — Task-Weighted Context Hypothesis

One fixed noise-generating process.
Three different task sensitivities h:
    GLOBAL
    LOCAL
    MIXED

Four controller arms:
    B1
    GLOBAL
    LOCAL
    HYBRID

Pipeline:
    TRAIN telemetry
        -> Sigma_hat
        -> structured decomposition
        -> task relevance
        -> frozen controller selection
        -> selection.json + SHA256

    ONLY THEN:

    TEST trajectories
        -> evaluate all four arms
        -> oracle mean fidelity
        -> regret of frozen selector
        -> comparator regrets

No v25 regardless of result.
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
# Frozen configuration
# ============================================================

@dataclass(frozen=True)
class Config:
    n_qubits: int = 20
    steps: int = 120

    train_trajectories: int = 300
    test_trajectories: int = 1000

    dt: float = 0.04
    seed: int = 20260924

    # Noise
    rho_global: float = 0.985
    rho_local: float = 0.94

    sigma_global: float = 0.035
    sigma_local: float = 0.025
    sigma_independent: float = 0.012
    sigma_probe: float = 0.055

    g_min: float = 0.70
    g_max: float = 1.30

    local_coupling: float = 0.32

    # Controllers
    b1_alpha: float = 0.16
    global_alpha: float = 0.14
    local_alpha: float = 0.18

    hybrid_global_gain: float = 1.0
    hybrid_local_gain: float = 1.0

    u_max: float = 1.50

    # Frozen selector thresholds
    q_min: float = 0.50
    relevance_threshold: float = 0.75

    # Statistics
    bootstrap_samples: int = 10000
    ci_alpha: float = 0.05

    output_dir: str = "results_v24"


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
# Geometry
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)

    if n <= 1e-15:
        raise ValueError("Cannot normalize zero vector")

    return v / n


def ring_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)

    for i in range(n):
        A[i, (i - 1) % n] = 1.0
        A[i, (i + 1) % n] = 1.0

    A /= A.sum(axis=1, keepdims=True)

    return A


def build_modes(cfg: Config):
    """
    g:
        true/predefined global sensitivity shape.

    l:
        fixed graph-like zero-global-overlap mode.

    Important:
        l is Gram-Schmidt orthogonalized against g.
    """

    N = cfg.n_qubits

    g = np.linspace(
        cfg.g_min,
        cfg.g_max,
        N,
        dtype=float,
    )

    g_hat = normalize(g)

    # Alternating mode has local structure.
    raw_l = np.asarray(
        [
            1.0 if i % 2 == 0 else -1.0
            for i in range(N)
        ],
        dtype=float,
    )

    # Remove global-mode component.
    raw_l = (
        raw_l
        - np.dot(raw_l, g_hat) * g_hat
    )

    l_hat = normalize(raw_l)

    # Mixed task has both components.
    m_hat = normalize(
        g_hat + l_hat
    )

    # Numerical sanity.
    assert abs(
        np.dot(g_hat, l_hat)
    ) < 1e-12

    return {
        "GLOBAL": g_hat,
        "LOCAL": l_hat,
        "MIXED": m_hat,
    }


# ============================================================
# Environment
# ============================================================

def generate_environment(
    cfg: Config,
    rng: np.random.Generator,
):
    """
    epsilon(t) = G z(t) + L xi(t) + eta(t)

    Same environment family as v23.
    """

    T = cfg.steps
    N = cfg.n_qubits

    graph = ring_adjacency(N)

    g = np.linspace(
        cfg.g_min,
        cfg.g_max,
        N,
    )

    z = np.zeros(T)
    xi = np.zeros((T, N))
    eta = np.zeros((T, N))

    for t in range(T):
        if t == 0:
            z[t] = rng.normal(
                0.0,
                cfg.sigma_global,
            )

            xi[t] = rng.normal(
                0.0,
                cfg.sigma_local,
                N,
            )

        else:
            z[t] = (
                cfg.rho_global * z[t - 1]
                + rng.normal(
                    0.0,
                    cfg.sigma_global,
                )
            )

            xi[t] = (
                cfg.rho_local * xi[t - 1]
                + rng.normal(
                    0.0,
                    cfg.sigma_local,
                    N,
                )
            )

        eta[t] = rng.normal(
            0.0,
            cfg.sigma_independent,
            N,
        )

    local_structured = (
        xi
        + cfg.local_coupling
        * (xi @ graph.T)
    )

    global_component = (
        z[:, None]
        * g[None, :]
    )

    detuning = (
        global_component
        + local_structured
        + eta
    )

    return {
        "detuning": detuning,
        "z": z,
        "xi": xi,
        "g": g,
        "graph": graph,
    }


def generate_probes(
    cfg: Config,
    detuning: np.ndarray,
    rng: np.random.Generator,
):
    return (
        detuning
        + rng.normal(
            0.0,
            cfg.sigma_probe,
            detuning.shape,
        )
    )


# ============================================================
# Controllers
# ============================================================

class B1Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.x = np.zeros(
            cfg.n_qubits
        )

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


class GlobalController:
    def __init__(self, cfg):
        self.cfg = cfg

        self.g = np.linspace(
            cfg.g_min,
            cfg.g_max,
            cfg.n_qubits,
        )

        self.z_hat = 0.0

    def estimate_global(
        self,
        observation,
    ):
        z_obs = (
            np.dot(
                self.g,
                observation
            )
            / np.dot(
                self.g,
                self.g
            )
        )

        a = self.cfg.global_alpha

        self.z_hat = (
            (1.0 - a) * self.z_hat
            + a * z_obs
        )

        return (
            self.g * self.z_hat
        )

    def update(self, observation):
        estimate = (
            self.estimate_global(
                observation
            )
        )

        return np.clip(
            -estimate,
            -self.cfg.u_max,
            self.cfg.u_max,
        )


class LocalController:
    def __init__(self, cfg):
        self.cfg = cfg

        self.graph = ring_adjacency(
            cfg.n_qubits
        )

        self.local_hat = np.zeros(
            cfg.n_qubits
        )

    def estimate_local(
        self,
        observation,
    ):
        # Remove global mean-like component.
        residual = (
            observation
            - np.mean(observation)
        )

        graph_signal = (
            residual
            + self.cfg.local_coupling
            * (
                self.graph
                @ residual
            )
        )

        a = self.cfg.local_alpha

        self.local_hat = (
            (1.0 - a)
            * self.local_hat
            + a * graph_signal
        )

        return self.local_hat.copy()

    def update(self, observation):
        estimate = (
            self.estimate_local(
                observation
            )
        )

        return np.clip(
            -estimate,
            -self.cfg.u_max,
            self.cfg.u_max,
        )


class HybridController:
    def __init__(self, cfg):
        self.cfg = cfg

        self.global_part = (
            GlobalController(cfg)
        )

        self.local_part = (
            LocalController(cfg)
        )

    def update(self, observation):
        global_hat = (
            self.global_part
            .estimate_global(
                observation
            )
        )

        residual = (
            observation
            - global_hat
        )

        local_hat = (
            self.local_part
            .estimate_local(
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


def make_controller(
    name: str,
    cfg: Config,
):
    if name == "B1":
        return B1Controller(cfg)

    if name == "GLOBAL":
        return GlobalController(cfg)

    if name == "LOCAL":
        return LocalController(cfg)

    if name == "HYBRID":
        return HybridController(cfg)

    raise ValueError(name)


CONTROLLERS = (
    "B1",
    "GLOBAL",
    "LOCAL",
    "HYBRID",
)


# ============================================================
# Task-sensitive fidelity
# ============================================================

def task_fidelity(
    phase: float,
) -> float:
    return float(
        np.cos(
            0.5 * phase
        ) ** 2
    )


def run_arm(
    cfg: Config,
    detuning: np.ndarray,
    probes: np.ndarray,
    controller,
    h: np.ndarray,
):
    """
    Task only changes h.

    Same physical residual vector.
    Same observations.
    Same controller.

    phase accumulates:
        phi += h^T residual * dt
    """

    phase = 0.0

    trace = np.zeros(
        cfg.steps
    )

    energy = 0.0

    for t in range(cfg.steps):
        u = controller.update(
            probes[t]
        )

        residual = (
            detuning[t]
            + u
        )

        phase += (
            np.dot(
                h,
                residual
            )
            * cfg.dt
        )

        trace[t] = (
            task_fidelity(
                phase
            )
        )

        energy += (
            np.dot(u, u)
            * cfg.dt
        )

    return {
        "F":
            float(trace[-1]),

        "FI":
            float(np.mean(trace)),

        "E":
            float(energy),
    }


# ============================================================
# TRAINING: estimate Sigma
# ============================================================

def estimate_training_covariance(
    cfg: Config,
    seeds: list[int],
):
    """
    Selector sees noisy telemetry only.

    We concatenate centered probe vectors
    across training trajectories/time.
    """

    samples = []

    for seed in seeds:
        seq = np.random.SeedSequence(
            seed
        )

        env_seq, probe_seq = (
            seq.spawn(2)
        )

        env_rng = (
            np.random.default_rng(
                env_seq
            )
        )

        probe_rng = (
            np.random.default_rng(
                probe_seq
            )
        )

        truth = (
            generate_environment(
                cfg,
                env_rng
            )
        )

        probes = (
            generate_probes(
                cfg,
                truth["detuning"],
                probe_rng,
            )
        )

        # Center in time per trajectory,
        # preserving cross-qubit covariance.
        centered = (
            probes
            - np.mean(
                probes,
                axis=0,
                keepdims=True,
            )
        )

        samples.append(
            centered
        )

    X = np.concatenate(
        samples,
        axis=0,
    )

    Sigma = np.cov(
        X,
        rowvar=False,
    )

    return Sigma


# ============================================================
# Decomposition + selection
# ============================================================

def decompose_covariance(
    cfg: Config,
    Sigma: np.ndarray,
    modes: dict[str, np.ndarray],
):
    """
    Simple frozen decomposition.

    Global covariance:
        projection onto known global direction.

    Local covariance:
        residual PSD approximation after removing
        global mode and isotropic floor.

    This is intentionally simple; v24 tests the
    task-weighting logic, not structure discovery.
    """

    g = modes["GLOBAL"]

    # Variance along global direction.
    lambda_g = float(
        g @ Sigma @ g
    )

    Sigma_g = (
        lambda_g
        * np.outer(g, g)
    )

    residual = (
        Sigma - Sigma_g
    )

    # Symmetrize numerical noise.
    residual = (
        0.5
        * (residual + residual.T)
    )

    # Isotropic floor estimated from smallest
    # eigenvalue, never negative.
    eigvals = np.linalg.eigvalsh(
        residual
    )

    sigma2 = max(
        0.0,
        float(np.min(eigvals))
    )

    Sigma_iso = (
        sigma2
        * np.eye(
            cfg.n_qubits
        )
    )

    Sigma_local_raw = (
        residual - Sigma_iso
    )

    # PSD projection.
    vals, vecs = np.linalg.eigh(
        Sigma_local_raw
    )

    vals = np.maximum(
        vals,
        0.0
    )

    Sigma_local = (
        vecs
        @ np.diag(vals)
        @ vecs.T
    )

    structured_trace = (
        np.trace(Sigma_g)
        + np.trace(Sigma_local)
    )

    total_trace = float(
        np.trace(Sigma)
    )

    q = (
        float(
            structured_trace
            / total_trace
        )
        if total_trace > 0
        else 0.0
    )

    # Numerical bound.
    q = float(
        np.clip(q, 0.0, 1.0)
    )

    return {
        "Sigma_global":
            Sigma_g,

        "Sigma_local":
            Sigma_local,

        "Sigma_iso":
            Sigma_iso,

        "lambda_global":
            lambda_g,

        "sigma2_iso":
            sigma2,

        "quality_q":
            q,
    }


def relevance(
    h: np.ndarray,
    decomposition: dict,
):
    Sg = decomposition[
        "Sigma_global"
    ]

    Sl = decomposition[
        "Sigma_local"
    ]

    Rg = float(
        h @ Sg @ h
    )

    Rl = float(
        h @ Sl @ h
    )

    denom = (
        Rg + Rl + 1e-15
    )

    return {
        "R_global": Rg,
        "R_local": Rl,
        "r_global":
            float(Rg / denom),
        "r_local":
            float(Rl / denom),
    }


def choose_controller(
    cfg: Config,
    q: float,
    rel: dict,
):
    if q < cfg.q_min:
        return "B1"

    if (
        rel["r_global"]
        >= cfg.relevance_threshold
    ):
        return "GLOBAL"

    if (
        rel["r_local"]
        >= cfg.relevance_threshold
    ):
        return "LOCAL"

    return "HYBRID"


# ============================================================
# Statistics
# ============================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    samples: int,
    alpha: float,
):
    rng = np.random.default_rng(
        seed
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    boot = np.empty(
        samples
    )

    for b in range(samples):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[b] = np.mean(
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
# Main
# ============================================================

def main():
    cfg = CFG

    source = Path(__file__).resolve()

    out = Path(
        cfg.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    modes = build_modes(
        cfg
    )

    print(
        "Q-IGL v24 FINAL H4 Test"
    )
    print(
        "======================="
    )

    print(
        "SOURCE_SHA256 =",
        sha256_file(source)
    )

    print(
        "CONFIG_SHA256 =",
        config_hash(cfg)
    )

    print(
        "GIT_COMMIT    =",
        git_commit()
    )

    # --------------------------------------------------------
    # Seeds are separated BEFORE training.
    # --------------------------------------------------------

    master = np.random.SeedSequence(
        cfg.seed
    )

    train_root, test_root = (
        master.spawn(2)
    )

    train_seq = train_root.spawn(
        cfg.train_trajectories
    )

    test_seq = test_root.spawn(
        cfg.test_trajectories
    )

    train_seeds = [
        int(
            s.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for s in train_seq
    ]

    test_seeds = [
        int(
            s.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for s in test_seq
    ]

    # ========================================================
    # PHASE 1: TRAIN
    # ========================================================

    print(
        "\nPHASE 1: TRAINING TELEMETRY"
    )

    Sigma_hat = (
        estimate_training_covariance(
            cfg,
            train_seeds,
        )
    )

    decomposition = (
        decompose_covariance(
            cfg,
            Sigma_hat,
            modes,
        )
    )

    selections = {}

    for task_name, h in modes.items():
        rel = relevance(
            h,
            decomposition,
        )

        selected = (
            choose_controller(
                cfg,
                decomposition[
                    "quality_q"
                ],
                rel,
            )
        )

        selections[
            task_name
        ] = {
            **rel,
            "selected_controller":
                selected,
        }

    selection_payload = {
        "config_sha256":
            config_hash(cfg),

        "quality_q":
            decomposition[
                "quality_q"
            ],

        "lambda_global":
            decomposition[
                "lambda_global"
            ],

        "sigma2_iso":
            decomposition[
                "sigma2_iso"
            ],

        "tasks":
            selections,
    }

    selection_path = (
        out / "selection.json"
    )

    selection_path.write_text(
        json.dumps(
            selection_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    selection_sha = (
        sha256_file(
            selection_path
        )
    )

    print(
        "TRAIN quality q =",
        f"{decomposition['quality_q']:.6f}"
    )

    for task, item in selections.items():
        print(
            f"{task:6s} -> "
            f"{item['selected_controller']:6s} | "
            f"rG={item['r_global']:.4f} "
            f"rL={item['r_local']:.4f}"
        )

    print(
        "SELECTION_SHA256 =",
        selection_sha
    )

    # Save covariance as training artifact.
    np.save(
        out / "sigma_hat.npy",
        Sigma_hat,
    )

    print(
        "SIGMA_SHA256     =",
        sha256_file(
            out / "sigma_hat.npy"
        )
    )

    print(
        "\nSelection frozen. "
        "Beginning TEST."
    )

    # ========================================================
    # PHASE 2: TEST
    # ========================================================

    rows = []

    for trajectory, seed in enumerate(
        test_seeds
    ):
        seq = np.random.SeedSequence(
            seed
        )

        env_seq, probe_seq = (
            seq.spawn(2)
        )

        env_rng = (
            np.random.default_rng(
                env_seq
            )
        )

        probe_rng = (
            np.random.default_rng(
                probe_seq
            )
        )

        truth = (
            generate_environment(
                cfg,
                env_rng
            )
        )

        probes = (
            generate_probes(
                cfg,
                truth["detuning"],
                probe_rng,
            )
        )

        row = {
            "trajectory":
                trajectory,
            "seed":
                seed,
        }

        # Same physical trajectory for
        # every task and every arm.
        for task_name, h in modes.items():

            for arm in CONTROLLERS:
                controller = (
                    make_controller(
                        arm,
                        cfg,
                    )
                )

                result = run_arm(
                    cfg,
                    truth[
                        "detuning"
                    ],
                    probes,
                    controller,
                    h,
                )

                prefix = (
                    f"{task_name}_{arm}"
                )

                row[
                    f"{prefix}_F"
                ] = result["F"]

                row[
                    f"{prefix}_FI"
                ] = result["FI"]

                row[
                    f"{prefix}_E"
                ] = result["E"]

        rows.append(row)

        if (
            trajectory + 1
        ) % 100 == 0:
            print(
                f"{trajectory + 1}/"
                f"{cfg.test_trajectories}"
            )

    # ---------------- CSV ----------------

    csv_path = (
        out / "test_trajectories.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # Analysis
    # ========================================================

    def col(name):
        return np.asarray(
            [
                float(r[name])
                for r in rows
            ],
            dtype=float,
        )

    task_results = {}

    policy_regrets = {
        "SELECTOR": [],
        "ALWAYS_B1": [],
        "ALWAYS_GLOBAL": [],
        "ALWAYS_LOCAL": [],
        "ALWAYS_HYBRID": [],
        "RANDOM_EXPECTED": [],
    }

    rng_random = (
        np.random.default_rng(
            cfg.seed + 777
        )
    )

    for task_name in modes:
        means = {}

        arrays = {}

        for arm in CONTROLLERS:
            arr = col(
                f"{task_name}_{arm}_F"
            )

            arrays[arm] = arr

            means[arm] = float(
                np.mean(arr)
            )

        oracle_arm = max(
            means,
            key=means.get,
        )

        oracle_f = (
            means[oracle_arm]
        )

        selected_arm = (
            selections[
                task_name
            ][
                "selected_controller"
            ]
        )

        selected_f = (
            means[selected_arm]
        )

        selector_regret = (
            oracle_f - selected_f
        )

        task_results[
            task_name
        ] = {
            "means": means,
            "oracle_arm":
                oracle_arm,
            "oracle_fidelity":
                oracle_f,
            "selected_arm":
                selected_arm,
            "selected_fidelity":
                selected_f,
            "selector_regret":
                selector_regret,
        }

        policy_regrets[
            "SELECTOR"
        ].append(
            selector_regret
        )

        policy_regrets[
            "ALWAYS_B1"
        ].append(
            oracle_f
            - means["B1"]
        )

        policy_regrets[
            "ALWAYS_GLOBAL"
        ].append(
            oracle_f
            - means["GLOBAL"]
        )

        policy_regrets[
            "ALWAYS_LOCAL"
        ].append(
            oracle_f
            - means["LOCAL"]
        )

        policy_regrets[
            "ALWAYS_HYBRID"
        ].append(
            oracle_f
            - means["HYBRID"]
        )

        random_mean = float(
            np.mean(
                list(
                    means.values()
                )
            )
        )

        policy_regrets[
            "RANDOM_EXPECTED"
        ].append(
            oracle_f
            - random_mean
        )

    mean_regrets = {
        policy: float(
            np.mean(values)
        )
        for policy, values
        in policy_regrets.items()
    }

    # --------------------------------------------------------
    # Bootstrap selector-vs-policy regret difference
    #
    # We bootstrap trajectories and recompute oracle means,
    # respecting pairing.
    # --------------------------------------------------------

    rng_boot = (
        np.random.default_rng(
            cfg.seed + 999
        )
    )

    comparators = (
        "ALWAYS_B1",
        "ALWAYS_GLOBAL",
        "ALWAYS_LOCAL",
        "ALWAYS_HYBRID",
        "RANDOM_EXPECTED",
    )

    bootstrap_diffs = {
        c: []
        for c in comparators
    }

    n = cfg.test_trajectories

    # Cache all arrays.
    cache = {}

    for task in modes:
        cache[task] = {
            arm: col(
                f"{task}_{arm}_F"
            )
            for arm in CONTROLLERS
        }

    for _ in range(
        cfg.bootstrap_samples
    ):
        idx = rng_boot.integers(
            0,
            n,
            size=n,
        )

        selector_regrets_b = []

        comparator_regrets_b = {
            c: []
            for c in comparators
        }

        for task in modes:
            means_b = {
                arm: float(
                    np.mean(
                        cache[
                            task
                        ][arm][idx]
                    )
                )
                for arm in CONTROLLERS
            }

            oracle_b = max(
                means_b.values()
            )

            selected_arm = (
                selections[
                    task
                ][
                    "selected_controller"
                ]
            )

            selector_regrets_b.append(
                oracle_b
                - means_b[
                    selected_arm
                ]
            )

            comparator_regrets_b[
                "ALWAYS_B1"
            ].append(
                oracle_b
                - means_b["B1"]
            )

            comparator_regrets_b[
                "ALWAYS_GLOBAL"
            ].append(
                oracle_b
                - means_b["GLOBAL"]
            )

            comparator_regrets_b[
                "ALWAYS_LOCAL"
            ].append(
                oracle_b
                - means_b["LOCAL"]
            )

            comparator_regrets_b[
                "ALWAYS_HYBRID"
            ].append(
                oracle_b
                - means_b["HYBRID"]
            )

            comparator_regrets_b[
                "RANDOM_EXPECTED"
            ].append(
                oracle_b
                - np.mean(
                    list(
                        means_b.values()
                    )
                )
            )

        sr = float(
            np.mean(
                selector_regrets_b
            )
        )

        for comp in comparators:
            cr = float(
                np.mean(
                    comparator_regrets_b[
                        comp
                    ]
                )
            )

            # Positive means selector has LOWER regret.
            bootstrap_diffs[
                comp
            ].append(
                cr - sr
            )

    comparator_ci = {}

    for comp in comparators:
        values = np.asarray(
            bootstrap_diffs[
                comp
            ]
        )

        comparator_ci[
            comp
        ] = {
            "mean_advantage":
                float(
                    np.mean(values)
                ),

            "ci95": [
                float(
                    np.quantile(
                        values,
                        cfg.ci_alpha / 2,
                    )
                ),
                float(
                    np.quantile(
                        values,
                        1 - cfg.ci_alpha / 2,
                    )
                ),
            ],
        }

    # ========================================================
    # Report
    # ========================================================

    print(
        "\nTEST RESULTS"
    )
    print(
        "============"
    )

    for task, result in (
        task_results.items()
    ):
        print(
            f"\nTask {task}"
        )
        print(
            "-" * (
                5 + len(task)
            )
        )

        for arm in CONTROLLERS:
            print(
                f"{arm:8s}: "
                f"{result['means'][arm]:.6f}"
            )

        print(
            "Selected :",
            result["selected_arm"]
        )

        print(
            "Oracle   :",
            result["oracle_arm"]
        )

        print(
            "Regret   :",
            f"{result['selector_regret']:.6f}"
        )

    print(
        "\nMEAN REGRET"
    )
    print(
        "==========="
    )

    for policy, value in (
        mean_regrets.items()
    ):
        print(
            f"{policy:16s}: "
            f"{value:.6f}"
        )

    print(
        "\nSELECTOR ADVANTAGE "
        "(comparator regret - selector regret)"
    )
    print(
        "=========================================="
    )

    for comp, info in (
        comparator_ci.items()
    ):
        lo, hi = info["ci95"]

        print(
            f"{comp:16s}: "
            f"{info['mean_advantage']:+.6f} "
            f"CI [{lo:+.6f}, {hi:+.6f}]"
        )

    summary = {
        "source_sha256":
            sha256_file(source),

        "config_sha256":
            config_hash(cfg),

        "git_commit":
            git_commit(),

        "selection_sha256":
            selection_sha,

        "csv_sha256":
            sha256_file(csv_path),

        "training":
            selection_payload,

        "task_results":
            task_results,

        "mean_regrets":
            mean_regrets,

        "selector_advantage":
            comparator_ci,
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

    print(
        "\nTEST_CSV_SHA256 =",
        sha256_file(csv_path)
    )

    print(
        "SUMMARY_SHA256  =",
        sha256_file(
            summary_path
        )
    )

    print(
        "\nFINAL EXPERIMENT COMPLETE."
    )
    print(
        "No automatic H4 verdict "
        "is generated by this program."
    )


if __name__ == "__main__":
    main()