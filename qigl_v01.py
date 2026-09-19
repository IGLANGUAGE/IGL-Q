"""
Q-IGL Digital Testbed v0.1
H1: persistent correlated device model vs independent local calibration.

Requires:
    numpy

Run:
    python qigl_v01.py

Design contract:
- 3 qubits
- target: GHZ
- paired Monte Carlo trajectories
- same probe data and control budget for B0/B1/K1
- controller never receives ground-truth local/common drift
- regimes A/B/C
- D_wrong condition
"""

from dataclasses import dataclass
import numpy as np


# ============================================================
# Frozen configuration
# ============================================================

@dataclass(frozen=True)
class Config:
    n_qubits: int = 3
    steps: int = 100
    dt: float = 0.05

    # Monte Carlo: specification target = 1000.
    # Set lower during development if desired.
    trajectories: int = 1000
    seed: int = 20260916

    # Physical noise
    sigma_local_process: float = 0.025
    sigma_common_process: float = 0.035
    rho_common: float = 0.98

    # Probe noise: identical observations supplied to B1/K1
    sigma_probe: float = 0.10

    # Sensitivity to shared latent drift
    g_before: tuple = (1.0, 0.8, 1.2)

    # Regime C after t*=50
    g_after: tuple = (0.55, 1.35, 0.75)
    rho_after: float = 0.85
    shift_step: int = 50

    # Controller
    u_max: float = 2.0

    # B1 independent exponential filter
    b1_alpha: float = 0.16

    # K1 assumed state-space model
    k1_rho: float = 0.98
    k1_sigma_local: float = 0.025
    k1_sigma_common: float = 0.035

    # Wrong D
    wrong_rho: float = 0.50
    wrong_g: tuple = (-0.4, 1.8, 0.25)
    wrong_covariance_scale: float = 0.02


CFG = Config()


# ============================================================
# Quantum target: GHZ
#
# Under Z detuning only, populations stay unchanged.
# GHZ coherence acquires phase phi = integral sum_i detuning_i dt.
#
# F_GHZ = |<GHZ|psi>|^2 = cos^2(phi / 2)
#
# This is analytically equivalent to density-matrix evolution for
# this deliberately narrow v0.1 experiment and makes 1000 paired
# trajectories cheap.
# ============================================================

def ghz_fidelity(total_phase: float) -> float:
    return float(np.cos(0.5 * total_phase) ** 2)


# ============================================================
# Ground-truth environment
# ============================================================

def generate_environment(cfg: Config, regime: str, rng: np.random.Generator):
    """
    Returns ground truth:
        local[t, i]
        common[t]
        g[t, i]
        true_detuning[t, i]

    Controllers MUST NOT receive these arrays.
    """
    T, N = cfg.steps, cfg.n_qubits

    local = np.zeros((T, N))
    common = np.zeros(T)
    g = np.zeros((T, N))

    g1 = np.asarray(cfg.g_before, dtype=float)
    g2 = np.asarray(cfg.g_after, dtype=float)

    for t in range(T):
        if regime == "A":
            rho = 0.0
            g[t] = 0.0

        elif regime == "B":
            rho = cfg.rho_common
            g[t] = g1

        elif regime == "C":
            if t < cfg.shift_step:
                rho = cfg.rho_common
                g[t] = g1
            else:
                rho = cfg.rho_after
                g[t] = g2
        else:
            raise ValueError(f"Unknown regime {regime}")

        # Independent local AR-like drift
        if t == 0:
            local[t] = rng.normal(0.0, cfg.sigma_local_process, N)
        else:
            local[t] = (
                0.92 * local[t - 1]
                + rng.normal(0.0, cfg.sigma_local_process, N)
            )

        # Shared latent drift
        if regime == "A":
            common[t] = 0.0
        elif t == 0:
            common[t] = rng.normal(0.0, cfg.sigma_common_process)
        else:
            common[t] = (
                rho * common[t - 1]
                + rng.normal(0.0, cfg.sigma_common_process)
            )

    detuning = local + g * common[:, None]
    return local, common, g, detuning


def generate_probes(cfg: Config, detuning, rng):
    """
    Equal measurement budget for B1 and K1.

    probe[t,i] = noisy local observation of total detuning.

    This is the ONLY environment information exposed to controllers.
    """
    noise = rng.normal(
        0.0,
        cfg.sigma_probe,
        size=detuning.shape
    )
    return detuning + noise


# ============================================================
# Controllers
# ============================================================

class B0Controller:
    """No adaptation, no persistent D."""

    def __init__(self, cfg):
        self.cfg = cfg

    def update(self, observation):
        return np.zeros(self.cfg.n_qubits)


class B1IndependentController:
    """
    Independent local estimator.
    No representation of a shared latent variable z.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.estimate = np.zeros(cfg.n_qubits)

    def update(self, observation):
        a = self.cfg.b1_alpha
        self.estimate = (
            (1.0 - a) * self.estimate
            + a * observation
        )
        return np.clip(
            -self.estimate,
            -self.cfg.u_max,
            self.cfg.u_max
        )


class K1CorrelatedController:
    """
    Q-IGL D = (x_hat, P, model parameters).

    Hidden model state:
        x = [d1, d2, d3, z]

    observation_i = d_i + g_i * z + measurement_noise

    This controller NEVER receives true d_i or true z.
    """

    def __init__(self, cfg, wrong=False):
        self.cfg = cfg
        N = cfg.n_qubits

        self.x = np.zeros(N + 1)

        if wrong:
            self.g = np.asarray(cfg.wrong_g, dtype=float)
            self.rho = cfg.wrong_rho
            self.P = np.eye(N + 1) * cfg.wrong_covariance_scale
        else:
            self.g = np.asarray(cfg.g_before, dtype=float)
            self.rho = cfg.k1_rho
            self.P = np.eye(N + 1) * 0.5

        self.wrong = wrong

    def update(self, observation):
        cfg = self.cfg
        N = cfg.n_qubits

        # ---------- L: predict D ----------
        F = np.eye(N + 1)
        F[:N, :N] *= 0.92
        F[-1, -1] = self.rho

        Q = np.diag(
            [cfg.k1_sigma_local ** 2] * N
            + [cfg.k1_sigma_common ** 2]
        )

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Measurement model y_i = d_i + g_i*z
        H = np.zeros((N, N + 1))
        H[:, :N] = np.eye(N)
        H[:, -1] = self.g

        R = np.eye(N) * cfg.sigma_probe ** 2

        # ---------- Kalman update ----------
        innovation = observation - H @ x_pred
        S = H @ P_pred @ H.T + R

        KG = P_pred @ H.T @ np.linalg.inv(S)

        self.x = x_pred + KG @ innovation

        # Joseph form: numerically safer covariance update
        I = np.eye(N + 1)
        KH = KG @ H
        self.P = (
            (I - KH) @ P_pred @ (I - KH).T
            + KG @ R @ KG.T
        )

        # Estimated total physical detuning
        estimate = self.x[:N] + self.g * self.x[-1]

        # ---------- K: context/control selection ----------
        return np.clip(
            -estimate,
            -cfg.u_max,
            cfg.u_max
        )


# ============================================================
# Single paired trajectory
# ============================================================

def run_controller(cfg, detuning, probes, controller):
    total_phase = 0.0
    integrated_fidelity = 0.0
    control_energy = 0.0

    fidelity_trace = np.zeros(cfg.steps)

    for t in range(cfg.steps):
        # The controller sees probes[t], never ground truth.
        u = controller.update(probes[t])

        # Environment applies actual hidden detuning.
        residual = detuning[t] + u

        # GHZ relative phase: sum of local Z detunings.
        total_phase += np.sum(residual) * cfg.dt

        f = ghz_fidelity(total_phase)
        fidelity_trace[t] = f
        integrated_fidelity += f
        control_energy += np.sum(u ** 2) * cfg.dt

    return {
        "final_fidelity": fidelity_trace[-1],
        "integrated_fidelity": integrated_fidelity / cfg.steps,
        "control_energy": control_energy,
        "trace": fidelity_trace,
    }


def run_paired_trajectory(cfg, regime, seed, wrong=False):
    """
    Same physical environment and probes for every controller.
    """
    ss = np.random.SeedSequence(seed)
    env_rng, probe_rng = [
        np.random.default_rng(s)
        for s in ss.spawn(2)
    ]

    _, _, _, detuning = generate_environment(
        cfg, regime, env_rng
    )
    probes = generate_probes(
        cfg, detuning, probe_rng
    )

    b0 = run_controller(
        cfg, detuning, probes,
        B0Controller(cfg)
    )

    b1 = run_controller(
        cfg, detuning, probes,
        B1IndependentController(cfg)
    )

    k1 = run_controller(
        cfg, detuning, probes,
        K1CorrelatedController(cfg, wrong=wrong)
    )

    return b0, b1, k1


# ============================================================
# Statistics
# ============================================================

def bootstrap_ci(values, rng, n_boot=3000, alpha=0.05):
    values = np.asarray(values)
    n = len(values)

    means = np.empty(n_boot)

    for k in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[k] = np.mean(sample)

    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)

    return float(lo), float(hi)


def paired_effect_size(diff):
    diff = np.asarray(diff)
    sd = np.std(diff, ddof=1)

    if sd == 0:
        return np.inf if np.mean(diff) != 0 else 0.0

    return float(np.mean(diff) / sd)


def summarize(name, b1_values, k1_values, seed):
    b1_values = np.asarray(b1_values)
    k1_values = np.asarray(k1_values)

    diff = k1_values - b1_values

    rng = np.random.default_rng(seed)
    ci = bootstrap_ci(diff, rng)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"B1 mean F      : {np.mean(b1_values):.6f}")
    print(f"K1 mean F      : {np.mean(k1_values):.6f}")
    print(f"mean ΔF        : {np.mean(diff):+.6f}")
    print(f"median ΔF      : {np.median(diff):+.6f}")
    print(f"95% bootstrap  : [{ci[0]:+.6f}, {ci[1]:+.6f}]")
    print(f"paired effect d: {paired_effect_size(diff):+.3f}")
    print(f"P(ΔF > 0)      : {np.mean(diff > 0):.3f}")


# ============================================================
# Frozen H1 experiment
# ============================================================

def experiment(cfg):
    master = np.random.SeedSequence(cfg.seed)

    conditions = [
        ("A", False),
        ("B", False),
        ("C", False),
        ("B", True),
        ("C", True),
    ]

    child_seeds = master.spawn(
        len(conditions) * cfg.trajectories
    )

    seed_cursor = 0

    print("Q-IGL Digital Testbed v0.1")
    print("===========================")
    print(f"trajectories/condition: {cfg.trajectories}")
    print(f"steps                 : {cfg.steps}")
    print(f"master seed           : {cfg.seed}")

    for condition_index, (regime, wrong) in enumerate(conditions):
        b1_f = []
        k1_f = []

        b1_integrated = []
        k1_integrated = []

        b1_energy = []
        k1_energy = []

        for _ in range(cfg.trajectories):
            seed = int(
                child_seeds[seed_cursor]
                .generate_state(1, dtype=np.uint64)[0]
            )
            seed_cursor += 1

            _, b1, k1 = run_paired_trajectory(
                cfg,
                regime,
                seed,
                wrong=wrong
            )

            b1_f.append(b1["final_fidelity"])
            k1_f.append(k1["final_fidelity"])

            b1_integrated.append(
                b1["integrated_fidelity"]
            )
            k1_integrated.append(
                k1["integrated_fidelity"]
            )

            b1_energy.append(b1["control_energy"])
            k1_energy.append(k1["control_energy"])

        label = (
            f"Regime {regime}"
            + (" / D_wrong" if wrong else "")
        )

        summarize(
            label,
            b1_f,
            k1_f,
            cfg.seed + condition_index + 1
        )

        print(
            "Integrated F   : "
            f"B1={np.mean(b1_integrated):.6f}, "
            f"K1={np.mean(k1_integrated):.6f}"
        )

        print(
            "Control energy : "
            f"B1={np.mean(b1_energy):.6f}, "
            f"K1={np.mean(k1_energy):.6f}"
        )


# ============================================================
# Anti-cheating structural test
# ============================================================

def test_no_cheating_interface():
    """
    Controllers accept only observations.
    Ground-truth detuning/common/local never enters update().
    """
    import inspect

    for controller in (
        B0Controller,
        B1IndependentController,
        K1CorrelatedController,
    ):
        sig = inspect.signature(controller.update)
        params = list(sig.parameters.keys())

        assert params == ["self", "observation"], (
            controller.__name__,
            params
        )


if __name__ == "__main__":
    test_no_cheating_interface()
    experiment(CFG)