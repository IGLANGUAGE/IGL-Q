"""
Q-IGL Digital Testbed v0.2 (Forensic Ready)
H1 + Ablation (Reset vs Persistent)
"""
import numpy as np
import hashlib
import csv
import os
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass(frozen=True)
class Config:
    n_qubits: int = 3
    steps: int = 100
    dt: float = 0.05
    trajectories: int = 1000
    seed: int = 20260916
    sigma_local_process: float = 0.025
    sigma_common_process: float = 0.035
    rho_common: float = 0.98
    sigma_probe: float = 0.10
    g_before: tuple = (1.0, 0.8, 1.2)
    g_after: tuple = (0.55, 1.35, 0.75)
    rho_after: float = 0.85
    shift_step: int = 50
    u_max: float = 2.0
    b1_alpha: float = 0.16
    k1_rho: float = 0.98
    k1_sigma_local: float = 0.025
    k1_sigma_common: float = 0.035
    wrong_rho: float = 0.50
    wrong_g: tuple = (-0.4, 1.8, 0.25)
    wrong_covariance_scale: float = 0.02

CFG = Config()

# --- FORENSIC SELF-HASH ---
def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

# --- QUANTUM SYSTEM ---
def ghz_fidelity(total_phase):
    return float(np.cos(0.5 * total_phase) ** 2)

def generate_environment(cfg, regime, rng):
    T, N = cfg.steps, cfg.n_qubits
    local = np.zeros((T, N))
    common = np.zeros(T)
    g = np.zeros((T, N))
    g1 = np.asarray(cfg.g_before, dtype=float)
    g2 = np.asarray(cfg.g_after, dtype=float)

    for t in range(T):
        if regime == "A":
            rho = 0.0; g[t] = 0.0
        elif regime == "B":
            rho = cfg.rho_common; g[t] = g1
        elif regime == "C":
            if t < cfg.shift_step: rho = cfg.rho_common; g[t] = g1
            else: rho = cfg.rho_after; g[t] = g2
        
        if t == 0: local[t] = rng.normal(0.0, cfg.sigma_local_process, N)
        else: local[t] = 0.92 * local[t - 1] + rng.normal(0.0, cfg.sigma_local_process, N)
        
        if regime == "A": common[t] = 0.0
        elif t == 0: common[t] = rng.normal(0.0, cfg.sigma_common_process)
        else: common[t] = rho * common[t - 1] + rng.normal(0.0, cfg.sigma_common_process)

    detuning = local + g * common[:, None]
    return local, common, g, detuning

def generate_probes(cfg, detuning, rng):
    noise = rng.normal(0.0, cfg.sigma_probe, size=detuning.shape)
    return detuning + noise

# --- CONTROLLERS ---
class B1Controller:
    def __init__(self, cfg):
        self.cfg = cfg; self.estimate = np.zeros(cfg.n_qubits)
    def update(self, observation):
        a = self.cfg.b1_alpha
        self.estimate = (1.0 - a) * self.estimate + a * observation
        return np.clip(-self.estimate, -self.cfg.u_max, self.cfg.u_max)

class K1Controller:
    def __init__(self, cfg, wrong=False):
        self.cfg = cfg; N = cfg.n_qubits
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

    def reset_state(self):
        """Для ablation: сброс памяти между эпизодами"""
        self.x = np.zeros(len(self.x))
        self.P = np.eye(len(self.x)) * 0.5

    def update(self, observation):
        cfg = self.cfg; N = cfg.n_qubits
        F = np.eye(N + 1); F[:N, :N] *= 0.92; F[-1, -1] = self.rho
        Q = np.diag([cfg.k1_sigma_local ** 2] * N + [cfg.k1_sigma_common ** 2])
        x_pred = F @ self.x; P_pred = F @ self.P @ F.T + Q
        H = np.zeros((N, N + 1)); H[:, :N] = np.eye(N); H[:, -1] = self.g
        R = np.eye(N) * cfg.sigma_probe ** 2
        innovation = observation - H @ x_pred
        S = H @ P_pred @ H.T + R
        KG = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + KG @ innovation
        I_mat = np.eye(N + 1); KH = KG @ H
        self.P = (I_mat - KH) @ P_pred @ (I_mat - KH).T + KG @ R @ KG.T
        estimate = self.x[:N] + self.g * self.x[-1]
        return np.clip(-estimate, -cfg.u_max, cfg.u_max)

def run_trajectory(cfg, regime, seed, controller_type, wrong=False):
    ss = np.random.SeedSequence(seed)
    env_rng, probe_rng = [np.random.default_rng(s) for s in ss.spawn(2)]
    _, _, _, detuning = generate_environment(cfg, regime, env_rng)
    probes = generate_probes(cfg, detuning, probe_rng)
    
    ctrl = K1Controller(cfg, wrong=wrong) if controller_type.startswith('K1') else B1Controller(cfg)
    
    # Исправление: сбрасываем состояние только если это K1_reset
    if controller_type == 'K1_reset' and hasattr(ctrl, 'reset_state'):
        ctrl.reset_state()

    total_phase = 0.0; integrated_f = 0.0; energy = 0.0
    for t in range(cfg.steps):
        u = ctrl.update(probes[t])
        residual = detuning[t] + u
        total_phase += np.sum(residual) * cfg.dt
        f = ghz_fidelity(total_phase)
        integrated_f += f; energy += np.sum(u ** 2) * cfg.dt
    
    return {
        "final_f": ghz_fidelity(total_phase),
        "int_f": integrated_f / cfg.steps,
        "energy": energy
    }

# --- MAIN EXPERIMENT ---
if __name__ == "__main__":
    SOURCE_HASH = get_self_hash()
    print(f"SOURCE_SHA256 = {SOURCE_HASH}")
    
    conditions = [
        ("A", False, False),      # Regime A, Normal, Persistent
        ("B", False, False),      # Regime B, Normal, Persistent
        ("C", False, False),      # Regime C, Normal, Persistent
        ("B", True, False),       # Regime B, Wrong Model, Persistent
        ("B", False, True),       # Regime B, Normal, RESET (Ablation)
    ]
    
    master = np.random.SeedSequence(CFG.seed)
    all_seeds = master.spawn(len(conditions) * CFG.trajectories)
    seed_idx = 0
    
    csv_file = "primary_results_v02.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition','regime','wrong','reset','trajectory','seed','B1_F','K1_F','B1_int','K1_int','B1_e','K1_e'])
        
        for cond_idx, (regime, wrong, is_reset) in enumerate(conditions):
            label = f"{regime}_{'Wrong' if wrong else 'Normal'}_{'Reset' if is_reset else 'Persist'}"
            seeds_chunk = [int(all_seeds[seed_idx + i].generate_state(1, dtype=np.uint64)[0]) for i in range(CFG.trajectories)]
            seed_idx += CFG.trajectories
            
            print(f"Condition {label}: first_5_seeds = {seeds_chunk[:5]}")
            
            for traj_idx, seed in enumerate(seeds_chunk):
                res_b1 = run_trajectory(CFG, regime, seed, 'B1')
                ctrl_type = 'K1_reset' if is_reset else 'K1'
                res_k1 = run_trajectory(CFG, regime, seed, ctrl_type, wrong=wrong)
                
                writer.writerow([
                    cond_idx, regime, wrong, is_reset, traj_idx, seed,
                    f"{res_b1['final_f']:.6f}", f"{res_k1['final_f']:.6f}",
                    f"{res_b1['int_f']:.6f}", f"{res_k1['int_f']:.6f}",
                    f"{res_b1['energy']:.6f}", f"{res_k1['energy']:.6f}"
                ])
    
    print(f"Results saved to {csv_file}")