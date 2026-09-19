"""
Q-IGL Digital Testbed v0.4 (Scale Test: 30 Qubits) - FIXED
"""
import numpy as np
import hashlib
import csv
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass
class Config:  # Убрали frozen=True, чтобы можно было задать g динамически
    n_qubits: int = 30
    steps: int = 50
    dt: float = 0.05
    trajectories: int = 100
    seed: int = 20260916
    
    sigma_local_process: float = 0.025
    sigma_common_process: float = 0.035
    rho_common: float = 0.98
    sigma_probe: float = 0.10
    
    g_before: tuple = None 
    g_after: tuple = None
    rho_after: float = 0.85
    shift_step: int = 25
    
    u_max: float = 2.0
    b1_alpha: float = 0.16
    k1_rho: float = 0.98
    k1_sigma_local: float = 0.025
    k1_sigma_common: float = 0.035
    
    wrong_rho: float = 0.50
    wrong_covariance_scale: float = 0.02

CFG = Config()

def generate_g_vectors(n, seed):
    rng = np.random.default_rng(seed)
    g_before = rng.uniform(0.5, 1.5, n)
    g_after = rng.uniform(0.5, 1.5, n)
    return tuple(g_before), tuple(g_after)

# Теперь это сработает
g_b, g_a = generate_g_vectors(CFG.n_qubits, CFG.seed)
CFG.g_before = g_b
CFG.g_after = g_a

# --- FORENSIC SELF-HASH ---
def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

# --- QUANTUM SYSTEM (Phase-Only Approximation for GHZ) ---
# For GHZ state |0...0> + |1...1>, dephasing noise Z_i adds phase to the coherence term.
# Fidelity = |<GHZ|psi>|^2 = cos^2(Phi_total / 2)
# Phi_total = sum(phi_i) where d(phi_i)/dt = detuning_i - control_i

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
        if regime == "B":
            rho = cfg.rho_common
            g[t] = g1
        elif regime == "C":
            if t < cfg.shift_step:
                rho = cfg.rho_common
                g[t] = g1
            else:
                rho = cfg.rho_after
                g[t] = g2
        
        # Local drift AR(1)
        if t == 0:
            local[t] = rng.normal(0.0, cfg.sigma_local_process, N)
        else:
            local[t] = 0.92 * local[t - 1] + rng.normal(0.0, cfg.sigma_local_process, N)
            
        # Common drift AR(1)
        if t == 0:
            common[t] = rng.normal(0.0, cfg.sigma_common_process)
        else:
            common[t] = rho * common[t - 1] + rng.normal(0.0, cfg.sigma_common_process)

    detuning = local + g * common[:, None]
    return detuning, g

def generate_probes(cfg, detuning, rng):
    noise = rng.normal(0.0, cfg.sigma_probe, size=detuning.shape)
    return detuning + noise

# --- CONTROLLERS ---

class B1Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.estimate = np.zeros(cfg.n_qubits)

    def update(self, observation):
        a = self.cfg.b1_alpha
        self.estimate = (1.0 - a) * self.estimate + a * observation
        return np.clip(-self.estimate, -self.cfg.u_max, self.cfg.u_max)

class K1Controller:
    def __init__(self, cfg, wrong=False):
        self.cfg = cfg
        N = cfg.n_qubits
        self.x = np.zeros(N + 1) # [delta_1..delta_N, z]
        
        # Initialize P matrix (Identity * scale)
        # For large N, we can use diagonal approximation for speed if needed, 
        # but let's try full for N=30 first.
        self.P = np.eye(N + 1) * 0.5
        
        if wrong:
            # Wrong model: random g, wrong rho
            rng_wrong = np.random.default_rng(123)
            self.g = rng_wrong.uniform(0.5, 1.5, N)
            self.rho = cfg.wrong_rho
        else:
            self.g = np.asarray(cfg.g_before, dtype=float)
            self.rho = cfg.k1_rho
            
        self.N = N

    def update(self, observation):
        cfg = self.cfg
        N = self.N
        
        # Prediction Step
        F = np.eye(N + 1)
        F[:N, :N] *= 0.92 # Local decay
        F[-1, -1] = self.rho # Global decay
        
        Q = np.diag([cfg.k1_sigma_local ** 2] * N + [cfg.k1_sigma_common ** 2])
        
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        
        # Measurement Model: y = Hx + v
        # y_i = delta_i + g_i * z
        H = np.zeros((N, N + 1))
        H[:, :N] = np.eye(N)
        H[:, -1] = self.g
        
        R = np.eye(N) * cfg.sigma_probe ** 2
        
        # Kalman Update
        innovation = observation - H @ x_pred
        S = H @ P_pred @ H.T + R
        
        # Solve for Kalman Gain KG = P_pred * H^T * S^-1
        # For N=30, S is 30x30. Inversion is O(N^3) which is fine for 30.
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        KG = P_pred @ H.T @ S_inv
        
        self.x = x_pred + KG @ innovation
        
        # Joseph form for P update
        I_mat = np.eye(N + 1)
        KH = KG @ H
        self.P = (I_mat - KH) @ P_pred @ (I_mat - KH).T + KG @ R @ KG.T
        
        # Estimate total detuning for control
        estimate = self.x[:N] + self.g * self.x[-1]
        
        return np.clip(-estimate, -cfg.u_max, cfg.u_max)

def run_trajectory(cfg, regime, seed, controller_type, wrong=False):
    ss = np.random.SeedSequence(seed)
    env_rng, probe_rng = [np.random.default_rng(s) for s in ss.spawn(2)]
    
    detuning, g_true = generate_environment(cfg, regime, env_rng)
    probes = generate_probes(cfg, detuning, probe_rng)
    
    if controller_type == 'B1':
        ctrl = B1Controller(cfg)
    elif controller_type == 'K1':
        ctrl = K1Controller(cfg, wrong=wrong)
        
    total_phase = 0.0
    integrated_f = 0.0
    energy = 0.0
    
    for t in range(cfg.steps):
        u = ctrl.update(probes[t])
        # Residual phase accumulation
        residual = detuning[t] + u
        total_phase += np.sum(residual) * cfg.dt
        
        f = ghz_fidelity(total_phase)
        integrated_f += f
        energy += np.sum(u ** 2) * cfg.dt
        
    return {
        "final_f": ghz_fidelity(total_phase),
        "int_f": integrated_f / cfg.steps,
        "energy": energy
    }

# --- MAIN EXPERIMENT ---
if __name__ == "__main__":
    SOURCE_HASH = get_self_hash()
    print(f"SOURCE_SHA256 = {SOURCE_HASH}")
    print(f"N_QUBITS = {CFG.n_qubits}")
    print(f"TRAJECTORIES = {CFG.trajectories}")
    
    with open("source_hash_v04.txt", "w") as hf:
        hf.write(SOURCE_HASH + "\n")
        
    conditions = [
        ("B", False), # Correlated Drift
        ("C", False), # Regime Shift
    ]
    
    master = np.random.SeedSequence(CFG.seed)
    all_seeds = master.spawn(len(conditions) * CFG.trajectories)
    seed_idx = 0
    
    csv_file = "primary_results_v04_scale.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition','regime','wrong','trajectory','seed','B1_F','K1_F','B1_int','K1_int','B1_e','K1_e'])
        
        for cond_idx, (regime, wrong) in enumerate(conditions):
            label = f"{regime}_{'Wrong' if wrong else 'Normal'}"
            seeds_chunk = [int(all_seeds[seed_idx + i].generate_state(1, dtype=np.uint64)[0]) for i in range(CFG.trajectories)]
            seed_idx += CFG.trajectories
            
            print(f"Starting Condition {label}...")
            
            b1_scores = []
            k1_scores = []
            
            for traj_idx, seed in enumerate(seeds_chunk):
                res_b1 = run_trajectory(CFG, regime, seed, 'B1')
                res_k1 = run_trajectory(CFG, regime, seed, 'K1', wrong=wrong)
                
                b1_scores.append(res_b1['final_f'])
                k1_scores.append(res_k1['final_f'])
                
                writer.writerow([
                    cond_idx, regime, wrong, traj_idx, seed,
                    f"{res_b1['final_f']:.6f}", f"{res_k1['final_f']:.6f}",
                    f"{res_b1['int_f']:.6f}",  f"{res_k1['int_f']:.6f}",
                    f"{res_b1['energy']:.6f}", f"{res_k1['energy']:.6f}",
                ])
                
            mean_b1 = np.mean(b1_scores)
            mean_k1 = np.mean(k1_scores)
            delta = mean_k1 - mean_b1
            
            print(f"  Mean B1 Fidelity: {mean_b1:.6f}")
            print(f"  Mean K1 Fidelity: {mean_k1:.6f}")
            print(f"  Delta (K1-B1):    {delta:+.6f}")
            
    print(f"Results saved to {csv_file}")