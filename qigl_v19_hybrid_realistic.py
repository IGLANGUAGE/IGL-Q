"""
Q-IGL Digital Testbed v2.0 (Hybrid Realistic Model)
Focus: Multi-scale noise control (Global Drift + Local Fluctuations).
System: 20 Qubits with complex crosstalk topology.
"""
import numpy as np
import hashlib
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass
class Config:
    n_qubits: int = 20
    steps: int = 40
    dt: float = 0.05
    trajectories: int = 150
    seed: int = 20260916
    
    # Noise Parameters
    sigma_global: float = 0.08   # Slow global drift
    rho_global: float = 0.99     # High correlation in time
    sigma_local: float = 0.03    # Fast local noise
    
    # Crosstalk Topology (Adjacency-like weights)
    crosstalk_strength: float = 0.05
    
    # Control Params
    u_max: float = 2.0
    pid_kp: float = 0.5          # Proportional gain for local control
    igl_lr: float = 0.05         # Learning rate for global model

CFG = Config()

def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def generate_crosstalk_matrix(n):
    """Generate a realistic sparse crosstalk matrix."""
    W = np.zeros((n, n))
    for i in range(n):
        # Neighbors have strong crosstalk
        if i > 0: W[i, i-1] = CFG.crosstalk_strength
        if i < n-1: W[i, i+1] = CFG.crosstalk_strength
        # Random long-range weak crosstalk
        if i > 2: W[i, i-3] = CFG.crosstalk_strength * 0.2
    return W

W_MATRIX = generate_crosstalk_matrix(CFG.n_qubits)

def generate_environment(cfg, rng):
    T, N = cfg.steps, cfg.n_qubits
    global_drift = np.zeros(T)
    local_noise = np.zeros((T, N))
    
    for t in range(T):
        if t == 0:
            global_drift[t] = rng.normal(0, cfg.sigma_global)
            local_noise[t] = rng.normal(0, cfg.sigma_local, N)
        else:
            global_drift[t] = cfg.rho_global * global_drift[t-1] + rng.normal(0, cfg.sigma_global * 0.1)
            local_noise[t] = 0.5 * local_noise[t-1] + rng.normal(0, cfg.sigma_local, N)
            
    return global_drift, local_noise

class BaselineController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_local = np.zeros(cfg.n_qubits)
        
    def step(self, phase_errors, raw_probe):
        # Simple local PID-like control
        error = raw_probe
        self.est_local = self.est_local + self.cfg.pid_kp * error * self.cfg.dt
        return np.clip(-self.est_local, -self.cfg.u_max, self.cfg.u_max)

class HybridIGLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_global_z = 0.0
        self.est_local = np.zeros(cfg.n_qubits)
        
    def step(self, phase_errors, raw_probe):
        # 1. Global IGL: Estimate common mode drift from mean of all probes
        mean_probe = np.mean(raw_probe)
        self.est_global_z = (1 - self.cfg.igl_lr) * self.est_global_z + self.cfg.igl_lr * mean_probe
        
        # 2. Local Control: Compensate residual after global subtraction
        residual = raw_probe - self.est_global_z
        self.est_local = self.est_local + self.cfg.pid_kp * residual * self.cfg.dt
        
        # 3. Combine: Global compensation + Local fine-tuning
        total_correction = -self.est_global_z - self.est_local
        return np.clip(total_correction, -self.cfg.u_max, self.cfg.u_max)

def apply_crosstalk(phase_errors, W):
    """Apply crosstalk influence based on weight matrix."""
    return phase_errors + W @ phase_errors

def run_trajectory(cfg, seed, ctrl_type):
    ss = np.random.SeedSequence(seed)
    env_rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    global_drift, local_noise = generate_environment(cfg, env_rng)
    phase_errors = np.zeros(cfg.n_qubits)
    
    ctrl = BaselineController(cfg) if ctrl_type == 'B1' else HybridIGLController(cfg)
    
    fidelities = []
    
    for t in range(cfg.steps):
        # 1. Natural Evolution
        detuning = global_drift[t] + local_noise[t]
        phase_errors += detuning * cfg.dt
        
        # 2. Crosstalk
        phase_errors = apply_crosstalk(phase_errors, W_MATRIX)
        
        # 3. Noisy Probe
        probe_noise = np.random.normal(0, 0.01, cfg.n_qubits)
        raw_probe = phase_errors + probe_noise
        
        # 4. Control Action
        u = ctrl.step(phase_errors, raw_probe)
        phase_errors += u * cfg.dt
        
        # 5. Fidelity Calculation (GHZ-like metric for simplicity)
        # Penalize variance and mean deviation
        var_err = np.var(phase_errors)
        mean_err = np.mean(np.abs(phase_errors))
        f = np.exp(-10 * var_err - 5 * mean_err)
        fidelities.append(f)
        
    return np.mean(fidelities)

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1 = []
    res_hybrid = []
    
    print(f"Starting Hybrid Realistic Test (20 Qubits, {CFG.trajectories} trajectories)...")
    
    for i in range(CFG.trajectories):
        s_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s_hyb = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        f_b1 = run_trajectory(CFG, s_b1, 'B1')
        f_hyb = run_trajectory(CFG, s_hyb, 'HYBRID')
        
        res_b1.append(f_b1)
        res_hybrid.append(f_hyb)
        
        if (i+1) % 30 == 0:
            print(f"  {i+1}/{CFG.trajectories}")

    m_b1 = np.mean(res_b1)
    m_hyb = np.mean(res_hybrid)
    
    print(f"\nResults:")
    print(f"Baseline (B1) Mean Fidelity : {m_b1:.6f}")
    print(f"Hybrid-IGL Mean Fidelity    : {m_hyb:.6f}")
    print(f"Delta (Hybrid-B1)           : {m_hyb - m_b1:+.6f}")
    
    if m_hyb > m_b1:
        print("✅ Hybrid IGL successfully handles multi-scale noise!")
    else:
        print("⚠️ No significant improvement in realistic model.")