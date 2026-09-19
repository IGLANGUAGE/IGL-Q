"""
Q-IGL Digital Testbed v2.0 (Scale to 100 Qubits)
Focus: Hybrid Multi-Scale Architecture at Large Scale.
System: 100 Qubits with complex crosstalk topology.
"""
import numpy as np
import hashlib
from dataclasses import dataclass
import time

# --- CONFIGURATION ---
@dataclass
class Config:
    n_qubits: int = 100
    steps: int = 30     # Reduced steps for speed
    dt: float = 0.05
    trajectories: int = 50 # Reduced trajectories for speed
    seed: int = 20260916
    
    # Noise Parameters
    sigma_global: float = 0.08
    rho_global: float = 0.99
    sigma_local: float = 0.03
    
    # Crosstalk Topology
    crosstalk_strength: float = 0.02 # Reduced strength to prevent immediate blow-up
    
    # Control Params
    u_max: float = 2.0
    pid_kp: float = 0.5
    igl_lr: float = 0.05

CFG = Config()

def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def generate_crosstalk_matrix(n):
    """Generate a sparse crosstalk matrix using vectorized operations."""
    W = np.zeros((n, n))
    indices = np.arange(n)
    
    # Neighbors
    mask_right = indices < n - 1
    W[indices[mask_right], indices[mask_right] + 1] = CFG.crosstalk_strength
    mask_left = indices > 0
    W[indices[mask_left], indices[mask_left] - 1] = CFG.crosstalk_strength
    
    # Weak long-range (every 5th qubit)
    mask_long = indices < n - 5
    W[indices[mask_long], indices[mask_long] + 5] = CFG.crosstalk_strength * 0.1
    
    return W

W_MATRIX = generate_crosstalk_matrix(CFG.n_qubits)

def generate_environment(cfg, rng):
    T, N = cfg.steps, cfg.n_qubits
    global_drift = np.zeros(T)
    local_noise = np.zeros((T, N))
    
    # Vectorized generation for speed
    white_noise_g = rng.normal(0, cfg.sigma_global * 0.1, T)
    white_noise_l = rng.normal(0, cfg.sigma_local, (T, N))
    
    global_drift[0] = rng.normal(0, cfg.sigma_global)
    local_noise[0] = rng.normal(0, cfg.sigma_local, N)
    
    for t in range(1, T):
        global_drift[t] = cfg.rho_global * global_drift[t-1] + white_noise_g[t]
        local_noise[t] = 0.5 * local_noise[t-1] + white_noise_l[t]
            
    return global_drift, local_noise

class BaselineController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_local = np.zeros(cfg.n_qubits)
        
    def step(self, phase_errors, raw_probe):
        error = raw_probe
        self.est_local += self.cfg.pid_kp * error * self.cfg.dt
        return np.clip(-self.est_local, -self.cfg.u_max, self.cfg.u_max)

class HybridIGLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_global_z = 0.0
        self.est_local = np.zeros(cfg.n_qubits)
        
    def step(self, phase_errors, raw_probe):
        mean_probe = np.mean(raw_probe)
        self.est_global_z = (1 - self.cfg.igl_lr) * self.est_global_z + self.cfg.igl_lr * mean_probe
        
        residual = raw_probe - self.est_global_z
        self.est_local += self.cfg.pid_kp * residual * self.cfg.dt
        
        total_correction = -self.est_global_z - self.est_local
        return np.clip(total_correction, -self.cfg.u_max, self.cfg.u_max)

def run_trajectory(cfg, seed, ctrl_type):
    ss = np.random.SeedSequence(seed)
    env_rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    global_drift, local_noise = generate_environment(cfg, env_rng)
    phase_errors = np.zeros(cfg.n_qubits)
    
    ctrl = BaselineController(cfg) if ctrl_type == 'B1' else HybridIGLController(cfg)
    
    fidelities = []
    
    for t in range(cfg.steps):
        detuning = global_drift[t] + local_noise[t]
        phase_errors += detuning * cfg.dt
        
        # Vectorized crosstalk application
        phase_errors = phase_errors + W_MATRIX @ phase_errors
        
        probe_noise = np.random.normal(0, 0.01, cfg.n_qubits)
        raw_probe = phase_errors + probe_noise
        
        u = ctrl.step(phase_errors, raw_probe)
        phase_errors += u * cfg.dt
        
        var_err = np.var(phase_errors)
        mean_err = np.mean(np.abs(phase_errors))
        f = np.exp(-10 * var_err - 5 * mean_err)
        fidelities.append(f)
        
    return np.mean(fidelities)

if __name__ == "__main__":
    start_time = time.time()
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    print(f"Starting 100-Qubit Hybrid Test ({CFG.trajectories} trajectories)...")
    print("This may take several minutes...")
    
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1 = []
    res_hybrid = []
    
    for i in range(CFG.trajectories):
        s_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s_hyb = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        f_b1 = run_trajectory(CFG, s_b1, 'B1')
        f_hyb = run_trajectory(CFG, s_hyb, 'HYBRID')
        
        res_b1.append(f_b1)
        res_hybrid.append(f_hyb)
        
        elapsed = time.time() - start_time
        print(f"  {i+1}/{CFG.trajectories} | Est. remaining: {elapsed/(i+1)*(CFG.trajectories-i-1):.0f}s")

    m_b1 = np.mean(res_b1)
    m_hyb = np.mean(res_hybrid)
    
    print(f"\nResults:")
    print(f"Baseline (B1) Mean Fidelity : {m_b1:.6f}")
    print(f"Hybrid-IGL Mean Fidelity    : {m_hyb:.6f}")
    print(f"Delta (Hybrid-B1)           : {m_hyb - m_b1:+.6f}")
    
    if m_hyb > m_b1:
        print("✅ Hybrid IGL scales successfully to 100 qubits!")
    else:
        print("⚠️ No significant improvement at this scale.")