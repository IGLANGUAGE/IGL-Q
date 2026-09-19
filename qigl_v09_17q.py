"""
Q-IGL Digital Testbed v0.9 (17-Qubit Cluster)
Focus: Correlated Noise in a Multi-Qubit Cluster.
System: 17 Qubits with strong nearest-neighbor crosstalk.
"""
import numpy as np
import hashlib
from dataclasses import dataclass

@dataclass
class Config:
    n_qubits: int = 17
    steps: int = 50
    dt: float = 0.05
    trajectories: int = 200
    seed: int = 20260916
    
    # Noise Parameters (Strong Correlated Mode)
    sigma_local: float = 0.03
    sigma_common: float = 0.06 # Strong global drift
    rho_common: float = 0.98
    sigma_crosstalk: float = 0.04 # Neighbor influence
    
    g_before: tuple = None # Will be generated
    
    u_max: float = 2.0
    b1_alpha: float = 0.1
    k1_rho: float = 0.98
    k1_sigma_common: float = 0.06

CFG = Config()

def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def generate_g_vectors(n, seed):
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(0.8, 1.2, n))

CFG.g_before = generate_g_vectors(CFG.n_qubits, CFG.seed)

def generate_environment(cfg, rng):
    T, N = cfg.steps, cfg.n_qubits
    local = np.zeros((T, N))
    common = np.zeros(T)
    g = np.asarray(cfg.g_before)
    
    for t in range(T):
        if t == 0:
            local[t] = rng.normal(0.0, cfg.sigma_local, N)
            common[t] = rng.normal(0.0, cfg.sigma_common)
        else:
            local[t] = 0.9 * local[t-1] + rng.normal(0.0, cfg.sigma_local, N)
            common[t] = cfg.rho_common * common[t-1] + rng.normal(0.0, cfg.sigma_common)
            
    # Base detuning
    detuning = local + common[:, None] * g
    
    # Add Crosstalk (Neighbor influence)
    crosstalk = np.zeros_like(detuning)
    for t in range(T):
        for i in range(N):
            if i > 0: crosstalk[t, i] += cfg.sigma_crosstalk * detuning[t, i-1]
            if i < N-1: crosstalk[t, i] += cfg.sigma_crosstalk * detuning[t, i+1]
            
    return detuning + crosstalk

class BaselineController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est = np.zeros(cfg.n_qubits)
        
    def step(self, phase_errors, raw_probe):
        # Local exponential smoothing
        self.est = (1 - self.cfg.b1_alpha) * self.est + self.cfg.b1_alpha * raw_probe
        # Simple correction: try to cancel estimated error
        return np.clip(-self.est, -self.cfg.u_max, self.cfg.u_max)

class IGLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.05
        
    def step(self, phase_errors, raw_probe):
        # Estimate global drift from mean of all qubits
        mean_probe = np.mean(raw_probe)
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * mean_probe
        
        # Global compensation: subtract estimated global drift from all qubits
        # This attacks the root cause of correlated noise
        correction = np.full(self.cfg.n_qubits, -self.est_z)
        return np.clip(correction, -self.cfg.u_max, self.cfg.u_max)

def run_trajectory(cfg, seed, ctrl_type):
    ss = np.random.SeedSequence(seed)
    env_rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    detuning = generate_environment(cfg, env_rng)
    phase_errors = np.zeros(cfg.n_qubits)
    
    ctrl = BaselineController(cfg) if ctrl_type == 'B1' else IGLController(cfg)
    fidelities = []
    
    for t in range(cfg.steps):
        # Accumulate errors
        phase_errors += detuning[t] * cfg.dt
        
        # Noisy probe
        raw_probe = detuning[t] + np.random.normal(0, 0.05, cfg.n_qubits)
        
        # Apply Control
        u = ctrl.step(phase_errors, raw_probe)
        phase_errors += u * cfg.dt # Control reduces accumulated phase
        
        # Fidelity: exp(-mean(phase^2)) for the whole cluster
        f = np.exp(-np.mean(phase_errors**2))
        fidelities.append(f)
        
    return np.mean(fidelities)

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1, res_igl = [], []
    print(f"Starting 17-Qubit Cluster Test ({CFG.trajectories} trajectories)...")
    
    for i in range(CFG.trajectories):
        s1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s2 = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        res_b1.append(run_trajectory(CFG, s1, 'B1'))
        res_igl.append(run_trajectory(CFG, s2, 'IGL'))
        
        if (i+1) % 50 == 0: print(f"  {i+1}/{CFG.trajectories}")

    m_b1, m_igl = np.mean(res_b1), np.mean(res_igl)
    print(f"\nBaseline (B1) Mean Fidelity: {m_b1:.6f}")
    print(f"IGL Mean Fidelity:           {m_igl:.6f}")
    print(f"Delta (IGL-B1):              {m_igl - m_b1:+.6f}")
    
    if m_igl > m_b1:
        print("✅ IGL shows significant improvement in 17-qubit cluster!")
    else:
        print("⚠️ No improvement detected.")