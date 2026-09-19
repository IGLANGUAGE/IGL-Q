"""
Q-IGL Digital Testbed v0.10 (50-Qubit Resonance Cascade)
Focus: Non-linear error propagation (Resonance).
System: 50 Qubits with non-linear crosstalk threshold.
"""
import numpy as np
import hashlib
from dataclasses import dataclass

@dataclass
class Config:
    n_qubits: int = 50
    steps: int = 30
    dt: float = 0.05
    trajectories: int = 100
    seed: int = 20260916
    
    # Noise Parameters
    sigma_local: float = 0.02
    sigma_common: float = 0.05
    rho_common: float = 0.98
    
    # Resonance Parameters
    resonance_threshold: float = 0.5 # Phase threshold for cascade
    crosstalk_factor: float = 0.15   # How much energy is passed to neighbors
    
    u_max: float = 2.0
    b1_alpha: float = 0.1
    k1_rho: float = 0.98
    k1_sigma_common: float = 0.05

CFG = Config()

def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def generate_environment(cfg, rng):
    T, N = cfg.steps, cfg.n_qubits
    local = np.zeros((T, N))
    common = np.zeros(T)
    
    for t in range(T):
        if t == 0:
            local[t] = rng.normal(0.0, cfg.sigma_local, N)
            common[t] = rng.normal(0.0, cfg.sigma_common)
        else:
            local[t] = 0.9 * local[t-1] + rng.normal(0.0, cfg.sigma_local, N)
            common[t] = cfg.rho_common * common[t-1] + rng.normal(0.0, cfg.sigma_common)
            
    return local + common[:, None]

class BaselineController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est = np.zeros(cfg.n_qubits)
        
    def step(self, phase_errors, raw_probe):
        self.est = (1 - self.cfg.b1_alpha) * self.est + self.cfg.b1_alpha * raw_probe
        return np.clip(-self.est, -self.cfg.u_max, self.cfg.u_max)

class IGLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.05
        
    def step(self, phase_errors, raw_probe):
        mean_probe = np.mean(raw_probe)
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * mean_probe
        # Global pre-compensation
        correction = np.full(self.cfg.n_qubits, -self.est_z)
        return np.clip(correction, -self.cfg.u_max, self.cfg.u_max)

def apply_resonance(phase_errors, cfg):
    """Non-linear crosstalk: if phase > threshold, it kicks neighbors."""
    new_errors = phase_errors.copy()
    abs_phases = np.abs(phase_errors)
    
    # Find qubits that exceeded threshold
    excited = abs_phases > cfg.resonance_threshold
    
    for i in range(cfg.n_qubits):
        if excited[i]:
            # Transfer energy to neighbors
            if i > 0: new_errors[i-1] += cfg.crosstalk_factor * phase_errors[i]
            if i < cfg.n_qubits - 1: new_errors[i+1] += cfg.crosstalk_factor * phase_errors[i]
            
    return new_errors

def run_trajectory(cfg, seed, ctrl_type):
    ss = np.random.SeedSequence(seed)
    env_rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    detuning = generate_environment(cfg, env_rng)
    phase_errors = np.zeros(cfg.n_qubits)
    
    ctrl = BaselineController(cfg) if ctrl_type == 'B1' else IGLController(cfg)
    fidelities = []
    
    for t in range(cfg.steps):
        # 1. Accumulate natural errors
        phase_errors += detuning[t] * cfg.dt
        
        # 2. Apply Control
        raw_probe = detuning[t] + np.random.normal(0, 0.02, cfg.n_qubits)
        u = ctrl.step(phase_errors, raw_probe)
        phase_errors += u * cfg.dt
        
        # 3. Apply Non-linear Resonance (The "Butterfly Effect")
        phase_errors = apply_resonance(phase_errors, cfg)
        
        # 4. Calculate Fidelity (Strict: any large error kills it)
        max_error = np.max(np.abs(phase_errors))
        f = np.exp(-max_error**2 * 5) # Penalize peaks heavily
        fidelities.append(f)
        
    return np.mean(fidelities)

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1, res_igl = [], []
    print(f"Starting 50-Qubit Resonance Test ({CFG.trajectories} trajectories)...")
    
    for i in range(CFG.trajectories):
        s1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s2 = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        res_b1.append(run_trajectory(CFG, s1, 'B1'))
        res_igl.append(run_trajectory(CFG, s2, 'IGL'))
        
        if (i+1) % 20 == 0: print(f"  {i+1}/{CFG.trajectories}")

    m_b1, m_igl = np.mean(res_b1), np.mean(res_igl)
    print(f"\nBaseline (B1) Mean Fidelity: {m_b1:.6f}")
    print(f"IGL Mean Fidelity:           {m_igl:.6f}")
    print(f"Delta (IGL-B1):              {m_igl - m_b1:+.6f}")
    
    if m_igl > m_b1:
        print("✅ IGL successfully prevented resonance cascade!")
    else:
        print("⚠️ Resonance overwhelmed both controllers.")