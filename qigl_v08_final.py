"""
Q-IGL Digital Testbed v0.8 (Final QEC Attempt)
Strategy: Adaptive Syndrome Thresholding based on Global Context.
System: 3-Qubit Repetition Code.
"""
import numpy as np
import hashlib
from dataclasses import dataclass

@dataclass
class Config:
    n_physical_qubits: int = 3
    steps: int = 50
    dt: float = 0.05
    trajectories: int = 500
    seed: int = 20260916
    
    # Noise (Correlated Mode B)
    sigma_local: float = 0.05
    sigma_common: float = 0.08
    rho_common: float = 0.98
    sigma_probe: float = 0.10
    
    b1_alpha: float = 0.16
    k1_rho: float = 0.98
    k1_sigma_common: float = 0.08

CFG = Config()

def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def generate_environment(cfg, rng):
    T = cfg.steps
    local = np.zeros((T, cfg.n_physical_qubits))
    common = np.zeros(T)
    for t in range(T):
        if t == 0:
            local[t] = rng.normal(0.0, cfg.sigma_local, cfg.n_physical_qubits)
            common[t] = rng.normal(0.0, cfg.sigma_common)
        else:
            local[t] = 0.92 * local[t-1] + rng.normal(0.0, cfg.sigma_local, cfg.n_physical_qubits)
            common[t] = cfg.rho_common * common[t-1] + rng.normal(0.0, cfg.sigma_common)
    return local + common[:, None]

class BaselineQEC:
    def step(self, phase_errors):
        s1 = np.sign(phase_errors[0] - phase_errors[1])
        s2 = np.sign(phase_errors[1] - phase_errors[2])
        corrected = phase_errors.copy()
        if s1 != 0 and s2 == 0: corrected[0] = 0
        elif s1 != 0 and s2 != 0: corrected[1] = 0
        elif s1 == 0 and s2 != 0: corrected[2] = 0
        return corrected

class AdaptiveQEC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.var_z = 0.0
        self.lr = 0.05
        
    def step(self, phase_errors, raw_probe):
        # Update IGL Context Model
        new_z = (1 - self.lr) * self.est_z + self.lr * np.mean(raw_probe)
        innovation = np.mean(raw_probe) - self.est_z
        self.var_z = (1 - self.lr) * self.var_z + self.lr * (innovation ** 2)
        self.est_z = new_z
        
        # ADAPTIVE THRESHOLDING
        # If global context is unstable (high variance), raise threshold for correction
        # to avoid false positives from common mode noise.
        threshold = 0.0 + 2.0 * np.sqrt(self.var_z) 
        
        diff1 = phase_errors[0] - phase_errors[1]
        diff2 = phase_errors[1] - phase_errors[2]
        
        corrected = phase_errors.copy()
        
        # Only correct if difference exceeds adaptive threshold
        if abs(diff1) > threshold and abs(diff2) <= threshold:
            corrected[0] = 0
        elif abs(diff1) > threshold and abs(diff2) > threshold:
            corrected[1] = 0
        elif abs(diff1) <= threshold and abs(diff2) > threshold:
            corrected[2] = 0
            
        return corrected

def run_trajectory(cfg, seed, ctrl_type):
    ss = np.random.SeedSequence(seed)
    env_rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    detuning = generate_environment(cfg, env_rng)
    phase_errors = np.zeros(cfg.n_physical_qubits)
    
    ctrl = BaselineQEC() if ctrl_type == 'B1' else AdaptiveQEC(cfg)
    fidelities = []
    
    for t in range(cfg.steps):
        phase_errors += detuning[t] * cfg.dt
        raw_probe = detuning[t] + np.random.normal(0, cfg.sigma_probe, cfg.n_physical_qubits)
        
        if ctrl_type == 'B1':
            phase_errors = ctrl.step(phase_errors)
        else:
            phase_errors = ctrl.step(phase_errors, raw_probe)
            
        fidelities.append(np.exp(-np.mean(phase_errors**2)))
        
    return np.mean(fidelities)

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1, res_adapt = [], []
    print("Running Final QEC Test (Adaptive Threshold)...")
    
    for i in range(CFG.trajectories):
        s1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s2 = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        res_b1.append(run_trajectory(CFG, s1, 'B1'))
        res_adapt.append(run_trajectory(CFG, s2, 'ADAPT'))
        if (i+1) % 100 == 0: print(f"  {i+1}/{CFG.trajectories}")

    m_b1, m_adapt = np.mean(res_b1), np.mean(res_adapt)
    print(f"\nBaseline Fidelity: {m_b1:.6f}")
    print(f"Adaptive Fidelity: {m_adapt:.6f}")
    print(f"Delta: {m_adapt - m_b1:+.6f}")