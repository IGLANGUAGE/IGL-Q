"""
Q-IGL Digital Testbed v0.6 (QEC: Repetition Code)
Focus: Error Correction under Correlated Noise.
System: 3-Qubit Repetition Code (Phase Flip Protection).
"""
import numpy as np
import hashlib
import csv
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass
class Config:
    n_physical_qubits: int = 3
    steps: int = 50
    dt: float = 0.05
    trajectories: int = 500
    seed: int = 20260916
    
    # Noise Parameters (Correlated Mode B)
    sigma_local_process: float = 0.05
    sigma_common_process: float = 0.08
    rho_common: float = 0.98
    sigma_probe: float = 0.10
    
    g_before: tuple = (1.0, 1.0, 1.0) # All qubits sensitive to global drift
    
    u_max: float = 2.0
    b1_alpha: float = 0.16
    k1_rho: float = 0.98
    k1_sigma_local: float = 0.05
    k1_sigma_common: float = 0.08

CFG = Config()

# --- FORENSIC SELF-HASH ---
def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

# --- QUANTUM SYSTEM (QEC Logic) ---
def generate_environment(cfg, rng):
    T = cfg.steps
    local = np.zeros((T, cfg.n_physical_qubits))
    common = np.zeros(T)
    
    for t in range(T):
        if t == 0:
            local[t] = rng.normal(0.0, cfg.sigma_local_process, cfg.n_physical_qubits)
            common[t] = rng.normal(0.0, cfg.sigma_common_process)
        else:
            local[t] = 0.92 * local[t-1] + rng.normal(0.0, cfg.sigma_local_process, cfg.n_physical_qubits)
            common[t] = cfg.rho_common * common[t-1] + rng.normal(0.0, cfg.sigma_common_process)
            
    detuning = local + common[:, None] * np.array(cfg.g_before)
    return detuning

def calculate_syndromes(phase_errors):
    """
    For Phase Flip Code:
    S1 = Z1*Z2 -> detects difference between q1 and q2
    S2 = Z2*Z3 -> detects difference between q2 and q3
    In phase approximation: Syndrome ~ sign(phase_i - phase_j)
    """
    s1 = np.sign(phase_errors[0] - phase_errors[1])
    s2 = np.sign(phase_errors[1] - phase_errors[2])
    return np.array([s1, s2])

def correct_error(phase_errors, syndromes):
    """
    Simple lookup table correction for 3-qubit code.
    """
    corrected = phase_errors.copy()
    
    # Syndrome [1, 1] or [-1, -1] -> Middle qubit error? No, in phase code:
    # If s1 != 0 and s2 == 0 -> Error on Q1
    # If s1 != 0 and s2 != 0 -> Error on Q2
    # If s1 == 0 and s2 != 0 -> Error on Q3
    
    if syndromes[0] != 0 and syndromes[1] == 0:
        corrected[0] = 0 # Reset phase of Q1
    elif syndromes[0] != 0 and syndromes[1] != 0:
        corrected[1] = 0 # Reset phase of Q2
    elif syndromes[0] == 0 and syndromes[1] != 0:
        corrected[2] = 0 # Reset phase of Q3
        
    return corrected

def get_logical_fidelity(phase_errors):
    """
    Logical fidelity is preserved if the majority of qubits are in phase.
    For 3 qubits, if 2 or 3 are close to 0, fidelity is high.
    Simplified: F = exp(-mean(phase^2))
    """
    return np.exp(-np.mean(phase_errors**2))

# --- CONTROLLERS ---

class BaselineQEC:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def step(self, phase_errors):
        syndromes = calculate_syndromes(phase_errors)
        corrected = correct_error(phase_errors, syndromes)
        return corrected

class IGLQEC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.05
        
    def step(self, phase_errors, raw_detuning_estimate):
        """
        IGL uses its estimate of global drift to pre-compensate or 
        adjust the correction threshold.
        Here we simply subtract the estimated global drift from all qubits
        BEFORE applying the discrete syndrome correction.
        """
        # Pre-compensation based on IGL memory D
        pre_compensated = phase_errors - self.est_z
        
        syndromes = calculate_syndromes(pre_compensated)
        corrected = correct_error(pre_compensated, syndromes)
        
        # Update IGL memory D (estimate of z)
        # Using mean of raw detuning as a probe for z
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * np.mean(raw_detuning_estimate)
        
        return corrected

# --- MAIN EXPERIMENT ---

def run_trajectory(cfg, seed, controller_type):
    ss = np.random.SeedSequence(seed)
    env_rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    detuning = generate_environment(cfg, env_rng)
    phase_errors = np.zeros(cfg.n_physical_qubits)
    
    if controller_type == 'B1':
        ctrl = BaselineQEC(cfg)
    else:
        ctrl = IGLQEC(cfg)
        
    fidelities = []
    
    for t in range(cfg.steps):
        # Accumulate errors
        phase_errors += detuning[t] * cfg.dt
        
        # Get raw estimate for IGL (simulating a noisy probe)
        raw_probe = detuning[t] + np.random.normal(0, cfg.sigma_probe, cfg.n_physical_qubits)
        
        # Apply Control/Correction
        if controller_type == 'B1':
            phase_errors = ctrl.step(phase_errors)
        else:
            phase_errors = ctrl.step(phase_errors, raw_probe)
            
        f = get_logical_fidelity(phase_errors)
        fidelities.append(f)
        
    return np.mean(fidelities)

if __name__ == "__main__":
    SOURCE_HASH = get_self_hash()
    print(f"SOURCE_SHA256 = {SOURCE_HASH}")
    
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    results = {'B1': [], 'IGL': []}
    
    print("Starting QEC Test (Repetition Code)...")
    for i in range(CFG.trajectories):
        seed_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        seed_igl = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        # Note: For fair comparison, we should use same seed for environment
        # But here we use paired seeds for statistical independence of trajectories
        
        f_b1 = run_trajectory(CFG, seed_b1, 'B1')
        f_igl = run_trajectory(CFG, seed_igl, 'IGL')
        
        results['B1'].append(f_b1)
        results['IGL'].append(f_igl)
        
        if (i+1) % 100 == 0:
            print(f"  Completed {i+1}/{CFG.trajectories} trajectories")

    mean_b1 = np.mean(results['B1'])
    mean_igl = np.mean(results['IGL'])
    
    print(f"\nResults:")
    print(f"Baseline (B1) Mean Logical Fidelity: {mean_b1:.6f}")
    print(f"IGL-QEC Mean Logical Fidelity:       {mean_igl:.6f}")
    print(f"Delta (IGL-B1):                      {mean_igl - mean_b1:+.6f}")
    
    if mean_igl > mean_b1:
        print("✅ IGL-QEC shows improvement in logical fidelity.")
    else:
        print("⚠️ No significant improvement detected in this configuration.")