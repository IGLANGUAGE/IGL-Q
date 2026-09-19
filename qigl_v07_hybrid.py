"""
Q-IGL Digital Testbed v0.7 (Hybrid QEC)
Focus: Hybrid approach (IGL Pre-compensation + Local Syndrome Correction).
System: 3-Qubit Repetition Code.
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
    
    g_before: tuple = (1.0, 1.0, 1.0)
    
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
    s1 = np.sign(phase_errors[0] - phase_errors[1])
    s2 = np.sign(phase_errors[1] - phase_errors[2])
    return np.array([s1, s2])

def correct_error(phase_errors, syndromes):
    corrected = phase_errors.copy()
    if syndromes[0] != 0 and syndromes[1] == 0:
        corrected[0] = 0 
    elif syndromes[0] != 0 and syndromes[1] != 0:
        corrected[1] = 0 
    elif syndromes[0] == 0 and syndromes[1] != 0:
        corrected[2] = 0 
    return corrected

def get_logical_fidelity(phase_errors):
    return np.exp(-np.mean(phase_errors**2))

# --- CONTROLLERS ---

class BaselineQEC:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def step(self, phase_errors):
        syndromes = calculate_syndromes(phase_errors)
        corrected = correct_error(phase_errors, syndromes)
        return corrected

class HybridQEC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.05
        
    def step(self, phase_errors, raw_detuning_estimate):
        # 1. IGL Part: Estimate global drift and pre-compensate
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * np.mean(raw_detuning_estimate)
        
        # Subtract estimated global drift from all qubits
        pre_compensated = phase_errors - self.est_z
        
        # 2. Baseline Part: Local syndrome correction on the residual
        syndromes = calculate_syndromes(pre_compensated)
        corrected = correct_error(pre_compensated, syndromes)
        
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
        ctrl = HybridQEC(cfg)
        
    fidelities = []
    
    for t in range(cfg.steps):
        # Accumulate errors
        phase_errors += detuning[t] * cfg.dt
        
        # Get raw estimate for IGL (noisy probe)
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
    
    results = {'B1': [], 'HYBRID': []}
    
    print("Starting Hybrid QEC Test...")
    for i in range(CFG.trajectories):
        seed_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        seed_hybrid = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        f_b1 = run_trajectory(CFG, seed_b1, 'B1')
        f_hybrid = run_trajectory(CFG, seed_hybrid, 'HYBRID')
        
        results['B1'].append(f_b1)
        results['HYBRID'].append(f_hybrid)
        
        if (i+1) % 100 == 0:
            print(f"  Completed {i+1}/{CFG.trajectories} trajectories")

    mean_b1 = np.mean(results['B1'])
    mean_hybrid = np.mean(results['HYBRID'])
    
    print(f"\nResults:")
    print(f"Baseline (B1) Mean Logical Fidelity: {mean_b1:.6f}")
    print(f"Hybrid-QEC Mean Logical Fidelity:    {mean_hybrid:.6f}")
    print(f"Delta (Hybrid-B1):                   {mean_hybrid - mean_b1:+.6f}")
    
    if mean_hybrid > mean_b1:
        print("✅ Hybrid approach shows improvement!")
    else:
        print("⚠️ No improvement detected.")