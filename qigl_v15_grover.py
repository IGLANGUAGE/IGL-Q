"""
Q-IGL Digital Testbed v0.15 (Grover's Algorithm)
Focus: Search accuracy under correlated phase noise.
System: 2-Qubit Grover Search (N=4 items, 1 marked).
"""
import numpy as np
import hashlib
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass
class Config:
    n_qubits: int = 2
    n_iterations: int = 1 # For N=4, one iteration is optimal
    trajectories: int = 500
    seed: int = 20260916
    
    # Noise Parameters (Correlated Mode B)
    sigma_common: float = 0.1 # Stronger noise to test robustness
    rho_common: float = 0.98
    sigma_probe: float = 0.02
    
    # IGL Params
    k1_rho: float = 0.98
    k1_sigma_common: float = 0.1

CFG = Config()

def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

# --- QUANTUM OPERATORS ---

def get_hadamard(n):
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H = H1
    for _ in range(n-1):
        H = np.kron(H, H1)
    return H

def get_oracle(target_idx, n):
    """Phase flip on target state."""
    dim = 2**n
    O = np.eye(dim, dtype=complex)
    O[target_idx, target_idx] = -1
    return O

def get_diffuser(n):
    """Reflection about the mean."""
    dim = 2**n
    H = get_hadamard(n)
    # |0><0| matrix
    zero_state = np.zeros((dim, 1), dtype=complex)
    zero_state[0, 0] = 1
    proj_0 = zero_state @ zero_state.conj().T
    # 2|0><0| - I
    inner = 2 * proj_0 - np.eye(dim, dtype=complex)
    return H @ inner @ H

# --- NOISY EXECUTION ---

def apply_noisy_rotation(state, angle_error, axis='Z'):
    """Simulate phase error as a small unwanted rotation."""
    n = int(np.log2(len(state)))
    dim = 2**n
    # For simplicity, we apply a global phase error proportional to the drift
    # In real hardware, this would be per-qubit, but global drift affects all.
    U_err = np.eye(dim, dtype=complex) * np.exp(1j * angle_error)
    return U_err @ state

def run_grover_iteration(state, oracle, diffuser, z_drift, ctrl_type, ctrl=None):
    """One step of Grover with IGL compensation."""
    
    # 1. Oracle Step
    # IGL tries to compensate the phase flip precision
    effective_drift = z_drift
    if ctrl_type == 'IGL' and ctrl:
        # IGL estimates z and subtracts it from the operation error
        effective_drift = z_drift - ctrl.est_z
        
    state = oracle @ state
    state = apply_noisy_rotation(state, effective_drift * 0.1) # Noise scales with drift
    
    # 2. Diffuser Step
    state = diffuser @ state
    state = apply_noisy_rotation(state, effective_drift * 0.1)
    
    return state

class IGLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.1
        
    def update(self, probe):
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * probe

# --- MAIN EXPERIMENT ---

def run_trajectory(cfg, seed, ctrl_type):
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    n = cfg.n_qubits
    dim = 2**n
    target_idx = 3 # |11>
    
    # Initial state |+>^n
    H = get_hadamard(n)
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    psi = H @ psi
    
    oracle = get_oracle(target_idx, n)
    diffuser = get_diffuser(n)
    
    ctrl = None
    if ctrl_type == 'IGL':
        ctrl = IGLController(cfg)
        
    z_drift = 0.0
    
    for step in range(cfg.n_iterations):
        # Evolve Drift
        if step == 0:
            z_drift = rng.normal(0, cfg.sigma_common)
        else:
            z_drift = cfg.rho_common * z_drift + rng.normal(0, cfg.sigma_common)
            
        # Probe for IGL
        probe = z_drift + rng.normal(0, cfg.sigma_probe)
        if ctrl:
            ctrl.update(probe)
            
        # Execute Grover
        psi = run_grover_iteration(psi, oracle, diffuser, z_drift, ctrl_type, ctrl)
        
    # Calculate Fidelity (Probability of finding target)
    fidelity = np.abs(psi[target_idx])**2
    return fidelity

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1 = []
    res_igl = []
    
    print(f"Starting Grover Search Test ({CFG.trajectories} trajectories)...")
    
    for i in range(CFG.trajectories):
        s_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s_igl = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        f_b1 = run_trajectory(CFG, s_b1, 'B1')
        f_igl = run_trajectory(CFG, s_igl, 'IGL')
        
        res_b1.append(f_b1)
        res_igl.append(f_igl)
        
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{CFG.trajectories}")

    mean_b1 = np.mean(res_b1)
    mean_igl = np.mean(res_igl)
    std_b1 = np.std(res_b1)
    std_igl = np.std(res_igl)
    
    print(f"\nResults (Probability of Success):")
    print(f"Baseline (B1) Mean P : {mean_b1:.4f} +/- {std_b1:.4f}")
    print(f"IGL-Grover Mean P    : {mean_igl:.4f} +/- {std_igl:.4f}")
    print(f"Delta (IGL-B1)       : {mean_igl - mean_b1:+.4f}")
    
    if mean_igl > mean_b1:
        print("✅ IGL improves search accuracy!")
    else:
        print("⚠️ No significant improvement detected.")