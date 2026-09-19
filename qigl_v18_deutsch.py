"""
Q-IGL Digital Testbed v0.18 (Deutsch-Jozsa Algorithm)
Focus: Correct classification under correlated phase noise.
System: 10-Qubit Deutsch-Jozsa.
"""
import numpy as np
import hashlib
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass
class Config:
    n_qubits: int = 10 # Serious scale for DJ
    trajectories: int = 300
    seed: int = 20260916
    
    # Noise Parameters (Strong Correlated Mode B)
    sigma_common: float = 0.1 
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

def apply_oracle(state, n, is_balanced=True):
    """
    For Balanced: Apply Z to the last qubit if first qubit is 1 (simple example).
    For Constant: Do nothing (Identity).
    """
    dim = 2**n
    if not is_balanced:
        return state # Constant oracle does nothing to phase in this representation
        
    # Simple balanced oracle: Flip phase of states where qubit 0 is |1>
    # This corresponds to f(x) = x_0
    U_oracle = np.eye(dim, dtype=complex)
    half = dim // 2
    U_oracle[half:, half:] *= -1
    
    return U_oracle @ state

def apply_noisy_hadamards(state, n, z_drift, ctrl_type, ctrl=None):
    """Apply final Hadamards with noise compensation."""
    
    effective_drift = z_drift
    if ctrl_type == 'IGL' and ctrl:
        effective_drift = z_drift - ctrl.est_z
        
    # Apply global phase error due to drift before H
    dim = len(state)
    phases = np.exp(1j * np.arange(dim) * effective_drift * 0.1)
    U_noise = np.diag(phases)
    
    noisy_state = U_noise @ state
    
    # Apply Final Hadamards on input register (first n-1 qubits usually, but here we use n qubits for simplicity)
    H = get_hadamard(n)
    final_state = H @ noisy_state
    
    return final_state

class IGLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.1
        
    def update(self, probe):
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * probe

# --- MAIN EXPERIMENT ---

def run_dj_trajectory(cfg, seed, ctrl_type, is_balanced=True):
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    n = cfg.n_qubits
    dim = 2**n
    
    # Initial state |0...0>
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    # 1. Apply Initial Hadamards
    H = get_hadamard(n)
    psi = H @ psi
    
    # 2. Apply Oracle
    psi = apply_oracle(psi, n, is_balanced)
    
    # 3. Evolve Drift (Noise happens during computation)
    z_drift = rng.normal(0, cfg.sigma_common)
    
    # Probe for IGL
    probe = z_drift + rng.normal(0, cfg.sigma_probe)
    ctrl = None
    if ctrl_type == 'IGL':
        ctrl = IGLController(cfg)
        ctrl.update(probe)
        
    # 4. Apply Final Hadamards with Noise
    psi = apply_noisy_hadamards(psi, n, z_drift, ctrl_type, ctrl)
    
    # 5. Measure
    # For Balanced function, we expect NOT |00...0>. 
    # For Constant, we expect |00...0>.
    # Let's test Balanced case. Success is measuring anything BUT 0.
    # Actually, standard DJ: if result is |0...0> -> Constant. Else -> Balanced.
    
    probs = np.abs(psi)**2
    # Probability of measuring |0...0>
    p_zero = probs[0]
    
    # If it's balanced, p_zero should be 0. If we measure 0, it's an error.
    # Success probability for Balanced case = 1 - p_zero
    success_prob = 1.0 - p_zero
    
    return success_prob

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1 = []
    res_igl = []
    
    print(f"Starting 10-Qubit Deutsch-Jozsa Test ({CFG.trajectories} trajectories)...")
    
    for i in range(CFG.trajectories):
        s_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s_igl = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        # Test Balanced Case
        p_b1 = run_dj_trajectory(CFG, s_b1, 'B1', is_balanced=True)
        p_igl = run_dj_trajectory(CFG, s_igl, 'IGL', is_balanced=True)
        
        res_b1.append(p_b1)
        res_igl.append(p_igl)
        
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{CFG.trajectories}")

    mean_b1 = np.mean(res_b1)
    mean_igl = np.mean(res_igl)
    std_b1 = np.std(res_b1)
    std_igl = np.std(res_igl)
    
    print(f"\nResults (Probability of Correct Classification for Balanced Function):")
    print(f"Baseline (B1) Mean P : {mean_b1:.4f} +/- {std_b1:.4f}")
    print(f"IGL-DJ Mean P        : {mean_igl:.4f} +/- {std_igl:.4f}")
    print(f"Delta (IGL-B1)       : {mean_igl - mean_b1:+.4f}")
    
    if mean_igl > mean_b1:
        print("✅ IGL improves algorithm reliability!")
    else:
        print("⚠️ No significant improvement detected.")