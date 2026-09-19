"""
Q-IGL Digital Testbed v0.16 (Shor's Algorithm: N=15)
Focus: Period finding stability under correlated phase noise.
System: 4-Qubit Period Register for factoring 15.
"""
import numpy as np
import hashlib
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass
class Config:
    n_qubits: int = 4 # Register size for period finding
    trajectories: int = 300
    seed: int = 20260916
    
    # Noise Parameters (Correlated Mode B)
    sigma_common: float = 0.15 # Strong noise to test QFT robustness
    rho_common: float = 0.98
    sigma_probe: float = 0.02
    
    # IGL Params
    k1_rho: float = 0.98
    k1_sigma_common: float = 0.15

CFG = Config()

def get_self_hash():
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

# --- QUANTUM OPERATORS ---

def get_qft_matrix(n):
    """Generate Quantum Fourier Transform matrix for n qubits."""
    dim = 2**n
    qft = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            qft[i, j] = np.exp(2j * np.pi * i * j / dim)
    return qft / np.sqrt(dim)

def apply_noisy_qft(state, qft_mat, z_drift, ctrl_type, ctrl=None):
    """Apply QFT with phase noise compensation."""
    
    # In real hardware, noise happens during gate execution.
    # Here we model it as a global phase distortion before the final transform.
    
    effective_drift = z_drift
    if ctrl_type == 'IGL' and ctrl:
        # IGL compensates the estimated drift
        effective_drift = z_drift - ctrl.est_z
        
    # Apply noise as a diagonal phase error matrix
    dim = len(state)
    phases = np.exp(1j * np.arange(dim) * effective_drift * 0.5)
    U_noise = np.diag(phases)
    
    # Apply Noise -> Then QFT
    noisy_state = U_noise @ state
    final_state = qft_mat @ noisy_state
    
    return final_state

class IGLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.1
        
    def update(self, probe):
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * probe

# --- MAIN EXPERIMENT ---

def run_shor_trajectory(cfg, seed, ctrl_type):
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    n = cfg.n_qubits
    dim = 2**n
    
    # For Shor's algorithm on N=15, we look for period of a=7 or a=11.
    # The ideal state before QFT is a superposition of states |0>, |r>, |2r>...
    # For simplicity, we simulate the "Period Finding" part directly.
    # Let's assume we are looking for period r=4 (which happens for a=4 mod 15? No, let's use standard example).
    # For a=7, period r=4. The state before QFT is roughly sum |k*r>.
    
    # Prepare ideal periodic state (simplified for simulation)
    # We create a state that has peaks at multiples of r=4.
    psi = np.zeros(dim, dtype=complex)
    r = 4 # Target period
    for k in range(dim // r):
        psi[k * r] = 1.0 / np.sqrt(dim // r)
        
    qft_mat = get_qft_matrix(n)
    
    ctrl = None
    if ctrl_type == 'IGL':
        ctrl = IGLController(cfg)
        
    z_drift = rng.normal(0, cfg.sigma_common)
    
    # Probe for IGL
    probe = z_drift + rng.normal(0, cfg.sigma_probe)
    if ctrl:
        ctrl.update(probe)
        
    # Execute Noisy QFT
    final_state = apply_noisy_qft(psi, qft_mat, z_drift, ctrl_type, ctrl)
    
    # Measure probability of finding the correct peak.
    # For r=4 and t=4 qubits, QFT should map this to peaks at 0, dim/r, 2*dim/r...
    # dim=16, r=4 -> peaks at 0, 4, 8, 12.
    # Success is measuring one of these states.
    
    probs = np.abs(final_state)**2
    success_states = [0, 4, 8, 12]
    success_prob = sum(probs[s] for s in success_states)
    
    return success_prob

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1 = []
    res_igl = []
    
    print(f"Starting Shor's Period Finding Test ({CFG.trajectories} trajectories)...")
    
    for i in range(CFG.trajectories):
        s_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s_igl = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        p_b1 = run_shor_trajectory(CFG, s_b1, 'B1')
        p_igl = run_shor_trajectory(CFG, s_igl, 'IGL')
        
        res_b1.append(p_b1)
        res_igl.append(p_igl)
        
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{CFG.trajectories}")

    mean_b1 = np.mean(res_b1)
    mean_igl = np.mean(res_igl)
    std_b1 = np.std(res_b1)
    std_igl = np.std(res_igl)
    
    print(f"\nResults (Probability of Correct Period):")
    print(f"Baseline (B1) Mean P : {mean_b1:.4f} +/- {std_b1:.4f}")
    print(f"IGL-Shor Mean P      : {mean_igl:.4f} +/- {std_igl:.4f}")
    print(f"Delta (IGL-B1)       : {mean_igl - mean_b1:+.4f}")
    
    if mean_igl > mean_b1:
        print("✅ IGL improves period finding accuracy!")
    else:
        print("⚠️ No significant improvement detected.")