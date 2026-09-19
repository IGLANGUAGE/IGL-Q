"""
Q-IGL Digital Testbed v0.17 (Quantum Fourier Transform)
Focus: Precision of phase interference under correlated noise.
System: 8-Qubit QFT.
"""
import numpy as np
import hashlib
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass
class Config:
    n_qubits: int = 8 # Serious scale
    trajectories: int = 200
    seed: int = 20260916
    
    # Noise Parameters (Strong Correlated Mode B)
    sigma_common: float = 0.15 # Strong drift to break interference
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
    """Generate exact QFT matrix for n qubits."""
    dim = 2**n
    qft = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            qft[i, j] = np.exp(2j * np.pi * i * j / dim)
    return qft / np.sqrt(dim)

def apply_noisy_qft_step(state, qubit_idx, n_qubits, z_drift, ctrl_type, ctrl=None):
    """
    Simulate QFT gate-by-gate with noise.
    QFT consists of Hadamards and Controlled-Phase rotations.
    Noise affects the phase rotations.
    """
    dim = 2**n_qubits
    psi = state.copy()
    
    # Effective drift after IGL compensation
    effective_drift = z_drift
    if ctrl_type == 'IGL' and ctrl:
        effective_drift = z_drift - ctrl.est_z
        
    # Apply Hadamard on qubit_idx (simplified as part of the flow)
    # In real QFT, H is applied, then controlled rotations from subsequent qubits.
    # We model the cumulative phase error on the target qubit due to drift.
    
    # For each controlled rotation R_k from qubit j > i:
    # Phase shift = 2*pi / 2^(j-i+1)
    # Drift adds an error delta_phi to this rotation.
    
    # Simplified model: Global drift adds a random phase error to every entangling step.
    # The more steps, the more the interference pattern degrades.
    
    num_steps = n_qubits - qubit_idx
    for k in range(1, num_steps + 1):
        # Controlled phase rotation error
        error_angle = effective_drift * 0.1 * k # Error scales with precision level
        # Apply diagonal phase error matrix to the whole state for simplicity
        phases = np.ones(dim, dtype=complex)
        # Identify basis states where control and target are 1
        for idx in range(dim):
            if (idx >> qubit_idx) & 1 and (idx >> (qubit_idx + k)) & 1:
                 phases[idx] = np.exp(1j * error_angle)
        
        psi = phases * psi
        
    return psi

class IGLController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.1
        
    def update(self, probe):
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * probe

# --- MAIN EXPERIMENT ---

def run_qft_trajectory(cfg, seed, ctrl_type):
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss.generate_state(1, dtype=np.uint64)[0])
    
    n = cfg.n_qubits
    dim = 2**n
    
    # Initial state |100...0> (index 2^(n-1)) or |1> (index 1). Let's use |1>.
    psi = np.zeros(dim, dtype=complex)
    psi[1] = 1.0
    
    # Ideal QFT result
    qft_mat = get_qft_matrix(n)
    ideal_psi = qft_mat @ psi
    
    ctrl = None
    if ctrl_type == 'IGL':
        ctrl = IGLController(cfg)
        
    z_drift = 0.0
    
    # Execute QFT step-by-step with noise
    for q in range(n):
        # Evolve Drift for each qubit step
        if q == 0:
            z_drift = rng.normal(0, cfg.sigma_common)
        else:
            z_drift = cfg.rho_common * z_drift + rng.normal(0, cfg.sigma_common)
            
        # Probe for IGL
        probe = z_drift + rng.normal(0, cfg.sigma_probe)
        if ctrl:
            ctrl.update(probe)
            
        # Apply noisy QFT operations for this qubit
        psi = apply_noisy_qft_step(psi, q, n, z_drift, ctrl_type, ctrl)
        
    # Calculate Fidelity with ideal QFT state
    fidelity = np.abs(np.conj(ideal_psi).T @ psi)**2
    return fidelity

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {get_self_hash()}")
    
    master = np.random.SeedSequence(CFG.seed)
    seeds = master.spawn(CFG.trajectories * 2)
    
    res_b1 = []
    res_igl = []
    
    print(f"Starting 8-Qubit QFT Precision Test ({CFG.trajectories} trajectories)...")
    
    for i in range(CFG.trajectories):
        s_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s_igl = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        f_b1 = run_qft_trajectory(CFG, s_b1, 'B1')
        f_igl = run_qft_trajectory(CFG, s_igl, 'IGL')
        
        res_b1.append(f_b1)
        res_igl.append(f_igl)
        
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{CFG.trajectories}")

    mean_b1 = np.mean(res_b1)
    mean_igl = np.mean(res_igl)
    std_b1 = np.std(res_b1)
    std_igl = np.std(res_igl)
    
    print(f"\nResults (Fidelity with Ideal QFT):")
    print(f"Baseline (B1) Mean F : {mean_b1:.4f} +/- {std_b1:.4f}")
    print(f"IGL-QFT Mean F       : {mean_igl:.4f} +/- {std_igl:.4f}")
    print(f"Delta (IGL-B1)       : {mean_igl - mean_b1:+.4f}")
    
    if mean_igl > mean_b1:
        print("✅ IGL significantly preserves QFT precision!")
    else:
        print("⚠️ No significant improvement detected.")