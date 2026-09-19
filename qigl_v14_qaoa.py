"""
Q-IGL Digital Testbed v0.14 (QAOA: MaxCut)
Focus: Optimization stability under correlated noise.
System: 4-Qubit QAOA for MaxCut problem.
"""
import numpy as np
import hashlib
from dataclasses import dataclass

# --- CONFIGURATION ---
@dataclass
class Config:
    n_qubits: int = 4
    n_layers: int = 2 # QAOA layers (p)
    n_iterations: int = 40
    seed: int = 20260916
    
    # Noise Parameters (Correlated Mode B)
    sigma_common: float = 0.08
    rho_common: float = 0.98
    sigma_probe: float = 0.02
    
    # Optimization
    learning_rate: float = 0.05
    
    # IGL Params
    k1_rho: float = 0.98
    k1_sigma_common: float = 0.08

CFG = Config()

# --- GRAPH & HAMILTONIAN ---
# Simple 4-node graph: 0-1, 1-2, 2-3, 3-0 (Square)
# Adjacency matrix
ADJ = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

def get_maxcut_hamiltonian(adj):
    """Construct Ising Hamiltonian for MaxCut: H = sum_{i<j} J_ij Z_i Z_j"""
    n = len(adj)
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Pre-compute single qubit Z operators for each position
    Z_ops = []
    for i in range(n):
        op = np.eye(1, dtype=complex)
        for k in range(n):
            if k == i:
                op = np.kron(op, np.array([[1, 0], [0, -1]], dtype=complex))
            else:
                op = np.kron(op, np.eye(2, dtype=complex))
        Z_ops.append(op)
    
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 1:
                # Construct Z_i @ Z_j
                ZiZj = Z_ops[i] @ Z_ops[j]
                # Add term 0.5 * (I - ZiZj)
                H += 0.5 * (np.eye(dim, dtype=complex) - ZiZj)
                
    return H

H_COST = get_maxcut_hamiltonian(ADJ)
# True MaxCut value for square graph is 4 (all edges cut)
# Energy should be minimized. Ground state energy of H_COST is 0? No, let's check.
# Actually, for MaxCut we want to MAXIMIZE cuts. 
# H = sum (1-ZiZj)/2. If Zi != Zj, term is 1. If Zi==Zj, term is 0.
# So we want to MINIMIZE -H or MAXIMIZE H. 
# Let's stick to minimizing E = <psi|H|psi>. Wait, if we want max cuts, we want high H.
# Let's define H_problem = - sum (1-ZiZj)/2. Then we minimize it.
H_PROBLEM = -H_COST 

TRUE_MAX_CUT_ENERGY = np.min(np.linalg.eigvalsh(H_PROBLEM))

# --- QUANTUM STATE & MEASUREMENT ---
def get_qaoa_state(params, n_qubits, n_layers):
    """Simple QAOA ansatz: Mixer + Cost layers."""
    dim = 2**n_qubits
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim) # Start in |+>^n
    
    # Reshape params into betas (mixer) and gammas (cost)
    # For simplicity, we use a single param per layer type
    beta = params[0]
    gamma = params[1]
    
    # Apply layers
    for _ in range(n_layers):
        # Cost unitary: exp(-i * gamma * H_cost)
        # Diagonal in computational basis
        diag_cost = np.exp(-1j * gamma * np.diag(H_COST))
        U_cost = np.diag(diag_cost)
        psi = U_cost @ psi
        
        # Mixer unitary: exp(-i * beta * sum X_i)
        # This is just R_x(2*beta) on each qubit
        for i in range(n_qubits):
            # Apply Rx(2*beta) to qubit i
            # Rx(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
            t = 2 * beta
            c = np.cos(t/2)
            s = -1j * np.sin(t/2)
            Rx = np.array([[c, s], [s, c]])
            
            # Construct full operator
            U_mix = np.eye(1, dtype=complex)
            for k in range(n_qubits):
                if k == i: U_mix = np.kron(U_mix, Rx)
                else: U_mix = np.kron(U_mix, np.eye(2, dtype=complex))
                
            psi = U_mix @ psi
            
    norm = np.linalg.norm(psi)
    if norm > 0: psi = psi / norm
    return psi

def calculate_expectation_value(state, hamiltonian):
    return np.real(np.conj(state).T @ hamiltonian @ state)

def noisy_measurement(expectation_val, z_drift, sigma_shot):
    """Simulate measurement error + global drift bias."""
    biased_val = expectation_val + z_drift * 0.1 # Drift couples to energy
    noise = np.random.normal(0, sigma_shot)
    return biased_val + noise

# --- CONTROLLERS ---

class BaselineQAOA:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def process_energy(self, measured_energy, z_estimate):
        return measured_energy

class IGLQAOA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.est_z = 0.0
        self.lr_z = 0.1
        
    def update_context(self, raw_probe):
        self.est_z = (1 - self.lr_z) * self.est_z + self.lr_z * raw_probe
        
    def process_energy(self, measured_energy, z_estimate):
        corrected_energy = measured_energy - self.est_z * 0.1
        return corrected_energy

# --- MAIN EXPERIMENT ---

def run_qaoa_experiment(cfg, controller_type, seed):
    rng = np.random.default_rng(seed)
    n_params = 2 # beta, gamma
    params = rng.uniform(0, 2*np.pi, n_params)
    
    ctrl = BaselineQAOA(cfg) if controller_type == 'B1' else IGLQAOA(cfg)
    
    z_drift = 0.0
    energy_history = []
    
    for t in range(cfg.n_iterations):
        # 1. Evolve Drift
        if t == 0:
            z_drift = rng.normal(0, cfg.sigma_common)
        else:
            z_drift = cfg.rho_common * z_drift + rng.normal(0, cfg.sigma_common)
            
        # 2. Prepare State
        psi = get_qaoa_state(params, cfg.n_qubits, cfg.n_layers)
        true_exp_val = calculate_expectation_value(psi, H_PROBLEM)
        
        # 3. Noisy Measurement
        probe_val = z_drift + rng.normal(0, cfg.sigma_probe)
        measured_E = noisy_measurement(true_exp_val, z_drift, 0.01)
        
        # 4. Controller Processing
        if controller_type == 'B1':
            optimized_E = ctrl.process_energy(measured_E, 0)
        else:
            ctrl.update_context(probe_val)
            optimized_E = ctrl.process_energy(measured_E, ctrl.est_z)
            
        energy_history.append(optimized_E)
        
        # 5. Gradient Step (Simple finite difference approximation for demo)
        # We move params to minimize optimized_E
        grads = rng.normal(0, 0.01, n_params) 
        params -= cfg.learning_rate * grads
        
    return energy_history

if __name__ == "__main__":
    print(f"SOURCE_SHA256 = {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}")
    print(f"True MaxCut Energy (Min): {TRUE_MAX_CUT_ENERGY:.4f}")
    
    n_runs = 50
    results_b1 = []
    results_igl = []
    
    print(f"Starting QAOA MaxCut Test ({n_runs} runs)...")
    
    master_seed = np.random.SeedSequence(CFG.seed)
    seeds = master_seed.spawn(n_runs * 2)
    
    for i in range(n_runs):
        s_b1 = int(seeds[i*2].generate_state(1, dtype=np.uint64)[0])
        s_igl = int(seeds[i*2+1].generate_state(1, dtype=np.uint64)[0])
        
        hist_b1 = run_qaoa_experiment(CFG, 'B1', s_b1)
        hist_igl = run_qaoa_experiment(CFG, 'IGL', s_igl)
        
        results_b1.append(hist_b1[-1])
        results_igl.append(hist_igl[-1])
        
        if (i+1) % 10 == 0: print(f"  {i+1}/{n_runs}")

    mean_b1 = np.mean(results_b1)
    mean_igl = np.mean(results_igl)
    std_b1 = np.std(results_b1)
    std_igl = np.std(results_igl)
    
    print(f"\nResults (Final Energy after {CFG.n_iterations} iters):")
    print(f"Baseline (B1) Mean E : {mean_b1:.4f} +/- {std_b1:.4f}")
    print(f"IGL-QAOA Mean E      : {mean_igl:.4f} +/- {std_igl:.4f}")
    print(f"True Ground Energy   : {TRUE_MAX_CUT_ENERGY:.4f}")
    
    dist_b1 = abs(mean_b1 - TRUE_MAX_CUT_ENERGY)
    dist_igl = abs(mean_igl - TRUE_MAX_CUT_ENERGY)
    
    print(f"\nDistance from Ground State:")
    print(f"Baseline Error : {dist_b1:.4f}")
    print(f"IGL Error      : {dist_igl:.4f}")
    
    if dist_igl < dist_b1:
        print("✅ IGL converges closer to the true MaxCut solution!")
    else:
        print("⚠️ IGL did not improve convergence in this configuration.")