#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import FEM


def run_simulation(N=40, resonance_factor=1.0, t_max=150.0, eF_override=None):
    """
    Run forced oscillations simulation.
    
    Args:
        N: number of elements per side
        resonance_factor: 1.0 for ω=ΔE/ℏ, 0.5 for ω=ΔE/(2ℏ)
        t_max: total simulation time
        eF_override: override default field amplitude
    """
    
    L = 2.0
    eF = eF_override if eF_override is not None else (1.0 / L)
    dt = 0.25
    hbar = 1.0
    
    print(f"\n{'='*70}")
    print(f"Forced Oscillations: N={N}, resonance_factor={resonance_factor}")
    print(f"{'='*70}")
    
    # Initialize FEM
    fem = FEM.FEM(L, N, 0.0, 0.0, FEM.BasisType.Quadratic)
    fem.build_mesh()
    fem.assemble_stiffness_matrix()
    fem.assemble_overlap_matrix()
    fem.apply_bc_eigenvalue()
    fem.assemble_potential_matrix(eF)
    
    num_nodes = fem.get_num_nodes()
    print(f"Total nodes: {num_nodes}")
    
    # Convert to scipy sparse matrices
    def to_csr(row, col, val):
        return sp.csr_matrix((np.array(val), np.array(col), np.array(row)), 
                             shape=(num_nodes, num_nodes))
    
    K = to_csr(fem.get_csr_row(), fem.get_csr_col(), fem.get_csr_val())
    O = to_csr(fem.get_overlap_csr_row(), fem.get_overlap_csr_col(), fem.get_overlap_csr_val())
    V = to_csr(fem.get_potential_csr_row(), fem.get_potential_csr_col(), fem.get_potential_csr_val())
    
    print(f"K: {K.nnz} nonzeros")
    print(f"O: {O.nnz} nonzeros")
    print(f"V: {V.nnz} nonzeros")
    
    # Stationary Hamiltonian (H0 = 0.5 * K)
    H0 = 0.5 * K
    
    print("\nComputing eigenvalues and eigenvectors...")
    # Solve generalized eigenvalue problem: K*c = E*O*c (or H0*c = E*O*c with H0 = K/2)
    try:
        # Request more than we need to ensure we get positive ones
        n_request = min(10, num_nodes - 2)
        eigenvalues, eigenvectors = spla.eigsh(H0, k=n_request, M=O, which='SM', tol=1e-6)
    except Exception as e:
        print(f"Error in eigsh: {e}")
        return None
    
    # Filter positive eigenvalues (boundary conditions introduce spurious negative ones)
    valid_idx = eigenvalues > 0.01
    if np.sum(valid_idx) < 2:
        # If filtering too strict, just keep smallest positive-ish ones
        valid_idx = eigenvalues > eigenvalues[0] - 0.1
    
    eigenvalues = eigenvalues[valid_idx]
    eigenvectors = eigenvectors[:, valid_idx]
    
    if len(eigenvalues) < 2:
        print("ERROR: Could not find at least 2 valid eigenvalues!")
        return None
    
    # Normalize eigenvectors with respect to overlap matrix
    for i in range(eigenvectors.shape[1]):
        norm = np.sqrt(np.real(eigenvectors[:, i].conj() @ O @ eigenvectors[:, i]))
        if norm > 1e-10:
            eigenvectors[:, i] /= norm
    
    print(f"Valid eigenvalues (first 3): {eigenvalues[:3]}")
    
    E0 = eigenvalues[0]
    E1 = eigenvalues[1]
    delta_E = E1 - E0
    omega = resonance_factor * delta_E / hbar
    
    print(f"E0 = {E0:.6f}")
    print(f"E1 = {E1:.6f}")
    print(f"ΔE = {delta_E:.6f}")
    print(f"ω = {omega:.6f} (resonance_factor={resonance_factor})")
    
    if delta_E <= 0.0:
        print("ERROR: Invalid eigenvalue ordering!")
        return None
    
    # Initial state: ground state
    c = eigenvectors[:, 0].astype(np.complex128)
    
    # Time evolution loop
    time_steps = np.arange(0, t_max, dt)
    n_steps = len(time_steps)
    n_states = min(6, eigenvectors.shape[1])
    
    # Storage for results
    history_norm = np.zeros(n_steps)
    history_projections = np.zeros((n_steps, n_states), dtype=np.complex128)
    
    print(f"\nEvolving state over {n_steps} time steps...")
    
    for step, t in enumerate(time_steps):
        if step % max(1, n_steps // 10) == 0:
            print(f"  Step {step}/{n_steps} (t={t:.2f})")
        
        # Store current norm
        N_t = np.real(c.conj() @ O @ c)
        history_norm[step] = N_t
        
        # Project onto eigenstates
        for i in range(n_states):
            history_projections[step, i] = eigenvectors[:, i].conj() @ O @ c
        
        # Crank-Nicolson step
        # [O - (dt/(2i*hbar))*H(t+dt)] c(t+dt) = [O + (dt/(2i*hbar))*H(t)] c(t)
        
        sin_t = np.sin(omega * t)
        sin_t_next = np.sin(omega * (t + dt))
        
        H_t = H0 + V * sin_t
        H_t_next = H0 + V * sin_t_next
        
        alpha = dt / (2.0j * hbar)
        
        # LHS: O - alpha*H(t+dt)
        LHS = O - alpha * H_t_next
        
        # RHS: (O + alpha*H(t)) * c(t)
        RHS = (O + alpha * H_t) @ c
        
        # Solve using sparse solver
        try:
            c = spla.spsolve(LHS.tocsr(), RHS)
            c = np.asarray(c).flatten()
        except Exception as e:
            print(f"Warning: solver failed at t={t}: {e}")
            pass
    
    # Compute probabilities
    probabilities = np.abs(history_projections)**2
    norm_safe = np.maximum(history_norm, 1e-14)
    probabilities /= norm_safe[:, np.newaxis]
    
    return {
        'time': time_steps,
        'norm': history_norm,
        'projections': history_projections,
        'probabilities': probabilities,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'omega': omega,
        'delta_E': delta_E,
        'eF': eF,
        'resonance_factor': resonance_factor,
        'L': L,
        'N': N,
    }


def plot_resonance(result):
    """Deliverable 1: Full resonance"""
    if result is None:
        print("Skipping plot (no result)")
        return
    
    time = result['time']
    prob = result['probabilities']
    
    # Plot 1: State populations
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, prob[:, 0], 'b-', linewidth=2.5, label='Ground state |1,1⟩')
    if prob.shape[1] > 1:
        ax.plot(time, prob[:, 1], 'r-', linewidth=2.5, label='Excited state |2,1⟩')
    if prob.shape[1] > 2:
        ax.plot(time, prob[:, 2], 'g-', linewidth=2.5, label='Excited state |1,2⟩')
    if prob.shape[1] > 3:
        ax.plot(time, np.sum(prob[:, 3:], axis=1), 'orange', linewidth=1.5, 
                linestyle='--', label='Higher states (leakage)')
    
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Probability $|p_i(t)|^2$', fontsize=12)
    ax.set_title(f'Resonant Dynamics: State Populations (N={result["N"]}, eF={result["eF"]:.4f})', 
                 fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/01a_resonant_populations.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Norm conservation
    fig, ax = plt.subplots(figsize=(12, 6))
    t_max_plot = 100.0
    idx_100 = np.where(time <= t_max_plot)[0]
    ax.plot(time[idx_100], result['norm'][idx_100], 'k-', linewidth=2)
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Norm N(t) = $c^T Oc$', fontsize=12)
    ax.set_title(f'Resonant Dynamics: Norm Conservation (N={result["N"]}, eF={result["eF"]:.4f})', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/01b_resonant_norm.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results
    print(f"\n{'='*70}")
    print(f"DELIVERABLE 1: Full Resonance (ω = ΔE/ℏ)")
    print(f"{'='*70}")
    if prob.shape[1] >= 3:
        max_excited_prob = np.max(prob[:, 1] + prob[:, 2])
        print(f"Maximum combined probability in degenerate excited states: {max_excited_prob:.6f}")
    print(f"Maximum ground state probability: {np.max(prob[:, 0]):.6f}")
    print(f"Minimum ground state probability: {np.min(prob[:, 0]):.6f}")
    print(f"Norm conservation: min={np.min(result['norm']):.6f}, max={np.max(result['norm']):.6f}")


def plot_half_resonance(result):
    """Deliverable 2: Half resonance"""
    if result is None:
        print("Skipping plot (no result)")
        return
    
    time = result['time']
    prob = result['probabilities']
    
    # Plot 1: State populations
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, prob[:, 0], 'b-', linewidth=2.5, label='Ground state |1,1⟩')
    if prob.shape[1] > 1:
        ax.plot(time, prob[:, 1], 'r-', linewidth=2.5, label='Excited state |2,1⟩')
    if prob.shape[1] > 2:
        ax.plot(time, prob[:, 2], 'g-', linewidth=2.5, label='Excited state |1,2⟩')
    if prob.shape[1] > 3:
        ax.plot(time, np.sum(prob[:, 3:], axis=1), 'orange', linewidth=1.5, 
                linestyle='--', label='Higher states (leakage)')
    
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Probability $|p_i(t)|^2$', fontsize=12)
    ax.set_title(f'Half-Resonant Dynamics: State Populations (N={result["N"]}, eF={result["eF"]:.4f})', 
                 fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/02a_half_resonance_populations.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Norm conservation
    fig, ax = plt.subplots(figsize=(12, 6))
    t_max_plot = 100.0
    idx_100 = np.where(time <= t_max_plot)[0]
    ax.plot(time[idx_100], result['norm'][idx_100], 'k-', linewidth=2)
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Norm N(t) = $c^T Oc$', fontsize=12)
    ax.set_title(f'Half-Resonant Dynamics: Norm Conservation (N={result["N"]}, eF={result["eF"]:.4f})', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/02b_half_resonance_norm.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"DELIVERABLE 2: Half Resonance (ω = ΔE/(2ℏ))")
    print(f"{'='*70}")
    if prob.shape[1] >= 3:
        max_excited_prob = np.max(prob[:, 1] + prob[:, 2])
        print(f"Maximum combined probability in degenerate excited states: {max_excited_prob:.6f}")
    print(f"Maximum ground state probability: {np.max(prob[:, 0]):.6f}")
    print(f"Minimum ground state probability: {np.min(prob[:, 0]):.6f}")
    print(f"Norm conservation: min={np.min(result['norm']):.6f}, max={np.max(result['norm']):.6f}")


def plot_leakage_analysis(results_eF):
    """Deliverable 3: Leakage vs field amplitude with time evolution"""
    if not results_eF or any(r is None for r in results_eF):
        print("Skipping leakage plot (no valid results)")
        return
    
    eF_values = [r['eF'] for r in results_eF]
    max_ground = []
    min_ground = []
    max_leakage = []
    oscillation_amplitude = []
    
    for result in results_eF:
        prob = result['probabilities']
        ground = prob[:, 0]
        
        max_ground.append(np.max(ground))
        min_ground.append(np.min(ground))
        oscillation_amplitude.append(np.max(ground) - np.min(ground))
        
        # Leakage to higher states
        if prob.shape[1] > 3:
            leakage = np.sum(prob[:, 3:], axis=1)
            max_leakage.append(np.max(leakage))
        else:
            max_leakage.append(0.0)
    
    # Plot 1: Leakage vs field amplitude
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eF_values, max_leakage, 'o-', linewidth=2.5, markersize=8, color='red')
    ax.set_xlabel('Field amplitude eF', fontsize=12)
    ax.set_ylabel('Max leakage to higher states', fontsize=12)
    ax.set_title('Max Leakage vs Field Amplitude (at resonance)', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/03a_leakage_vs_eF.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Rabi oscillation amplitude vs field
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eF_values, oscillation_amplitude, 's-', linewidth=2.5, markersize=8, color='green')
    ax.set_xlabel('Field amplitude eF', fontsize=12)
    ax.set_ylabel('Ground state oscillation amplitude', fontsize=12)
    ax.set_title('Rabi Oscillation Amplitude vs Field Strength', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/03b_rabi_amplitude.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Leakage probability time evolution for different eFs
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_eF)))
    for i, result in enumerate(results_eF):
        time = result['time']
        prob = result['probabilities']
        leakage_prob = np.sum(prob[:, 3:], axis=1) if prob.shape[1] > 3 else np.zeros_like(time)
        idx_100 = np.where(time <= 100.0)[0]
        ax.plot(time[idx_100], leakage_prob[idx_100], linewidth=2.5, 
                color=colors[i], label=f'eF={result["eF"]:.1f}')
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Leakage probability', fontsize=12)
    ax.set_title('Leakage Probability vs Time (for different eF values)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/03c_leakage_vs_time.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Ground state probability for different eFs
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, result in enumerate(results_eF):
        time = result['time']
        prob = result['probabilities']
        ground_prob = prob[:, 0]
        idx_100 = np.where(time <= 100.0)[0]
        ax.plot(time[idx_100], ground_prob[idx_100], linewidth=2.5, 
                color=colors[i], label=f'eF={result["eF"]:.1f}')
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Ground state probability', fontsize=12)
    ax.set_title('Ground State Probability vs Time (for different eF values)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/03d_ground_vs_time.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"DELIVERABLE 3: Leakage Analysis (Varying Field Amplitude)")
    print(f"{'='*70}")
    print(f"{'eF':<10} {'Max Ground':<15} {'Min Ground':<15} {'Oscillation':<15} {'Max Leakage':<15}")
    print(f"{'-'*70}")
    for i, eF in enumerate(eF_values):
        print(f"{eF:<10.5f} {max_ground[i]:<15.6f} {min_ground[i]:<15.6f} "
              f"{oscillation_amplitude[i]:<15.6f} {max_leakage[i]:<15.6f}")


def plot_omega_comparison():
    """Plot comparison of different omega values"""
    print("\n" + "="*70)
    print("ADDITIONAL: Omega Comparison (Different Resonance Factors)")
    print("="*70)
    
    resonance_factors = [0.25, 0.5, 1.0, 1.5, 2.0]
    results_omega = []
    
    for rf in resonance_factors:
        result = run_simulation(N=20, resonance_factor=rf, t_max=100.0)
        results_omega.append(result)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_omega)))
    
    # Plot 1: Ground state populations for different omegas
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, result in enumerate(results_omega):
        time = result['time']
        prob = result['probabilities']
        ground_prob = prob[:, 0]
        idx_100 = np.where(time <= 100.0)[0]
        ax.plot(time[idx_100], ground_prob[idx_100], linewidth=2.5, 
                color=colors[i], label=f'ω/ω₀={result["resonance_factor"]:.2f}')
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Ground state probability $|p_0(t)|^2$', fontsize=12)
    ax.set_title('Ground State Population for Different Resonance Factors', fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/04a_omega_ground_state.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Leakage for different omegas
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, result in enumerate(results_omega):
        time = result['time']
        prob = result['probabilities']
        leakage_prob = np.sum(prob[:, 3:], axis=1) if prob.shape[1] > 3 else np.zeros_like(time)
        idx_100 = np.where(time <= 100.0)[0]
        ax.plot(time[idx_100], leakage_prob[idx_100], linewidth=2.5, 
                color=colors[i], label=f'ω/ω₀={result["resonance_factor"]:.2f}')
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Leakage probability', fontsize=12)
    ax.set_title('Leakage to Higher States for Different Resonance Factors', fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/michal/AGH/sem6/CP2/Labs/6-FEM-Forced_oscillations/run/04b_omega_leakage.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Run all deliverables and additional plots"""
    
    # Deliverable 1: Full resonance
    print("\n" + "="*70)
    print("DELIVERABLE 1: Full Resonance (ω = ΔE/ℏ)")
    print("="*70)
    result_resonance = run_simulation(N=30, resonance_factor=1.0, t_max=150.0)
    plot_resonance(result_resonance)
    
    # Deliverable 2: Half resonance
    print("\n" + "="*70)
    print("DELIVERABLE 2: Half Resonance (ω = ΔE/(2ℏ))")
    print("="*70)
    result_half = run_simulation(N=30, resonance_factor=0.5, t_max=150.0)
    plot_half_resonance(result_half)
    
    # Deliverable 3: Leakage analysis
    print("\n" + "="*70)
    print("DELIVERABLE 3: Leakage Analysis (Varying eF)")
    print("="*70)
    eF_variations = []
    eF_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for eF_val in eF_values:
        result = run_simulation(N=30, resonance_factor=1.0, t_max=150.0, eF_override=eF_val)
        eF_variations.append(result)
    
    plot_leakage_analysis(eF_variations)
    
    # Additional: Omega comparison
    plot_omega_comparison()
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("Generated plots:")
    print("  1a. 01a_resonant_populations.png")
    print("  1b. 01b_resonant_norm.png")
    print("  2a. 02a_half_resonance_populations.png")
    print("  2b. 02b_half_resonance_norm.png")
    print("  3a. 03a_leakage_vs_eF.png")
    print("  3b. 03b_rabi_amplitude.png")
    print("  3c. 03c_leakage_vs_time.png")
    print("  3d. 03d_ground_vs_time.png")
    print("  4a. 04a_omega_ground_state.png")
    print("  4b. 04b_omega_leakage.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
