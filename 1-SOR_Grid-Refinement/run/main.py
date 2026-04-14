
import poisson_solver
import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor

# Physical & Analytical Constants
DOMAIN_SIZE = 3.0 
S_ANALYTICAL = -(3.0 * np.pi / 2.0)**2

# Module-level helper for ProcessPoolExecutor (must be picklable)
def count_iterations_to_convergence(omega, n_power=6, max_iters=10000, tol=1e-3):
    """Run SOR and return iterations to convergence."""
    solver = poisson_solver.PoissonSolver2D(n_power)
    solver.solve_sor(max_iters=max_iters, tol=tol, omega=omega)
    iters = len(solver.get_residual_history()) * 10
    return iters

# ============ Plotting Functions ============

def plot_convergence_gs(n_power=6):
    """Gauss-Seidel convergence and residual history."""
    print(f"--- Running Gauss-Seidel for N=2^{n_power} ---")
    solver = poisson_solver.PoissonSolver2D(n_power, log_every=10)
    solver.solve_gauss_seidel(max_iters=5000, tol=1e-5)
    
    res_hist = solver.get_residual_history()
    en_hist = solver.get_energy_history()
    iters = np.arange(len(res_hist)) * 10

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max Residual', color='tab:red')
    ax1.semilogy(iters, res_hist, color='tab:red', label='Residual')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Energy S', color='tab:blue')
    ax2.plot(iters, en_hist, color='tab:blue', linestyle='--', label='Energy')
    ax2.axhline(y=S_ANALYTICAL, color='black', linestyle=':', label='Analytical S')
    
    plt.title(f"Gauss-Seidel Convergence (N=2^{n_power})")
    fig.tight_layout()
    plt.savefig("gauss_seidel_convergence.png")
    plt.show()

def plot_energy_vs_n():
    """Energy vs grid size (discretization error)."""
    ns = [4, 5, 6, 7]
    energies = []
    
    for n in ns:
        solver = poisson_solver.PoissonSolver2D(n)
        solver.solve_gauss_seidel(max_iters=10000, tol=1e-3)
        energies.append(solver.compute_energy())
        
    errors = [abs(e - S_ANALYTICAL) for e in energies]
    n_values = [2**n for n in ns]

    plt.figure(figsize=(8, 5))
    plt.loglog(n_values, errors, 'o-', label='|S_calc - S_analytical|')
    plt.xlabel('N (Grid Points)')
    plt.ylabel('Energy Error')
    plt.title('Discretization Error: Energy vs N')
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("energy_vs_n.png")
    plt.show()

def test_sor(omega, n_power=6, max_iters=10000, tol=1e-3):
    """SOR convergence with energy and residual plots."""
    print(f"Testing SOR with ω={omega:.4f} on N=2^{n_power} grid...")
    solver = poisson_solver.PoissonSolver2D(n_power, log_every=10)
    solver.solve_sor(max_iters=max_iters, tol=tol, omega=omega)
    
    res_hist = solver.get_residual_history()
    en_hist = solver.get_energy_history()
    iters = np.arange(len(res_hist)) * 10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Energy convergence
    ax1.semilogy(iters, np.abs(np.array(en_hist) - S_ANALYTICAL), 'b-o', 
                label='|S - S_analytical|', markersize=4)
    ax1.axhline(y=1e-5, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy Error (log)')
    ax1.set_title(f'Energy Convergence (ω={omega:.4f})')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()
    
    # Residual convergence
    ax2.semilogy(iters, res_hist, 'r-o', label='Max Residual δ', markersize=4)
    ax2.axhline(y=tol, color='b', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max Residual δ (log)')
    ax2.set_title(f'Residual Convergence (ω={omega:.4f})')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"sor_convergence_omega_{omega:.4f}.png", dpi=150)
    plt.show()

def plot_error_map(n_power=6, omega=1.8):
    """Error between numerical and analytical solution."""
    print(f"Computing error map for N=2^{n_power}...")
    solver = poisson_solver.PoissonSolver2D(n_power)
    solver.solve_sor(max_iters=5000, tol=1e-4, omega=omega)
    
    u_numerical = np.array(solver.get_solution())
    N = u_numerical.shape[0] - 1
    
    x = np.linspace(0, 3, N + 1)
    y = np.linspace(0, 3, N + 1)
    xx, yy = np.meshgrid(x, y)
    u_analytical = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    
    error = np.abs(u_numerical - u_analytical)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    im1 = axes[0].imshow(u_numerical, origin='lower', cmap='RdBu_r', aspect='auto')
    axes[0].set_title(f'Numerical Solution (N=2^{n_power})')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(u_analytical, origin='lower', cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Analytical Solution')
    fig.colorbar(im2, ax=axes[1])
    
    error_nonzero = np.maximum(error, 1e-16)
    im3 = axes[2].imshow(np.log10(error_nonzero), origin='lower', cmap='hot', aspect='auto')
    axes[2].set_title('Pointwise Error log₁₀(|error|)')
    fig.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'Error Analysis (N=2^{n_power}, ω={omega:.4f})')
    plt.tight_layout()
    plt.savefig(f"error_map_n{n_power}_omega{omega:.4f}.png", dpi=150)
    plt.show()
    
    max_error = error.max()
    mean_error = error.mean()
    l2_error = np.sqrt(np.mean(error**2))
    print(f"Error statistics: max={max_error:.2e}, mean={mean_error:.2e}, L2={l2_error:.2e}")

def compare_convergence_rates(n_power=9):
    """GS (ω=1) vs optimized SOR convergence comparison."""
    print(f"--- Convergence Rate Comparison (N=2^{n_power}) ---")
    
    # Estimate optimal omega
    n_coarse = max(2, n_power - 2)
    omegas_scan = np.linspace(1.0, 1.99, 8)
    results = {}
    for w in omegas_scan:
        solver_temp = poisson_solver.PoissonSolver2D(n_coarse)
        solver_temp.solve_sor(max_iters=2000, tol=1e-3, omega=w)
        iters = len(solver_temp.get_residual_history()) * 10
        results[w] = iters
    
    opt_w = min(results, key=results.get)
    print(f"Estimated optimal ω: {opt_w:.4f}")
    
    # Run GS
    solver_gs = poisson_solver.PoissonSolver2D(n_power, log_every=10)
    solver_gs.solve_sor(max_iters=100000, tol=1e-5, omega=1.0)
    res_gs = solver_gs.get_residual_history()
    iters_gs = np.arange(len(res_gs)) * 10
    
    # Run optimized SOR
    solver_sor = poisson_solver.PoissonSolver2D(n_power, log_every=10)
    solver_sor.solve_sor(max_iters=10000, tol=1e-5, omega=opt_w)
    res_sor = solver_sor.get_residual_history()
    iters_sor = np.arange(len(res_sor)) * 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(iters_gs, res_gs, 'b-o', linewidth=2, markersize=5, 
               label=f'Gauss-Seidel (ω=1, {len(res_gs)*10} iters)')
    ax.semilogy(iters_sor, res_sor, 'r-s', linewidth=2, markersize=5, 
               label=f'SOR (ω={opt_w:.4f}, {len(res_sor)*10} iters)')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Maximum Residual δ (log)', fontsize=12)
    ax.set_title(f'Convergence Rate: GS vs SOR (N=2^{n_power})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    speedup = (len(res_gs) * 10) / (len(res_sor) * 10)
    ax.text(0.5, 0.05, f'Speedup: {speedup:.2f}x', transform=ax.transAxes,
           fontsize=12, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f"convergence_comparison_n{n_power}.png", dpi=150)
    plt.show()

def compare_grid_refinement_vs_sor(final_n=10, omega=1.985, lower_n=5):
    """Compare grid refinement vs direct SOR convergence."""
    print(f"--- Grid Refinement vs Direct SOR (final N=2^{final_n}) ---")
    
    # Direct SOR
    solver_direct = poisson_solver.PoissonSolver2D(final_n, log_every=1)
    solver_direct.solve_sor(max_iters=100000, tol=1e-3, omega=omega)
    res_direct = solver_direct.get_residual_history()
    en_direct = solver_direct.get_energy_history()
    iters_direct = np.arange(len(res_direct))
    
    # Grid refinement
    solver_refine = poisson_solver.PoissonSolver2D(final_n, log_every=1)
    solver_refine.solve_with_refinement(lower_n, max_iters=100000, tol=1e-3, omega=omega)
    res_refine = solver_refine.get_residual_history()
    en_refine = solver_refine.get_energy_history()
    iters_refine = np.arange(len(res_refine))
    
    print(f"Direct SOR: {len(res_direct)} iterations")
    print(f"Grid refinement: {len(res_refine)} iterations")
    speedup = len(res_direct) / max(len(res_refine), 1)
    print(f"Speedup: {speedup:.2f}x")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residual (linear)
    ax = axes[0, 0]
    ax.plot(iters_direct, res_direct, 'b-', linewidth=1.5, alpha=0.7, label='Direct SOR')
    ax.plot(iters_refine, res_refine, 'r-', linewidth=1.5, alpha=0.7, label='Grid Refinement')
    ax.axhline(y=1e-3, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Residual δ')
    ax.set_title('Residual Convergence (Linear)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual (log)
    ax = axes[0, 1]
    ax.semilogy(iters_direct, res_direct, 'b-o', linewidth=1.5, markersize=2, alpha=0.7)
    ax.semilogy(iters_refine, res_refine, 'r-s', linewidth=1.5, markersize=2, alpha=0.7)
    ax.axhline(y=1e-3, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Residual δ (log)')
    ax.set_title('Residual Convergence (Log)')
    ax.grid(True, alpha=0.3, which='both')
    
    # Energy (linear)
    ax = axes[1, 0]
    ax.plot(iters_direct, en_direct, 'b-', linewidth=1.5, alpha=0.7)
    ax.plot(iters_refine, en_refine, 'r-', linewidth=1.5, alpha=0.7)
    ax.axhline(y=S_ANALYTICAL, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy S')
    ax.set_title('Energy Convergence (Linear)')
    ax.grid(True, alpha=0.3)
    
    # Energy error (log)
    ax = axes[1, 1]
    en_error_direct = np.abs(np.array(en_direct) - S_ANALYTICAL)
    en_error_refine = np.abs(np.array(en_refine) - S_ANALYTICAL)
    ax.semilogy(iters_direct, np.maximum(en_error_direct, 1e-15), 'b-o', linewidth=1.5, markersize=2)
    ax.semilogy(iters_refine, np.maximum(en_error_refine, 1e-15), 'r-s', linewidth=1.5, markersize=2)
    ax.axhline(y=1e-3, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|S - S_analytical| (log)')
    ax.set_title('Energy Error Convergence (Log)')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(f'Grid Refinement vs Direct SOR (N=2^{final_n})', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"refinement_vs_sor_n{final_n}.png", dpi=150)
    plt.show()

def find_optimal_omega_parallel(n_power=6, num_workers=6):
    """Find optimal ω using parallel coarse scan + fine golden-section search."""
    print(f"--- Finding optimal ω for N=2^{n_power} (parallel) ---")
    
    # Phase 1: Coarse parallel scan
    n_power_coarse = max(2, n_power - 2)
    print(f"Phase 1: Parallel scan on N=2^{n_power_coarse}...")
    omegas_coarse = np.linspace(1.0, 1.99, 12)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = {executor.submit(count_iterations_to_convergence, w, n_power_coarse): w 
                for w in omegas_coarse}
        results = {}
        for future in tasks:
            w = tasks[future]
            iters = future.result()
            results[w] = iters
            print(f"  ω={w:.4f}: {iters:.0f} iterations")
    
    opt_w_coarse = min(results, key=results.get)
    print(f"Coarse optimum: ω={opt_w_coarse:.4f}")
    
    # Phase 2: Fine golden-section search
    print("\nPhase 2: Fine golden-section search...")
    a = max(1.0, opt_w_coarse - 0.2)
    b = min(1.99, opt_w_coarse + 0.2)
    
    def golden_search(func, a, b, tol=1e-3, max_iter=30):
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi
        x_history, f_history = [], []
        
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        f1 = func(x1)
        f2 = func(x2)
        
        for iteration in range(max_iter):
            x_history.append((x1 + x2) / 2)
            f_history.append(min(f1, f2))
            print(f"  Iter {iteration}: x1={x1:.6f} (f={f1:.1f}), x2={x2:.6f} (f={f2:.1f})")
            
            if b - a < tol:
                x_opt = (a + b) / 2
                f_opt = func(x_opt)
                return x_opt, f_opt, x_history, f_history
            
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = a + resphi * (b - a)
                f1 = func(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = b - resphi * (b - a)
                f2 = func(x2)
        
        x_opt = (a + b) / 2
        f_opt = func(x_opt)
        return x_opt, f_opt, x_history, f_history
    
    opt_w, min_iters, x_hist, f_hist = golden_search(
        lambda w: count_iterations_to_convergence(w, n_power=n_power),
        a, b, tol=1e-2, max_iter=20
    )
    
    print(f"\nOptimal ω: {opt_w:.6f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    omegas_sorted = sorted(results.keys())
    iters_sorted = [results[w] for w in omegas_sorted]
    ax1.plot(omegas_sorted, iters_sorted, 'g-o', label='Coarse scan')
    ax1.axvline(x=opt_w_coarse, color='orange', linestyle='--', label=f'Coarse optimum')
    ax1.set_xlabel('ω')
    ax1.set_ylabel('Iterations to convergence')
    ax1.set_title('Phase 1: Coarse Parallel Scan')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(x_hist, f_hist, 'b-o', label='Golden-section search')
    ax2.axvline(x=opt_w, color='r', linestyle='--', label=f'Optimal ω={opt_w:.4f}')
    ax2.set_xlabel('ω')
    ax2.set_ylabel('Iterations to convergence')
    ax2.set_title('Phase 2: Fine Golden-Section Search')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("optimal_omega_parallel.png")
    plt.show()
    
    return opt_w

def plot_timing_refinement_vs_direct(final_n=10, omega=1.985, lower_n=5):
    """Timing comparison: direct SOR vs grid refinement."""
    print(f"--- Timing Comparison (final N=2^{final_n}) ---")
    
    # Direct SOR
    solver_direct = poisson_solver.PoissonSolver2D(final_n)
    t0 = time.time()
    solver_direct.solve_sor(max_iters=100000, tol=1e-3, omega=omega)
    direct_time_s = time.time() - t0
    
    # Grid refinement
    solver_ref = poisson_solver.PoissonSolver2D(final_n)
    solver_ref.solve_with_refinement(lower_n, max_iters=100000, tol=1e-3, omega=omega)
    refine_times_ms = solver_ref.get_refinement_timing_history()
    refine_times_s = [t / 1000.0 for t in refine_times_ms]
    cumulative_refine_time_s = sum(refine_times_s)
    
    # Build labels
    level_labels = [f"2^{lower_n + k}" for k in range(len(refine_times_s))]
    all_labels = ['Direct SOR', 'Cumulative GR'] + level_labels
    all_times = [direct_time_s, cumulative_refine_time_s] + refine_times_s
    colors = ['tab:blue', 'tab:orange'] + ['tab:green'] * len(refine_times_s)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(all_labels, all_times, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Timing: Direct SOR vs Grid Refinement (ω={omega:.4f})', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"timing_refinement_vs_direct_n{final_n}.png", dpi=150)
    plt.show()
    
    print(f"Direct SOR: {direct_time_s:.3f} s")
    print(f"Cumulative GR: {cumulative_refine_time_s:.3f} s")
    for label, t_s in zip(level_labels, refine_times_s):
        print(f"  Level {label}: {t_s:.3f} s")
    speedup = direct_time_s / cumulative_refine_time_s if cumulative_refine_time_s > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

def test_grid_refinement(final_n=10, lower_n=5, omega=1.985):
    """Grid refinement test with timing and residual plots."""
    print(f"--- Grid Refinement Test (2^{lower_n} → 2^{final_n}, ω={omega:.4f}) ---")
    
    # Solve with finer logging (log_every=1 for 10x more history)
    solver = poisson_solver.PoissonSolver2D(final_n, log_every=100, history_stride=50)
    
    t0 = time.time()
    solver.solve_with_refinement(lower_n, max_iters=100000, tol=1e-3, omega=omega)
    total_time = time.time() - t0
    
    res_hist = solver.get_residual_history()
    en_hist = solver.get_energy_history()
    
    # Convert iteration count to time (assume linear time per iteration)
    iters = np.arange(len(res_hist))
    times = iters * (total_time / max(len(res_hist), 1))
    
    print(f"Converged in {len(res_hist)} steps ({total_time:.3f} s)")
    print(f"Final residual: {res_hist[-1]:.2e}")
    print(f"Final energy: {en_hist[-1]:.6f}")
    print(f"Analytical energy: {S_ANALYTICAL:.6f}")
    print(f"Energy error: {abs(en_hist[-1] - S_ANALYTICAL):.2e}")
    
    # Plot timing vs residual (log-log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Time vs Residual (log-log)
    ax1.loglog(times, res_hist, 'b-o', linewidth=1.5, markersize=3, label='Residual')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Max Residual δ (log)', fontsize=12)
    ax1.set_title(f'Time vs Residual (2^{lower_n}→2^{final_n})')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()
    
    # Right: Iteration vs Residual (log-log)
    ax2.loglog(iters + 1, res_hist, 'r-o', linewidth=1.5, markersize=3, label='Residual')
    ax2.set_xlabel('Iteration (log)', fontsize=12)
    ax2.set_ylabel('Max Residual δ (log)', fontsize=12)
    ax2.set_title(f'Iteration vs Residual (2^{lower_n}→2^{final_n})')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"test_grid_refinement_n{final_n}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    opt_w = 1.985 # found by find_optimal_omega_parallel
    
    # plot_convergence_gs(n_power=6)
    # plot_energy_vs_n()
    # test_sor(opt_w, n_power=9)
    # plot_error_map(n_power=9, omega=opt_w)
    # compare_convergence_rates(n_power=9)
    # compare_grid_refinement_vs_sor(final_n=10, omega=opt_w, lower_n=5)
    # opt_w = find_optimal_omega_parallel(n_power=10, num_workers=6)
    test_grid_refinement(final_n=13, lower_n=9, omega=opt_w)
    
    # plot_timing_refinement_vs_direct(final_n=11, omega=opt_w, lower_n=5)
