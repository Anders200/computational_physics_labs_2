import numpy as np
import matplotlib.pyplot as plt
import TdSE  # Your compiled pybind11 module
from find_eigenstates import run_itm_solver

def main():
    # -------------------------------------------------------------------------
    # 1. Assignment Parameters (rt.pdf Page 1)
    # -------------------------------------------------------------------------
    # Sweet spot for spatial accuracy and physical stability without micro-steps
    N = 52
    dt = 0.001         # Stiffer grids (N=128) require a smaller time-step
    omega_x = 1.0       
    omega_y = 1.001     
    m = 1.0             

    nominal_period = 2 * np.pi
    t_max = 2 * nominal_period
    total_steps = int(t_max / dt)  

    # -------------------------------------------------------------------------
    # 2. Compute or Load Eigenstates using ITM (find_eigenstates.py)
    # -------------------------------------------------------------------------
    print("--- Step 1: Finding first 3 eigenstates using ITM ---")
    itm_results = run_itm_solver(
        n_points=N, 
        omega_x=omega_x, 
        omega_y=omega_y, 
        num_states=3
    )

    wavefunctions = itm_results["psi"]
    psi0 = wavefunctions[0].tolist()
    psi1 = wavefunctions[1].tolist()
    psi2 = wavefunctions[2].tolist()

    # -------------------------------------------------------------------------
    # 3. Setup and Run Time-Dependent Schrödinger Equation (TdSE)
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Initializing TdSE Crank-Nicolson Simulator ---")
    sim = TdSE.tdse(N=N, dt=dt, omega_x=omega_x, omega_y=omega_y, m=m)
    sim.set_initial_state(psi0, psi1, psi2)

    initial_psi = np.array(psi0).reshape(N, N)
    dx = 4.0 / N
    x_coord = (np.arange(N) - N / 2.0) * dx
    y_coord = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x_coord, y_coord)
    prob = np.abs(initial_psi)**2
    norm_0 = np.sum(prob)
    x_avg_0 = np.sum(X.T * prob) / norm_0 if norm_0 > 0 else 0
    y_avg_0 = np.sum(Y.T * prob) / norm_0 if norm_0 > 0 else 0

    print(f"Evolving system for {total_steps} steps (Max Time: {t_max:.4f} a.u.)...")
    
    sim.evolve(total_steps)

    # -------------------------------------------------------------------------
    # 4. Extract History Observables
    # -------------------------------------------------------------------------
    norm_history   = np.array(sim.get_norm_history())
    energy_history = np.array(sim.get_energy_history())
    x_avg_history  = np.array(sim.get_x_avg_history())
    y_avg_history  = np.array(sim.get_y_avg_history())

    norm_history = np.concatenate(([norm_0], norm_history))
    energy_history = np.concatenate(([energy_history[0]] if len(energy_history) > 0 else [0], energy_history))
    x_avg_history = np.concatenate(([x_avg_0], x_avg_history))
    y_avg_history = np.concatenate(([y_avg_0], y_avg_history))
    
    # Use actual length of history arrays
    num_recorded_steps = len(norm_history)
    actual_time_reached = (num_recorded_steps - 1) * dt
    time_axis = np.linspace(0, actual_time_reached, num_recorded_steps)

    # -------------------------------------------------------------------------
    # 5. Generate Graphs
    # -------------------------------------------------------------------------
    print(f"\n--- Step 3: Generating Graphs (Plotting up to t = {actual_time_reached:.4f} a.u.) ---")

    # Plot 1: Norm and Average Energy vs Time
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Time (atomic units)')
    ax1.set_ylabel('Wavefunction Norm', color=color)
    line1 = ax1.plot(time_axis, norm_history, color=color, lw=2, label='Norm')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(min(norm_history) - 0.001, max(norm_history) + 0.001)
    ax1.grid(True, linestyle=':')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Average Energy (atomic units)', color=color)
    line2 = ax2.plot(time_axis, energy_history, color=color, linestyle='--', lw=2, label='Energy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Wavefunction Norm & Average Energy vs Time')
    fig.tight_layout()
    plt.savefig('norm_energy_vs_time.png', dpi=300)

    # Plot 2: Position expectation values <x>(t) and <y>(t)
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, x_avg_history, label=r'$\langle x \rangle(t)$', color='blue', lw=1.5)
    plt.plot(time_axis, y_avg_history, label=r'$\langle y \rangle(t)$', color='darkorange', linestyle='--', lw=1.5)
    plt.xlabel('Time (atomic units)')
    plt.ylabel('Position Expectation Value')
    plt.title(r'Position Expectation Values $\langle x \rangle$ and $\langle y \rangle$ vs Time')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig('position_vs_time.png', dpi=300)

    # Plot 3: Trajectory y(x)
    plt.figure(figsize=(6, 6))
    plt.plot(x_avg_history, y_avg_history, color='purple', lw=1.5, label='Packet Trajectory')
    plt.scatter(x_avg_history[0], y_avg_history[0], color='green', marker='o', s=100, zorder=5, label='Start (t=0)')
    plt.scatter(x_avg_history[-1], y_avg_history[-1], color='red', marker='X', s=100, zorder=5, label=f'End (t={actual_time_reached:.2f})')
    plt.xlabel(r'$\langle x \rangle$')
    plt.ylabel(r'$\langle y \rangle$')
    plt.title(r'Trajectory of the Wave Packet Center $\langle y \rangle$ vs $\langle x \rangle$')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('trajectory_y_vs_x.png', dpi=300)

    # Plot 4: Final wavefunction probability density |Psi(x, y)|^2
    final_psi = np.array(sim.get_psi()).reshape(N, N)
    final_density = np.abs(final_psi) ** 2

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        final_density.T,
        extent=[x_coord[0], x_coord[-1], y_coord[0], y_coord[-1]],
        origin='lower',
        cmap='magma',
        aspect='equal'
    )
    plt.colorbar(im, label=r'$|\Psi(x, y)|^2$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(rf'Final Wavefunction Probability Density at $t={actual_time_reached:.2f}$ a.u.')
    plt.tight_layout()
    plt.savefig('wavefunction_density.png', dpi=300)
    
    print("All plots saved successfully.")

if __name__ == "__main__":
    main()
