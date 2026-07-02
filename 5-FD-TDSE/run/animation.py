import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import TdSE  # Your compiled pybind11 module
from find_eigenstates import run_itm_solver

def main():
    # -------------------------------------------------------------------------
    # 1. System Parameters (Scales beautifully to large N now!)
    # -------------------------------------------------------------------------
    N = 100
    dt = 0.0001
    omega_x = 1.0       
    omega_y = 1.001
    m = 1.0             

    nominal_period = 2 * np.pi
    t_max = 2 * nominal_period
    total_steps = int(t_max / dt)

    dx = 4.0 / N
    x_coord = (np.arange(N) - N / 2.0) * dx
    y_coord = (np.arange(N) - N / 2.0) * dx

    # -------------------------------------------------------------------------
    # 2. Get Eigenstates from ITM & Plot Spatial Profiles
    # -------------------------------------------------------------------------
    print(f"--- Step 1: Finding and Reshaping ITM Eigenstates (N={N}) ---")
    itm_results = run_itm_solver(n_points=N, omega_x=omega_x, omega_y=omega_y, num_states=3)
    wavefunctions = itm_results["psi"]

    psi0_2d = wavefunctions[0].reshape(N, N)
    psi1_2d = wavefunctions[1].reshape(N, N)
    psi2_2d = wavefunctions[2].reshape(N, N)

    fig_static, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    states_data = [psi0_2d, psi1_2d, psi2_2d]
    titles = ["Ground State ($\Psi_0$)", "1st Excited State ($\Psi_1$)", "2nd Excited State ($\Psi_2$)"]

    for ax, data, title in zip(axes, states_data, titles):
        im = ax.imshow(data.T, extent=[x_coord[0], x_coord[-1], y_coord[0], y_coord[-1]], 
                       origin='lower', cmap='seismic')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig_static.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"ITM Computed Eigenstates (Spatial Profiles, N={N})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("static_eigenstates.png", dpi=300)
    print("-> Saved static eigenstates visualization to: static_eigenstates.png")

    # -------------------------------------------------------------------------
    # 3. Memory-Safe TdSE Setup (No massive array accumulation)
    # -------------------------------------------------------------------------
    print(f"\n--- Step 2: Initializing TdSE Crank-Nicolson Simulator (N={N}) ---")
    sim = TdSE.tdse(N=N, dt=dt, omega_x=omega_x, omega_y=omega_y, m=m)
    sim.set_initial_state(wavefunctions[0].tolist(), wavefunctions[1].tolist(), wavefunctions[2].tolist())
    
    step_stride = 500
    num_frames = total_steps // step_stride

    print("Preparing 2D Dynamic Animation Canvas...")
    fig_anim, ax_anim = plt.subplots(figsize=(7, 6))
    
    # Run the very first stride to get an initial state scale
    sim.evolve(step_stride)
    init_psi = np.array(sim.get_psi()).reshape(N, N)
    init_density = np.abs(init_psi)**2

    im_anim = ax_anim.imshow(init_density.T, extent=[x_coord[0], x_coord[-1], y_coord[0], y_coord[-1]], 
                             origin='lower', cmap='magma', vmin=0, vmax=np.max(init_density) * 0.9)
    
    ax_anim.set_xlabel('Position x (a.u.)', fontsize=11)
    ax_anim.set_ylabel('Position y (a.u.)', fontsize=11)
    title_text = ax_anim.set_title("Time-Dependent Probability Density $|\Psi(x,y)|^2$\nTime = 0.00 a.u.", fontweight='bold')
    fig_anim.colorbar(im_anim, ax=ax_anim, label='Probability Density')

    # -------------------------------------------------------------------------
    # 4. Generator-Based Animation (Zero Memory Footprint)
    # -------------------------------------------------------------------------
    def frame_generator():
        """Generator that evolves the C++ state sequentially, 
        yielding only one snapshot at a time to keep RAM empty."""
        current_time = step_stride * dt
        for frame in range(1, num_frames):
            sim.evolve(step_stride) # Step the C++ engine forward incrementally
            
            # Fetch and immediate reshape
            current_psi = np.array(sim.get_psi()).reshape(N, N)
            prob_density = np.abs(current_psi)**2
            
            current_time += step_stride * dt
            yield prob_density, current_time

    def update_frame(data):
        """Receives only the single current frame data from the generator"""
        prob_density, current_time = data
        im_anim.set_array(prob_density.T)
        title_text.set_text(f"Time-Dependent Probability Density $|\Psi(x,y)|^2$\nTime = {current_time:.2f} a.u.")
        return [im_anim, title_text]

    # Passing the generator instead of a pre-allocated array list saves your RAM
    ani = animation.FuncAnimation(
        fig_anim, 
        update_frame, 
        frames=frame_generator, 
        save_count=num_frames,
        interval=30, 
        blit=True, 
        repeat=False
    )

    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=30)

    ani.save(
        "tdse_evolution.gif",
        writer=writer,
        dpi=150
    )

    print("Saved GIF as tdse_evolution.gif")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
