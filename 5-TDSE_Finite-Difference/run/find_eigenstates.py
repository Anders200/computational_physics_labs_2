import os
import numpy as np
import itm  

def run_itm_solver(n_points=100, omega_x=1.0, omega_y=2.0, num_states=5, force_recompute=False):
    """
    Runs the Imaginary Time Method to find the lowest eigenstates.
    If states for these parameters already exist on disk, it loads them instantly.
    """
    output_path = os.path.join("states", "eigenstates.npz")

    # 1. Try to load existing states to save time
    if not force_recompute and os.path.exists(output_path):
        data = np.load(output_path)
        
        # Verify the saved data matches our current physical parameters
        if (data['N'] == n_points and 
            np.isclose(data['omega_x'], omega_x) and 
            np.isclose(data['omega_y'], omega_y) and 
            len(data['psi']) >= num_states):
            
            print(f"Found cached states at {output_path}! Loading instantly...")
            return {
                "psi": data['psi'][:num_states],  # Slice just the states we need
                "energies": data['energies'][:num_states],
                "dx": data['dx'].item(),          # .item() unwraps 0D numpy arrays back to python scalars
                "N": data['N'].item()
            }
        else:
            print("Cached states exist, but parameters differ. Recomputing...")

    # 2. If no valid cache exists, compute them
    alpha_factor = 0.9 
    max_iter = 20000
    tolerance = 1e-10

    solver = itm.Solver(n_points, alpha_factor, omega_x, omega_y, max_iter, tolerance)
    
    print(f"Computing {num_states} eigenstates using ITM...")
    solver.solve(num_states - 1)  

    # 3. Extract Results
    energies = solver.get_state_energies()
    states = solver.get_states()
    dx = solver.grid_dx()
    wavefunctions = np.array(states)

    # 4. Save to Disk for next time
    os.makedirs("states", exist_ok=True)
    np.savez(
        output_path,
        psi=wavefunctions,
        energies=np.array(energies),
        N=n_points,
        dx=dx,
        omega_x=omega_x,
        omega_y=omega_y
    )
    print(f"States saved to {output_path}")

    return {
        "psi": wavefunctions,
        "energies": energies,
        "dx": dx,
        "N": n_points
    }

if __name__ == "__main__":
    run_itm_solver()
