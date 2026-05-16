import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import os

# Assuming the FEMPersistence class is in a file named persistence.py
from persistence import FEMPersistence
import FEM as fem_module

def solve_stationary_system(L, N, basis_type):
    """
    Handles the logic for checking, solving, and saving the stationary 
    Schrödinger eigenstates.
    """
    db = FEMPersistence()
    
    # 1. Initialize the FEM object from your C++ module
    # rc_x and rc_y are 0.0 as they are for Laplace mode
    fem = fem_module.FEM(L, N, 0.0, 0.0, basis_type)
    fem.build_mesh()
    
    # Check if we already have this configuration saved
    try:
        data = db.load_stationary_results(fem, basis_type)
        print(f"--- Found existing results for N={N}, Basis={basis_type}. Loading... ---")
        return fem, data
    except FileNotFoundError:
        print(f"--- No existing results found. Solving stationary EVP for N={N}... ---")

    # 2. Assemble Matrices
    fem.assemble_stiffness_matrix()
    fem.assemble_overlap_matrix()
    
    # 3. Apply Boundary Conditions for Eigenvalue Problem
    # Sets H_ii = -1 and O_ii = 1 for boundary nodes 
    fem.apply_bc_eigenvalue()
    
    # 4. Convert CSR data to Scipy Sparse for Eigensolver
    # H = 1/2 * Stiffness Matrix 
    h_val = np.array(fem.get_csr_val()) * 0.5 
    h_col = np.array(fem.get_csr_col())
    h_row = np.array(fem.get_csr_row())
    H = csr_matrix((h_val, h_col, h_row))

    o_val = np.array(fem.get_overlap_csr_val())
    o_col = np.array(fem.get_overlap_csr_col())
    o_row = np.array(fem.get_overlap_csr_row())
    O = csr_matrix((o_val, o_col, o_row))

    # 5. Solve the Generalized Eigenvalue Problem: Hc = EOc [cite: 53]
    # We look for the smallest magnitude eigenvalues (SM)
    # k=10 should be enough to capture the first few physical states
    eigenvalues, eigenvectors = eigsh(H, M=O, k=10, which='SM')

    # 6. Filter spurious boundary states
    # Boundary nodes were assigned eigenvalue -1 
    # Physical eigenvalues for 2D well are positive 
    physical_mask = eigenvalues > 0
    eigenvalues = eigenvalues[physical_mask]
    eigenvectors = eigenvectors[:, physical_mask]

    # 7. Save and Return
    db.save_stationary_results(fem, basis_type, eigenvalues, eigenvectors)
    
    data = {
        "eigenvalues": eigenvalues,
        "ground_state": eigenvectors[:, 0],
        "first_excited": eigenvectors[:, 1],
        "H0": H,
        "O": O
    }
    
    return fem, data

# Example Usage:
if __name__ == "__main__":
    L_val = 5.0  # Side length from assignment [cite: 49]
    N_elements = 500
    basis = fem_module.BasisType.Bilinear

    fem_obj, stationary_data = solve_stationary_system(L_val, N_elements, basis)

    print(f"Ground State Energy (E0): {stationary_data['eigenvalues'][0]}")
    print(f"First Excited Energy (E1): {stationary_data['eigenvalues'][1]}")
    
    # Now the FEM object and ground state are ready for the Forced assignment.
    # You can set the ground state into the C++ object like this:
    fem_obj.set_coefficients(stationary_data["ground_state"])
