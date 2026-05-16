import os
import numpy as np
from scipy.sparse import csr_matrix

class FEMPersistence:
    def __init__(self, base_dir="previous_states"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_path(self, fem_obj, basis_type):
        """Creates a unique directory name based on N and Basis Type."""
        folder_name = f"N_{fem_obj.get_N()}_basis_{basis_type}"
        path = os.path.join(self.base_dir, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_stationary_results(self, fem_obj, basis_type, eigenvalues, eigenvectors):
        """
        Saves matrices and state data.
        eigenvectors: 2D array where each column is an eigenstate.
        """
        path = self._get_path(fem_obj, basis_type)
        
        # 1. Save Eigenvalues and Ground State (First column)
        np.save(os.path.join(path, "eigenvalues.npy"), eigenvalues)
        np.save(os.path.join(path, "ground_state.npy"), eigenvectors[:, 0])
        np.save(os.path.join(path, "first_excited.npy"), eigenvectors[:, 1])

        # 2. Save Matrices (CSR Format)
        # Extracting CSR data from your C++ FEM object [cite: 31, 32]
        h0_data = {
            'data': np.array(fem_obj.get_csr_val()),
            'indices': np.array(fem_obj.get_csr_col()),
            'indptr': np.array(fem_obj.get_csr_row())
        }
        o_data = {
            'data': np.array(fem_obj.get_overlap_csr_val()),
            'indices': np.array(fem_obj.get_overlap_csr_col()),
            'indptr': np.array(fem_obj.get_overlap_csr_row())
        }
        
        np.savez(os.path.join(path, "matrices.npz"), 
                 h0_val=h0_data['data'], h0_col=h0_data['indices'], h0_row=h0_data['indptr'],
                 o_val=o_data['data'], o_col=o_data['indices'], o_row=o_data['indptr'])
        
        print(f"Results saved to: {path}")

    def load_stationary_results(self, fem_obj, basis_type):
        """Loads data for the forced oscillation assignment."""
        path = self._get_path(fem_obj, basis_type)
        
        data = {
            "eigenvalues": np.load(os.path.join(path, "eigenvalues.npy")),
            "ground_state": np.load(os.path.join(path, "ground_state.npy")),
            "first_excited": np.load(os.path.join(path, "first_excited.npy"))
        }
        
        # Load matrices for Crank-Nicolson math 
        m = np.load(os.path.join(path, "matrices.npz"))
        data["H0"] = csr_matrix((m['h0_val'], m['h0_col'], m['h0_row']))
        data["O"] = csr_matrix((m['o_val'], m['o_col'], m['o_row']))
        
        return data
