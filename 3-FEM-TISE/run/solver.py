"""
2-D infinite square quantum well solved with FEM.

Solves  -∇²/2 Ψ = E Ψ  on [0,L]×[0,L] with Ψ = 0 on the boundary.
"""
import sys
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
sys.path.append('.')

try:
    import FEM
except ImportError:
    logging.error("Could not import the FEM module. Compile the C++ pybind11 extension first.")
    sys.exit(1)



def exact_eigenvalues(L: float, n_max: int = 6):
    """
    Return sorted list of (E, nx, ny) tuples for the 2-D infinite square well.

    E_{nx,ny} = (nx² + ny²) π² / (2 L²)
    """
    entries = []
    for nx in range(1, n_max + 1):
        for ny in range(1, n_max + 1):
            E = (nx**2 + ny**2) * np.pi**2 / (2.0 * L**2)
            entries.append((E, nx, ny))
    entries.sort(key=lambda t: t[0])
    return entries


def exact_wavefunction(x, y, nx: int, ny: int, L: float):
    """Normalised eigenfunction: (2/L) sin(nx π x/L) sin(ny π y/L)."""
    return (2.0 / L) * np.sin(nx * np.pi * x / L) * np.sin(ny * np.pi * y / L)

def plot_wavefunction_error_map(solver, fem_vec, nx, ny, L, grid_res=150,
                                output_dir=None, basis_name="", N=None):
    """
    Computes and plots the absolute spatial error |Exact - FEM|.
    """
    X, Y, Z_fem = evaluate_wavefunction_on_grid(solver, fem_vec, grid_res)

    Z_exact = exact_wavefunction(X, Y, nx, ny, L)

    if np.sum(Z_fem * Z_exact) < 0:
        Z_fem = -Z_fem

    error_map = np.abs(Z_exact - Z_fem)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(X, Y, error_map, cmap='viridis', shading='auto')
    fig.colorbar(mesh, ax=ax, label='|$\Psi_{exact} - \Psi_{FEM}$|')

    ax.set_title(f"Absolute Error Distribution (State $n_x={nx}, n_y={ny}$)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('scaled')
    fig.tight_layout()

    path = None
    if output_dir:
        basis_prefix = f"{basis_name.lower()}_" if basis_name else ""
        n_suffix = f"_N{N}" if N is not None else ""
        path = os.path.join(output_dir, f"{basis_prefix}wavefunction_error{n_suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logging.debug(f"Saved: {path}")

    return fig, path


def run_schrodinger(L: float, N: int, basis, n_eigs: int = 40):
    """
    Run the full FEM pipeline and return (solver, eigenvalues, eigenvectors).

    The returned eigenvalues are sorted and physical (BC spurious values
    ≈ -1 are discarded).  Eigenvectors are columns of the returned array.
    """
    basis_name = "Bilinear" if basis == FEM.BasisType.Bilinear else "Quadratic"
    logging.debug(f"EVP: {basis_name} N={N}")

    solver = FEM.FEM(L, N, 0.0, 0.0, basis)
    solver.build_mesh()
    solver.assemble_stiffness_matrix()
    solver.assemble_overlap_matrix()
    solver.apply_bc_eigenvalue()

    n = solver.get_num_nodes()

    K = sp.csr_matrix(
        (np.array(solver.get_csr_val()),
         np.array(solver.get_csr_col()),
         np.array(solver.get_csr_row())),
        shape=(n, n), dtype=float)

    O = sp.csr_matrix(
        (np.array(solver.get_overlap_csr_val()),
         np.array(solver.get_overlap_csr_col()),
         np.array(solver.get_overlap_csr_row())),
        shape=(n, n), dtype=float)

    H = 0.5 * K

    k_req = min(n_eigs, n - 2)
    vals, vecs = spla.eigsh(H.tocsc(), k=k_req, M=O.tocsc(),
                             sigma=1e-4, which='LM', tol=1e-10, maxiter=10000)

    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    physical = vals > 0.0
    vals, vecs = vals[physical], vecs[:, physical]

    logging.debug(f"  → {len(vals)} physical eigenvalues found "
                  f"(lowest: {vals[0]:.6f})")
    return solver, vals, vecs



def plot_spectrum_convergence(L, N_values, basis, output_dir=None):
    """
    For each N compute the FEM eigenvalues and compare with exact ones.
    Produces two panels:
      left  – absolute eigenvalue vs N (FEM lines, exact dashed)
      right – relative error |E_FEM - E_exact| / E_exact vs N (log scale)
    """
    basis_name = "Bilinear" if basis == FEM.BasisType.Bilinear else "Quadratic"

    exact_list = exact_eigenvalues(L, n_max=8)
    exact_energies = [e for e, nx, ny in exact_list]
    n_levels = min(8, len(exact_energies))
    exact_show = np.array(exact_energies[:n_levels])

    fem_results = {}
    for N in N_values:
        _, vals, _ = run_schrodinger(L, N, basis, n_eigs=max(50, n_levels + 10))
        fem_results[N] = vals

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Spectrum convergence — {basis_name} basis  (L={L})", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_levels))

    ax = axes[0]
    for li in range(n_levels):
        E_ex = exact_show[li]
        fem_vals_for_level = []
        for N in N_values:
            vals = fem_results[N]
            if li < len(vals):
                fem_vals_for_level.append(vals[li])
            else:
                fem_vals_for_level.append(np.nan)
        ax.plot(N_values, fem_vals_for_level, 'o-', color=colors[li],
                label=f"level {li+1} (FEM)")
        ax.axhline(E_ex, color=colors[li], linestyle='--', linewidth=0.8)

    ax.set_xlabel("N  (elements per side)")
    ax.set_ylabel("Energy eigenvalue E")
    ax.set_title("FEM eigenvalues (solid) vs exact (dashed)")
    ax.set_xscale('log')
    ax.set_xticks(N_values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    ax2 = axes[1]
    for li in range(n_levels):
        E_ex = exact_show[li]
        rel_errors = []
        for N in N_values:
            vals = fem_results[N]
            if li < len(vals):
                rel_errors.append(abs(vals[li] - E_ex) / E_ex)
            else:
                rel_errors.append(np.nan)
        ax2.plot(N_values, rel_errors, 'o-', color=colors[li],
                 label=f"level {li+1}")

    ax2.set_xlabel("N  (elements per side)")
    ax2.set_ylabel("|E_FEM − E_exact| / E_exact")
    ax2.set_title("Relative error (log-log)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks(N_values)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()

    path = None
    if output_dir:
        path = os.path.join(output_dir, f"{basis_name.lower()}_convergence.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logging.debug(f"Saved: {path}")
    return fig, path



def evaluate_wavefunction_on_grid(solver, vec, grid_res=80):
    """
    Set the FEM coefficient vector to *vec* and evaluate on a uniform grid.
    Returns (X, Y, Z) where Z[i,j] = Ψ(X[i,j], Y[i,j]).
    """
    L = solver.get_L()
    solver.set_coefficients(list(vec))
    x_vals = np.linspace(0, L, grid_res)
    y_vals = np.linspace(0, L, grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(solver.evaluate)(X, Y)
    return X, Y, Z


def plot_wavefunctions(solver, vals, vecs, basis_name, N, L,
                       n_states=6, output_dir=None):
    """
    Plot |Ψ|² for the first *n_states* physical eigenstates alongside
    the corresponding exact |Ψ_exact|².
    """
    exact_list = exact_eigenvalues(L, n_max=8)
    exact_energies_sorted = sorted(set(round(e, 10) for e, _, _ in exact_list))

    n_plot = min(n_states, len(vals))
    ncols = 3
    nrows = int(np.ceil(n_plot / ncols))

    fig_fem, axes_fem = plt.subplots(nrows, ncols,
                                     figsize=(5 * ncols, 4 * nrows))
    axes_fem = np.array(axes_fem).reshape(-1)
    fig_fem.suptitle(f"FEM |Ψ|²  —  {basis_name} basis  N={N}", fontsize=14)

    for idx in range(n_plot):
        vec = vecs[:, idx]
        X, Y, Z = evaluate_wavefunction_on_grid(solver, vec)
        ax = axes_fem[idx]
        if Z[Z.shape[0]//2, Z.shape[1]//2] < 0:
            Z = -Z
        c = ax.contourf(X, Y, Z**2, levels=30, cmap='inferno')
        E_label = vals[idx]
        ax.set_title(f"E_{idx+1} = {E_label:.4f}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig_fem.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_plot, len(axes_fem)):
        axes_fem[idx].set_visible(False)
    fig_fem.tight_layout()

    fig_ex, axes_ex = plt.subplots(nrows, ncols,
                                   figsize=(5 * ncols, 4 * nrows))
    axes_ex = np.array(axes_ex).reshape(-1)
    fig_ex.suptitle(f"Exact |Ψ|²  —  {basis_name} basis reference  N={N}",
                    fontsize=14)

    x_vals = np.linspace(0, L, 80)
    y_vals = np.linspace(0, L, 80)
    X, Y = np.meshgrid(x_vals, y_vals)

    quantum_numbers = []
    used = {}
    for idx in range(n_plot):
        E_fem = vals[idx]
        closest = min(exact_list, key=lambda t: abs(t[0] - E_fem))
        E_ex, nx, ny = closest
        key = (min(nx, ny), max(nx, ny))
        if key in used:
            nx, ny = ny, nx
        used[key] = True
        quantum_numbers.append((E_ex, nx, ny))

    for idx in range(n_plot):
        E_ex, nx, ny = quantum_numbers[idx]
        Z_ex = exact_wavefunction(X, Y, nx, ny, L)
        ax = axes_ex[idx]
        c = ax.contourf(X, Y, Z_ex**2, levels=30, cmap='inferno')
        ax.set_title(f"E_exact ({nx},{ny}) = {E_ex:.4f}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig_ex.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_plot, len(axes_ex)):
        axes_ex[idx].set_visible(False)
    fig_ex.tight_layout()

    paths = []
    if output_dir:
        p1 = os.path.join(output_dir, f"{basis_name.lower()}_N{N}_wavefunction_fem.png")
        p2 = os.path.join(output_dir, f"{basis_name.lower()}_N{N}_wavefunction_exact.png")
        fig_fem.savefig(p1, dpi=150, bbox_inches='tight')
        fig_ex.savefig(p2,  dpi=150, bbox_inches='tight')
        logging.debug(f"Saved: {p1}")
        logging.debug(f"Saved: {p2}")
        paths = [p1, p2]

    return fig_fem, fig_ex, paths



def print_spectrum_table(vals, L, n_levels=10, basis_name=""):
    exact_list = exact_eigenvalues(L, n_max=8)
    exact_energies = [e for e, nx, ny in exact_list]
    header = f"{'':>4}  {'E_FEM':>12}  {'E_exact':>12}  {'rel. err':>12}"
    print(f"\n{basis_name} spectrum")
    print(header)
    print("─" * len(header))
    for i in range(min(n_levels, len(vals), len(exact_energies))):
        E_fem = vals[i]
        E_ex  = exact_energies[i]
        err   = abs(E_fem - E_ex) / E_ex
        print(f"{i+1:>4}  {E_fem:>12.6f}  {E_ex:>12.6f}  {err:>12.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FEM Schrödinger solver: 2-D infinite square quantum well"
    )
    parser.add_argument("--L", type=float, default=5.0,
                        help="Domain side length (default: 5.0)")
    parser.add_argument("--N-convergence", type=int, nargs="+",
                        default=[5, 10, 20, 40],
                        help="N values for convergence study (default: 5 10 20 40)")
    parser.add_argument("--N-wavefunction", type=int, default=30,
                        help="N used for wave-function plots (default: 30)")
    parser.add_argument("--n-states", type=int, default=6,
                        help="Number of eigenstates to plot (default: 6)")
    parser.add_argument("--basis", type=str,
                        choices=["bilinear", "quadratic", "both"], default="both",
                        help="Basis type (default: both)")
    parser.add_argument("--plot-show", action="store_true",
                        help="Display plots on screen")
    parser.add_argument("--plot-save", action="store_true",
                        help="Save plots to output directory")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory for saved plots (default: ./output)")

    args = parser.parse_args()

    if args.L <= 0:
        logging.error("L must be positive"); sys.exit(1)

    output_dir = None
    if args.plot_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Output directory: {output_dir.absolute()}")

    basis_map = {
        "bilinear":  [FEM.BasisType.Bilinear],
        "quadratic": [FEM.BasisType.Quadratic],
        "both":      [FEM.BasisType.Bilinear, FEM.BasisType.Quadratic],
    }
    bases = basis_map[args.basis]

    all_figs = []

    for basis in bases:
        basis_name = "Bilinear" if basis == FEM.BasisType.Bilinear else "Quadratic"

        logging.debug(f"=== {basis_name}: spectrum convergence study ===")
        fig_conv, _ = plot_spectrum_convergence(
            args.L, args.N_convergence, basis, output_dir)
        all_figs.append(fig_conv)

        logging.debug(f"=== {basis_name}: wave functions (N={args.N_wavefunction}) ===")
        solver, vals, vecs = run_schrodinger(
            args.L, args.N_wavefunction, basis,
            n_eigs=max(50, args.n_states + 10))

        print_spectrum_table(vals, args.L, n_levels=10, basis_name=basis_name)

        fig_fem, fig_ex, _ = plot_wavefunctions(
            solver, vals, vecs, basis_name,
            args.N_wavefunction, args.L,
            n_states=args.n_states,
            output_dir=output_dir)

        error_fig, _ = plot_wavefunction_error_map(
            solver, vecs[:, 0], 1, 1, args.L,
            grid_res=150,
            output_dir=output_dir,
            basis_name=basis_name,
            N=args.N_wavefunction,
        )

        all_figs.extend([fig_fem, fig_ex, error_fig])

    if args.plot_show:
        plt.show()
    else:
        for f in all_figs:
            plt.close(f)

    logging.debug("Done.")
