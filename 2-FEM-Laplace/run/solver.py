import sys
import argparse
import logging
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Configure logging to debug level
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Ensure the module can be found if running directly inside the 'run' dir
sys.path.append('.')

try:
    import FEM
except ImportError:
    logging.error("Could not import the FEM module.")
    logging.error("Make sure you have compiled the C++ code into a shared library (.so, .pyd, or .dll) "
                  "using pybind11 and placed it in the 'run' directory.")
    sys.exit(1)

def run_and_evaluate(L=5.0, N=10, rc_x=-1.0, rc_y=0.0, basis=FEM.BasisType.Bilinear):
    """
    Initializes the FEM solver, runs the pipeline, and evaluates the error.
    """
    basis_name = "Bilinear" if basis == FEM.BasisType.Bilinear else "Quadratic"
    logging.debug(f"Running FEM Solver with {basis_name} Basis (L={L}, N={N}, rc=({rc_x}, {rc_y}))")
    
    # 1. Initialize solver
    solver = FEM.FEM(L, N, rc_x, rc_y, basis)
    
    # 2. Run the pipeline
    solver.build_mesh()
    solver.assemble_stiffness_matrix()
    solver.apply_boundary_conditions()
    solver.solve()
    
    # 3. Evaluate precision over a grid
    grid_res = 50
    x_vals = np.linspace(0, L, grid_res)
    y_vals = np.linspace(0, L, grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    Z_approx = np.zeros_like(X)
    Z_exact = np.zeros_like(X)
    
    for i in range(grid_res):
        for j in range(grid_res):
            Z_approx[i, j] = solver.evaluate(X[i, j], Y[i, j])
            Z_exact[i, j] = solver.exact_solution(X[i, j], Y[i, j])
            
    # Calculate the average square of the difference (MSE)
    diff = Z_approx - Z_exact
    mse = np.mean(diff**2)
    
    logging.debug(f"{basis_name} Solution MSE: {mse:.6e}")
    
    return X, Y, Z_approx, Z_exact, diff, basis_name, mse

def plot_approximate(X, Y, Z_approx, title, N, output_dir=None):
    """
    Creates figure for approximate solution.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    c = ax.contourf(X, Y, Z_approx, levels=50, cmap='viridis')
    ax.set_title(f"{title} Basis - FEM Approximate Solution (N={N})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, f"{title.lower()}_N{N}_approximate.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logging.debug(f"Saved: {path}")
        return fig, path
    return fig, None

def plot_exact(X, Y, Z_exact, title, N, output_dir=None):
    """
    Creates figure for exact solution.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    c = ax.contourf(X, Y, Z_exact, levels=50, cmap='viridis')
    ax.set_title(f"{title} Basis - Exact Solution (N={N})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, f"{title.lower()}_N{N}_exact.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logging.debug(f"Saved: {path}")
        return fig, path
    return fig, None

def plot_error(X, Y, diff, title, mse, N, output_dir=None):
    """
    Creates figure for error distribution.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    c = ax.contourf(X, Y, diff, levels=50, cmap='coolwarm')
    ax.set_title(f"{title} Basis - Error Distribution (MSE: {mse:.6e}, N={N})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, f"{title.lower()}_N{N}_error.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logging.debug(f"Saved: {path}")
        return fig, path
    return fig, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FEM Laplace Solver with Bilinear and Quadratic basis functions"
    )
    parser.add_argument(
        "--L",
        type=float,
        default=5.0,
        help="Domain size [0, L] × [0, L] (default: 5.0)"
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Number of elements per side (can specify multiple values, default: 50 100 200)"
    )
    parser.add_argument(
        "--rc-x",
        type=float,
        default=-1.0,
        help="Point charge x-coordinate (default: -1.0)"
    )
    parser.add_argument(
        "--rc-y",
        type=float,
        default=0.0,
        help="Point charge y-coordinate (default: 0.0)"
    )
    parser.add_argument(
        "--basis",
        type=str,
        choices=["bilinear", "quadratic", "both"],
        default="both",
        help="Basis type to use: bilinear, quadratic, or both (default: both)"
    )
    parser.add_argument(
        "--plot-show",
        action="store_true",
        help="Display plots on screen"
    )
    parser.add_argument(
        "--plot-save",
        action="store_true",
        help="Save plots to output directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save plots (default: ./output)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.L <= 0:
        logging.error("L must be positive")
        sys.exit(1)
    
    if any(n <= 0 for n in args.N):
        logging.error("All N values must be positive")
        sys.exit(1)

    logging.debug(f"Configuration: L={args.L}, N={args.N}, rc=({args.rc_x}, {args.rc_y}), "
                  f"basis={args.basis}, plot_show={args.plot_show}, plot_save={args.plot_save}")
    
    # Create output directory if saving is enabled
    output_dir = None
    if args.plot_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Output directory: {output_dir.absolute()}")

    # Determine which basis functions to run
    basis_to_run = []
    if args.basis in ["bilinear", "both"]:
        basis_to_run.append(FEM.BasisType.Bilinear)
    if args.basis in ["quadratic", "both"]:
        basis_to_run.append(FEM.BasisType.Quadratic)

    # Run solver for each N value and basis type
    for N in args.N:
        logging.debug(f"Processing N={N}")
        
        for basis in basis_to_run:
            X, Y, Z_approx, Z_exact, diff, basis_name, mse = run_and_evaluate(
                args.L, N, args.rc_x, args.rc_y, basis
            )
            
            if args.plot_show or args.plot_save:
                # Plot approximate solution
                fig1, path1 = plot_approximate(X, Y, Z_approx, basis_name, N,
                                             output_dir if args.plot_save else None)
                
                # Plot exact solution
                fig2, path2 = plot_exact(X, Y, Z_exact, basis_name, N,
                                        output_dir if args.plot_save else None)
                
                # Plot error
                fig3, path3 = plot_error(X, Y, diff, basis_name, mse, N,
                                        output_dir if args.plot_save else None)
                
                if args.plot_show:
                    plt.show()
                else:
                    # Close figures if not showing to save memory
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)
