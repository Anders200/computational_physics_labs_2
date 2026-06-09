#!/usr/bin/env python3
"""Driver script for the CUDA FEM membrane solver."""

import sys, os, math, time

sys.path.insert(0, os.path.dirname(__file__))

try:
    import membrane
except ImportError as e:
    sys.exit(f"ERROR: cannot import membrane module ({e}).\n"
             "Did you build with cmake and place the .so in run/?")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


L     = 5.0
v     = 1.0
N     = 150


def run_case(omega_label: str, omega: float, n_periods: int = 6):
    """Run one driving-frequency case and return time-series of field_max."""

    dt = membrane.cfl_dt(L, v, N) * 0.05

    print(f"\n{'='*60}")
    print(f"  ω = {omega_label}  ({omega:.6f} rad/s)")
    print(f"  CFL dt = {membrane.cfl_dt(L, v, N):.6f}   using dt = {dt:.6f}")

    s = membrane.Solver(L, v, omega, dt, N)
    s.initialize_mesh()

    T_drive  = 2 * math.pi / omega
    n_total  = int(n_periods * T_drive / dt)

    times = [s.get_time()]
    norms = [s.field_norm()]

    t0 = time.perf_counter()
    for _ in range(n_total):
        s.step()
        norms.append(s.field_norm())
        times.append(s.get_time())
    elapsed = time.perf_counter() - t0

    print(f"  {n_total} steps in {elapsed:.2f}s  "
          f"({n_total/elapsed:.0f} steps/s)")
    print(f"  Final field_max = {s.field_max():.4e}")

    if HAS_NUMPY and HAS_MPL:
        Np1  = s.get_nodes_per_side()
        psi  = np.array(s.get_field()).reshape(Np1, Np1)
        x    = np.array(s.get_x_coords()).reshape(Np1, Np1)
        y    = np.array(s.get_y_coords()).reshape(Np1, Np1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"2-D FEM membrane  ω = {omega_label}  t = {s.get_time():.3f}", fontsize=13)

        ax = axes[0]
        im = ax.imshow(
            psi,
            origin="lower",
            extent=(0, L, 0, L),
            cmap="RdBu_r",
            vmin=psi.min(),
            vmax=psi.max(),
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label="Ψ")
        ax.set_title("Displacement field Ψ(x,y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.plot(L / 2, L / 2, "ko", ms=5, label="driven node")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        ax2.plot(times, norms, lw=1.4)
        ax2.set_xlabel("time"); ax2.set_ylabel("‖Ψ‖₂")
        ax2.set_title("L2 norm vs time")
        ax2.grid(True, alpha=0.4)

        safe_label = omega_label.replace('/', '_').replace(' ', '')
        fname = os.path.join(os.path.dirname(__file__), f"result_{safe_label}.png")
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  Saved → {fname}")

    return times, norms


def test_dt_stability(omega_label: str, omega: float, n_periods: int = 4,
                      scales=None, norm2_limit: float = 1e6):
    """Run multiple dt scales and report stability by ‖Ψ‖₂² threshold."""
    if scales is None:
        scales = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

    print(f"\nDT stability sweep  ω = {omega_label}")
    for scale in scales:
        dt = membrane.cfl_dt(L, v, N) * scale
        s = membrane.Solver(L, v, omega, dt, N)
        s.initialize_mesh()

        T_drive = 2 * math.pi / omega
        n_total = int(n_periods * T_drive / dt)

        stable = True
        for _ in range(n_total):
            s.step()
            norm2 = s.field_norm() ** 2
            if norm2 > norm2_limit:
                stable = False
                break

        status = "stable" if stable else "unstable"
        print(f"  {scale:.2f} - {status}")


if __name__ == "__main__":
    cases = [
        ("π/L",       math.pi / L),
        # ("2π/L",      2 * math.pi / L),
        # ("√2·π/L",    math.sqrt(2.0) * math.pi / L),
    ]

    for label, omega in cases:
        run_case(label, omega)

    # test_dt_stability("π/L", math.pi / L)

    print("\nDone.")