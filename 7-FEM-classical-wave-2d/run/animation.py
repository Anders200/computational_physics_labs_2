#!/usr/bin/env python3
"""Matplotlib animation for the CUDA FEM membrane solver."""

import sys, os, math, time, argparse

sys.path.insert(0, os.path.dirname(__file__))

try:
    import membrane
except ImportError as e:
    sys.exit(
        f"ERROR: cannot import membrane module ({e}).\n"
        "Did you build with cmake and place the .so in run/?"
    )

try:
    import numpy as np
except ImportError as e:
    sys.exit("ERROR: numpy is required for animation.")

try:
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception as e:
    sys.exit(f"ERROR: matplotlib import failed ({e}).")


def parse_omega(omega_str: str, L: float) -> float:
    """Parse omega from a string like 'pi/L', '2*pi/L', or a float."""
    s = omega_str.strip().lower().replace(" ", "")
    if s in {"pi/l", "π/l"}:
        return math.pi / L
    if s in {"2pi/l", "2*pi/l", "2π/l"}:
        return 2.0 * math.pi / L
    if s in {"sqrt2*pi/l", "sqrt(2)*pi/l", "√2*pi/l", "√2·pi/l"}:
        return math.sqrt(2.0) * math.pi / L
    return float(s)


def main():
    parser = argparse.ArgumentParser(description="Animate 2-D FEM membrane evolution.")
    parser.add_argument("--L", type=float, default=5.0)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--omega", type=str, default="pi/L",
                        help="Driving frequency (e.g. pi/L, 2*pi/L, sqrt(2)*pi/L, or float).")
    parser.add_argument("--periods", type=float, default=3.0)
    parser.add_argument("--dt-scale", type=float, default=0.10)
    parser.add_argument("--steps-per-frame", type=int, default=0,
                        help="Override steps per frame (0 = auto).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", type=str, default="",
                        help="Optional output file (.mp4 or .gif).")
    args = parser.parse_args()

    L = args.L
    v = args.v
    N = args.N
    omega = parse_omega(args.omega, L)
    omega = math.sqrt(2) * math.pi / L if omega == 0 else omega

    dt = membrane.cfl_dt(L, v, N) * args.dt_scale
    s = membrane.Solver(L, v, omega, dt, N)
    s.initialize_mesh()

    T_drive = 2 * math.pi / omega
    n_total = max(1, int(args.periods * T_drive / dt))

    if args.steps_per_frame and args.steps_per_frame > 0:
        steps_per_frame = args.steps_per_frame
    else:
        target_frames = min(300, n_total)
        steps_per_frame = max(1, n_total // target_frames)

    n_frames = max(1, n_total // steps_per_frame)

    Np1 = s.get_nodes_per_side()
    x = np.array(s.get_x_coords()).reshape(Np1, Np1)
    y = np.array(s.get_y_coords()).reshape(Np1, Np1)

    psi = np.array(s.get_field()).reshape(Np1, Np1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"2-D FEM membrane  ω = {args.omega}  t = {s.get_time():.3f}", fontsize=13)

    im = ax0.imshow(
        psi,
        origin="lower",
        extent=(0, L, 0, L),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        animated=True,
    )
    fig.colorbar(im, ax=ax0, label="Ψ")
    ax0.set_title("Displacement field Ψ(x,y)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_aspect("equal")
    ax0.plot(L / 2, L / 2, "ko", ms=5, label="driven node")
    ax0.legend(fontsize=8)

    times = [s.get_time()]
    norms = [s.field_norm()]
    (line,) = ax1.plot(times, norms, lw=1.4)
    ax1.set_xlabel("time")
    ax1.set_ylabel("‖Ψ‖₂")
    ax1.set_title("L2 norm vs time")
    ax1.grid(True, alpha=0.4)

    def update(frame_idx):
        for _ in range(steps_per_frame):
            s.step()

        psi = np.array(s.get_field()).reshape(Np1, Np1)
        im.set_data(psi)

        t = s.get_time()
        times.append(t)
        norms.append(s.field_norm())
        line.set_data(times, norms)
        ax1.relim()
        ax1.autoscale_view()

        fig.suptitle(f"2-D FEM membrane  ω = {args.omega}  t = {t:.3f}", fontsize=13)
        return (im, line)

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / max(1, args.fps),
        blit=False,
    )

    if not os.environ.get("DISPLAY") and not args.save:
        print("WARNING: No DISPLAY detected; use --save to write an animation file.")

    if args.save:
        outfile = os.path.join(os.path.dirname(__file__), args.save)
        ani.save(outfile, fps=args.fps)
        print(f"Saved animation → {outfile}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
