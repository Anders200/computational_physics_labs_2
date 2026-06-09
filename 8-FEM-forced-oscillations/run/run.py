import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from fd_solver import FD_solver

L = 1.0
v = 1.0
N = 100
T_FINAL = 80.0


def simulate(omega: float, damping: float, t_final: float = T_FINAL):
    solver = FD_solver(N, L, v, damping, omega)
    solver.evolve(t_final)

    energy = np.asarray(solver.get_energy_history(), dtype=float)
    disp = np.asarray(solver.get_displacement_history(), dtype=float)

    dx = L / N
    dt = dx / (2.0 * v)
    t = np.arange(len(energy), dtype=float) * dt

    return t, energy, disp, dt


def average_energy(t: np.ndarray, energy: np.ndarray, t0: float = 10.0, t1: float = 80.0) -> float:
    mask = (t >= t0) & (t <= t1)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapz(energy[mask], t[mask]) / (t1 - t0))


def estimate_periodicity(t: np.ndarray, signal: np.ndarray, min_time: float = 10.0):
    mask = t >= min_time
    sig = signal[mask]
    sig = sig - sig.mean()
    if sig.size < 8:
        return None, None

    dt = t[1] - t[0]
    freq = np.fft.rfftfreq(sig.size, d=dt)
    power = np.abs(np.fft.rfft(sig)) ** 2

    if power.size <= 1:
        return None, None

    peak_idx = int(np.argmax(power[1:]) + 1)
    peak_freq = freq[peak_idx]
    if peak_freq <= 0:
        return None, None

    return 1.0 / peak_freq, peak_freq


def local_maxima(x: np.ndarray, y: np.ndarray, rel_threshold: float = 0.2):
    if y.size < 3:
        return []
    max_y = y.max() if y.size else 0.0
    peaks = []
    for i in range(1, y.size - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] >= rel_threshold * max_y:
            peaks.append((x[i], y[i]))
    return peaks


def main():
    out_dir = Path(__file__).parent

    omega_pi = math.pi
    damping_base = 1.0

    t, energy, disp, dt = simulate(omega_pi, damping_base)

    period_est, freq_est = estimate_periodicity(t, disp)
    if period_est is None:
        periodic_statement = "Insufficient data to assess periodicity."
    else:
        driving_period = 2.0 * math.pi / omega_pi
        rel_err = abs(period_est - driving_period) / driving_period
        periodic_statement = (
            f"Estimated period ≈ {period_est:.3f}, driving period = {driving_period:.3f}. "
            f"Relative error ≈ {rel_err:.2%}."
        )

    print("Deliverable: periodicity for ω = π")
    print(periodic_statement)
    print()

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, energy, lw=1.2)
    plt.xlabel("t")
    plt.ylabel("E(t)")
    plt.title("Energy vs time (ω = π)")
    plt.tight_layout()
    plt.savefig(out_dir / "energy_omega_pi.png", dpi=160)
    plt.close()

    omegas = np.linspace(1.5, 15.0, 40)
    avg_energy = np.zeros_like(omegas)
    for i, omega in enumerate(omegas):
        t_i, e_i, _, _ = simulate(float(omega), damping_base)
        avg_energy[i] = average_energy(t_i, e_i)

    plt.figure(figsize=(8, 4.5))
    plt.plot(omegas, avg_energy, lw=1.2)
    plt.xlabel("ω")
    plt.ylabel("⟨E⟩(ω)")
    plt.title("Average energy over t ∈ (10, 80)")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_energy_vs_omega.png", dpi=160)
    plt.close()

    peaks = local_maxima(omegas, avg_energy, rel_threshold=0.2)
    if peaks:
        peak_str = ", ".join([f"{p[0]:.3f}" for p in peaks])
    else:
        peak_str = "No clear peaks detected."

    print("Deliverable: resonant frequencies (approx.)")
    print(peak_str)
    print()

    damping_values = [0.2, 1.0, 2.0]
    plt.figure(figsize=(8, 4.5))
    for d in damping_values:
        avg_d = np.zeros_like(omegas)
        for i, omega in enumerate(omegas):
            t_i, e_i, _, _ = simulate(float(omega), float(d))
            avg_d[i] = average_energy(t_i, e_i)
        plt.plot(omegas, avg_d, lw=1.2, label=f"damping={d}")

    plt.xlabel("ω")
    plt.ylabel("⟨E⟩(ω)")
    plt.title("Resonant peaks vs damping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "avg_energy_vs_omega_damping.png", dpi=160)
    plt.close()

    print("Deliverable: damping effect")
    print("Generated avg_energy_vs_omega_damping.png (lower damping -> sharper, higher peaks).")


if __name__ == "__main__":
    main()
