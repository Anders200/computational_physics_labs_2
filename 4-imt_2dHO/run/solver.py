"""


Usage
-----
    cd run/
    python3 solver.py

The .so is built by CMake (pybind11_add_module target 'itm') and
placed in run/ automatically — this script just imports it and runs.

Outputs written to run/output/:
    convergence_alpha.png   — ⟨E⟩ vs iteration for several α values
    convergence_excited.png — ⟨E⟩ vs iteration for excited states 1-4
    wavefunctions.png       — 2D colour maps of Ψ₀ … Ψ₄
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# The .so lives next to this script (run/); make sure Python finds it.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT  = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

import itm                                           # built by CMake as 'itm'


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Parameters
# ─────────────────────────────────────────────────────────────────────────────

N        = 300
OMEGA_X  = 1.0
OMEGA_Y  = 2.0
N_STATES = 4     # solve states 0 … N_STATES

def exact_spectrum(count, omega_x, omega_y):
    spectrum = []
    max_quantum_number = count - 1
    for nx in range(max_quantum_number + 1):
        for ny in range(max_quantum_number + 1):
            energy = omega_x * (nx + 0.5) + omega_y * (ny + 0.5)
            spectrum.append((energy, (nx, ny)))

    spectrum.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    return spectrum[:count]


EXACT_SPECTRUM = exact_spectrum(N_STATES + 1, OMEGA_X, OMEGA_Y)
EXACT = [energy for energy, _ in EXACT_SPECTRUM]
EXACT_LABELS = [f"({nx},{ny})" for _, (nx, ny) in EXACT_SPECTRUM]


def save_solutions(path, solver, histories=None):
    np.savez_compressed(
        path,
        energies=np.asarray(solver.get_state_energies(), dtype=np.float64),
        states=np.asarray(solver.get_states(), dtype=np.float64),
        dx=np.float64(solver.grid_dx()),
        N=np.int64(solver.grid_size()),
        omega_x=np.float64(OMEGA_X),
        omega_y=np.float64(OMEGA_Y),
        histories=np.asarray(histories, dtype=object) if histories is not None else np.asarray([], dtype=object),
    )


def load_solutions(path, solver):
    data = np.load(path, allow_pickle=True)
    solver.set_state_energies(data["energies"].tolist())
    solver.set_states(data["states"].tolist())
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Figure 1 — convergence vs α
# ─────────────────────────────────────────────────────────────────────────────
#
# alpha_factors = [0.3, 0.6, 0.9, 0.95, 1.01]
# colors        = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_factors)))
#
# fig, ax = plt.subplots(figsize=(8, 5))
# for af, col in zip(alpha_factors, colors):
#     try:
#         s    = itm.Solver(N, af, OMEGA_X, OMEGA_Y, 100000, 1e-6)
#         hist = s.solve(0)
#         lbl  = f"α = {af:.2f}·αc"
#         dE = np.abs(np.diff(np.asarray(hist, dtype=np.float64)))
#         if dE.size == 0:
#             continue
#         dE = np.where(dE > 0, dE, 1e-16)
#         ax.semilogy(dE, color=col, lw=1.5, label=lbl)
#     except Exception as e:
#         print(f"α_factor={af} diverged: {e}")
#
# ax.axhline(1e-6, ls="--", color="k", lw=0.8, label="tol = 1e-6")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("|ΔE|")
# ax.set_title("Ground-state convergence for various α")
# ax.legend(fontsize=9)
# ax.set_xlim(left=0)
# fig.tight_layout()
# fig.savefig(os.path.join(OUT, "convergence_alpha.png"), dpi=150)
# plt.close(fig)
# print("Saved", os.path.join(OUT, "convergence_alpha.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main solve — states 0 … N_STATES with safe α = 0.9 αc
# ─────────────────────────────────────────────────────────────────────────────


solver = itm.Solver(N, 0.9, OMEGA_X, OMEGA_Y, 10000000, 1e-8)
histories = []
solve_times = []
for k in range(N_STATES + 1):
    t0 = time.time()
    histories.append(solver.solve(k))
    t1 = time.time()
    solve_times.append(t1 - t0)

dx     = solver.grid_dx()
E_num  = solver.get_state_energies()
states = solver.get_states()

print("\n=== Eigenvalues ===")
print(f"{'State':>6}  {'E_numerical':>14}  {'E_exact':>10}  {'|error|':>10}")
for k in range(N_STATES + 1):
    print(f"{k:>6}  {E_num[k]:>14.8f}  {EXACT[k]:>10.4f}  {abs(E_num[k]-EXACT[k]):>10.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Figure 2 — excited-state convergence
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
colors2 = plt.cm.plasma(np.linspace(0.1, 0.85, N_STATES))
for k in range(1, N_STATES + 1):
    hist = np.asarray(histories[k], dtype=np.float64)
    dE = np.abs(np.diff(hist))
    if dE.size == 0:
        continue
    dE = np.where(dE > 0, dE, 1e-16)
    ax.semilogy(dE, color=colors2[k-1], lw=1.5, label=f"State {k}")

ax.axhline(1e-7, ls="--", color="k", lw=0.8, label="tol = 1e-7")
ax.set_xlabel("Iteration")
ax.set_ylabel("|⟨E⟩ − E_exact|")
ax.set_title("Convergence of excited states (α = 0.9 αc)")
ax.legend(fontsize=9)
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "convergence_excited.png"), dpi=150)
plt.close(fig)
print("Saved", os.path.join(OUT, "convergence_excited.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figure 3 — wavefunction visualisation
# ─────────────────────────────────────────────────────────────────────────────

x = (np.arange(N) - N / 2.0) * dx
X, Y = np.meshgrid(x, x, indexing="ij")

fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(2, N_STATES + 1, figure=fig,
                        hspace=0.45, wspace=0.35)

for k in range(N_STATES + 1):
    psi = np.array(states[k]).reshape(N, N)
    # Canonicalise sign so the dominant lobe is positive
    if psi.max() < -psi.min():
        psi = -psi

    ax  = fig.add_subplot(gs[0, k])
    lim = np.abs(psi).max()
    im  = ax.pcolormesh(X, Y, psi, cmap="RdBu_r",
                        vmin=-lim, vmax=lim, shading="auto")
    ax.set_aspect("equal")
    ax.set_title(f"Ψ_{k}  {EXACT_LABELS[k]}\nE={E_num[k]:.4f}", fontsize=8)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Bottom row: probability density |Ψ|²
for k in range(N_STATES + 1):
    psi  = np.array(states[k]).reshape(N, N)
    prob = psi ** 2
    ax   = fig.add_subplot(gs[1, k])
    im   = ax.pcolormesh(X, Y, prob, cmap="hot_r", shading="auto")
    ax.set_aspect("equal")
    ax.set_title(f"|Ψ_{k}|²", fontsize=8)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Eigenfunctions of the 2D anisotropic harmonic oscillator  "
             f"(ωx={OMEGA_X}, ωy={OMEGA_Y})", fontsize=11)
fig.savefig(os.path.join(OUT, "wavefunctions.png"), dpi=150)
plt.close(fig)
print("Saved", os.path.join(OUT, "wavefunctions.png"))


save_solutions(os.path.join(OUT, "solutions.npz"), solver, histories=histories)
print("Saved", os.path.join(OUT, "solutions.npz"))



for k, t in enumerate(solve_times):
    print(f"Time for solver.solve({k}): {t:.4f} seconds")
print(f"Total time for solver steps: {sum(solve_times):.4f} seconds")
