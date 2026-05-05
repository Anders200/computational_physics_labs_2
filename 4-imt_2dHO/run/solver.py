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

# The .so lives next to this script (run/); make sure Python finds it.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT  = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

import itm                                           # built by CMake as 'itm'


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Parameters
# ─────────────────────────────────────────────────────────────────────────────

N        = 100
OMEGA_X  = 1.0
OMEGA_Y  = 2.0
N_STATES = 10      # solve states 0 … 4

# Exact energies: E(nx,ny) = ωx(nx+½) + ωy(ny+½)
# Sorted ascending for ωx=1, ωy=2:
#   (0,0)→1.5  (1,0)→2.5  (0,1)→3.5=(2,0)  (1,1)→4.5
EXACT = [
    OMEGA_X*(nx+.5) + OMEGA_Y*(ny+.5)
    for nx, ny in [(0,0),(1,0),(0,1),(2,0),(1,1)]
]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Figure 1 — convergence vs α
# ─────────────────────────────────────────────────────────────────────────────

alpha_factors = [0.3, 0.6, 0.9, 0.95, 1.01]
colors        = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_factors)))

fig, ax = plt.subplots(figsize=(8, 5))
for af, col in zip(alpha_factors, colors):
    try:
        s    = itm.Solver(N, af, OMEGA_X, OMEGA_Y, 3000, 1e-9)
        hist = s.solve(0)
        lbl  = f"α = {af:.2f}·αc"
        ax.semilogy(np.abs(np.array(hist) - EXACT[0]),
                    color=col, lw=1.5, label=lbl)
    except Exception as e:
        print(f"α_factor={af} diverged: {e}")

ax.axhline(1e-9, ls="--", color="k", lw=0.8, label="tol = 1e-9")
ax.set_xlabel("Iteration")
ax.set_ylabel("|⟨E⟩ − E_exact|")
ax.set_title("Ground-state convergence for various α")
ax.legend(fontsize=9)
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "convergence_alpha.png"), dpi=150)
plt.close(fig)
print("Saved", os.path.join(OUT, "convergence_alpha.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main solve — 5 states with safe α = 0.9 αc
# ─────────────────────────────────────────────────────────────────────────────

solver = itm.Solver(N, 0.9, OMEGA_X, OMEGA_Y, 10000, 1e-10)

# Collect per-state histories by solving incrementally
histories = []
for k in range(N_STATES + 1):
    s    = itm.Solver(N, 0.9, OMEGA_X, OMEGA_Y, 100000, 1e-9)
    hist = s.solve(k)
    histories.append(hist)            # history of state k

# Final solve retaining all wavefunctions
solver = itm.Solver(N, 0.9, OMEGA_X, OMEGA_Y, 10000, 1e-10)
solver.solve(N_STATES)

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
    err = np.abs(np.array(histories[k]) - EXACT[k])
    err = np.where(err > 0, err, 1e-16)
    ax.semilogy(err, color=colors2[k-1], lw=1.5, label=f"State {k}")

ax.axhline(1e-10, ls="--", color="k", lw=0.8, label="tol = 1e-10")
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

labels = ["(0,0)", "(1,0)", "(0,1) or (2,0)", "(2,0) or (0,1)", "(1,1)"]

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
    ax.set_title(f"Ψ_{k}  {labels[k]}\nE={E_num[k]:.4f}", fontsize=8)
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
