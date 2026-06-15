import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import FEM_solver as fem

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=60, metavar="N",
                    help="mesh cells per axis (default: 60)")
args = parser.parse_args()

N  = args.n
L  = 5.0
dt = 0.02

print(f"FEM Advection-Diffusion  |  N={N}, L={L}, dt={dt}")

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
})

def save(name):
    path = os.path.join(HERE, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  → {name}")

def to_grid(result):
    x = np.array(result.x_nodes)
    y = np.array(result.y_nodes)
    u = np.array(result.u_final)
    # globalIdx(i,j)=j*N+i  → reshape gives [j,i] = [y-row, x-col]: correct
    return x.reshape(N, N), y.reshape(N, N), u.reshape(N, N)

def field_plot(xg, yg, ug, title, cmap="plasma"):
    xg2 = np.block([[xg,   xg+L],
                    [xg,   xg+L]])
    yg2 = np.block([[yg,   yg  ],
                    [yg+L, yg+L]])
    ug2 = np.tile(ug, (2, 2))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    pm = ax.pcolormesh(xg2, yg2, ug2, cmap=cmap, shading="auto")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax

def plot_trajectory(cx, cy, t, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    dx = np.abs(np.diff(cx))
    dy = np.abs(np.diff(cy))
    breaks = np.where((dx > L/2) | (dy > L/2))[0] + 1
    segs = np.split(np.arange(len(cx)), breaks)

    from matplotlib.collections import LineCollection
    points = np.array([cx, cy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    jump_mask = np.zeros(len(segments), dtype=bool)
    for b in breaks - 1:
        if 0 <= b < len(jump_mask):
            jump_mask[b] = True

    norm = plt.Normalize(t.min(), t.max())
    for seg_idx in segs:
        if len(seg_idx) < 2:
            continue
        s = seg_idx
        lc = LineCollection(
            np.array([[[cx[i], cy[i]], [cx[i+1], cy[i+1]]]
                      for i in s[:-1]]),
            cmap="viridis", norm=norm, lw=1.5
        )
        lc.set_array(t[s[:-1]])
        ax.add_collection(lc)

    sc = ax.scatter(cx, cy, c=t, cmap="viridis", norm=norm, s=12, zorder=3)
    ax.plot(cx[0], cy[0], "wo", ms=7, zorder=5, label="$t=0$")
    ax.plot(cx[-1], cy[-1], "r*", ms=10, zorder=5, label=f"$t={t[-1]:.2f}$")
    plt.colorbar(sc, ax=ax, label="time", fraction=0.046, pad=0.04)
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("centroid $x$"); ax.set_ylabel("centroid $y$")
    ax.set_title(title)
    ax.legend(fontsize=9)
    return fig, ax


print("\nDeliverable 1: pure advection")

vx, vy = 1.0, 1.0
speed  = np.hypot(vx, vy)
T      = L / speed

solver = fem.FEMSolver(N, L)
r1 = solver.run_pure_advection(vx, vy, dt, steps_per_snapshot=5)

t1  = np.array(r1.time)
cx1 = np.array(r1.center_x)
cy1 = np.array(r1.center_y)
mn1 = np.array(r1.u_min)
mx1 = np.array(r1.u_max)

# 1a – final field (tiled)
xg, yg, ug = to_grid(r1)
field_plot(xg, yg, ug, f"Pure advection — field at $t = {t1[-1]:.2f}$ (≈ $2T$)")
save("deliverable1_field.png")

# 1b – centroid trajectory
plot_trajectory(cx1, cy1, t1, "Centroid trajectory — pure advection")
save("deliverable1_trajectory.png")

# 1c – max / min u vs time
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(t1, mx1, color="crimson",   lw=1.8, label="max $u$")
ax.plot(t1, mn1, color="steelblue", lw=1.8, label="min $u$")
ax.axvline(T,   color="grey", ls="--", lw=1.1, label="$T$")
ax.axvline(2*T, color="grey", ls=":",  lw=1.1, label="$2T$")
ax.set_xlabel("time"); ax.set_ylabel("$u$")
ax.set_title("Max / min nodal $u$ — pure advection")
ax.legend(fontsize=9)
save("deliverable1_minmax.png")


print("\nDeliverable 2: pure diffusion")

D2    = 0.1
t_end = 5.0

solver = fem.FEMSolver(N, L)
r2 = solver.run_pure_diffusion(D2, t_end, dt, steps_per_snapshot=5)

t2  = np.array(r2.time)
mn2 = np.array(r2.u_min)
mx2 = np.array(r2.u_max)

k       = 2.0
sigma20 = 1.0 / (2.0 * k)
theory2 = sigma20 / (sigma20 + 2.0 * D2 * t2) * mx2[0]

# 2a – max u log + theory
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogy(t2, mx2,     color="crimson", lw=1.8, label="FEM max $u$")
ax.semilogy(t2, theory2, color="grey",    lw=1.2, ls="--", label="Gaussian theory")
ax.set_xlabel("time"); ax.set_ylabel("max $u$  (log scale)")
ax.set_title("Maximum nodal $u$ — pure diffusion")
ax.legend(fontsize=9)
save("deliverable2_maxu.png")

# 2b – min u
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(t2, mn2, color="steelblue", lw=1.8)
ax.set_xlabel("time"); ax.set_ylabel("min $u$")
ax.set_title("Minimum nodal $u$ — pure diffusion")
save("deliverable2_minu.png")

# 2c – final field (no tiling needed: blob is centred, not near boundary)
xg, yg, ug = to_grid(r2)
field_plot(xg, yg, ug, f"Pure diffusion — field at $t = {t2[-1]:.1f}$", cmap="inferno")
save("deliverable2_field.png")


print("\nDeliverable 3: advection-diffusion")

vx3, vy3 = 1.0, 1.0
D3       = 0.1
T3       = L / np.hypot(vx3, vy3)

solver = fem.FEMSolver(N, L)
r3 = solver.run_advection_diffusion(vx3, vy3, D3, T3, dt, steps_per_snapshot=5)

t3  = np.array(r3.time)
cx3 = np.array(r3.center_x)
cy3 = np.array(r3.center_y)

cx3_th = (L/2 + vx3 * t3) % L
cy3_th = (L/2 + vy3 * t3) % L

# 3a – final field (tiled)
xg, yg, ug = to_grid(r3)
field_plot(xg, yg, ug, f"Advection-diffusion — field at $t = {t3[-1]:.2f}$")
save("deliverable3_field.png")

# 3b – centroid trajectory
plot_trajectory(cx3, cy3, t3, "Centroid vs theory — advection-diffusion")
ax = plt.gcf().get_axes()[0]  
fig, ax = plot_trajectory(cx3, cy3, t3, "Centroid vs theory — advection-diffusion")

dx_th = np.abs(np.diff(cx3_th))
breaks_th = np.where(dx_th > L/2)[0] + 1
segs_th = np.split(np.arange(len(cx3_th)), breaks_th)
for s in segs_th:
    ax.plot(cx3_th[s], cy3_th[s], "r--", lw=1.2, alpha=0.8)
ax.plot([], [], "r--", lw=1.2, label="theory")
ax.legend(fontsize=9)
save("deliverable3_trajectory.png")

# 3c – centroid x vs time
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(t3, cx3,    color="steelblue",  lw=1.8, label="FEM $c_x$")
ax.plot(t3, cx3_th, color="crimson", ls="--", lw=1.2, label="theory")
ax.set_xlabel("time"); ax.set_ylabel("centroid $x$")
ax.set_title("Centroid $x$ vs time")
ax.legend(fontsize=9)
save("deliverable3_cx.png")

# 3d – centroid y vs time
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(t3, cy3,    color="darkorange", lw=1.8, label="FEM $c_y$")
ax.plot(t3, cy3_th, color="crimson", ls="--", lw=1.2, label="theory")
ax.set_xlabel("time"); ax.set_ylabel("centroid $y$")
ax.set_title("Centroid $y$ vs time")
ax.legend(fontsize=9)
save("deliverable3_cy.png")

print("\nDone.")
