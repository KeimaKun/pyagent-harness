"""
pendulum_sim.py
===============
Simple pendulum simulation — mass 0.5 kg, rod length 1 m.

Physics
-------
Equation of motion:  d²θ/dt² = -(g/L) sin(θ)
State vector: [θ, ω]  where ω = dθ/dt

Solved with scipy.integrate.solve_ivp (RK45).
Initial conditions: θ₀ = 45°, ω₀ = 0 rad/s.

Output
------
Saves 300-frame animation to pendulum_sim.gif (10 s × 30 fps).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

# ── Parameters ───────────────────────────────────────────────────────────────
MASS   = 0.5        # kg
LENGTH = 1.0        # m
G      = 9.81       # m/s²
THETA0 = np.radians(45)   # initial angle (rad)
OMEGA0 = 0.0              # initial angular velocity (rad/s)

DURATION = 10       # seconds
FPS      = 30       # frames per second
N_FRAMES = DURATION * FPS   # 300 frames

# ── Solve ODE ─────────────────────────────────────────────────────────────────
def pendulum_ode(t, y):
    theta, omega = y
    dtheta = omega
    domega = -(G / LENGTH) * np.sin(theta)
    return [dtheta, domega]

t_eval = np.linspace(0, DURATION, N_FRAMES)
sol = solve_ivp(
    pendulum_ode,
    t_span=(0, DURATION),
    y0=[THETA0, OMEGA0],
    t_eval=t_eval,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
)

theta = sol.y[0]          # angle array
omega = sol.y[1]          # angular velocity array

# Cartesian position of the bob (pivot at origin)
bob_x = LENGTH * np.sin(theta)
bob_y = -LENGTH * np.cos(theta)

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#1a1a2e")

# Left: animation panel
ax_anim = axes[0]
ax_anim.set_facecolor("#16213e")
ax_anim.set_xlim(-1.5, 1.5)
ax_anim.set_ylim(-1.5, 0.4)
ax_anim.set_aspect("equal")
ax_anim.set_title("Simple Pendulum  (m=0.5 kg, L=1 m)", color="white", fontsize=11)
ax_anim.tick_params(colors="grey")
for spine in ax_anim.spines.values():
    spine.set_edgecolor("#444")

# Pivot marker
ax_anim.plot(0, 0, "s", color="#e94560", markersize=8, zorder=5)

# Rod + bob (mutable artists)
rod_line, = ax_anim.plot([], [], "-", color="#e2e2e2", lw=2.5, zorder=3)
bob_dot,  = ax_anim.plot([], [], "o", color="#0f3460", markersize=18,
                          markeredgecolor="#e94560", markeredgewidth=2, zorder=4)
trail,    = ax_anim.plot([], [], "-", color="#e94560", lw=0.8, alpha=0.4, zorder=2)

time_text = ax_anim.text(
    0.02, 0.96, "", transform=ax_anim.transAxes,
    color="white", fontsize=9, va="top",
)

# Right: phase-space panel (θ vs ω)
ax_phase = axes[1]
ax_phase.set_facecolor("#16213e")
ax_phase.set_xlabel("θ  (rad)", color="white")
ax_phase.set_ylabel("ω  (rad/s)", color="white")
ax_phase.set_title("Phase Space", color="white", fontsize=11)
ax_phase.tick_params(colors="grey")
for spine in ax_phase.spines.values():
    spine.set_edgecolor("#444")

# Full phase-space trajectory (faint background)
ax_phase.plot(theta, omega, "-", color="#444", lw=1, alpha=0.5)
phase_dot, = ax_phase.plot([], [], "o", color="#e94560", markersize=8, zorder=5)
phase_trail, = ax_phase.plot([], [], "-", color="#e94560", lw=1.5, alpha=0.6, zorder=4)

TRAIL_LEN = 40   # frames of trail to keep

# ── Animation ─────────────────────────────────────────────────────────────────
def init():
    rod_line.set_data([], [])
    bob_dot.set_data([], [])
    trail.set_data([], [])
    phase_dot.set_data([], [])
    phase_trail.set_data([], [])
    time_text.set_text("")
    return rod_line, bob_dot, trail, phase_dot, phase_trail, time_text


def update(frame):
    x, y = bob_x[frame], bob_y[frame]

    # Rod from pivot (0,0) to bob
    rod_line.set_data([0, x], [0, y])
    bob_dot.set_data([x], [y])

    # Trail (last TRAIL_LEN frames)
    start = max(0, frame - TRAIL_LEN)
    trail.set_data(bob_x[start:frame + 1], bob_y[start:frame + 1])

    # Phase space
    phase_dot.set_data([theta[frame]], [omega[frame]])
    phase_trail.set_data(theta[start:frame + 1], omega[start:frame + 1])

    time_text.set_text(f"t = {sol.t[frame]:.2f} s")
    return rod_line, bob_dot, trail, phase_dot, phase_trail, time_text


ani = FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    init_func=init,
    interval=1000 / FPS,
    blit=True,
)

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = "pendulum_sim.gif"
writer = PillowWriter(fps=FPS)
ani.save(output_path, writer=writer, dpi=90)
print(f"Saved: {output_path}")
plt.close(fig)