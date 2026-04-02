# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: model-predictive-control (3.12.1)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Closed-Loop Model Predictive Control for a 3D Quadrotor Tracking an S-Shaped Trajectory
#
# This notebook demonstrates how to formulate and solve a closed-loop model predictive control (MPC) problem for a full 3D quadrotor tracking a time-varying reference trajectory using the `MPC` and `OCP` classes from the `model_predictive_control` package.

# %%
from collections.abc import Callable

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from model_predictive_control.mpc import MPC
from model_predictive_control.ocp import (
    OCP,
    control_bounds_constraints,
    state_bounds_constraints,
    terminal_tracking_objective,
    tracking_objective,
)
from model_predictive_control.plots import plot_controls, plot_mpc_trajectories

# %% [markdown]
# ## 1. System Dynamics
#
# We define the physical parameters of the quadrotor and derive the non-linear 3D equations of motion.

# %%
# Physical parameters
m = 1.0  # Mass (kg)
g = 9.81  # Gravity (m/s^2)
l = 0.25  # Arm length (m)  # noqa: E741
c_tau = 0.01  # Thrust-to-torque coefficient

# Moments of inertia
J_x = 0.02
J_y = 0.02
J_z = 0.04
J = np.diag([J_x, J_y, J_z])
inv_J = np.linalg.inv(J)

# Dimensions
nx = 12
nu = 4

x = ca.MX.sym("x", nx)
u = ca.MX.sym("u", nu)

# Unpack states
p_pos = x[0:3]  # [x, y, z]
v_vel = x[3:6]  # [v_x, v_y, v_z]
eta = x[6:9]  # [phi, theta, psi]
omega = x[9:12]  # [p, q, r]

# Unpack controls
T1, T2, T3, T4 = u[0], u[1], u[2], u[3]
F_thrust = T1 + T2 + T3 + T4

# Torques
tau_x = l * (T4 - T2)
tau_y = l * (T1 - T3)
tau_z = c_tau * (T1 - T2 + T3 - T4)
tau = ca.vertcat(tau_x, tau_y, tau_z)

# Kinematics
phi, theta, psi = eta[0], eta[1], eta[2]

R_z = ca.vcat([ca.hcat([ca.cos(psi), -ca.sin(psi), 0]), ca.hcat([ca.sin(psi), ca.cos(psi), 0]), ca.hcat([0, 0, 1])])
R_y = ca.vcat(
    [ca.hcat([ca.cos(theta), 0, ca.sin(theta)]), ca.hcat([0, 1, 0]), ca.hcat([-ca.sin(theta), 0, ca.cos(theta)])]
)
R_x = ca.vcat([ca.hcat([1, 0, 0]), ca.hcat([0, ca.cos(phi), -ca.sin(phi)]), ca.hcat([0, ca.sin(phi), ca.cos(phi)])])
R_IB = R_z @ R_y @ R_x

# Translational dynamics
g_vec = ca.vertcat(0, 0, -g)
T_B = ca.vertcat(0, 0, F_thrust)
v_dot = g_vec + (R_IB @ T_B) / m

# Rotational kinematics matrix
W = ca.vcat(
    [
        ca.hcat([1, ca.sin(phi) * ca.tan(theta), ca.cos(phi) * ca.tan(theta)]),
        ca.hcat([0, ca.cos(phi), -ca.sin(phi)]),
        ca.hcat([0, ca.sin(phi) / ca.cos(theta), ca.cos(phi) / ca.cos(theta)]),
    ]
)
eta_dot = W @ omega

# Rotational dynamics (Euler's equations)
omega_dot = inv_J @ (tau - ca.cross(omega, J @ omega))

# Full state derivative
x_dot = ca.vertcat(v_vel, v_dot, eta_dot, omega_dot)
dynamics = ca.Function("dynamics", [x, u], [x_dot])

# %% [markdown]
# ## 2. Objective and Constraints

# %%
# State weights
Q_diag = [
    100.0,
    100.0,
    200.0,  # Position: [x, y, z]
    10.0,
    10.0,
    10.0,  # Velocity: [v_x, v_y, v_z]
    10.0,
    10.0,
    50.0,  # Angles:   [phi, theta, psi]
    1.0,
    1.0,
    1.0,  # Rates:    [p, q, r]
]
Q = np.diag(Q_diag)
R = np.diag([1.0, 1.0, 1.0, 1.0])

objective = tracking_objective(Q, R)

Qf = Q * 5.0
terminal_objective = terminal_tracking_objective(Qf, np.zeros(nx))

u_min_val = 0.0
u_max_val = 3.0
u_min = np.array([u_min_val] * nu)
u_max = np.array([u_max_val] * nu)

inf = 1e9
x_min = np.array([-inf, -inf, 0.0, -inf, -inf, -inf, -np.pi / 3, -np.pi / 3, -inf, -inf, -inf, -inf])
x_max = np.array([inf, inf, inf, inf, inf, inf, np.pi / 3, np.pi / 3, inf, inf, inf, inf])

state_bounds = state_bounds_constraints(x_min, x_max, nu)
control_bounds = control_bounds_constraints(u_min, u_max, nx)

in_eq_constraints = ca.Function("in_eq", [x, u], [ca.vertcat(state_bounds(x, u), control_bounds(x, u))])

# %% [markdown]
# ## 3. Reference Trajectory and MPC Setup

# %%
N = 40  # OCP Horizon length
N_sim = N * 3  # Closed loop simulation steps
dt = 0.1
time_sim = np.arange(N_sim + 1) * dt
T_ref_total = (N_sim + N) * dt
time_ref = np.arange(0, N_sim + N + 1) * dt

X_ref_full = np.zeros((N_sim + N + 1, nx))
U_ref_full = np.zeros((N_sim + N, nu))

hover_thrust = m * g / 4.0

for k in range(N_sim + N + 1):
    t = time_ref[k]

    # S-shape trajectory
    X_ref_full[k, 0] = 1.0 - 1.0 * t
    X_ref_full[k, 1] = 2.0 * np.sin(2 * np.pi * t / (N * 3 * dt))
    X_ref_full[k, 2] = 1.0 + 0.5 * t

    X_ref_full[k, 3] = 1.0
    X_ref_full[k, 4] = 2.0 * (2 * np.pi / (N * 3 * dt)) * np.cos(2 * np.pi * t / (N * 3 * dt))
    X_ref_full[k, 5] = 0.0

for k in range(N_sim + N):
    U_ref_full[k, :] = hover_thrust

# Initialize OCP and MPC
ocp = OCP(
    N=N,
    dt=dt,
    objective=objective,
    dynamics=dynamics,
    in_eq_constraints=in_eq_constraints,
    terminal_objective=terminal_objective,
)

setup_args = {
    "method": "collocation",
    "dynamics_type": "continuous",
    "solver": "ipopt",
    "solver_opts": {"print_level": 0},
}

mpc = MPC(ocp, setup_args=setup_args, X_guess=X_ref_full[0 : N + 1, :], U_guess=U_ref_full[0:N, :])

# %% [markdown]
# ## 4. Closed-Loop Simulation

# %%
X_closed_loop = np.zeros((N_sim + 1, nx))
U_closed_loop = np.zeros((N_sim, nu))
X_open_loop = np.zeros((N_sim, N + 1, nx))

x_current = X_ref_full[0, :].copy()
X_closed_loop[0, :] = x_current


# Use a discrete-time integration scheme (RK4) for the "true" plant
def rk4_step(
    dyn_func: Callable[[ArrayLike, ArrayLike], ArrayLike], x: ArrayLike, u: ArrayLike, dt: float
) -> np.ndarray:
    k1 = np.array(dyn_func(x, u)).flatten()
    k2 = np.array(dyn_func(x + dt / 2 * k1, u)).flatten()
    k3 = np.array(dyn_func(x + dt / 2 * k2, u)).flatten()
    k4 = np.array(dyn_func(x + dt * k3, u)).flatten()
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


print("Running MPC closed-loop simulation...")
for k in range(N_sim):
    # Slice the reference for the current horizon
    x_ref_horizon = X_ref_full[k : k + N + 1, :]
    u_ref_horizon = U_ref_full[k : k + N, :]

    # Step MPC
    u_k = mpc.step(x_current=x_current, x_ref=x_ref_horizon, u_ref=u_ref_horizon)

    # Store open loop predictions
    X_open_loop[k, :, :] = mpc.last_X_opt

    # Step simulation
    x_next = rk4_step(dynamics, x_current, u_k, dt)

    X_closed_loop[k + 1, :] = x_next
    U_closed_loop[k, :] = u_k
    x_current = x_next

print("Simulation finished.")

# %% [markdown]
# ## 5. Visualize Results

# %%
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Position X, Y, Z vs Reference
for idx, label in enumerate(["X [m]", "Y [m]", "Z [m]"]):
    axs[0].plot(time_sim, X_closed_loop[:, idx], label=f"Closed-Loop {label}", linewidth=2)
    axs[0].plot(
        time_ref[: N_sim + 1], X_ref_full[: N_sim + 1, idx], label=f"Reference {label}", linestyle="--", alpha=0.7
    )
axs[0].set_title("Position Tracking")
axs[0].set_ylabel("Position")
axs[0].legend(loc="upper right", ncol=3)
axs[0].grid(True)

# Euler Angles with open-loop predictions
plot_mpc_trajectories(
    time_sim,
    X_closed_loop,
    X_open_loop,
    indices=[6, 7, 8],
    labels=[r"Roll ($\phi$)", r"Pitch ($\theta$)", r"Yaw ($\psi$)"],
    fig=fig,
    ax=axs[1],
    title="Euler Angles [rad] with Open-Loop Predictions",
    ylabel="Angle [rad]",
    bounds=[(-np.pi / 3, np.pi / 3), (-np.pi / 3, np.pi / 3), None],
    step_interval=5,
)

# Controls
plot_controls(
    time_sim,
    U_closed_loop,
    labels=["$T_1$", "$T_2$", "$T_3$", "$T_4$"],
    fig=fig,
    ax=axs[2],
    title="Motor Thrusts [N]",
    bounds=[(u_min_val, u_max_val)] * 4,
    step=True,
)

plt.tight_layout()
plt.show()

# 3D Path Plot
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection="3d")
ax_3d.set_aspect("equal")

ax_3d.plot(
    X_ref_full[: N_sim + 1, 0],
    X_ref_full[: N_sim + 1, 1],
    X_ref_full[: N_sim + 1, 2],
    "--",
    color="gray",
    label="Reference Path",
)
ax_3d.plot(X_closed_loop[:, 0], X_closed_loop[:, 1], X_closed_loop[:, 2], "-b", linewidth=2, label="Closed-Loop Path")
ax_3d.set_xlabel("X [m]")
ax_3d.set_ylabel("Y [m]")
ax_3d.set_zlabel("Z [m]")
ax_3d.set_title("3D Quadrotor Trajectory (MPC)")
ax_3d.legend()
plt.show()
