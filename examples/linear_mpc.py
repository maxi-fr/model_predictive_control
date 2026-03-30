# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Closed-Loop Linear Model Predictive Control
#
# This notebook demonstrates a closed-loop Model Predictive Control (MPC) simulation for a simple unstable linear 2D system using the `OCP` class.

# %%
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from model_predictive_control.ocp import (
    OCP,
    control_bounds_constraints,
    linear_dynamics,
    quadratic_objective,
    state_bounds_constraints,
    terminal_quadratic_objective,
)
from model_predictive_control.plots import plot_controls, plot_mpc_trajectories

# %% [markdown]
# ## 1. Linear System Dynamics
#
# We define a generic unstable linear system $x_{k+1} = A x_k + B u_k$.

# %%
A = np.array([[1.0, 0.1], [0.5, 1.0]])
B = np.array([[0.0], [0.1]])

nx = A.shape[1]
nu = B.shape[1]

# Create discrete dynamics function
dynamics = linear_dynamics(A, B)

# %% [markdown]
# ## 2. Objective Function and Constraints
#
# We use a standard quadratic objective to penalize state deviations and control effort, and box constraints for safety.

# %%
# Objective matrices
Q = np.diag([100.0, 10.0])
R = np.array([[0.1]])

q_term = np.zeros(nx)
r_term = np.zeros(nu)
N_cross = np.zeros((nx, nu))

objective = quadratic_objective(Q, R, q_term, r_term, N_cross)

# Terminal objective
Qf = np.diag([1000.0, 100.0])
terminal_objective = terminal_quadratic_objective(Qf, q_term)

# Constraints
u_max_val = 50.0
u_min = np.array([-u_max_val])
u_max = np.array([u_max_val])

x_max_val = 2.0
x_min = np.array([-x_max_val, -x_max_val])
x_max = np.array([x_max_val, x_max_val])

state_bounds = state_bounds_constraints(x_min, x_max, nu)
control_bounds = control_bounds_constraints(u_min, u_max, nx)

x = ca.MX.sym("x", nx)
u = ca.MX.sym("u", nu)

in_eq_constraints = ca.Function("in_eq", [x, u], [ca.vertcat(state_bounds(x, u), control_bounds(x, u))])

# %% [markdown]
# ## 3. OCP Setup and Closed-Loop Simulation

# %%
N_horizon = 20
N_sim = 40
dt = 0.1

ocp = OCP(
    N=N_horizon,
    dt=dt,
    objective=objective,
    dynamics=dynamics,
    terminal_objective=terminal_objective,
    in_eq_constraints=in_eq_constraints,
)

# Setup using multiple shooting and ipopt
ocp.setup(method="multiple_shooting", dynamics_type="discrete", solver="ipopt", solver_opts={"print_level": 0})

# Simulation loop
x0_val = np.array([1.5, 0.0])  # Start near the bound
X_closed_loop = np.zeros((nx, N_sim + 1))
U_closed_loop = np.zeros((nu, N_sim))
X_open_loop = np.zeros((N_sim, nx, N_horizon + 1))

X_closed_loop[:, 0] = x0_val
current_x = x0_val

for k in range(N_sim):
    X_opt, U_opt, status = ocp.solve(current_x)

    # Extract first control action
    u_k = U_opt[:, 0]

    # Store predictions for plotting
    X_open_loop[k, :, :] = X_opt

    # Apply control to system
    x_next = A @ current_x + B @ u_k

    # Store results
    U_closed_loop[:, k] = u_k
    X_closed_loop[:, k + 1] = x_next

    # Update current state
    current_x = x_next

print("Simulation finished.")

# %% [markdown]
# ## 4. Plot Results

# %%
time = np.arange(N_sim + 1) * dt

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot states with open loop predictions
plot_mpc_trajectories(
    time,
    X_closed_loop,
    X_open_loop,
    labels=["State 1", "State 2"],
    fig=fig,
    ax=axs[0],
    title="Closed-Loop MPC Trajectories with Open-Loop Predictions",
    bounds=[(x_min[0], x_max[0]), (x_min[1], x_max[1])],
    step_interval=4,
)

# Plot controls
plot_controls(
    time,
    U_closed_loop,
    labels=["Control"],
    fig=fig,
    ax=axs[1],
    title="Closed-Loop Control Action",
    bounds=[(u_min[0], u_max[0])],
)

plt.tight_layout()
plt.show()
