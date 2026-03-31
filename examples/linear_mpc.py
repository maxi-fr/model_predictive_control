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
# # Closed-Loop Linear Model Predictive Control
#
# This notebook demonstrates a closed-loop Model Predictive Control (MPC) simulation for a simple unstable linear 2D system using the `LinearOCP` class, which uses a QP formulation.

# %%
import matplotlib.pyplot as plt
import numpy as np

from model_predictive_control.ocp import LinearOCP
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

# %% [markdown]
# ## 2. Objective Function and Constraints
#
# We use a standard quadratic objective to penalize state deviations and control effort, and box constraints for safety.

# %%
# Objective matrices
Q = np.diag([100.0, 10.0])
R = np.array([[0.1]])

q = np.zeros(nx)
r = np.zeros(nu)
N_cross = np.zeros((nx, nu))

# Terminal objective
Qf = np.diag([1000.0, 100.0])

# Constraints
# F x + G u <= h
u_max_val = 50.0
x_max_val = 2.0

# Box constraints:
# x_1 <= 2.0  ->  [1, 0] x + 0 u <= 2.0
# -x_1 <= 2.0 -> [-1, 0] x + 0 u <= 2.0
# x_2 <= 2.0  ->  [0, 1] x + 0 u <= 2.0
# -x_2 <= 2.0 ->  [0, -1] x + 0 u <= 2.0
# u <= 50.0   ->  [0, 0] x + 1 u <= 50.0
# -u <= 50.0  ->  [0, 0] x - 1 u <= 50.0

F = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]])

G = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [-1.0]])

h = np.array([x_max_val, x_max_val, x_max_val, x_max_val, u_max_val, u_max_val])

# Terminal constraints (only on state)
F_term = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

h_term = np.array([x_max_val, x_max_val, x_max_val, x_max_val])

# Bounds for plotting
x_min = np.array([-x_max_val, -x_max_val])
x_max = np.array([x_max_val, x_max_val])
u_min = np.array([-u_max_val])
u_max = np.array([u_max_val])

# %% [markdown]
# ## 3. OCP Setup and Closed-Loop Simulation

# %%
N_horizon = 20
N_sim = 40
dt = 0.1

ocp = LinearOCP(
    N=N_horizon,
    dt=dt,
    A=A,
    B=B,
    Q=Q,
    R=R,
    q=q,
    r=r,
    N_cross=N_cross,
    Qf=Qf,
    qf=q,
    F=F,
    G=G,
    h=h,
    F_term=F_term,
    h_term=h_term,
)

# Setup using multiple shooting (sparse) and qrqp backend
ocp.setup(
    method="multiple_shooting",
    dynamics_type="discrete",
    solver="qrqp",
    solver_opts={"print_iter": False, "print_header": False},
)

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
