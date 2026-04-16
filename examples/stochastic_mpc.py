# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %% [markdown]
# # Chance Constrained Model Predictive Control
# This notebook demonstrates a Chance Constrained MPC formulation for a 1D mass-spring-damper system subject to additive Gaussian noise. We compare a nominal MPC (which ignores noise) with a chance-constrained MPC (which tightens constraints to account for the noise variance).

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from model_predictive_control.constraints import ConstraintList, LinearConstraint
from model_predictive_control.dynamics import LinearDynamics
from model_predictive_control.mpc import LinearMPC
from model_predictive_control.ocp import LinearOCP

# %% [markdown]
# ## 1. System Dynamics
# We define a 1D mass-spring-damper system:
# $m \ddot{p} + c \dot{p} + k p = u$
#
# where $p$ is position, $v$ is velocity, and $u$ is the control force.
# We discretize this using exact forward Euler or simple zero-order hold approximation.

# %%
# Parameters
m = 1.0  # mass
k = 0.5  # spring constant
c = 0.1  # damping coefficient
dt = 0.1  # sampling time

# Continuous-time matrices: x = [p, v]^T
Ac = np.array([[0, 1], [-k / m, -c / m]])
Bc = np.array([[0], [1 / m]])

# Discrete-time matrices (Euler approximation for simplicity)
A = np.eye(2) + Ac * dt
B = Bc * dt

nx = A.shape[1]
nu = B.shape[1]

dynamics = LinearDynamics(A=A, B=B)

# %% [markdown]
# ## 2. Noise Characteristics
# We assume additive zero-mean Gaussian noise on the states:
# $x_{k+1} = A x_k + B u_k + w_k$ where $w_k \sim \mathcal{N}(0, \Sigma_w)$

# %%
# Noise covariance
sigma_w_pos = 0.05
sigma_w_vel = 0.02
Sigma_w = np.diag([sigma_w_pos**2, sigma_w_vel**2])

# Set a random seed for reproducible comparisons
np.random.seed(42)

# %% [markdown]
# ## 3. Formulate Nominal MPC
# We design a standard MPC controller aiming to regulate the system to the origin, subject to state bounds (position bounds) and control bounds.

# %%
# Objective
Q = np.diag([10.0, 1.0])
R = np.array([[0.1]])

# Constraints
p_max = 1.0
p_min = -1.0
u_max = 2.0
u_min = -2.0

N = 20

nominal_state_constraints = ConstraintList()
# We use LinearConstraint for position limits. x = [p, v]^T
# p <= p_max => F x <= h where F = [1, 0], h = [p_max]
# -p <= -p_min => F x <= h where F = [-1, 0], h = [-p_min]
nominal_state_constraints.add(LinearConstraint(h=np.array([p_max]), F=np.array([[1.0, 0.0]]), nu=nu), slice(1, None))
nominal_state_constraints.add(LinearConstraint(h=np.array([-p_min]), F=np.array([[-1.0, 0.0]]), nu=nu), slice(1, None))

# Control limits
nominal_state_constraints.add(LinearConstraint(h=np.array([u_max]), G=np.array([[1.0]]), nx=nx), slice(0, N))
nominal_state_constraints.add(LinearConstraint(h=np.array([-u_min]), G=np.array([[-1.0]]), nx=nx), slice(0, N))

# We need to use LinearOCP for the formulation
nominal_ocp = LinearOCP(dynamics=dynamics, Q=Q, R=R, N=N, dt=dt, constraints=nominal_state_constraints)

nominal_mpc = LinearMPC(linear_ocp=nominal_ocp, setup_args={"solver": "osqp"})

# %% [markdown]
# ## 4. Formulate Chance Constrained MPC
# For chance constraints $P(p_k \le p_{max}) \ge 1 - \epsilon$, we tighten the constraint.
# Since $x_{k+1} = A x_k + B u_k + w_k$, the error covariance propagates as $\Sigma_{k+1} = A \Sigma_k A^T + \Sigma_w$.
# For simplicity in this example, we assume open-loop variance propagation over the horizon to compute the tightening.

# %%
epsilon = 0.05  # 5% violation probability
z_val = st.norm.ppf(1 - epsilon)

# Compute variance propagation over horizon N
Sigma_k = np.zeros((2, 2))
pos_variance_horizon = []

for _i in range(N):
    pos_variance_horizon.append(Sigma_k[0, 0])
    Sigma_k = A @ Sigma_k @ A.T + Sigma_w

# For a strict chance constraint, we tighten the bounds by z_val * std_dev
# To keep it simple, we'll take the maximum tightening over the horizon and apply it uniformly
max_std_dev = np.sqrt(max(pos_variance_horizon))
tightening = z_val * max_std_dev

print(f"Constraint tightening margin: {tightening:.3f}")

cc_p_max = p_max - tightening
cc_p_min = p_min + tightening

cc_state_constraints = ConstraintList()
cc_state_constraints.add(LinearConstraint(h=np.array([cc_p_max]), F=np.array([[1.0, 0.0]]), nu=nu), slice(1, None))
cc_state_constraints.add(LinearConstraint(h=np.array([-cc_p_min]), F=np.array([[-1.0, 0.0]]), nu=nu), slice(1, None))

# Control limits (unchanged)
cc_state_constraints.add(LinearConstraint(h=np.array([u_max]), G=np.array([[1.0]]), nx=nx), slice(0, N))
cc_state_constraints.add(LinearConstraint(h=np.array([-u_min]), G=np.array([[-1.0]]), nx=nx), slice(0, N))

cc_ocp = LinearOCP(dynamics=dynamics, Q=Q, R=R, N=N, dt=dt, constraints=cc_state_constraints)

cc_mpc = LinearMPC(linear_ocp=cc_ocp, setup_args={"solver": "osqp"})

# %% [markdown]
# ## 5. Closed-Loop Simulation
# We simulate both controllers starting from an initial condition close to the bound. The same noise realization will be applied to both.

# %%
n_steps = 50
x0 = np.array([0.8, 0.0])  # Start near the boundary

# Generate identical noise sequence for fair comparison
noise_seq = np.random.multivariate_normal(np.zeros(2), Sigma_w, size=n_steps)


def run_simulation(mpc_controller):
    x = x0.copy()
    states = [x]
    controls = []

    for _i in range(n_steps):
        # Solve MPC using wrapper step
        try:
            u_k = mpc_controller.step(x)
        except RuntimeError as e:
            print(f"MPC solve failed at step {_i}: {e}")
            break

        # Apply control and noise
        x_next = A @ x + B.flatten() * u_k + noise_seq[_i]

        states.append(x_next)
        controls.append(u_k)
        x = x_next

    return np.array(states), np.array(controls)


print("Running Nominal MPC...")
states_nom, controls_nom = run_simulation(nominal_mpc)

print("Running Chance Constrained MPC...")
states_cc, controls_cc = run_simulation(cc_mpc)

# %% [markdown]
# ## 6. Results
# We plot the position over time to observe if the bounds are violated.

# %%
time = np.arange(len(states_nom)) * dt

plt.figure(figsize=(10, 6))

plt.plot(time, states_nom[:, 0], "r-", label="Nominal MPC")
if len(states_cc) > 0:
    plt.plot(np.arange(len(states_cc)) * dt, states_cc[:, 0], "b-", label="Chance Constrained MPC")

# Bounds
plt.axhline(p_max, color="k", linestyle="--", label="Actual Bound ($p_{max}$)")
plt.axhline(cc_p_max, color="g", linestyle=":", label="Tightened Bound")

plt.xlabel("Time [s]")
plt.ylabel("Position $p$")
plt.legend()
plt.title("Position vs Time")
plt.grid(True)
plt.show()
