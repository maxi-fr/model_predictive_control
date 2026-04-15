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
from model_predictive_control.constraints import LinearChanceConstraint

epsilon = 0.05  # 5% violation probability

cc_state_constraints = ConstraintList()
cc_state_constraints.add(
    LinearChanceConstraint(
        h=np.array([p_max]), F=np.array([[1.0, 0.0]]), A=A, Sigma_w=Sigma_w, epsilon=epsilon, N=N, nu=nu
    ),
    slice(1, None),
)
cc_state_constraints.add(
    LinearChanceConstraint(
        h=np.array([-p_min]), F=np.array([[-1.0, 0.0]]), A=A, Sigma_w=Sigma_w, epsilon=epsilon, N=N, nu=nu
    ),
    slice(1, None),
)

# Control limits (unchanged)
cc_state_constraints.add(LinearConstraint(h=np.array([u_max]), G=np.array([[1.0]]), nx=nx), slice(0, N))
cc_state_constraints.add(LinearConstraint(h=np.array([-u_min]), G=np.array([[-1.0]]), nx=nx), slice(0, N))

cc_ocp = LinearOCP(dynamics=dynamics, Q=Q, R=R, N=N, dt=dt, constraints=cc_state_constraints)

cc_mpc = LinearMPC(linear_ocp=cc_ocp, setup_args={"solver": "osqp"})

# %% [markdown]
# ## 5. Closed-Loop Simulation
# We simulate both controllers starting from an initial condition close to the bound. The same noise realization will be applied to both.

# %%
from model_predictive_control.simulation import simulate

n_steps = 50
x0 = np.array([0.8, 0.0])  # Start near the boundary


def noisy_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    x_next_nom = dynamics(x, u)
    noise = np.random.multivariate_normal(np.zeros(2), Sigma_w)
    return np.array(x_next_nom).flatten() + noise


print("Running Nominal MPC...")
np.random.seed(42)  # Seed for reproducible comparison
res_nom = simulate(nominal_mpc, noisy_dynamics, x0, num_steps=n_steps)
states_nom = res_nom.X
controls_nom = res_nom.U

print("Running Chance Constrained MPC...")
np.random.seed(42)  # Reset seed for fair comparison
try:
    res_cc = simulate(cc_mpc, noisy_dynamics, x0, num_steps=n_steps)
    states_cc = res_cc.X
    controls_cc = res_cc.U
except RuntimeError as e:
    print(f"MPC solve failed: {e}")
    states_cc = np.array([])
    controls_cc = np.array([])

# Recover tightening bounds for plot
cc_p_max = cc_mpc.ocp.constraints.constraints[0][0].h[0]  # type: ignore
cc_p_min = -cc_mpc.ocp.constraints.constraints[1][0].h[0]  # type: ignore

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
