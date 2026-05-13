# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chance Constrained Model Predictive Control
# This notebook demonstrates a Chance Constrained MPC formulation for a 1D mass-spring-damper system subject to additive Gaussian noise. We compare a nominal MPC (which ignores noise) with a chance-constrained MPC (which tightens constraints to account for the noise variance).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from simulate.estimator import IdentityEstimator
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor
from simulate.simulation import Simulation

from model_predictive_control.constraints import ConstraintList, LinearConstraint
from model_predictive_control.dynamics import DynamicsLog, LinearDynamics
from model_predictive_control.mpc import LinearMPC
from model_predictive_control.ocp import LinearOCP
from model_predictive_control.plots import plot_controls

# %% [markdown]
# ## 1. System Dynamics
# We define a 1D mass-spring-damper system:
# $m \ddot{p} + c \dot{p} + k p = u$
#
# where $p$ is position, $v$ is velocity, and $u$ is the control force.

# %%
# Parameters
m = 1.0  # mass
k = 0.5  # spring constant
c = 0.1  # damping coefficient
dt = 0.1  # sampling time

# Continuous-time matrices: x = [p, v]^T
Ac = np.array([[0, 1], [-k / m, -c / m]])
Bc = np.array([[0], [1 / m]])

# Discrete-time matrices (Euler approximation)
A = np.eye(2) + Ac * dt
B = Bc * dt

nx = A.shape[1]
nu = B.shape[1]

# %% [markdown]
# ## 2. Stochastic Dynamics Implementation
# To simulate the system with noise using the `simulate` framework, we subclass `LinearDynamics` to include additive Gaussian noise in the `update` step.

# %%
class StochasticLinearDynamics(LinearDynamics):
    """Linear dynamics with additive Gaussian noise."""

    def __init__(self, A: np.ndarray, B: np.ndarray, Sigma_w: np.ndarray, dt: float = 0.1, seed: int = 42) -> None:
        super().__init__(A, B, dt=dt)
        self.Sigma_w = Sigma_w
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_rng(self) -> None:
        """Reset the random number generator to the initial seed."""
        self.rng = np.random.default_rng(self.seed)

    def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, DynamicsLog]:
        """Advance the dynamics and add noise."""
        u_vec = self.to_col_vec(u).flatten()
        # Nominal discrete step
        self.x = np.asarray(self.f(self.x, u_vec)).flatten()
        # Add additive noise: w ~ N(0, Sigma_w)
        noise = self.rng.multivariate_normal(np.zeros(self.nx), self.Sigma_w)
        self.x += noise
        return self.from_col_vec(self.x), DynamicsLog(x=self.x.copy())

# %% [markdown]
# ## 3. Noise Characteristics
# We assume additive zero-mean Gaussian noise on the states:
# $x_{k+1} = A x_k + B u_k + w_k$ where $w_k \sim \mathcal{N}(0, \Sigma_w)$

# %%
# Noise covariance
sigma_w_pos = 0.05
sigma_w_vel = 0.02
Sigma_w = np.diag([sigma_w_pos**2, sigma_w_vel**2])

# Create the plant
plant = StochasticLinearDynamics(A=A, B=B, Sigma_w=Sigma_w, dt=dt, seed=42)

# %% [markdown]
# ## 4. Formulate Nominal MPC
# Standard MPC controller aiming to regulate the system to the origin, subject to state and control bounds.

# %%
# Objective
Q = np.diag([3000.0, 0.0])
R = np.array([[0.1]])

# Constraints
p_max = 1.5
u_max = 2.0
u_min = -2.0

N = 10

nominal_state_constraints = ConstraintList()
# Position limits: p <= p_max
nominal_state_constraints.add(LinearConstraint(h=np.array([p_max]), F=np.array([[1.0, 0.0]]), nu=nu), slice(1, None))

# Control limits: u_min <= u <= u_max
nominal_state_constraints.add(LinearConstraint(h=np.array([u_max]), G=np.array([[1.0]]), nx=nx), slice(0, N))
nominal_state_constraints.add(LinearConstraint(h=np.array([-u_min]), G=np.array([[-1.0]]), nx=nx), slice(0, N))

nominal_ocp = LinearOCP(dynamics=plant, Q=Q, R=R, N=N, dt=dt, constraints=nominal_state_constraints)
nominal_mpc = LinearMPC(linear_ocp=nominal_ocp, dt=dt, setup_args={"solver": "osqp"})

# %%
n_steps = 50
t_end = n_steps * dt
x0 = np.array([0.0, 0.0])  # Start near the boundary

# Simulation components
ref = StepReference(dt=dt, step_value=np.array([1.0, 0.0]))
sensor = GaussianSensor(dt=dt, std_dev=0.0)  # Perfect sensing
estimator = IdentityEstimator(dt=dt)

# 1. Run Nominal MPC
plant.x = x0.copy()
plant.reset_rng()
sim_nom = Simulation(
    t_end=t_end, plant=plant, reference=ref, sensor=sensor, estimator=estimator, controller=nominal_mpc
)
print("Running Nominal MPC...")
sim_nom.run()

# %% [markdown]
# ## 5. Formulate Chance Constrained MPC
# For chance constraints $P(p_k \le p_{max}) \ge 1 - \epsilon$, we tighten the constraints using the propagated variance. 
# We use time-varying tightening to ensure feasibility near the initial condition.

# %%
epsilon = 0.05  # 5% violation probability
z_val = st.norm.ppf(1 - epsilon)

# Compute variance propagation over horizon N
Sigma_k = np.zeros((2, 2))
cc_state_constraints = ConstraintList()

for k in range(1, N + 1):
    # Propagate covariance: Sigma_{k+1} = A Sigma_k A^T + Sigma_w
    # Here Sigma_k represents the uncertainty at step k
    std_dev_pos = np.sqrt(Sigma_k[0, 0])
    tightening = z_val * std_dev_pos

    # Tighten state constraints
    cc_p_max = p_max - tightening
    print(cc_p_max)

    cc_state_constraints.add(LinearConstraint(h=np.array([cc_p_max]), F=np.array([[1.0, 0.0]]), nu=nu), k)

    # Update Sigma for next step
    Sigma_k = A @ Sigma_k @ A.T + Sigma_w

# Control limits (unchanged)
cc_state_constraints.add(LinearConstraint(h=np.array([u_max]), G=np.array([[1.0]]), nx=nx), slice(0, N))
cc_state_constraints.add(LinearConstraint(h=np.array([-u_min]), G=np.array([[-1.0]]), nx=nx), slice(0, N))

cc_ocp = LinearOCP(dynamics=plant, Q=Q, R=R, N=N, dt=dt, constraints=cc_state_constraints)
cc_mpc = LinearMPC(linear_ocp=cc_ocp, dt=dt, setup_args={"solver": "osqp"})

# %% [markdown]
# ## 6. Closed-Loop Simulation
# We simulate both controllers using the `simulate.Simulation` class. We reset the plant's RNG before each run to ensure they experience the same noise sequence.

# %%
# Simulation components
ref = StepReference(dt=dt, step_value=np.array([1.0, 0.0]))
sensor = GaussianSensor(dt=dt, std_dev=0.0)  # Perfect sensing
estimator = IdentityEstimator(dt=dt)

# 2. Run Chance Constrained MPC
plant = StochasticLinearDynamics(A=A, B=B, Sigma_w=Sigma_w, dt=dt, seed=42)
plant.x = x0.copy()
plant.reset_rng()
sim_cc = Simulation(t_end=t_end, plant=plant, reference=ref, sensor=sensor, estimator=estimator, controller=cc_mpc)
print("Running Chance Constrained MPC...")
sim_cc.run()

# %% [markdown]
# ## 7. Results
# We extract the results from the simulation loggers and plot them.

# %%
def extract_sim_results(sim: Simulation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    results_df = pd.DataFrame(sim.logger.universal_logs)
    time = results_df["t"].to_numpy()
    X = np.array([log["y"] for log in sim.logger.universal_logs])
    U = np.array([log["u"] for log in sim.logger.universal_logs])
    X_open_loop = np.array([log["X_opt"] for log in sim.logger.component_logs["controller"]])
    return time, X, U, X_open_loop


time_nom, X_nom, U_nom, X_ol_nom = extract_sim_results(sim_nom)
time_cc, X_cc, U_cc, X_ol_cc = extract_sim_results(sim_cc)

# Plotting Position
fig, ax = plt.subplots(figsize=(10, 6))

# Nominal Results
ax.plot(time_nom, X_nom[:, 0], "r-", label="Nominal MPC")
# Chance Constrained Results
ax.plot(time_cc, X_cc[:, 0], "b-", label="Chance Constrained MPC")

ax.axhline(np.atleast_1d(ref.step_value)[0], label="ref.", linestyle="--")

# Bounds
ax.axhline(p_max, color="k", linestyle="--", label="Actual Bound ($p_{max}$)")

ax.set_xlabel("Time [s]")
ax.set_ylabel("Position $p$")
ax.legend()
ax.set_title("Stochastic MPC: Position vs Time")
ax.grid(visible=True)
plt.show()

# Plotting Controls
fig, ax = plt.subplots(figsize=(10, 4))
plot_controls(time_nom, U_nom, labels=["Nominal Control"], fig=fig, ax=ax, title="Control Actions")
plot_controls(time_cc, U_cc, labels=["CC Control"], fig=fig, ax=ax)
plt.show()
