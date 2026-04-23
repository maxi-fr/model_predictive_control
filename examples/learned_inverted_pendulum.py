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
# # Learned Dynamics MPC for Inverted Pendulum
#
# This notebook demonstrates how to learn the dynamics of an inverted pendulum using PyTorch, wrap it using `LearnedDynamics`, and use it inside a Model Predictive Control (MPC) formulation. We then evaluate the performance of the learned MPC on the true dynamics using the `experiment` function.

# %%
import casadi as ca
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learning.dynamics import LearnedDynamics
from model_predictive_control.constraints import ConstraintList, ControlBoundConstraint, StateBoundConstraint
from model_predictive_control.dynamics import Dynamics
from model_predictive_control.mpc import MPC
from model_predictive_control.objective import QuadraticObjective
from model_predictive_control.ocp import OCP
from model_predictive_control.simulation import experiment

# %% [markdown]
# ## 1. True System Dynamics
#
# We define the true continuous-time dynamics of the inverted pendulum and discretize them using 4th-order Runge-Kutta (RK4).

# %%
# Physical parameters
M = 1.0  # Mass of the cart (kg)
m = 0.1  # Mass of the pendulum (kg)
l = 0.5  # Length of the pendulum (m)
g = 9.81  # Gravity (m/s^2)

nx = 4
nu = 1
dt = 0.05


def true_continuous_dynamics(x, u):
    p, v, theta, omega = x[0], x[1], x[2], x[3]

    sin_theta = ca.sin(theta) if isinstance(theta, ca.MX) else np.sin(theta)
    cos_theta = ca.cos(theta) if isinstance(theta, ca.MX) else np.cos(theta)

    denominator = M + m - m * cos_theta**2

    force = u[0]
    p_ddot = (force + m * l * omega**2 * sin_theta - m * g * sin_theta * cos_theta) / denominator
    theta_ddot = (-force * cos_theta - m * l * omega**2 * sin_theta * cos_theta + (M + m) * g * sin_theta) / (
        l * denominator
    )

    if isinstance(x, ca.MX):
        return ca.vertcat(v, p_ddot, omega, theta_ddot)
    return np.array([v, p_ddot, omega, theta_ddot])


def true_discrete_dynamics(x, u):
    """RK4 discretization"""
    k1 = true_continuous_dynamics(x, u)
    k2 = true_continuous_dynamics(x + 0.5 * dt * k1, u)
    k3 = true_continuous_dynamics(x + 0.5 * dt * k2, u)
    k4 = true_continuous_dynamics(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Create CasADi function for the true discrete dynamics
x_sym = ca.MX.sym("x", nx)
u_sym = ca.MX.sym("u", nu)
x_next_sym = true_discrete_dynamics(x_sym, u_sym)
true_dyn_func = ca.Function("true_dyn", [x_sym, u_sym], [x_next_sym])
true_dynamics_obj = Dynamics(true_dyn_func)

# %% [markdown]
# ## 2. Generate Training Data
#
# We simulate the true system with random control inputs to collect $(x_k, u_k, x_{k+1})$ transitions.

# %%
np.random.seed(42)
num_samples = 5000

X_data = []
U_data = []
Y_data = []

for _ in range(num_samples):
    # Random initial states
    p = np.random.uniform(-2.0, 2.0)
    v = np.random.uniform(-1.0, 1.0)
    theta = np.random.uniform(-np.pi, np.pi)
    omega = np.random.uniform(-2.0, 2.0)
    x_k = np.array([p, v, theta, omega])

    # Random control input
    u_k = np.random.uniform(-20.0, 20.0, size=(1,))

    # Next state
    x_next = true_discrete_dynamics(x_k, u_k)

    X_data.append(x_k)
    U_data.append(u_k)
    Y_data.append(x_next)

X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
U_tensor = torch.tensor(np.array(U_data), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)

dataset = TensorDataset(X_tensor, U_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# %% [markdown]
# ## 3. Define and Train Neural Network
#
# We learn the residual $x_{k+1} - x_k = f_\theta(x_k, u_k)$. Since we are operating on small timestep `dt`, predicting the change is often easier than predicting the absolute next state.

# %%
class ResNetDynamics(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.net = nn.Sequential(nn.Linear(nx + nu, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, nx))

    def forward(self, x, u):
        # We need to reshape x and u in case they come in unbatched from l4casadi
        x_flat = torch.reshape(x, (-1, self.nx))
        u_flat = torch.reshape(u, (-1, self.nu))

        xu = torch.cat([x_flat, u_flat], dim=-1)
        delta_x = self.net(xu)
        x_next = x_flat + delta_x

        # Keep shapes consistent
        if x.dim() == 1 or (x.dim() == 2 and x.shape[1] == 1):
            return torch.reshape(x_next, (self.nx, 1))
        return x_next


model = ResNetDynamics(nx, nu)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 150
for epoch in range(epochs):
    epoch_loss = 0.0
    for bx, bu, by in dataloader:
        optimizer.zero_grad()

        y_pred = model(bx, bu)
        loss = criterion(y_pred, by)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

# Wrap in LearnedDynamics
learned_dynamics = LearnedDynamics(model, nx, nu)

# %% [markdown]
# ## 4. OCP Setup with Learned Dynamics
#
# We configure the OCP to regulate the pendulum to the upright position using our learned dynamics.

# %%
N = 40

# State cost matrix
Q = np.diag([10.0, 1.0, 10.0, 1.0])
R = np.array([[0.1]])

# Terminal cost matrix
Qf = np.diag([100.0, 10.0, 100.0, 10.0])

q_term = np.zeros(nx)
r_term = np.zeros(nu)
N_cross = np.zeros((nx, nu))

objective = QuadraticObjective(Q, R, Qf, q_term, N, q_term, r_term, N_cross)

# Constraints
u_max_val = 20.0
u_min = np.array([-u_max_val])
u_max = np.array([u_max_val])

p_max_val = 2.0
inf = 1e9
x_min = np.array([-p_max_val, -inf, -inf, -inf])
x_max = np.array([p_max_val, inf, inf, inf])

state_bounds = StateBoundConstraint(x_min, x_max)
control_bounds = ControlBoundConstraint(u_min, u_max)

cl = ConstraintList()
cl.add(state_bounds, slice(None))
cl.add(control_bounds, slice(0, N))

# Note: For learned dynamics (l4casadi wrapped), we use 'discrete' dynamics_type
ocp = OCP(
    N=N,
    dt=dt,
    objective=objective,
    dynamics=learned_dynamics,
    constraints=cl,
)

# Ipopt with L-BFGS because the neural network doesn't easily provide exact Hessians
setup_args = {
    "method": "collocation",
    "dynamics_type": "discrete",
    "solver": "ipopt",
    "solver_opts": {"print_level": 0, "sb": "yes", "hessian_approximation": "limited-memory", "max_iter": 200},
}
mpc = MPC(ocp=ocp, setup_args=setup_args)

# Initialize guess
mpc.last_X_opt = np.zeros((N + 1, nx))
mpc.last_U_opt = np.zeros((N, nu))

# %% [markdown]
# ## 5. Evaluate in Simulation
#
# We run an experiment starting from slightly offset angles to see if the MPC based on *learned* dynamics can stabilize the *true* dynamics system.

# %%
# Initial conditions to test (cart at center, angle slightly offset)
x0_list = [np.array([0.0, 0.0, 0.2, 0.0]), np.array([0.0, 0.0, -0.3, 0.0]), np.array([0.0, 0.0, 0.5, 0.0])]

# Run the experiment simulating the TRUE discrete dynamics
df_results = experiment(
    mpc=mpc,
    dynamics=true_discrete_dynamics,
    x0_list=x0_list,
    num_steps=50,
)

print(df_results)
