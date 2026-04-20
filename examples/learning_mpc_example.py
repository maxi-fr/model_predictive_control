# %% [markdown]
# # Learned Components in MPC
#
# In this notebook, we show how to integrate PyTorch-based neural networks into the MPC formulation using `l4casadi` and our `learning` package.

# %%
import numpy as np
import torch
from torch import nn

# Learning setup imports
from learning.dynamics import LearnedDynamics

# MPC setup imports
from model_predictive_control.mpc import MPC
from model_predictive_control.objective import Objective, QuadraticCost, TerminalQuadraticCost
from model_predictive_control.ocp import OCP
from model_predictive_control.simulation import simulate

# %% [markdown]
# ## 1. Define a Learned Dynamics Model
# We will define a simple Multilayer Perceptron (MLP) to approximate a discrete-time dynamics model.

# %%
class SimpleMLPDynamics(nn.Module):
    def __init__(self, nx: int, nu: int):
        super().__init__()
        self.nx = nx
        self.nu = nu

        # A small 2-layer network. Input is [x, u] concatenated.
        self.net = nn.Sequential(nn.Linear(nx + nu, 16), nn.ReLU(), nn.Linear(16, nx))

        # We initialize it with some specific weights so it behaves predictably for the example,
        # rather than fully random. Let's make it act roughly like x_{k+1} = 0.9*x_k + 0.1*u_k
        with torch.no_grad():
            self.net[0].weight.data.fill_(0.0)  # type: ignore[operator]
            self.net[0].bias.data.fill_(0.0)  # type: ignore[operator]
            self.net[2].weight.data.fill_(0.0)  # type: ignore[operator]
            self.net[2].bias.data.fill_(0.0)  # type: ignore[operator]

            # Create a simple linear pass-through behavior roughly equivalent to A = 0.9, B = 0.1
            self.net[0].weight[0, 0] = 1.0  # type: ignore[operator]
            self.net[0].weight[1, 1] = 1.0  # type: ignore[operator]
            # Since ReLU zeroes out negative inputs, add a bias or use inputs >= 0.

            self.net[2].weight[0, 0] = 0.9  # type: ignore[operator]
            self.net[2].weight[0, 1] = 0.1  # type: ignore[operator]
            self.net[2].bias.data.fill_(0.0)  # type: ignore[operator]

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # We concatenate x and u internally, keeping the network interface clean
        x_u = torch.cat([torch.reshape(x, (-1,)), torch.reshape(u, (-1,))], dim=0)

        # Output must be a 2D matrix (nx, 1) for unbatched l4casadi
        # We must use torch.reshape to avoid issues with fx tracing mutations/views
        return torch.reshape(self.net(x_u), (self.nx, 1))


nx = 1
nu = 1
model = SimpleMLPDynamics(nx, nu)

# Wrap it in our LearnedDynamics class
dynamics = LearnedDynamics(model, nx, nu)

# %% [markdown]
# ## 2. Setup Objective and Constraints
# We'll use a standard quadratic objective to regulate the state to the origin.

# %%
Q = np.eye(nx) * 10.0
R = np.eye(nu) * 0.1
Qf = Q

stage_cost = QuadraticCost(Q, R)
terminal_cost = TerminalQuadraticCost(Qf, np.zeros((nx, 1)))

N = 10
objective = Objective(stage_cost, terminal_cost, N)

# %% [markdown]
# ## 3. Define and Solve the OCP
# We wrap the dynamics and objective in an OCP, and use the MPC wrapper for simulation.

# %%
dt = 0.1
ocp = OCP(N=N, dt=dt, dynamics=dynamics, objective=objective)
mpc = MPC(ocp=ocp, setup_args={"dynamics_type": "discrete", "solver_opts": {"hessian_approximation": "limited-memory"}})

# We need to initialize the guess to avoid NaN evaluation in the first solver step
x_guess = np.zeros((N + 1, nx))
u_guess = np.zeros((N, nu))
x0 = np.array([5.0])
mpc.last_X_opt = x_guess
mpc.last_U_opt = u_guess

# Simulate for a few steps using the library's utility
res = simulate(mpc=mpc, dynamics=dynamics, x0=x0, num_steps=15)

# %%
print("Final State:")
print(res.X[-1, -1])
print("Control Trajectory:")
print(res.U[:, 0].flatten())
