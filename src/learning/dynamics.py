import casadi as ca
import l4casadi as l4c
import torch

from model_predictive_control.dynamics import Dynamics

class LearnedDynamics(Dynamics):
    """Dynamics constraint: x_{k+1} = f_theta(x_k, u_k) wrapped using l4casadi."""

    def __init__(self, model: torch.nn.Module, nx: int, nu: int) -> None:
        """
        Initialize learned dynamics.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model representing the dynamics. It should take a concatenated tensor of (x, u) as input.
        nx : int
            Number of states.
        nu : int
            Number of controls.
        """
        self.model = model
        self.nx = nx
        self.nu = nu

        self.l4c_model = l4c.L4CasADi(model, batched=False)

        x = ca.MX.sym("x", nx)
        u = ca.MX.sym("u", nu)

        input_cat = ca.vertcat(x, u)
        out = self.l4c_model(input_cat)

        f = ca.Function("learned_dyn", [x, u], [out], ["x", "u"], ["f"])
        super().__init__(f)