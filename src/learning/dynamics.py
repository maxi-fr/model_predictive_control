import casadi as ca
import l4casadi as l4c  # type: ignore[import-untyped]
import torch

from model_predictive_control.dynamics import Dynamics


class DynamicsWrapper(torch.nn.Module):
    """Wrapper to handle concatenated input for l4casadi while exposing separate x, u to the underlying model."""

    def __init__(self, model: torch.nn.Module, nx: int, nu: int) -> None:
        super().__init__()
        self.model = model
        self.nx = nx
        self.nu = nu

    def forward(self, x_u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the dynamics model.

        Parameters
        ----------
            x_u: Concatenated state and control tensor.
        """
        x = x_u[: self.nx]
        u = x_u[self.nx : self.nx + self.nu]
        out: torch.Tensor = self.model(x, u)
        return out


class LearnedDynamics(Dynamics):
    """Dynamics constraint: x_{k+1} = f_theta(x_k, u_k) wrapped using l4casadi."""

    def __init__(self, model: torch.nn.Module, nx: int, nu: int) -> None:
        """
        Initialize learned dynamics.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model representing the dynamics. It should take tensors (x, u) as input.
        nx : int
            Number of states.
        nu : int
            Number of controls.
        """
        self.model = model
        self.nx = nx
        self.nu = nu

        self._wrapper = DynamicsWrapper(model, nx, nu)
        self.l4c_model = l4c.L4CasADi(self._wrapper, batched=False)

        x = ca.MX.sym("x", nx)
        u = ca.MX.sym("u", nu)

        input_cat = ca.vertcat(x, u)
        out = self.l4c_model(input_cat)

        f = ca.Function("learned_dyn", [x, u], [out], ["x", "u"], ["f"])
        super().__init__(f)
