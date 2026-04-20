import casadi as ca
import l4casadi as l4c  # type: ignore[import-untyped]
import torch

from model_predictive_control.constraints import Constraint, ControlConstraint, StateConstraint


class ConstraintWrapper(torch.nn.Module):
    """Wrapper to handle concatenated input for l4casadi while exposing separate x, u to the underlying model."""

    def __init__(self, model: torch.nn.Module, nx: int, nu: int) -> None:
        super().__init__()
        self.model = model
        self.nx = nx
        self.nu = nu

    def forward(self, x_u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the constraint model.

        Parameters
        ----------
            x_u: Concatenated state and control tensor.
        """
        x = x_u[: self.nx]
        u = x_u[self.nx : self.nx + self.nu]
        out: torch.Tensor = self.model(x, u)
        return out


class LearnedConstraint(Constraint):
    """Learned mixed constraint wrapping f_theta(x, u) <= 0 or == 0."""

    def __init__(self, model: torch.nn.Module, nx: int, nu: int, is_equality: bool = False) -> None:
        """
        Initialize the learned mixed constraint.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model representing the constraint. It should take separate tensors (x, u) as input.
        nx : int
            Number of states.
        nu : int
            Number of controls.
        is_equality : bool, optional
            Whether the constraint is an equality constraint (== 0) or inequality (<= 0). Default is False.
        """
        self.model = model
        self.nx = nx
        self.nu = nu

        self._wrapper = ConstraintWrapper(model, nx, nu)
        self.l4c_model = l4c.L4CasADi(self._wrapper, batched=False)

        x = ca.MX.sym("x", nx)
        u = ca.MX.sym("u", nu)

        input_cat = ca.vertcat(x, u)
        out = self.l4c_model(input_cat)

        f = ca.Function("learned_constraint", [x, u], [out], ["x", "u"], ["f"])
        super().__init__(f, is_equality=is_equality)


class LearnedStateConstraint(StateConstraint):
    """Learned state constraint wrapping f_theta(x) <= 0 or == 0."""

    def __init__(self, model: torch.nn.Module, nx: int, is_equality: bool = False) -> None:
        """
        Initialize the learned state constraint.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model representing the constraint. It should take x as input.
        nx : int
            Number of states.
        is_equality : bool, optional
            Whether the constraint is an equality constraint (== 0) or inequality (<= 0). Default is False.
        """
        self.model = model
        self.nx = nx

        self.l4c_model = l4c.L4CasADi(model, batched=False)

        x = ca.MX.sym("x", nx)
        out = self.l4c_model(x)

        f = ca.Function("learned_state_constraint", [x], [out], ["x"], ["f"])
        super().__init__(f, is_equality=is_equality)


class LearnedControlConstraint(ControlConstraint):
    """Learned control constraint wrapping f_theta(u) <= 0 or == 0."""

    def __init__(self, model: torch.nn.Module, nu: int, is_equality: bool = False) -> None:
        """
        Initialize the learned control constraint.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model representing the constraint. It should take u as input.
        nu : int
            Number of controls.
        is_equality : bool, optional
            Whether the constraint is an equality constraint (== 0) or inequality (<= 0). Default is False.
        """
        self.model = model
        self.nu = nu

        self.l4c_model = l4c.L4CasADi(model, batched=False)

        u = ca.MX.sym("u", nu)
        out = self.l4c_model(u)

        f = ca.Function("learned_control_constraint", [u], [out], ["u"], ["f"])
        super().__init__(f, is_equality=is_equality)
