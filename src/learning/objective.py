import casadi as ca
import l4casadi as l4c
import torch

from model_predictive_control.objective import CostFunction

class LearnedCostFunction(CostFunction):
    """Learned stage or terminal cost wrapped using l4casadi."""

    def __init__(self, model: torch.nn.Module, nx: int, nu: int | None = None, has_reference: bool = False) -> None:
        """
        Initialize learned cost function.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model representing the cost.
            If nu is provided, it should take a concatenated tensor of (x, u) or (x, u, x_ref, u_ref) if has_reference=True.
            If nu is None, it should take x, or (x, x_ref) if has_reference=True.
        nx : int
            Number of states.
        nu : int | None, optional
            Number of controls. If None, it's considered a terminal cost.
        has_reference : bool, optional
            Whether the cost depends on a reference trajectory.
        """
        self.model = model
        self.nx = nx
        self.nu = nu
        self._has_reference = has_reference

        self.l4c_model = l4c.L4CasADi(model, batched=False)

        x = ca.MX.sym("x", nx)
        inputs = [x]
        names_in = ["x"]

        cat_list = [x]

        if nu is not None:
            u = ca.MX.sym("u", nu)
            inputs.append(u)
            names_in.append("u")
            cat_list.append(u)

        if has_reference:
            x_ref = ca.MX.sym("x_ref", nx)
            inputs.append(x_ref)
            names_in.append("x_ref")
            cat_list.append(x_ref)

            if nu is not None:
                u_ref = ca.MX.sym("u_ref", nu)
                inputs.append(u_ref)
                names_in.append("u_ref")
                cat_list.append(u_ref)

        input_cat = ca.vertcat(*cat_list)
        out = self.l4c_model(input_cat)

        f = ca.Function("learned_cost", inputs, [out], names_in, ["f"])
        super().__init__(f)
        # Override the property locally since Base relies on n_in() > 2 which might not hold for a terminal cost with reference
        self._has_reference = has_reference