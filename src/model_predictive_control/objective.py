import typing
from typing import Any

import casadi as ca
import numpy as np


class CostFunction:
    """Base class for cost functions."""

    def __init__(self, f: ca.Function) -> None:
        self.f = f
        self._has_reference: bool = f.n_in() > 2

    @property
    def has_reference(self) -> bool:
        """Check if the cost function has a reference input."""
        return self._has_reference

    def __call__(self, *args: Any) -> ca.MX:  # noqa: ANN401
        """Evaluate the cost function."""
        return self.f(*args)

    def validate_dimensions(self, nx: int, nu: int | None = None) -> None:
        """Validate dimensions of the casadi function."""
        if self.f.size_in(0)[0] != nx:
            msg = f"Cost function state input size ({self.f.size_in(0)[0]}) must match state size ({nx})."
            raise ValueError(msg)

        if nu is not None:
            if self.f.size_in(1)[0] != nu:
                msg = f"Cost function control input size ({self.f.size_in(1)[0]}) must match control size ({nu})."
                raise ValueError(msg)

            if self.has_reference and self.f.n_in() == 4 and (self.f.size_in(2)[0] != nx or self.f.size_in(3)[0] != nu):
                msg = f"Cost function reference inputs must match state ({nx}) and control ({nu}) sizes."
                raise ValueError(msg)
        # Terminal cost
        elif self.has_reference:
            if self.f.n_in() == 2 and self.f.size_in(1)[0] != nx:
                msg = f"Cost function reference input must match state ({nx}) size."
                raise ValueError(msg)

        if self.f.size_out(0)[0] != 1:
            msg = "Cost function must return a scalar."
            raise ValueError(msg)


class LQRCost(CostFunction):
    """LQR Stage Cost."""

    def __init__(self, Q: np.ndarray, R: np.ndarray, N_cross: np.ndarray | None = None) -> None:
        nx = Q.shape[0]
        nu = R.shape[0]
        x = ca.MX.sym("x", nx)
        u = ca.MX.sym("u", nu)
        x_ref = ca.MX.sym("x_ref", nx)
        u_ref = ca.MX.sym("u_ref", nu)

        if N_cross is None:
            N_cross = np.zeros((nx, nu))

        if Q.shape[0] != Q.shape[1] or Q.shape[0] != nx:
            msg = "Matrix Q must be square and match state dimension."
            raise ValueError(msg)
        if R.shape[0] != R.shape[1] or R.shape[0] != nu:
            msg = "Matrix R must be square and match control dimension."
            raise ValueError(msg)
        if N_cross.shape[0] != nx or N_cross.shape[1] != nu:
            msg = "Matrix N_cross must match state and control dimensions."
            raise ValueError(msg)

        dx = x - x_ref
        du = u - u_ref

        f = ca.Function(
            "lqr_obj",
            [x, u, x_ref, u_ref],
            [dx.T @ Q @ dx + du.T @ R @ du + dx.T @ N_cross @ du],
            ["x", "u", "x_ref", "u_ref"],
            ["f"],
        )
        super().__init__(f)


class TerminalLQRCost(CostFunction):
    """LQR Terminal Cost."""

    def __init__(self, Q: np.ndarray) -> None:
        nx = Q.shape[0]

        if Q.shape[1] != nx:
            msg = "Matrix Q must be square."
            raise ValueError(msg)

        x = ca.MX.sym("x", nx)
        x_ref = ca.MX.sym("x_ref", nx)

        dx = x - x_ref

        f = ca.Function("term_lqr_obj", [x, x_ref], [dx.T @ Q @ dx], ["x", "x_ref"], ["f"])
        super().__init__(f)
        self._has_reference = True


class QuadraticCost(CostFunction):
    """Quadratic Stage Cost."""

    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        N_cross: np.ndarray | None = None,
    ) -> None:
        nx = Q.shape[0]
        nu = R.shape[0]
        x = ca.MX.sym("x", nx)
        u = ca.MX.sym("u", nu)

        if q is None:
            q = np.zeros((nx, 1))
        if r is None:
            r = np.zeros((nu, 1))
        if N_cross is None:
            N_cross = np.zeros((nx, nu))

        if Q.shape[0] != Q.shape[1] or Q.shape[0] != nx:
            msg = "Matrix Q must be square and match state dimension."
            raise ValueError(msg)
        if R.shape[0] != R.shape[1] or R.shape[0] != nu:
            msg = "Matrix R must be square and match control dimension."
            raise ValueError(msg)
        if q.shape[0] != nx:
            msg = "Vector q must match state dimension."
            raise ValueError(msg)
        if r.shape[0] != nu:
            msg = "Vector r must match control dimension."
            raise ValueError(msg)
        if N_cross.shape[0] != nx or N_cross.shape[1] != nu:
            msg = "Matrix N_cross must match state and control dimensions."
            raise ValueError(msg)

        f = ca.Function(
            "quadr_obj", [x, u], [x.T @ Q @ x + x.T @ q + u.T @ R @ u + u.T @ r + x.T @ N_cross @ u], ["x", "u"], ["f"]
        )
        super().__init__(f)


class TerminalQuadraticCost(CostFunction):
    """Quadratic Terminal Cost."""

    def __init__(self, Q: np.ndarray, q: np.ndarray) -> None:
        nx = Q.shape[0]

        if Q.shape[1] != nx:
            msg = "Matrix Q must be square."
            raise ValueError(msg)
        if q.shape[0] != nx:
            msg = "Vector q must have the same length as Q."
            raise ValueError(msg)

        x = ca.MX.sym("x", nx)

        f = ca.Function("term_quadr_obj", [x], [x.T @ Q @ x + x.T @ q], ["x"], ["f"])
        super().__init__(f)


class Objective:
    """Full cost structure over the horizon."""

    @typing.overload
    def __init__(self, cost: CostFunction, N: int) -> None: ...

    @typing.overload
    def __init__(self, cost: CostFunction, cost_term: CostFunction, N: int) -> None: ...

    @typing.overload
    def __init__(self, stage_costs: list[CostFunction]) -> None: ...

    @typing.overload
    def __init__(self, stage_costs: list[CostFunction], cost_term: CostFunction) -> None: ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.stage_costs: list[CostFunction] = []
        self.terminal_cost: CostFunction | None = None

        if len(args) == 2 and isinstance(args[0], CostFunction) and isinstance(args[1], int):
            cost, N = args
            self.stage_costs = [cost] * N
            self.terminal_cost = None
        elif (
            len(args) == 3
            and isinstance(args[0], CostFunction)
            and isinstance(args[1], CostFunction)
            and isinstance(args[2], int)
        ):
            cost, cost_term, N = args
            self.stage_costs = [cost] * N
            self.terminal_cost = cost_term
        elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], CostFunction):
            stage_costs, cost_term = args
            self.stage_costs = typing.cast("list[CostFunction]", list(stage_costs))
            self.terminal_cost = cost_term
        elif len(args) == 1 and isinstance(args[0], list):
            stage_costs = args[0]
            self.stage_costs = typing.cast("list[CostFunction]", list(stage_costs))
            self.terminal_cost = None
        else:
            msg = "Invalid arguments for Objective constructor."
            raise ValueError(msg)

    @property
    def has_reference(self) -> bool:
        """Check if any cost function in the objective requires a reference.

        Returns
        -------
            True if any stage cost or the terminal cost has a reference input.
        """
        if any(c.has_reference for c in self.stage_costs):
            return True
        return bool(self.terminal_cost is not None and self.terminal_cost.has_reference)

    def validate_dimensions(self, nx: int, nu: int) -> None:
        """Validate dimensions of all cost functions."""
        for c in self.stage_costs:
            c.validate_dimensions(nx, nu)
        if self.terminal_cost is not None:
            self.terminal_cost.validate_dimensions(nx)


class LQRObjective(Objective):
    """LQR Objective Factory."""

    def __init__(self, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray, N: int, N_cross: np.ndarray | None = None) -> None:
        stage_cost = LQRCost(Q, R, N_cross)
        terminal_cost = TerminalLQRCost(Qf)
        super().__init__(stage_cost, terminal_cost, N)


class QuadraticObjective(Objective):
    """Quadratic Objective Factory."""

    def __init__(  # noqa: PLR0913
        self,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        qf: np.ndarray,
        N: int,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        N_cross: np.ndarray | None = None,
    ) -> None:
        stage_cost = QuadraticCost(Q, R, q, r, N_cross)
        terminal_cost = TerminalQuadraticCost(Qf, qf)
        super().__init__(stage_cost, terminal_cost, N)
