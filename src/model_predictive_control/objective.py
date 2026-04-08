import typing
from typing import Any

import casadi as ca
import numpy as np

from model_predictive_control.ocp import (
    lqr_objective,
    quadratic_objective,
    terminal_lqr_objective,
    terminal_quadratic_objective,
)


class CostFunction:
    """Base class for cost functions."""

    def __init__(self, f: ca.Function) -> None:
        self.f = f
        self._has_reference: bool = f.n_in() > 2

    @property
    def has_reference(self) -> bool:
        return self._has_reference

    def __call__(self, *args: Any) -> ca.MX:
        return self.f(*args)

    def validate_dimensions(self, nx: int, nu: int | None = None) -> None:
        """Validate dimensions of the casadi function."""
        if self.f.size_in(0)[0] != nx:
            raise ValueError(f"Cost function state input size ({self.f.size_in(0)[0]}) must match state size ({nx}).")

        if nu is not None:
            if self.f.size_in(1)[0] != nu:
                raise ValueError(
                    f"Cost function control input size ({self.f.size_in(1)[0]}) must match control size ({nu})."
                )

            if self.has_reference:
                if self.f.n_in() == 4 and (self.f.size_in(2)[0] != nx or self.f.size_in(3)[0] != nu):
                    raise ValueError(
                        f"Cost function reference inputs must match state ({nx}) and control ({nu}) sizes."
                    )
        # Terminal cost
        elif self.has_reference:
            if self.f.n_in() == 2 and self.f.size_in(1)[0] != nx:
                raise ValueError(f"Cost function reference input must match state ({nx}) size.")

        if self.f.size_out(0)[0] != 1:
            raise ValueError("Cost function must return a scalar.")


class LQRCost(CostFunction):
    """LQR Stage Cost."""

    def __init__(self, Q: np.ndarray, R: np.ndarray, N_cross: np.ndarray | None = None) -> None:
        f = lqr_objective(Q, R, N_cross)
        super().__init__(f)


class TerminalLQRCost(CostFunction):
    """LQR Terminal Cost."""

    def __init__(self, Q: np.ndarray) -> None:
        f = terminal_lqr_objective(Q)
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
        f = quadratic_objective(Q, R, q, r, N_cross)
        super().__init__(f)


class TerminalQuadraticCost(CostFunction):
    """Quadratic Terminal Cost."""

    def __init__(self, Q: np.ndarray, q: np.ndarray) -> None:
        f = terminal_quadratic_objective(Q, q)
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
            self.stage_costs = list(stage_costs)
            self.terminal_cost = cost_term
        elif len(args) == 1 and isinstance(args[0], list):
            stage_costs = args[0]
            self.stage_costs = list(stage_costs)
            self.terminal_cost = None
        else:
            raise ValueError("Invalid arguments for Objective constructor.")

    @property
    def has_reference(self) -> bool:
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

    def __init__(
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
