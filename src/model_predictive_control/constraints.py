from collections.abc import Iterable
from typing import Any

import casadi as ca
import numpy as np


class BaseConstraint:
    def __init__(self, is_equality: bool = False):
        self.is_equality = is_equality

    def validate_dimensions(self, nx: int, nu: int) -> None:
        raise NotImplementedError


class Constraint(BaseConstraint):
    """Base constraint wrapping f(x, u)"""

    def __init__(self, f: ca.Function, is_equality: bool = False):
        super().__init__(is_equality)
        self.f = f

    def validate_dimensions(self, nx: int, nu: int) -> None:
        if self.f.n_in() != 2:
            raise ValueError("Constraint must take exactly two arguments (state x and control u).")
        if self.f.size_in(0)[0] != nx or self.f.size_in(1)[0] != nu:
            raise ValueError(f"Constraint function inputs must match state ({nx}) and control ({nu}) sizes.")


class StateConstraint(Constraint):
    """Constraint wrapping f(x)"""

    def __init__(self, f: ca.Function, is_equality: bool = False):
        super(Constraint, self).__init__(is_equality)
        self.f = f

    def validate_dimensions(self, nx: int, nu: int) -> None:
        if self.f.n_in() != 1:
            raise ValueError("StateConstraint must take exactly one argument (state x).")
        if self.f.size_in(0)[0] != nx:
            raise ValueError(f"StateConstraint function input must match state ({nx}) size.")


class ControlConstraint(Constraint):
    """Constraint wrapping f(u)"""

    def __init__(self, f: ca.Function, is_equality: bool = False):
        super(Constraint, self).__init__(is_equality)
        self.f = f

    def validate_dimensions(self, nx: int, nu: int) -> None:
        if self.f.n_in() != 1:
            raise ValueError("ControlConstraint must take exactly one argument (control u).")
        if self.f.size_in(0)[0] != nu:
            raise ValueError(f"ControlConstraint function input must match control ({nu}) size.")


class LinearConstraint(BaseConstraint):
    """Constraint wrapping F*x + G*u <= h or == h"""

    def __init__(
        self,
        h: np.ndarray,
        F: np.ndarray | None = None,
        G: np.ndarray | None = None,
        is_equality: bool = False,
    ):
        super().__init__(is_equality)
        self.h = np.asarray(h, dtype=float)
        self.F = np.asarray(F, dtype=float) if F is not None else None
        self.G = np.asarray(G, dtype=float) if G is not None else None

        # Determine number of constraints from h
        # h could be (nc,) or (N, nc)
        self.nc = self.h.shape[-1] if self.h.ndim > 0 else 1

    def validate_dimensions(self, nx: int, nu: int) -> None:
        # Expected shapes: h is (nc,) or (N, nc)
        # F is (nc, nx) or (N, nc, nx)
        # G is (nc, nu) or (N, nc, nu)
        if self.F is not None:
            if self.F.shape[-1] != nx:
                raise ValueError(f"LinearConstraint F matrix last dimension must match state ({nx}) size.")
            if self.F.shape[-2] != self.nc:
                raise ValueError(f"LinearConstraint F matrix must have {self.nc} rows.")
        if self.G is not None:
            if self.G.shape[-1] != nu:
                raise ValueError(f"LinearConstraint G matrix last dimension must match control ({nu}) size.")
            if self.G.shape[-2] != self.nc:
                raise ValueError(f"LinearConstraint G matrix must have {self.nc} rows.")


class ConstraintList:
    def __init__(self) -> None:
        self.constraints: list[tuple[BaseConstraint, Any]] = []

    def add(self, constraint: BaseConstraint, time_indices: Any) -> None:
        self.constraints.append((constraint, time_indices))

    def resolve_indices(self, time_indices: Any, N: int) -> list[int]:
        if isinstance(time_indices, int):
            idx = time_indices if time_indices >= 0 else N + 1 + time_indices
            return [idx]
        if isinstance(time_indices, slice):
            return list(range(*time_indices.indices(N + 1)))
        if isinstance(time_indices, Iterable):
            return [i if i >= 0 else N + 1 + i for i in time_indices]
        raise ValueError(f"Unsupported time_indices format: {type(time_indices)}")

    def __iter__(self) -> Any:
        return iter(self.constraints)

    def __len__(self) -> int:
        return len(self.constraints)
