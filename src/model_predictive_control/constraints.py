from collections.abc import Iterable, Iterator

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

Index = int | slice | Iterable[int]


class BaseConstraint:
    """Base class for all constraints."""

    def __init__(self, is_equality: bool = False) -> None:
        """
        Initialize the base constraint.

        Parameters
        ----------
            is_equality: Whether the constraint is an equality constraint (== 0) or inequality (<= 0).
        """
        self.is_equality = is_equality

    def validate_dimensions(self, nx: int, nu: int) -> None:
        """
        Validate the dimensions of the constraint against the state and control sizes.

        Parameters
        ----------
            nx: Number of states.
            nu: Number of controls.
        """
        raise NotImplementedError


class Constraint(BaseConstraint):
    """Base constraint wrapping f(x, u)."""

    def __init__(self, f: ca.Function, is_equality: bool = False) -> None:
        """
        Initialize the constraint.

        Parameters
        ----------
            f: CasADi function representing the constraint, f(x, u).
            is_equality: Whether the constraint is an equality constraint (== 0) or inequality (<= 0).
        """
        super().__init__(is_equality)
        self.f = f

    def validate_dimensions(self, nx: int, nu: int) -> None:
        """
        Validate the dimensions of the constraint against the state and control sizes.

        Parameters
        ----------
            nx: Number of states.
            nu: Number of controls.

        Raises
        ------
            ValueError: If the constraint function inputs do not match the expected state and control sizes.
        """
        if self.f.n_in() != 2:
            msg = "Constraint must take exactly two arguments (state x and control u)."
            raise ValueError(msg)
        if self.f.size_in(0)[0] != nx or self.f.size_in(1)[0] != nu:
            msg = f"Constraint function inputs must match state ({nx}) and control ({nu}) sizes."
            raise ValueError(msg)


class StateConstraint(Constraint):
    """Constraint wrapping f(x)."""

    def __init__(self, f: ca.Function, is_equality: bool = False) -> None:
        """
        Initialize the state constraint.

        Parameters
        ----------
            f: CasADi function representing the constraint, f(x).
            is_equality: Whether the constraint is an equality constraint (== 0) or inequality (<= 0).
        """
        super().__init__(f, is_equality)

    def validate_dimensions(self, nx: int, nu: int) -> None:  # noqa: ARG002
        """
        Validate the dimensions of the constraint against the state and control sizes.

        Parameters
        ----------
            nx: Number of states.
            nu: Number of controls.

        Raises
        ------
            ValueError: If the constraint function inputs do not match the expected state size.
        """
        if self.f.n_in() != 1:
            msg = "StateConstraint must take exactly one argument (state x)."
            raise ValueError(msg)
        if self.f.size_in(0)[0] != nx:
            msg = f"StateConstraint function input must match state ({nx}) size."
            raise ValueError(msg)


class ControlConstraint(Constraint):
    """Constraint wrapping f(u)."""

    def __init__(self, f: ca.Function, is_equality: bool = False) -> None:
        """
        Initialize the control constraint.

        Parameters
        ----------
            f: CasADi function representing the constraint, f(u).
            is_equality: Whether the constraint is an equality constraint (== 0) or inequality (<= 0).
        """
        super().__init__(f, is_equality)

    def validate_dimensions(self, nx: int, nu: int) -> None:  # noqa: ARG002
        """
        Validate the dimensions of the constraint against the state and control sizes.

        Parameters
        ----------
            nx: Number of states.
            nu: Number of controls.

        Raises
        ------
            ValueError: If the constraint function inputs do not match the expected control size.
        """
        if self.f.n_in() != 1:
            msg = "ControlConstraint must take exactly one argument (control u)."
            raise ValueError(msg)
        if self.f.size_in(0)[0] != nu:
            msg = f"ControlConstraint function input must match control ({nu}) size."
            raise ValueError(msg)


class LinearConstraint(BaseConstraint):
    """Constraint wrapping F*x + G*u <= h or == h."""

    def __init__(
        self,
        h: ArrayLike,
        F: ArrayLike | None = None,
        G: ArrayLike | None = None,
        is_equality: bool = False,
    ) -> None:
        """
        Initialize the linear constraint.

        Parameters
        ----------
            h: Upper bound or target value array.
            F: State coefficient matrix.
            G: Control coefficient matrix.
            is_equality: Whether the constraint is an equality constraint (== h) or inequality (<= h).
        """
        super().__init__(is_equality)
        self.h = np.asarray(h, dtype=float)
        self.F = np.asarray(F, dtype=float) if F is not None else None
        self.G = np.asarray(G, dtype=float) if G is not None else None

        # Determine number of constraints from h
        # h could be (nc,) or (N, nc)
        self.nc = self.h.shape[-1] if self.h.ndim > 0 else 1

    def validate_dimensions(self, nx: int, nu: int) -> None:
        """
        Validate the dimensions of the linear constraint matrices against the state and control sizes.

        Parameters
        ----------
            nx: Number of states.
            nu: Number of controls.

        Raises
        ------
            ValueError: If the matrix dimensions do not match the expected state and control sizes.
        """
        # Expected shapes: h is (nc,) or (N, nc)
        # F is (nc, nx) or (N, nc, nx)
        # G is (nc, nu) or (N, nc, nu)
        if self.F is not None:
            if self.F.shape[-1] != nx:
                msg = f"LinearConstraint F matrix last dimension must match state ({nx}) size."
                raise ValueError(msg)
            if self.F.shape[-2] != self.nc:
                msg = f"LinearConstraint F matrix must have {self.nc} rows."
                raise ValueError(msg)
        if self.G is not None:
            if self.G.shape[-1] != nu:
                msg = f"LinearConstraint G matrix last dimension must match control ({nu}) size."
                raise ValueError(msg)
            if self.G.shape[-2] != self.nc:
                msg = f"LinearConstraint G matrix must have {self.nc} rows."
                raise ValueError(msg)


class ConstraintList:
    """A list of constraints applied over specific time indices."""

    def __init__(self) -> None:
        """Initialize the constraint list."""
        self.constraints: list[tuple[BaseConstraint, Index]] = []

    def add(self, constraint: BaseConstraint, time_indices: Index) -> None:
        """
        Add a constraint to the list.

        Parameters
        ----------
            constraint: The constraint to add.
            time_indices: Time indices where the constraint should be applied. Can be an integer, a slice, or an iterable of integers.
        """
        self.constraints.append((constraint, time_indices))

    def resolve_indices(self, time_indices: Index, N: int) -> list[int]:
        """
        Resolve the time indices format into a list of integers.

        Parameters
        ----------
            time_indices: Time indices where the constraint should be applied.
            N: The horizon length.

        Returns
        -------
            A list of explicit integer indices.

        Raises
        ------
            ValueError: If the time_indices format is unsupported.
        """
        if isinstance(time_indices, int):
            idx = time_indices if time_indices >= 0 else N + 1 + time_indices
            return [idx]
        if isinstance(time_indices, slice):
            return list(range(*time_indices.indices(N + 1)))
        if isinstance(time_indices, Iterable):
            return [i if i >= 0 else N + 1 + i for i in time_indices]

        msg = f"Unsupported time_indices format: {type(time_indices)}"  # type: ignore[unreachable]
        raise ValueError(msg)

    def __iter__(self) -> Iterator[tuple[BaseConstraint, Index]]:
        """
        Iterate over the constraints.

        Returns
        -------
            An iterator over the list of constraints and their time indices.
        """
        return iter(self.constraints)

    def __len__(self) -> int:
        """
        Get the number of constraints in the list.

        Returns
        -------
            The number of constraints.
        """
        return len(self.constraints)
