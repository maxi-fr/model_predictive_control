from collections.abc import Iterable, Iterator

import casadi as ca
import numpy as np
import scipy.stats as st
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


class StateBoundConstraint(StateConstraint):
    """State bounds constraint: x_min <= x <= x_max."""

    def __init__(self, x_min: ArrayLike, x_max: ArrayLike) -> None:
        """
        Initialize the state bound constraint.

        Parameters
        ----------
            x_min: Lower bounds array for states.
            x_max: Upper bounds array for states.
        """
        self.x_min = np.asarray(x_min, dtype=float)
        self.x_max = np.asarray(x_max, dtype=float)

        if self.x_min.shape != self.x_max.shape:
            msg = "x_min and x_max must have the same shape."
            raise ValueError(msg)

        nx = self.x_min.shape[0]
        x_sym = ca.MX.sym("x", nx)

        f_val = ca.vertcat(self.x_min - x_sym, x_sym - self.x_max)
        f_func = ca.Function("state_bounds", [x_sym], [f_val])

        super().__init__(f=f_func, is_equality=False)


class ControlBoundConstraint(ControlConstraint):
    """Control bounds constraint: u_min <= u <= u_max."""

    def __init__(self, u_min: ArrayLike, u_max: ArrayLike) -> None:
        """
        Initialize the control bound constraint.

        Parameters
        ----------
            u_min: Lower bounds array for controls.
            u_max: Upper bounds array for controls.
        """
        self.u_min = np.asarray(u_min, dtype=float)
        self.u_max = np.asarray(u_max, dtype=float)

        if self.u_min.shape != self.u_max.shape:
            msg = "u_min and u_max must have the same shape."
            raise ValueError(msg)

        nu = self.u_min.shape[0]
        u_sym = ca.MX.sym("u", nu)

        f_val = ca.vertcat(self.u_min - u_sym, u_sym - self.u_max)
        f_func = ca.Function("control_bounds", [u_sym], [f_val])

        super().__init__(f=f_func, is_equality=False)


class SphereConstraint(StateConstraint):
    """Sphere constraint: ||x[indices] - center||^2 <= radius^2 or >= radius^2."""

    def __init__(
        self,
        center: ArrayLike,
        radius: float,
        indices: list[int] | slice,
        nx: int,
        keepout: bool = False,
    ) -> None:
        """
        Initialize the sphere constraint.

        Parameters
        ----------
            center: Center of the sphere.
            radius: Radius of the sphere.
            indices: Indices of the state vector to apply the constraint to.
            nx: Number of states (needed to create the symbolic variable).
            keepout: If True, evaluates to radius^2 - ||x[indices] - center||^2 <= 0 (obstacle avoidance).
                     If False, evaluates to ||x[indices] - center||^2 - radius^2 <= 0 (containment).
        """
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.keepout = keepout

        x_sym = ca.MX.sym("x", nx)
        x_sub = x_sym[indices]

        if x_sub.shape[0] != self.center.shape[0]:
            msg = "Center dimension must match the number of specified indices."
            raise ValueError(msg)

        diff = x_sub - self.center
        sq_dist = ca.sumsqr(diff)

        # radius^2 <= sq_dist  =>  radius^2 - sq_dist <= 0
        # sq_dist <= radius^2  =>  sq_dist - radius^2 <= 0
        f_val = self.radius**2 - sq_dist if keepout else sq_dist - self.radius**2

        f_func = ca.Function("sphere_constraint", [x_sym], [f_val])
        super().__init__(f=f_func, is_equality=False)


class StateNormConstraint(StateConstraint):
    """State norm constraint: ||x[indices]||_p <= max_norm."""

    def __init__(self, max_norm: float, indices: list[int] | slice, nx: int, p: int = 2) -> None:
        """
        Initialize the state norm constraint.

        Parameters
        ----------
            max_norm: Maximum allowed norm.
            indices: Indices of the state vector to apply the constraint to.
            nx: Number of states (needed to create the symbolic variable).
            p: The order of the norm (default is 2).
        """
        self.max_norm = float(max_norm)
        self.p = p

        x_sym = ca.MX.sym("x", nx)
        x_sub = x_sym[indices]

        if p == 2:
            norm = ca.norm_2(x_sub)
        elif p == 1:
            norm = ca.norm_1(x_sub)
        else:
            msg = f"Norm of order {p} is not natively supported by CasADi helper functions."
            raise ValueError(msg)

        f_val = norm - self.max_norm
        f_func = ca.Function("state_norm", [x_sym], [f_val])
        super().__init__(f=f_func, is_equality=False)


class ControlNormConstraint(ControlConstraint):
    """Control norm constraint: ||u[indices]||_p <= max_norm."""

    def __init__(self, max_norm: float, indices: list[int] | slice, nu: int, p: int = 2) -> None:
        """
        Initialize the control norm constraint.

        Parameters
        ----------
            max_norm: Maximum allowed norm.
            indices: Indices of the control vector to apply the constraint to.
            nu: Number of controls (needed to create the symbolic variable).
            p: The order of the norm (default is 2).
        """
        self.max_norm = float(max_norm)
        self.p = p

        u_sym = ca.MX.sym("u", nu)
        u_sub = u_sym[indices]

        if p == 2:
            norm = ca.norm_2(u_sub)
        elif p == 1:
            norm = ca.norm_1(u_sub)
        else:
            msg = f"Norm of order {p} is not natively supported by CasADi helper functions."
            raise ValueError(msg)

        f_val = norm - self.max_norm
        f_func = ca.Function("control_norm", [u_sym], [f_val])
        super().__init__(f=f_func, is_equality=False)


class LinearConstraint(Constraint):
    """Constraint wrapping F*x + G*u <= h or == h."""

    def __init__(  # noqa: C901, PLR0913
        self,
        h: ArrayLike,
        F: ArrayLike | None = None,
        G: ArrayLike | None = None,
        is_equality: bool = False,
        nx: int | None = None,
        nu: int | None = None,
    ) -> None:
        """
        Initialize the linear constraint.

        Parameters
        ----------
            h: Upper bound or target value array.
            F: State coefficient matrix.
            G: Control coefficient matrix.
            is_equality: Whether the constraint is an equality constraint (== h) or inequality (<= h).
            nx: Number of states (required if F is None).
            nu: Number of controls (required if G is None).
        """
        self.h = np.asarray(h, dtype=float)

        if self.h.ndim > 1:
            msg = "LinearConstraint h array must be 1D. Time-varying bounds are not supported."
            raise ValueError(msg)

        self.nc = self.h.shape[0] if self.h.size > 0 else 0

        self.F = np.asarray(F, dtype=float) if F is not None else None
        self.G = np.asarray(G, dtype=float) if G is not None else None

        if self.F is not None and self.F.ndim > 2:
            msg = "LinearConstraint F array must be 2D. Time-varying matrices are not supported."
            raise ValueError(msg)

        if self.G is not None and self.G.ndim > 2:
            msg = "LinearConstraint G array must be 2D. Time-varying matrices are not supported."
            raise ValueError(msg)

        if self.F is not None:
            if self.F.ndim == 1:
                self.F = self.F.reshape(1, -1)
            nx = self.F.shape[1]
        elif nx is None:
            msg = "nx must be provided if F is None."
            raise ValueError(msg)

        if self.G is not None:
            if self.G.ndim == 1:
                self.G = self.G.reshape(1, -1)
            nu = self.G.shape[1]
        elif nu is None:
            msg = "nu must be provided if G is None."
            raise ValueError(msg)

        x_sym = ca.MX.sym("x", nx)
        u_sym = ca.MX.sym("u", nu)

        val = 0
        if self.F is not None:
            val += self.F @ x_sym
        if self.G is not None:
            val += self.G @ u_sym

        f_val = val - self.h
        f_func = ca.Function("linear_constraint", [x_sym, u_sym], [f_val])

        super().__init__(f=f_func, is_equality=is_equality)

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
        super().validate_dimensions(nx, nu)

        if self.F is not None:
            if self.F.shape[1] != nx:
                msg = f"LinearConstraint F matrix last dimension must match state ({nx}) size."
                raise ValueError(msg)
            if self.F.shape[0] != self.nc:
                msg = f"LinearConstraint F matrix must have {self.nc} rows."
                raise ValueError(msg)
        if self.G is not None:
            if self.G.shape[1] != nu:
                msg = f"LinearConstraint G matrix last dimension must match control ({nu}) size."
                raise ValueError(msg)
            if self.G.shape[0] != self.nc:
                msg = f"LinearConstraint G matrix must have {self.nc} rows."
                raise ValueError(msg)


class TerminalLinearConstraint(StateConstraint):
    """Terminal constraint wrapping F*x <= h or == h."""

    def __init__(
        self,
        h: ArrayLike,
        F: ArrayLike,
        is_equality: bool = False,
    ) -> None:
        """
        Initialize the terminal linear constraint.

        Parameters
        ----------
            h: Upper bound or target value array.
            F: State coefficient matrix.
            is_equality: Whether the constraint is an equality constraint (== h) or inequality (<= h).
        """
        self.h = np.asarray(h, dtype=float)
        self.F = np.asarray(F, dtype=float)

        if self.h.ndim > 1:
            msg = "TerminalLinearConstraint h array must be 1D. Time-varying bounds are not supported."
            raise ValueError(msg)

        if self.F.ndim > 2:
            msg = "TerminalLinearConstraint F array must be 2D. Time-varying matrices are not supported."
            raise ValueError(msg)

        if self.F.ndim == 1:
            self.F = self.F.reshape(1, -1)

        self.nc = self.h.shape[0] if self.h.size > 0 else 0
        nx = self.F.shape[1]

        x_sym = ca.MX.sym("x", nx)
        f_val = self.F @ x_sym - self.h
        f_func = ca.Function("terminal_linear_constraint", [x_sym], [f_val])

        super().__init__(f=f_func, is_equality=is_equality)

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
        super().validate_dimensions(nx, nu)

        if self.F.ndim == 1:
            self.F = self.F.reshape(1, -1)

        if self.F.shape[1] != nx:
            msg = f"TerminalLinearConstraint F matrix last dimension must match state ({nx}) size."
            raise ValueError(msg)
        if self.F.shape[0] != self.nc:
            msg = f"TerminalLinearConstraint F matrix must have {self.nc} rows."
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


class LinearChanceConstraint(LinearConstraint):
    """Linear chance constraint wrapping F*x + G*u <= h, tightened for Gaussian noise."""

    def __init__(  # noqa: PLR0913
        self,
        h: ArrayLike,
        A: ArrayLike,
        Sigma_w: ArrayLike,
        epsilon: float,
        N: int,
        F: ArrayLike | None = None,
        G: ArrayLike | None = None,
        nx: int | None = None,
        nu: int | None = None,
    ) -> None:
        """
        Initialize the linear chance constraint.

        Parameters
        ----------
            h: Upper bound array (will be tightened).
            A: System dynamics matrix (nx x nx) for variance propagation.
            Sigma_w: Process noise covariance matrix (nx x nx).
            epsilon: Maximum allowed violation probability (0 < epsilon < 1).
            N: Horizon length over which to propagate variance.
            F: State coefficient matrix.
            G: Control coefficient matrix.
            nx: Number of states (required if F is None).
            nu: Number of controls (required if G is None).
        """
        h_arr = np.asarray(h, dtype=float)
        A_arr = np.asarray(A, dtype=float)
        Sigma_w_arr = np.asarray(Sigma_w, dtype=float)
        F_arr = np.asarray(F, dtype=float) if F is not None else None

        nx_inferred = A_arr.shape[0]

        # Compute open-loop variance propagation over horizon N
        Sigma_k = np.zeros((nx_inferred, nx_inferred))
        max_variances = np.zeros(nx_inferred)

        for _ in range(N):
            max_variances = np.maximum(max_variances, np.diag(Sigma_k))
            Sigma_k = A_arr @ Sigma_k @ A_arr.T + Sigma_w_arr

        # We need to project the state variance into the constraint space defined by F.
        # Var(F x) = F * Var(x) * F^T. Wait, the notebook just takes max variance of position
        # but to be general, the constraint is F_i * x <= h_i.
        # The variance of F_i * x is F_i * Sigma_k * F_i^T.

        # In the notebook: F is [[1, 0]] or [[-1, 0]], so F_i * x is just p or -p.
        # The variance of p or -p is the same: F_i * Sigma_k * F_i^T.

        # Let's compute the maximum standard deviation for each constraint i over the horizon
        nc = h_arr.shape[0] if h_arr.size > 0 else 0
        h_tight = h_arr.copy()

        z_val = st.norm.ppf(1 - epsilon)

        if F_arr is not None:
            if F_arr.ndim == 1:
                F_arr = F_arr.reshape(1, -1)

            max_std_devs = np.zeros(nc)

            # Recalculate over horizon N and find max variance for each constraint
            Sigma_k = np.zeros((nx_inferred, nx_inferred))
            for _ in range(N):
                # Variance of F x is diag(F * Sigma_k * F^T)
                var_Fx = np.diag(F_arr @ Sigma_k @ F_arr.T)
                max_std_devs = np.maximum(max_std_devs, np.sqrt(var_Fx))
                Sigma_k = A_arr @ Sigma_k @ A_arr.T + Sigma_w_arr

            tightening = z_val * max_std_devs
            h_tight = h_arr - tightening

        super().__init__(h=h_tight, F=F, G=G, is_equality=False, nx=nx, nu=nu)
