from typing import Any

import casadi as ca
import numpy as np
from casadi.casadi import Function
from numpy._typing import ArrayLike


class OCP:
    def __init__(
        self,
        N: int,
        dt: float,
        objective: ca.Function,
        dynamics: ca.Function,
        eq_constraints: ca.Function | None = None,
        in_eq_constraints: ca.Function | None = None,
        terminal_objective: ca.Function | None = None,
        terminal_eq_constraints: ca.Function | None = None,
        terminal_in_eq_constraints: ca.Function | None = None,
    ) -> None:
        self.N = N
        self.dt = dt
        self.objective = objective
        self.dynamics = dynamics
        self.eq_constraints = eq_constraints
        self.in_eq_constraints = in_eq_constraints
        self.terminal_objective = terminal_objective
        self.terminal_eq_constraints = terminal_eq_constraints
        self.terminal_in_eq_constraints = terminal_in_eq_constraints

        self._opti: ca.Opti | None = None
        self._x0_param: ca.MX | None = None
        self._X: ca.MX | None = None
        self._U: ca.MX | None = None

        self._nx, self._nu = self._validate_dimensions()

    def _validate_dimensions(self) -> tuple[int, int]:
        """Validates all casadi functions and returns nx and nu."""

        if self.dynamics.n_in() < 2:
            raise ValueError("Dynamics function must take at least two arguments (state x and control u).")

        nx = self.dynamics.size_in(0)[0]
        nu = self.dynamics.size_in(1)[0]

        if self.dynamics.size_out(0)[0] != nx:
            raise ValueError(
                f"Dynamics function output size ({self.dynamics.size_out(0)[0]}) must match state size ({nx})."
            )

        if self.objective.size_in(0)[0] != nx or self.objective.size_in(1)[0] != nu:
            raise ValueError(f"Objective function inputs must match state ({nx}) and control ({nu}) sizes.")
        if self.objective.size_out(0)[0] != 1:
            raise ValueError("Objective function must return a scalar.")

        if (
            hasattr(self, "eq_constraints")
            and self.eq_constraints is not None
            and (self.eq_constraints.size_in(0)[0] != nx or self.eq_constraints.size_in(1)[0] != nu)
        ):
            raise ValueError(f"eq_constraints function inputs must match state ({nx}) and control ({nu}) sizes.")

        if (
            hasattr(self, "in_eq_constraints")
            and self.in_eq_constraints is not None
            and (self.in_eq_constraints.size_in(0)[0] != nx or self.in_eq_constraints.size_in(1)[0] != nu)
        ):
            raise ValueError(f"in_eq_constraints function inputs must match state ({nx}) and control ({nu}) sizes.")

        if hasattr(self, "terminal_objective") and self.terminal_objective is not None:
            if self.terminal_objective.size_in(0)[0] != nx:
                raise ValueError(f"terminal_objective function input must match state ({nx}) size.")
            if self.terminal_objective.size_out(0)[0] != 1:
                raise ValueError("terminal_objective function must return a scalar.")

        if (
            hasattr(self, "terminal_eq_constraints")
            and self.terminal_eq_constraints is not None
            and (self.terminal_eq_constraints.size_in(0)[0] != nx)
        ):
            raise ValueError(f"terminal_eq_constraints function input must match state ({nx}) size.")

        if (
            hasattr(self, "terminal_in_eq_constraints")
            and self.terminal_in_eq_constraints is not None
            and (self.terminal_in_eq_constraints.size_in(0)[0] != nx)
        ):
            raise ValueError(f"terminal_in_eq_constraints function input must match state ({nx}) size.")

        return nx, nu

    def setup(
        self,
        method: str = "multiple_shooting",
        dynamics_type: str = "continuous",
        integrator: str = "rk4",
        solver: str = "ipopt",
        plugin_opts: dict[Any, Any] | None = None,
        solver_opts: dict[Any, Any] | None = None,
    ) -> None:
        nx = self._nx
        nu = self._nu

        self._opti = ca.Opti()
        self._x0_param = self._opti.parameter(nx)

        if dynamics_type == "continuous":
            if integrator == "rk4":
                # Runge-Kutta 4 integration
                X0 = ca.MX.sym("X0", nx)
                U0 = ca.MX.sym("U0", nu)
                k1 = self.dynamics(X0, U0)
                k2 = self.dynamics(X0 + self.dt / 2.0 * k1, U0)
                k3 = self.dynamics(X0 + self.dt / 2.0 * k2, U0)
                k4 = self.dynamics(X0 + self.dt * k3, U0)
                X_next = X0 + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                dyn_func = ca.Function("dyn_rk4", [X0, U0], [X_next])
            else:
                # Forward Euler
                X0 = ca.MX.sym("X0", nx)
                U0 = ca.MX.sym("U0", nu)
                X_next = X0 + self.dt * self.dynamics(X0, U0)
                dyn_func = ca.Function("dyn_euler", [X0, U0], [X_next])
        elif dynamics_type == "discrete":
            dyn_func = self.dynamics
        else:
            raise ValueError(f"Unknown dynamics_type: {dynamics_type}")

        if method == "single_shooting":
            self._U = self._opti.variable(nu, self.N)
            self._X = self._opti.variable(nx, self.N + 1)  # Still define it for later extraction

            x_k = self._x0_param
            self._opti.subject_to(self._X[:, 0] == x_k)

            cost = 0.0
            for k in range(self.N):
                u_k = self._U[:, k]

                # Cost
                cost += self.objective(x_k, u_k)

                # Constraints
                if self.eq_constraints is not None:
                    self._opti.subject_to(self.eq_constraints(x_k, u_k) == 0)
                if self.in_eq_constraints is not None:
                    self._opti.subject_to(self.in_eq_constraints(x_k, u_k) <= 0)

                # Dynamics
                x_k = dyn_func(x_k, u_k)
                self._opti.subject_to(self._X[:, k + 1] == x_k)  # Link to X so we can extract it easily

        elif method == "multiple_shooting":
            self._X = self._opti.variable(nx, self.N + 1)
            self._U = self._opti.variable(nu, self.N)

            self._opti.subject_to(self._X[:, 0] == self._x0_param)

            cost = 0.0
            for k in range(self.N):
                x_k = self._X[:, k]
                u_k = self._U[:, k]

                # Cost
                cost += self.objective(x_k, u_k)

                # Constraints
                if self.eq_constraints is not None:
                    self._opti.subject_to(self.eq_constraints(x_k, u_k) == 0)
                if self.in_eq_constraints is not None:
                    self._opti.subject_to(self.in_eq_constraints(x_k, u_k) <= 0)

                # Dynamics gap closing
                x_next = dyn_func(x_k, u_k)
                self._opti.subject_to(self._X[:, k + 1] == x_next)

        elif method == "collocation":
            if dynamics_type == "discrete":
                raise ValueError("Collocation method is not applicable to discrete dynamics.")

            self._X = self._opti.variable(nx, self.N + 1)
            self._U = self._opti.variable(nu, self.N)

            self._opti.subject_to(self._X[:, 0] == self._x0_param)

            # Hermite-Simpson direct collocation
            cost = 0.0
            for k in range(self.N):
                x_k = self._X[:, k]
                x_k_next = self._X[:, k + 1]
                u_k = self._U[:, k]

                # Cost
                cost += self.objective(x_k, u_k)

                # Constraints
                if self.eq_constraints is not None:
                    self._opti.subject_to(self.eq_constraints(x_k, u_k) == 0)
                if self.in_eq_constraints is not None:
                    self._opti.subject_to(self.in_eq_constraints(x_k, u_k) <= 0)

                # Hermite-Simpson collocation point
                f_k = self.dynamics(x_k, u_k)
                f_k_next = self.dynamics(x_k_next, u_k)

                # State at midpoint
                x_c = 0.5 * (x_k + x_k_next) + (self.dt / 8.0) * (f_k - f_k_next)

                # Dynamics at midpoint
                f_c = self.dynamics(x_c, u_k)

                # Simpson's rule for state integration
                self._opti.subject_to(x_k_next == x_k + (self.dt / 6.0) * (f_k + 4 * f_c + f_k_next))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Terminal conditions
        x_N = self._X[:, self.N]
        if self.terminal_objective is not None:
            cost += self.terminal_objective(x_N)
        if self.terminal_eq_constraints is not None:
            self._opti.subject_to(self.terminal_eq_constraints(x_N) == 0)
        if self.terminal_in_eq_constraints is not None:
            self._opti.subject_to(self.terminal_in_eq_constraints(x_N) <= 0)

        self._opti.minimize(cost)

        # Set up solver options
        p_opts = {"expand": True}
        s_opts = {}

        # Add basic defaults for common ipopt usage if it's the chosen solver
        if solver == "ipopt":
            s_opts = {"max_iter": 1000, "print_level": 0}

        if plugin_opts is not None:
            p_opts.update(plugin_opts)

        if solver_opts is not None:
            s_opts.update(solver_opts)

        self._opti.solver(solver, p_opts, s_opts)

    def solve(self, x0: ArrayLike) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Solves the OCP for a given initial state.

        Args:
            x0: Initial state as a numpy array or list.

        Returns:
            Tuple of (X_opt, U_opt, status)
            - X_opt: numpy array of optimal state trajectory
            - U_opt: numpy array of optimal control trajectory
            - status: string indicating solver status
        """
        if self._opti is None:
            raise RuntimeError("OCP has not been set up. Call setup() first.")

        self._opti.set_value(self._x0_param, x0)

        try:
            sol = self._opti.solve()
            X_opt = sol.value(self._X)
            U_opt = sol.value(self._U)
            status: str = sol.stats()["return_status"]
        except Exception as e:
            # If solve fails, return the values at the last iteration
            X_opt = self._opti.debug.value(self._X)
            U_opt = self._opti.debug.value(self._U)
            status = f"Solve_Failed: {str(e)}"

        # Ensure 2D arrays even if nx=1 or nu=1
        if isinstance(X_opt, np.ndarray) and X_opt.ndim == 1:
            X_opt = X_opt.reshape(1, -1)
        if isinstance(U_opt, np.ndarray) and U_opt.ndim == 1:
            U_opt = U_opt.reshape(1, -1)

        return X_opt, U_opt, status


def quadratic_objective(
    Q: np.ndarray, R: np.ndarray, q: np.ndarray | None = None, r: np.ndarray | None = None, N: np.ndarray | None = None
) -> Function:
    nx = Q.shape[0]
    nu = R.shape[0]
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    if q is None:
        q = np.zeros((nx, 1))
    if r is None:
        r = np.zeros((nu, 1))
    if N is None:
        N = np.zeros((nx, nu))

    if Q.shape[0] != Q.shape[1] or Q.shape[0] != nx:
        raise ValueError("Matrix Q must be square and match state dimension.")
    if R.shape[0] != R.shape[1] or R.shape[0] != nu:
        raise ValueError("Matrix R must be square and match control dimension.")
    if q.shape[0] != nx:
        raise ValueError("Vector q must match state dimension.")
    if r.shape[0] != nu:
        raise ValueError("Vector r must match control dimension.")
    if N.shape[0] != nx or N.shape[1] != nu:
        raise ValueError("Matrix N must match state and control dimensions.")

    return ca.Function(
        "quadr_obj", [x, u], [x.T @ Q @ x + x.T @ q + u.T @ R @ u + u.T @ r + x.T @ N @ u], ["x", "u"], ["f"]
    )


def linear_constraints(F: np.ndarray, G: np.ndarray, h: np.ndarray) -> Function:
    nx = F.shape[1]
    nu = G.shape[1]

    if F.shape[0] != G.shape[0] or F.shape[0] != h.shape[0]:
        raise ValueError("The number of rows in F, G, and h must be equal.")

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    return ca.Function("lin_con", [x, u], [F @ x + G @ u - h], ["x", "u"], ["f"])


def linear_dynamics(A: np.ndarray, B: np.ndarray) -> Function:
    nx = A.shape[1]
    nu = B.shape[1]

    if A.shape[0] != nx:
        raise ValueError("Matrix A must be square.")
    if B.shape[0] != nx:
        raise ValueError("Matrix B must have the same number of rows as A.")

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    return ca.Function("lin_dyn", [x, u], [A @ x + B @ u], ["x", "u"], ["f"])


def state_bounds_constraints(x_min: np.ndarray, x_max: np.ndarray, nu: int) -> Function:
    nx = x_min.shape[0]
    if x_max.shape[0] != nx:
        raise ValueError("x_min and x_max must have the same length.")

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    return ca.Function("state_bounds", [x, u], [ca.vertcat(x_min - x, x - x_max)], ["x", "u"], ["f"])


def control_bounds_constraints(u_min: np.ndarray, u_max: np.ndarray, nx: int) -> Function:
    nu = u_min.shape[0]
    if u_max.shape[0] != nu:
        raise ValueError("u_min and u_max must have the same length.")

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    return ca.Function("control_bounds", [x, u], [ca.vertcat(u_min - u, u - u_max)], ["x", "u"], ["f"])


def terminal_quadratic_objective(Q: np.ndarray, q: np.ndarray) -> Function:
    nx = Q.shape[0]

    if Q.shape[1] != nx:
        raise ValueError("Matrix Q must be square.")
    if q.shape[0] != nx:
        raise ValueError("Vector q must have the same length as Q.")

    x = ca.MX.sym("x", nx)

    return ca.Function("term_quadr_obj", [x], [x.T @ Q @ x + x.T @ q], ["x"], ["f"])


def terminal_linear_constraints(F: np.ndarray, h: np.ndarray) -> Function:
    nx = F.shape[1]

    if F.shape[0] != h.shape[0]:
        raise ValueError("The number of rows in F and h must be equal.")

    x = ca.MX.sym("x", nx)

    return ca.Function("term_lin_con", [x], [F @ x - h], ["x"], ["f"])
