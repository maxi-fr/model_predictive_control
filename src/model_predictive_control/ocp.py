from typing import Any

import casadi as ca
import numpy as np
import scipy.linalg
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

    def linearize(
        self, x_bar: ArrayLike, u_bar: ArrayLike, dynamics_type: str = "continuous", integrator: str = "rk4"
    ) -> "LinearOCP":
        """
        Linearizes the OCP around a nominal state and control trajectory (or fixed point)
        and returns a LinearOCP instance.
        """
        x_bar = np.asarray(x_bar, dtype=float)
        u_bar = np.asarray(u_bar, dtype=float)

        nx = self._nx
        nu = self._nu
        N = self.N

        if x_bar.ndim == 1:
            if x_bar.shape[0] != nx:
                raise ValueError(f"x_bar fixed point must have size {nx}")
            X_bar = np.tile(x_bar, (N + 1, 1))
        else:
            if x_bar.shape != (N + 1, nx):
                raise ValueError(f"x_bar trajectory must have shape ({N + 1}, {nx})")
            X_bar = x_bar

        if u_bar.ndim == 1:
            if u_bar.shape[0] != nu:
                raise ValueError(f"u_bar fixed point must have size {nu}")
            U_bar = np.tile(u_bar, (N, 1))
        else:
            if u_bar.shape != (N, nu):
                raise ValueError(f"u_bar trajectory must have shape ({N}, {nu})")
            U_bar = u_bar

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

        # Dynamics symbolic derivatives
        x_sym = ca.MX.sym("x", nx)
        u_sym = ca.MX.sym("u", nu)

        f_val = dyn_func(x_sym, u_sym)
        A_func = ca.Function("A", [x_sym, u_sym], [ca.jacobian(f_val, x_sym)])
        B_func = ca.Function("B", [x_sym, u_sym], [ca.jacobian(f_val, u_sym)])

        # Objective symbolic derivatives
        L_val = self.objective(x_sym, u_sym)
        q_func = ca.Function("q", [x_sym, u_sym], [ca.jacobian(L_val, x_sym).T])
        r_func = ca.Function("r", [x_sym, u_sym], [ca.jacobian(L_val, u_sym).T])
        Q_func = ca.Function("Q", [x_sym, u_sym], [ca.hessian(L_val, x_sym)[0]])
        R_func = ca.Function("R", [x_sym, u_sym], [ca.hessian(L_val, u_sym)[0]])

        # N_cross is partial^2 L / (partial x partial u)
        grad_x = ca.jacobian(L_val, x_sym)
        N_cross_func = ca.Function("N_cross", [x_sym, u_sym], [ca.jacobian(grad_x, u_sym)])

        # Stage constraints
        C_funcs = []
        if self.in_eq_constraints is not None:
            C_val = self.in_eq_constraints(x_sym, u_sym)
            C_funcs.append(C_val)
        if self.eq_constraints is not None:
            E_val = self.eq_constraints(x_sym, u_sym)
            C_funcs.append(E_val)
            C_funcs.append(-E_val)

        has_stage_constraints = False
        nc = 0
        if len(C_funcs) > 0:
            C_total = ca.vertcat(*C_funcs)
            F_func = ca.Function("F", [x_sym, u_sym], [ca.jacobian(C_total, x_sym)])
            G_func = ca.Function("G", [x_sym, u_sym], [ca.jacobian(C_total, u_sym)])
            h_func_val = ca.Function("h_val", [x_sym, u_sym], [-C_total])
            has_stage_constraints = True
            nc = C_total.shape[0]

        # Arrays for the linear OCP
        A = np.zeros((N, nx, nx))
        B = np.zeros((N, nx, nu))
        Q = np.zeros((N, nx, nx))
        R = np.zeros((N, nu, nu))
        N_cross = np.zeros((N, nx, nu))
        q = np.zeros((N, nx))
        r = np.zeros((N, nu))

        if has_stage_constraints:
            F = np.zeros((N, nc, nx))
            G = np.zeros((N, nc, nu))
            h = np.zeros((N, nc))
        else:
            F, G, h = None, None, None

        for k in range(N):
            xk = X_bar[k]
            uk = U_bar[k]

            A[k] = np.array(A_func(xk, uk))
            B[k] = np.array(B_func(xk, uk))
            Q[k] = np.array(Q_func(xk, uk))
            R[k] = np.array(R_func(xk, uk))
            N_cross[k] = np.array(N_cross_func(xk, uk))
            q[k] = np.array(q_func(xk, uk)).flatten()
            r[k] = np.array(r_func(xk, uk)).flatten()

            if has_stage_constraints:
                F[k] = np.array(F_func(xk, uk))
                G[k] = np.array(G_func(xk, uk))
                h[k] = np.array(h_func_val(xk, uk)).flatten()

        # Terminal cost
        x_N_sym = ca.MX.sym("x_N", nx)
        if self.terminal_objective is not None:
            L_term_val = self.terminal_objective(x_N_sym)
            qf_func = ca.Function("qf", [x_N_sym], [ca.jacobian(L_term_val, x_N_sym).T])
            Qf_func = ca.Function("Qf", [x_N_sym], [ca.hessian(L_term_val, x_N_sym)[0]])

            xN = X_bar[N]
            Qf = np.array(Qf_func(xN))
            qf = np.array(qf_func(xN)).flatten()
        else:
            Qf = np.zeros((nx, nx))
            qf = np.zeros(nx)

        # Terminal constraints
        C_term_funcs = []
        if self.terminal_in_eq_constraints is not None:
            C_term_funcs.append(self.terminal_in_eq_constraints(x_N_sym))
        if self.terminal_eq_constraints is not None:
            E_term_val = self.terminal_eq_constraints(x_N_sym)
            C_term_funcs.append(E_term_val)
            C_term_funcs.append(-E_term_val)

        if len(C_term_funcs) > 0:
            C_term_total = ca.vertcat(*C_term_funcs)
            F_term_func = ca.Function("F_term", [x_N_sym], [ca.jacobian(C_term_total, x_N_sym)])
            h_term_func_val = ca.Function("h_term_val", [x_N_sym], [-C_term_total])

            xN = X_bar[N]
            F_term = np.array(F_term_func(xN))
            h_term = np.array(h_term_func_val(xN)).flatten()
        else:
            F_term, h_term = None, None

        return LinearOCP(
            N=N,
            dt=self.dt,
            A=A,
            B=B,
            Q=Q,
            R=R,
            q=q,
            r=r,
            N_cross=N_cross,
            Qf=Qf,
            qf=qf,
            F=F,
            G=G,
            h=h,
            F_term=F_term,
            h_term=h_term,
        )

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


class LinearOCP:
    def __init__(
        self,
        N: int,
        dt: float,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        N_cross: np.ndarray | None = None,
        Qf: np.ndarray | None = None,
        qf: np.ndarray | None = None,
        F: np.ndarray | None = None,
        G: np.ndarray | None = None,
        h: np.ndarray | None = None,
        F_term: np.ndarray | None = None,
        h_term: np.ndarray | None = None,
    ) -> None:
        """
        Initializes a Linear Optimal Control Problem.

        Cost stage: 0.5 * (x^T Q x + u^T R u) + x^T N_cross u + q^T x + r^T u
        Terminal cost: 0.5 * x_N^T Qf x_N + qf^T x_N
        Dynamics: x_{k+1} = A x_k + B u_k
        Constraints: F x_k + G u_k <= h
        Terminal constraints: F_term x_N <= h_term
        """
        self.N = N
        self.dt = dt

        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)

        self.nx = self.A.shape[-1]
        self.nu = self.B.shape[-1]

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        if q is None:
            self.q = np.zeros(self.nx) if self.Q.ndim == 2 else np.zeros((self.N, self.nx))
        else:
            self.q = np.asarray(q, dtype=float)

        if r is None:
            self.r = np.zeros(self.nu) if self.R.ndim == 2 else np.zeros((self.N, self.nu))
        else:
            self.r = np.asarray(r, dtype=float)

        if N_cross is None:
            self.N_cross = np.zeros((self.nx, self.nu)) if self.Q.ndim == 2 else np.zeros((self.N, self.nx, self.nu))
        else:
            self.N_cross = np.asarray(N_cross, dtype=float)

        self.Qf = (self.Q if self.Q.ndim == 2 else self.Q[-1]).copy() if Qf is None else np.asarray(Qf, dtype=float)
        self.qf = (self.q if self.q.ndim == 1 else self.q[-1]).copy() if qf is None else np.asarray(qf, dtype=float)

        self.F = None if F is None else np.asarray(F, dtype=float)
        self.G = None if G is None else np.asarray(G, dtype=float)
        self.h = None if h is None else np.asarray(h, dtype=float)

        self.F_term = None if F_term is None else np.asarray(F_term, dtype=float)
        self.h_term = None if h_term is None else np.asarray(h_term, dtype=float)

        self._validate_dimensions()

        self._method: str = ""
        self._solver_obj: ca.Function | None = None
        self._solver_args: dict[str, Any] = {}

        # We will save the parametric pieces of the QP to form arguments quickly in solve()
        self._qp_setup: dict[str, Any] = {}

    def _is_time_varying(self, arr: np.ndarray, expected_dims: int) -> bool:
        return bool(arr.ndim == expected_dims + 1 and arr.shape[0] == self.N)

    def _validate_dimensions(self) -> None:
        nx = self.nx
        nu = self.nu
        N = self.N

        def check_shape(arr: np.ndarray, base_shape: tuple[int, ...], name: str) -> None:
            if arr.shape != base_shape and arr.shape != (N,) + base_shape:
                raise ValueError(f"Array {name} must be {base_shape} or {(N,) + base_shape}")

        check_shape(self.A, (nx, nx), "A")
        check_shape(self.B, (nx, nu), "B")
        check_shape(self.Q, (nx, nx), "Q")
        check_shape(self.R, (nu, nu), "R")
        check_shape(self.q, (nx,), "q")
        check_shape(self.r, (nu,), "r")
        check_shape(self.N_cross, (nx, nu), "N_cross")

        if self.Qf.shape != (nx, nx):
            raise ValueError(f"Matrix Qf must be ({nx}, {nx})")
        if self.qf.shape != (nx,):
            raise ValueError(f"Vector qf must be ({nx},)")

        if self.F is not None or self.G is not None or self.h is not None:
            if self.F is None or self.G is None or self.h is None:
                raise ValueError("If any of F, G, h are provided, all three must be provided.")

            nc = self.F.shape[-2]
            check_shape(self.F, (nc, nx), "F")
            check_shape(self.G, (nc, nu), "G")
            check_shape(self.h, (nc,), "h")

        if self.F_term is not None or self.h_term is not None:
            if self.F_term is None or self.h_term is None:
                raise ValueError("If either F_term or h_term is provided, both must be provided.")
            nc_term = self.F_term.shape[0]
            if self.F_term.shape != (nc_term, nx):
                raise ValueError(f"Matrix F_term must be ({nc_term}, {nx})")
            if self.h_term.shape != (nc_term,):
                raise ValueError(f"Vector h_term must be ({nc_term},)")

    def _get_at_k(self, arr: np.ndarray, k: int) -> np.ndarray:
        if arr is None:
            return None
        # If the array has the time dimension, extract k-th slice
        # e.g., if A is (N, nx, nx) -> A.ndim == 3 -> returns A[k]
        # if A is (nx, nx) -> A.ndim == 2 -> returns A

        # Determine the base dimensions based on the attribute
        # For matrices like A, B, Q, R, N_cross, F, G, it's 2
        # For vectors like q, r, h, it's 1
        1 if len(arr.shape) == 1 or (
            len(arr.shape) == 2 and arr.shape[0] == self.N and arr.shape[1] != self.N and arr.shape[1] != arr.shape[0]
        ) else 2
        # Actually better to rely on known attributes

        arr.shape[-1] == self.nx or arr.shape[-1] == self.nu or (self.h is not None and np.array_equal(arr, self.h))
        if len(arr.shape) == 1:
            return arr
        elif len(arr.shape) == 2:
            if arr.shape[0] == self.N and arr.shape[1] in (self.nx, self.nu) and not np.array_equal(arr, self.A):
                # Time-varying vector
                return arr[k]
            elif self.h is not None and np.array_equal(arr, self.h):
                if arr.shape[0] == self.N:
                    return arr[k]
                else:
                    return arr
            else:
                # Constant matrix
                return arr
        elif len(arr.shape) == 3:
            # Time-varying matrix
            return arr[k]
        return arr

    def setup(
        self,
        method: str = "multiple_shooting",
        dynamics_type: str = "discrete",
        solver: str = "qrqp",
        plugin_opts: dict[str, Any] | None = None,
        solver_opts: dict[str, Any] | None = None,
    ) -> None:
        """
        Sets up the QP solver for the given method and solver backend.
        method: "multiple_shooting" (sparse) or "single_shooting" (condensed)
        dynamics_type: "discrete" or "continuous" (will be exactly discretized using ZOH)
        solver: The backend solver for ca.qpsol (e.g. 'qrqp', 'osqp')
        """
        self._method = method
        nx = self.nx
        nu = self.nu
        N = self.N

        # Compute A_d and B_d for all k
        A_d_list = []
        B_d_list = []
        for k in range(N):
            Ak = self.A[k] if self.A.ndim == 3 else self.A
            Bk = self.B[k] if self.B.ndim == 3 else self.B

            if dynamics_type == "continuous":
                M = np.zeros((nx + nu, nx + nu))
                M[:nx, :nx] = Ak
                M[:nx, nx:] = Bk
                M_d = scipy.linalg.expm(M * self.dt)
                A_d_list.append(M_d[:nx, :nx])
                B_d_list.append(M_d[:nx, nx:])
            elif dynamics_type == "discrete":
                A_d_list.append(Ak)
                B_d_list.append(Bk)
            else:
                raise ValueError(f"Unknown dynamics_type: {dynamics_type}")

        # "expand" is for nlpsol, conic doesn't need it.
        p_opts = {}
        s_opts = {}
        if plugin_opts is not None:
            p_opts.update(plugin_opts)
        if solver_opts is not None:
            s_opts.update(solver_opts)

        if method == "multiple_shooting":
            n_vars = (N + 1) * nx + N * nu

            H_sp = np.zeros((n_vars, n_vars))
            g_vec = np.zeros(n_vars)

            for k in range(N):
                Qk = self.Q[k] if self.Q.ndim == 3 else self.Q
                Rk = self.R[k] if self.R.ndim == 3 else self.R
                N_cross_k = self.N_cross[k] if self.N_cross.ndim == 3 else self.N_cross
                qk = self.q[k] if self.q.ndim == 2 else self.q
                rk = self.r[k] if self.r.ndim == 2 else self.r

                idx_x = k * (nx + nu)
                idx_u = idx_x + nx
                H_sp[idx_x : idx_x + nx, idx_x : idx_x + nx] = Qk
                H_sp[idx_u : idx_u + nu, idx_u : idx_u + nu] = Rk
                if np.any(N_cross_k):
                    H_sp[idx_x : idx_x + nx, idx_u : idx_u + nu] = N_cross_k
                    H_sp[idx_u : idx_u + nu, idx_x : idx_x + nx] = N_cross_k.T

                g_vec[idx_x : idx_x + nx] = qk
                g_vec[idx_u : idx_u + nu] = rk

            idx_xN = N * (nx + nu)
            H_sp[idx_xN : idx_xN + nx, idx_xN : idx_xN + nx] = self.Qf
            g_vec[idx_xN : idx_xN + nx] = self.qf

            n_eq = (N + 1) * nx
            A_eq = np.zeros((n_eq, n_vars))

            A_eq[:nx, :nx] = np.eye(nx)

            for k in range(N):
                row_idx = (k + 1) * nx
                idx_x = k * (nx + nu)
                idx_u = idx_x + nx
                idx_x_next = (k + 1) * (nx + nu)

                A_eq[row_idx : row_idx + nx, idx_x : idx_x + nx] = -A_d_list[k]
                A_eq[row_idx : row_idx + nx, idx_u : idx_u + nu] = -B_d_list[k]
                A_eq[row_idx : row_idx + nx, idx_x_next : idx_x_next + nx] = np.eye(nx)

            n_ineq = 0
            nc_term = 0
            if self.F is not None:
                if self.F.ndim == 3:
                    n_ineq += sum(self.F[k].shape[0] for k in range(N))
                else:
                    n_ineq += N * self.F.shape[0]
            if self.F_term is not None:
                nc_term = self.F_term.shape[0]
                n_ineq += nc_term

            if n_ineq > 0:
                A_ineq = np.zeros((n_ineq, n_vars))
                uba = np.zeros(n_eq + n_ineq)

                curr_row = 0
                for k in range(N):
                    if self.F is not None and self.G is not None and self.h is not None:
                        F_k = self.F[k] if self.F.ndim == 3 else self.F
                        G_k = self.G[k] if self.G.ndim == 3 else self.G
                        h_k = self.h[k] if self.h.ndim == 2 else self.h
                        nc = F_k.shape[0]

                        idx_x = k * (nx + nu)
                        idx_u = idx_x + nx
                        A_ineq[curr_row : curr_row + nc, idx_x : idx_x + nx] = F_k
                        A_ineq[curr_row : curr_row + nc, idx_u : idx_u + nu] = G_k
                        uba[n_eq + curr_row : n_eq + curr_row + nc] = h_k
                        curr_row += nc

                if self.F_term is not None and self.h_term is not None:
                    A_ineq[curr_row : curr_row + nc_term, idx_xN : idx_xN + nx] = self.F_term
                    uba[n_eq + curr_row : n_eq + curr_row + nc_term] = self.h_term

                A_c = np.vstack([A_eq, A_ineq])
            else:
                A_c = A_eq
                uba = np.zeros(n_eq)

            lba = np.zeros(n_eq + n_ineq)
            if n_ineq > 0:
                lba[n_eq:] = -1e9

            H_sp_ca = ca.DM(H_sp)
            A_c_ca = ca.DM(A_c)
            opts = {**p_opts, **s_opts}
            qp = {"h": H_sp_ca.sparsity(), "a": A_c_ca.sparsity()}
            self._solver_obj = ca.conic("solver", solver, qp, opts)
            self._qp_setup = {
                "h": H_sp_ca,
                "a": A_c_ca,
                "g": g_vec,
                "lba": lba,
                "uba": uba,
                "n_eq": n_eq,
                "n_vars": n_vars,
            }

        elif method == "single_shooting":
            n_vars = N * nu

            S_x = np.zeros(((N + 1) * nx, nx))
            S_u = np.zeros(((N + 1) * nx, N * nu))

            S_x[:nx, :] = np.eye(nx)

            for k in range(1, N + 1):
                S_x[k * nx : (k + 1) * nx, :] = A_d_list[k - 1] @ S_x[(k - 1) * nx : k * nx, :]

                for i in range(k):
                    if i == k - 1:
                        S_u[k * nx : (k + 1) * nx, i * nu : (i + 1) * nu] = B_d_list[k - 1]
                    else:
                        S_u[k * nx : (k + 1) * nx, i * nu : (i + 1) * nu] = (
                            A_d_list[k - 1] @ S_u[(k - 1) * nx : k * nx, i * nu : (i + 1) * nu]
                        )

            Q_bar = np.zeros(((N + 1) * nx, (N + 1) * nx))
            R_bar = np.zeros((N * nu, N * nu))
            N_bar = np.zeros(((N + 1) * nx, N * nu))
            q_bar = np.zeros((N + 1) * nx)
            r_bar = np.zeros(N * nu)

            for k in range(N):
                Q_bar[k * nx : (k + 1) * nx, k * nx : (k + 1) * nx] = self.Q[k] if self.Q.ndim == 3 else self.Q
                R_bar[k * nu : (k + 1) * nu, k * nu : (k + 1) * nu] = self.R[k] if self.R.ndim == 3 else self.R
                N_bar[k * nx : (k + 1) * nx, k * nu : (k + 1) * nu] = (
                    self.N_cross[k] if self.N_cross.ndim == 3 else self.N_cross
                )
                q_bar[k * nx : (k + 1) * nx] = self.q[k] if self.q.ndim == 2 else self.q
                r_bar[k * nu : (k + 1) * nu] = self.r[k] if self.r.ndim == 2 else self.r

            Q_bar[N * nx : (N + 1) * nx, N * nx : (N + 1) * nx] = self.Qf
            q_bar[N * nx : (N + 1) * nx] = self.qf

            H_u = S_u.T @ Q_bar @ S_u + R_bar + S_u.T @ N_bar + N_bar.T @ S_u
            H_sp_ca = ca.DM(H_u)

            n_ineq = 0
            nc_term = 0
            if self.F is not None:
                if self.F.ndim == 3:
                    n_ineq += sum(self.F[k].shape[0] for k in range(N))
                else:
                    n_ineq += N * self.F.shape[0]
            if self.F_term is not None:
                nc_term = self.F_term.shape[0]
                n_ineq += nc_term

            if n_ineq > 0:
                F_bar = np.zeros((n_ineq, (N + 1) * nx))
                G_bar = np.zeros((n_ineq, N * nu))
                h_bar = np.zeros(n_ineq)

                curr_row = 0
                for k in range(N):
                    if self.F is not None and self.G is not None and self.h is not None:
                        F_k = self.F[k] if self.F.ndim == 3 else self.F
                        G_k = self.G[k] if self.G.ndim == 3 else self.G
                        h_k = self.h[k] if self.h.ndim == 2 else self.h
                        nc = F_k.shape[0]

                        F_bar[curr_row : curr_row + nc, k * nx : (k + 1) * nx] = F_k
                        G_bar[curr_row : curr_row + nc, k * nu : (k + 1) * nu] = G_k
                        h_bar[curr_row : curr_row + nc] = h_k
                        curr_row += nc

                if self.F_term is not None and self.h_term is not None:
                    F_bar[curr_row : curr_row + nc_term, N * nx : (N + 1) * nx] = self.F_term
                    h_bar[curr_row : curr_row + nc_term] = self.h_term

                A_ineq_u = F_bar @ S_u + G_bar
                A_c_ca = ca.DM(A_ineq_u)
            else:
                A_c_ca = ca.DM.zeros(0, n_vars)
                F_bar = np.zeros((0, (N + 1) * nx))
                h_bar = np.zeros(0)

            opts = {**p_opts, **s_opts}
            qp = {"h": H_sp_ca.sparsity(), "a": A_c_ca.sparsity()}
            self._solver_obj = ca.conic("solver", solver, qp, opts)

            self._qp_setup = {
                "h": H_sp_ca,
                "a": A_c_ca,
                "S_x": S_x,
                "S_u": S_u,
                "S_xT_Q_bar_Su": S_x.T @ Q_bar @ S_u,
                "S_xT_N_bar": S_x.T @ N_bar,
                "q_barT_Su_plus_r_barT": q_bar.T @ S_u + r_bar.T,
                "F_bar_S_x": F_bar @ S_x if n_ineq > 0 else None,
                "h_bar": h_bar if n_ineq > 0 else None,
                "n_ineq": n_ineq,
                "n_vars": n_vars,
            }

        else:
            raise ValueError(f"Unknown method: {method}")

    def solve(self, x0: ArrayLike) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Solves the Linear OCP for a given initial state.
        """
        if self._solver_obj is None:
            raise RuntimeError("LinearOCP has not been set up. Call setup() first.")

        x0_arr = np.asarray(x0, dtype=float).flatten()
        if x0_arr.shape != (self.nx,):
            raise ValueError(f"Initial state must have length {self.nx}")

        if self._method == "multiple_shooting":
            lba = self._qp_setup["lba"].copy()
            uba = self._qp_setup["uba"].copy()

            # Update initial condition constraint: I * x_0 = x0
            lba[: self.nx] = x0_arr
            uba[: self.nx] = x0_arr

            res = self._solver_obj(
                h=self._qp_setup["h"], g=self._qp_setup["g"], a=self._qp_setup["a"], lba=lba, uba=uba
            )

            z_opt = np.array(res["x"]).flatten()
            status = self._solver_obj.stats()["return_status"]

            # Unpack z = [x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N]
            X_opt = np.zeros((self.nx, self.N + 1))
            U_opt = np.zeros((self.nu, self.N))

            for k in range(self.N):
                idx_x = k * (self.nx + self.nu)
                idx_u = idx_x + self.nx
                X_opt[:, k] = z_opt[idx_x : idx_x + self.nx]
                U_opt[:, k] = z_opt[idx_u : idx_u + self.nu]

            idx_xN = self.N * (self.nx + self.nu)
            X_opt[:, self.N] = z_opt[idx_xN : idx_xN + self.nx]

        elif self._method == "single_shooting":
            # g_u(x0) = (x_0^T S_x^T Q_bar S_u + x_0^T S_x^T N_bar + q_bar^T S_u + r_bar^T)^T
            g_u_T = (
                x0_arr.T @ self._qp_setup["S_xT_Q_bar_Su"]
                + x0_arr.T @ self._qp_setup["S_xT_N_bar"]
                + self._qp_setup["q_barT_Su_plus_r_barT"]
            )
            g_u = g_u_T.T

            if self._qp_setup["n_ineq"] > 0:
                # lba <= A U <= uba
                # A_ineq_u U <= h_bar - F_bar S_x x_0
                uba = self._qp_setup["h_bar"] - self._qp_setup["F_bar_S_x"] @ x0_arr
                lba = np.full(self._qp_setup["n_ineq"], -1e9)
                res = self._solver_obj(h=self._qp_setup["h"], g=g_u, a=self._qp_setup["a"], lba=lba, uba=uba)
            else:
                res = self._solver_obj(h=self._qp_setup["h"], g=g_u, a=self._qp_setup["a"])

            U_vec = np.array(res["x"]).flatten()
            status = self._solver_obj.stats()["return_status"]

            # Reconstruct X_opt and U_opt
            U_opt = U_vec.reshape((self.N, self.nu)).T
            X_vec = self._qp_setup["S_x"] @ x0_arr + self._qp_setup["S_u"] @ U_vec
            X_opt = X_vec.reshape((self.N + 1, self.nx)).T

        else:
            raise ValueError(f"Unknown method: {self._method}")

        return X_opt, U_opt, status
