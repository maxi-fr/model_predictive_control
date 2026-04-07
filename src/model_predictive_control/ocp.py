from typing import Any

import casadi as ca
import numpy as np
import numpy.typing as npt
import scipy.linalg
from casadi.casadi import Function
from numpy._typing import ArrayLike

from model_predictive_control.constraints import (
    Constraint,
    ConstraintList,
    ControlConstraint,
    LinearConstraint,
    StateConstraint,
)


class OCP:  # noqa: D101 TODO: add doc string
    def __init__(  # noqa: PLR0913
        self,
        N: int,
        dt: float,
        objective: ca.Function,
        dynamics: ca.Function,
        constraints: ConstraintList | None = None,
        terminal_objective: ca.Function | None = None,
    ) -> None:
        self.N = N
        self.dt = dt
        self.objective = objective
        self.dynamics = dynamics
        self.constraints = constraints if constraints is not None else ConstraintList()
        self.terminal_objective = terminal_objective

        self._opti: ca.Opti | None = None
        self._x0_param: ca.MX | None = None
        self._X: ca.MX | None = None
        self._U: ca.MX | None = None

        self._nx, self._nu = self.validate_dimensions()

    def validate_dimensions(self) -> tuple[int, int]:  # noqa: C901
        """Validate all casadi functions and returns nx and nu."""
        if self.dynamics.n_in() < 2:
            msg = "Dynamics function must take at least two arguments (state x and control u)."
            raise ValueError(msg)

        nx = self.dynamics.size_in(0)[0]
        nu = self.dynamics.size_in(1)[0]

        if self.dynamics.size_out(0)[0] != nx:
            msg = f"Dynamics function output size ({self.dynamics.size_out(0)[0]}) must match state size ({nx})."
            raise ValueError(msg)

        if self.objective.size_in(0)[0] != nx or self.objective.size_in(1)[0] != nu:
            msg = f"Objective function inputs must match state ({nx}) and control ({nu}) sizes."
            raise ValueError(msg)

        if self.objective.n_in() == 4 and (self.objective.size_in(2)[0] != nx or self.objective.size_in(3)[0] != nu):
            msg = f"Objective function reference inputs must match state ({nx}) and control ({nu}) sizes."
            raise ValueError(msg)

        if self.objective.size_out(0)[0] != 1:
            msg = "Objective function must return a scalar."
            raise ValueError(msg)

        if hasattr(self, "terminal_objective") and self.terminal_objective is not None:
            if self.terminal_objective.size_in(0)[0] != nx:
                msg = f"terminal_objective function input must match state ({nx}) size."
                raise ValueError(msg)

            if self.terminal_objective.n_in() == 2 and self.terminal_objective.size_in(1)[0] != nx:
                msg = f"terminal_objective function reference input must match state ({nx}) size."
                raise ValueError(msg)

            if self.terminal_objective.size_out(0)[0] != 1:
                msg = "terminal_objective function must return a scalar."
                raise ValueError(msg)

        if hasattr(self, "constraints") and self.constraints is not None:
            for constraint, _ in self.constraints:
                constraint.validate_dimensions(nx, nu)

        return nx, nu

    def linearize(  # noqa: PLR0915, PLR0912, PLR0913, C901
        self,
        x_bar: ArrayLike,
        u_bar: ArrayLike,
        dynamics_type: str = "continuous",
        integrator: str = "rk4",
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> "LinearOCP":
        """Linearize the OCP around a nominal state and control trajectory (or fixed point)."""
        x_bar = np.asarray(x_bar, dtype=float)
        u_bar = np.asarray(u_bar, dtype=float)

        nx = self._nx
        nu = self._nu
        N = self.N

        X_ref: np.ndarray | None = None
        U_ref: np.ndarray | None = None

        if self.objective.n_in() == 4:
            if x_ref is None:
                X_ref = np.zeros((N + 1, nx))
            else:
                X_ref = np.asarray(x_ref, dtype=float)
                if X_ref.shape != (N + 1, nx):
                    msg = f"x_ref trajectory must have shape ({N + 1}, {nx})"
                    raise ValueError(msg)

            if u_ref is None:
                U_ref = np.zeros((N, nu))
            else:
                U_ref = np.asarray(u_ref, dtype=float)
                if U_ref.shape != (N, nu):
                    msg = f"u_ref trajectory must have shape ({N}, {nu})"
                    raise ValueError(msg)
        elif (
            hasattr(self, "terminal_objective")
            and self.terminal_objective is not None
            and self.terminal_objective.n_in() == 2
        ):
            if x_ref is None:
                X_ref = np.zeros((N + 1, nx))
            else:
                X_ref = np.asarray(x_ref, dtype=float)
                if X_ref.shape != (N + 1, nx):
                    msg = f"x_ref trajectory must have shape ({N + 1}, {nx})"
                    raise ValueError(msg)

        if x_bar.ndim == 1:
            if x_bar.shape[0] != nx:
                msg = f"x_bar fixed point must have size {nx}"
                raise ValueError(msg)
            X_bar = np.tile(x_bar, (N + 1, 1))
        else:
            if x_bar.shape != (N + 1, nx):
                msg = f"x_bar trajectory must have shape ({N + 1}, {nx})"
                raise ValueError(msg)
            X_bar = x_bar

        if u_bar.ndim == 1:
            if u_bar.shape[0] != nu:
                msg = f"u_bar fixed point must have size {nu}"
                raise ValueError(msg)
            U_bar = np.tile(u_bar, (N, 1))
        else:
            if u_bar.shape != (N, nu):
                msg = f"u_bar trajectory must have shape ({N}, {nu})"
                raise ValueError(msg)
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
            msg = f"Unknown dynamics_type: {dynamics_type}"
            raise ValueError(msg)

        # Dynamics symbolic derivatives
        x_sym = ca.MX.sym("x", nx)
        u_sym = ca.MX.sym("u", nu)

        f_val = dyn_func(x_sym, u_sym)
        A_func = ca.Function("A", [x_sym, u_sym], [ca.jacobian(f_val, x_sym)])
        B_func = ca.Function("B", [x_sym, u_sym], [ca.jacobian(f_val, u_sym)])

        # Objective symbolic derivatives
        if self.objective.n_in() == 4:
            x_ref_sym = ca.MX.sym("x_ref", nx)
            u_ref_sym = ca.MX.sym("u_ref", nu)
            L_val = self.objective(x_sym, u_sym, x_ref_sym, u_ref_sym)
            q_func = ca.Function("q", [x_sym, u_sym, x_ref_sym, u_ref_sym], [ca.jacobian(L_val, x_sym).T])
            r_func = ca.Function("r", [x_sym, u_sym, x_ref_sym, u_ref_sym], [ca.jacobian(L_val, u_sym).T])
            Q_func = ca.Function("Q", [x_sym, u_sym, x_ref_sym, u_ref_sym], [ca.hessian(L_val, x_sym)[0]])
            R_func = ca.Function("R", [x_sym, u_sym, x_ref_sym, u_ref_sym], [ca.hessian(L_val, u_sym)[0]])
            grad_x = ca.jacobian(L_val, x_sym)
            N_cross_func = ca.Function("N_cross", [x_sym, u_sym, x_ref_sym, u_ref_sym], [ca.jacobian(grad_x, u_sym)])
        else:
            L_val = self.objective(x_sym, u_sym)
            q_func = ca.Function("q", [x_sym, u_sym], [ca.jacobian(L_val, x_sym).T])
            r_func = ca.Function("r", [x_sym, u_sym], [ca.jacobian(L_val, u_sym).T])
            Q_func = ca.Function("Q", [x_sym, u_sym], [ca.hessian(L_val, x_sym)[0]])
            R_func = ca.Function("R", [x_sym, u_sym], [ca.hessian(L_val, u_sym)[0]])
            grad_x = ca.jacobian(L_val, x_sym)
            N_cross_func = ca.Function("N_cross", [x_sym, u_sym], [ca.jacobian(grad_x, u_sym)])

        # Arrays for the linear OCP
        A = np.zeros((N, nx, nx))
        B = np.zeros((N, nx, nu))
        Q = np.zeros((N, nx, nx))
        R = np.zeros((N, nu, nu))
        N_cross = np.zeros((N, nx, nu))
        q = np.zeros((N, nx))
        r = np.zeros((N, nu))

        lin_constraints = ConstraintList()
        # Precompute constraint jacobian functions for faster evaluation
        constraint_funcs: list[dict[str, Any]] = []
        if hasattr(self, "constraints") and self.constraints is not None:
            for constraint, time_indices in self.constraints:
                # Need to linearize each constraint
                if isinstance(constraint, StateConstraint):
                    C_val = constraint.f(x_sym)
                    F_func = ca.Function("F", [x_sym], [ca.jacobian(C_val, x_sym)])
                    h_func_val = ca.Function("h_val", [x_sym], [-C_val])
                    constraint_funcs.append(
                        {"c": constraint, "F_func": F_func, "G_func": None, "h_func": h_func_val, "ti": time_indices}
                    )
                elif isinstance(constraint, ControlConstraint):
                    C_val = constraint.f(u_sym)
                    G_func = ca.Function("G", [u_sym], [ca.jacobian(C_val, u_sym)])
                    h_func_val = ca.Function("h_val", [u_sym], [-C_val])
                    constraint_funcs.append(
                        {"c": constraint, "F_func": None, "G_func": G_func, "h_func": h_func_val, "ti": time_indices}
                    )
                elif isinstance(constraint, Constraint):
                    C_val = constraint.f(x_sym, u_sym)
                    F_func = ca.Function("F", [x_sym, u_sym], [ca.jacobian(C_val, x_sym)])
                    G_func = ca.Function("G", [x_sym, u_sym], [ca.jacobian(C_val, u_sym)])
                    h_func_val = ca.Function("h_val", [x_sym, u_sym], [-C_val])
                    constraint_funcs.append(
                        {"c": constraint, "F_func": F_func, "G_func": G_func, "h_func": h_func_val, "ti": time_indices}
                    )
                elif isinstance(constraint, LinearConstraint):
                    constraint_funcs.append(
                        {"c": constraint, "F_func": None, "G_func": None, "h_func": None, "ti": time_indices}
                    )

        # Keep track of F, G, h for each constraint over N
        # We'll build a LinearConstraint for each original constraint
        lin_constraint_data: list[dict[str, Any]] = []
        for c_data in constraint_funcs:
            c = c_data["c"]
            time_indices = c_data["ti"]
            if isinstance(c, LinearConstraint):
                lin_constraint_data.append(c_data)
                continue

            resolved_indices = self.constraints.resolve_indices(time_indices, N)
            has_F = isinstance(c, (StateConstraint, Constraint))
            has_G = isinstance(c, (ControlConstraint, Constraint))
            nc = c.f.size_out(0)[0] if hasattr(c, "f") else 1

            F_arr = np.zeros((len(resolved_indices), nc, nx)) if has_F else None
            G_arr = np.zeros((len(resolved_indices), nc, nu)) if has_G else None
            h_arr = np.zeros((len(resolved_indices), nc))

            new_data = c_data.copy()
            new_data.update({"res_idx": resolved_indices, "F_arr": F_arr, "G_arr": G_arr, "h_arr": h_arr})
            lin_constraint_data.append(new_data)

        for k in range(N):
            xk = X_bar[k]
            uk = U_bar[k]

            A[k] = np.array(A_func(xk, uk))
            B[k] = np.array(B_func(xk, uk))
            if self.objective.n_in() == 4:
                assert U_ref is not None
                assert X_ref is not None
                x_ref_k = X_ref[k]
                u_ref_k = U_ref[k]
                Q[k] = np.array(Q_func(xk, uk, x_ref_k, u_ref_k))
                R[k] = np.array(R_func(xk, uk, x_ref_k, u_ref_k))
                N_cross[k] = np.array(N_cross_func(xk, uk, x_ref_k, u_ref_k))
                q[k] = np.array(q_func(xk, uk, x_ref_k, u_ref_k)).flatten()
                r[k] = np.array(r_func(xk, uk, x_ref_k, u_ref_k)).flatten()
            else:
                Q[k] = np.array(Q_func(xk, uk))
                R[k] = np.array(R_func(xk, uk))
                N_cross[k] = np.array(N_cross_func(xk, uk))
                q[k] = np.array(q_func(xk, uk)).flatten()
                r[k] = np.array(r_func(xk, uk)).flatten()

            for c_data in lin_constraint_data:
                c = c_data["c"]
                if isinstance(c, LinearConstraint):
                    continue

                F_func = c_data["F_func"]
                G_func = c_data["G_func"]
                h_func_val = c_data["h_func"]
                resolved_indices = c_data["res_idx"]
                F_arr = c_data["F_arr"]
                G_arr = c_data["G_arr"]
                h_arr = c_data["h_arr"]
                if k in resolved_indices:
                    idx = resolved_indices.index(k)
                    if isinstance(c, StateConstraint):
                        assert F_arr is not None
                        F_arr[idx] = np.array(F_func(xk))
                        h_arr[idx] = np.array(h_func_val(xk)).flatten()
                    elif isinstance(c, ControlConstraint):
                        assert G_arr is not None
                        G_arr[idx] = np.array(G_func(uk))
                        h_arr[idx] = np.array(h_func_val(uk)).flatten()
                    elif isinstance(c, Constraint):
                        assert F_arr is not None
                        assert G_arr is not None
                        F_arr[idx] = np.array(F_func(xk, uk))
                        G_arr[idx] = np.array(G_func(xk, uk))
                        h_arr[idx] = np.array(h_func_val(xk, uk)).flatten()

        # Terminal step (N) constraint evaluations
        xN = X_bar[N]
        for c_data in lin_constraint_data:
            c = c_data["c"]
            if isinstance(c, LinearConstraint):
                lin_constraints.add(c, c_data["ti"])
                continue

            F_func = c_data["F_func"]
            G_func = c_data["G_func"]
            h_func_val = c_data["h_func"]
            resolved_indices = c_data["res_idx"]
            F_arr = c_data["F_arr"]
            G_arr = c_data["G_arr"]
            h_arr = c_data["h_arr"]
            if N in resolved_indices:
                idx = resolved_indices.index(N)
                if isinstance(c, StateConstraint):
                    assert F_arr is not None
                    F_arr[idx] = np.array(F_func(xN))
                    h_arr[idx] = np.array(h_func_val(xN)).flatten()
                elif isinstance(c, Constraint):
                    # For terminal constraint on (x, u), we use u=0 as a dummy since u_N is not defined,
                    # but usually terminal constraints are StateConstraints.
                    assert F_arr is not None
                    assert G_arr is not None
                    dummy_u = np.zeros(nu)
                    F_arr[idx] = np.array(F_func(xN, dummy_u))
                    G_arr[idx] = np.array(G_func(xN, dummy_u))
                    h_arr[idx] = np.array(h_func_val(xN, dummy_u)).flatten()

            # Squeeze time dim if it's only 1 step long, just to keep arrays simpler
            if F_arr is not None and F_arr.shape[0] == 1:
                F_arr = F_arr[0]
            if G_arr is not None and G_arr.shape[0] == 1:
                G_arr = G_arr[0]
            if h_arr.shape[0] == 1:
                h_arr = h_arr[0]

            lin_c = LinearConstraint(h=h_arr, F=F_arr, G=G_arr, is_equality=c.is_equality)
            lin_constraints.add(lin_c, resolved_indices)

        # Terminal cost
        x_N_sym = ca.MX.sym("x_N", nx)
        if self.terminal_objective is not None:
            if self.terminal_objective.n_in() == 2:
                assert X_ref is not None
                x_ref_N_sym = ca.MX.sym("x_ref_N", nx)
                L_term_val = self.terminal_objective(x_N_sym, x_ref_N_sym)
                qf_func = ca.Function("qf", [x_N_sym, x_ref_N_sym], [ca.jacobian(L_term_val, x_N_sym).T])
                Qf_func = ca.Function("Qf", [x_N_sym, x_ref_N_sym], [ca.hessian(L_term_val, x_N_sym)[0]])

                xN = X_bar[N]
                x_ref_N = X_ref[N]
                Qf = np.array(Qf_func(xN, x_ref_N))
                qf = np.array(qf_func(xN, x_ref_N)).flatten()
            else:
                L_term_val = self.terminal_objective(x_N_sym)
                qf_func = ca.Function("qf", [x_N_sym], [ca.jacobian(L_term_val, x_N_sym).T])
                Qf_func = ca.Function("Qf", [x_N_sym], [ca.hessian(L_term_val, x_N_sym)[0]])

                xN = X_bar[N]
                Qf = np.array(Qf_func(xN))
                qf = np.array(qf_func(xN)).flatten()
        else:
            Qf = np.zeros((nx, nx))
            qf = np.zeros(nx)

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
            constraints=lin_constraints,
        )

    def setup(  # noqa: D102, PLR0915, PLR0912, PLR0913, C901 TODO: fix issues
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

        self._x_ref_param = None
        self._u_ref_param = None

        if self.objective.n_in() == 4:
            self._x_ref_param = self._opti.parameter(nx, self.N + 1)
            self._u_ref_param = self._opti.parameter(nu, self.N)
        elif (
            hasattr(self, "terminal_objective")
            and self.terminal_objective is not None
            and self.terminal_objective.n_in() == 2
        ):
            self._x_ref_param = self._opti.parameter(nx, self.N + 1)

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
            msg = f"Unknown dynamics_type: {dynamics_type}"
            raise ValueError(msg)

        if method == "single_shooting":
            self._U = self._opti.variable(nu, self.N)
            self._X = self._opti.variable(nx, self.N + 1)  # Still define it for later extraction

            x_k = self._x0_param
            self._opti.subject_to(self._X[:, 0] == x_k)

            cost = 0.0
            for k in range(self.N):
                u_k = self._U[:, k]

                # Cost
                if self.objective.n_in() == 4:
                    assert self._x_ref_param is not None
                    assert self._u_ref_param is not None
                    cost += self.objective(x_k, u_k, self._x_ref_param[:, k], self._u_ref_param[:, k])
                else:
                    cost += self.objective(x_k, u_k)

                # Constraints
                for constraint, time_indices in self.constraints:
                    resolved_indices = self.constraints.resolve_indices(time_indices, self.N)
                    if k in resolved_indices:
                        if isinstance(constraint, StateConstraint):
                            val = constraint.f(x_k)
                        elif isinstance(constraint, ControlConstraint):
                            val = constraint.f(u_k)
                        elif isinstance(constraint, Constraint):
                            val = constraint.f(x_k, u_k)

                        if constraint.is_equality:
                            self._opti.subject_to(val == 0)
                        else:
                            self._opti.subject_to(val <= 0)

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
                if self.objective.n_in() == 4:
                    assert self._x_ref_param is not None
                    assert self._u_ref_param is not None
                    cost += self.objective(x_k, u_k, self._x_ref_param[:, k], self._u_ref_param[:, k])
                else:
                    cost += self.objective(x_k, u_k)

                # Constraints
                for constraint, time_indices in self.constraints:
                    resolved_indices = self.constraints.resolve_indices(time_indices, self.N)
                    if k in resolved_indices:
                        if isinstance(constraint, StateConstraint):
                            val = constraint.f(x_k)
                        elif isinstance(constraint, ControlConstraint):
                            val = constraint.f(u_k)
                        elif isinstance(constraint, Constraint):
                            val = constraint.f(x_k, u_k)

                        if constraint.is_equality:
                            self._opti.subject_to(val == 0)
                        else:
                            self._opti.subject_to(val <= 0)

                # Dynamics gap closing
                x_next = dyn_func(x_k, u_k)
                self._opti.subject_to(self._X[:, k + 1] == x_next)

        elif method == "collocation":
            if dynamics_type == "discrete":
                msg = "Collocation method is not applicable to discrete dynamics."
                raise ValueError(msg)

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
                if self.objective.n_in() == 4:
                    assert self._x_ref_param is not None
                    assert self._u_ref_param is not None
                    cost += self.objective(x_k, u_k, self._x_ref_param[:, k], self._u_ref_param[:, k])
                else:
                    cost += self.objective(x_k, u_k)

                # Constraints
                for constraint, time_indices in self.constraints:
                    resolved_indices = self.constraints.resolve_indices(time_indices, self.N)
                    if k in resolved_indices:
                        if isinstance(constraint, StateConstraint):
                            val = constraint.f(x_k)
                        elif isinstance(constraint, ControlConstraint):
                            val = constraint.f(u_k)
                        elif isinstance(constraint, Constraint):
                            val = constraint.f(x_k, u_k)

                        if constraint.is_equality:
                            self._opti.subject_to(val == 0)
                        else:
                            self._opti.subject_to(val <= 0)

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
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        # Terminal conditions
        x_N = self._X[:, self.N]
        if self.terminal_objective is not None:
            if self.terminal_objective.n_in() == 2:
                assert self._x_ref_param is not None
                cost += self.terminal_objective(x_N, self._x_ref_param[:, self.N])
            else:
                cost += self.terminal_objective(x_N)

        for constraint, time_indices in self.constraints:
            resolved_indices = self.constraints.resolve_indices(time_indices, self.N)
            if self.N in resolved_indices:
                if isinstance(constraint, StateConstraint):
                    val = constraint.f(x_N)
                elif isinstance(constraint, Constraint):
                    # For terminal constraints that incorrectly use f(x, u), we can't properly evaluate u_N
                    # Since this is poor practice, we pass a dummy, but ideally users use StateConstraint
                    dummy_u = ca.MX.zeros(nu)
                    val = constraint.f(x_N, dummy_u)
                else:
                    continue  # ControlConstraint doesn't make sense at terminal step

                if constraint.is_equality:
                    self._opti.subject_to(val == 0)
                else:
                    self._opti.subject_to(val <= 0)

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

    def solve(  # noqa: PLR0912, C901, TODO: refactor to fix issues
        self,
        x0: ArrayLike,
        X_guess: ArrayLike | None = None,
        U_guess: ArrayLike | None = None,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str]:
        """
        Solves the OCP for a given initial state.

        Args:
            x0: Initial state as a numpy array or list.
            X_guess: Optional initial guess for state trajectory of shape (N + 1, nx).
            U_guess: Optional initial guess for control trajectory of shape (N, nu).
            x_ref: Optional time-varying state reference of shape (N + 1, nx).
            u_ref: Optional time-varying control reference of shape (N, nu).

        Returns
        -------
            Tuple of (X_opt, U_opt, status)
            - X_opt: numpy array of optimal state trajectory of shape (N + 1, nx)
            - U_opt: numpy array of optimal control trajectory of shape (N, nu)
            - status: string indicating solver status
        """
        if self._opti is None:
            msg = "OCP has not been set up. Call setup() first."
            raise RuntimeError(msg)

        self._opti.set_value(self._x0_param, x0)

        if self._x_ref_param is not None:
            if x_ref is None:
                x_ref = np.zeros((self.N + 1, self._nx))
            x_ref_arr = np.asarray(x_ref, dtype=float)
            if x_ref_arr.shape != (self.N + 1, self._nx):
                msg = f"x_ref must have shape ({self.N + 1}, {self._nx})"
                raise ValueError(msg)
            self._opti.set_value(self._x_ref_param, x_ref_arr.T)

        if self._u_ref_param is not None:
            if u_ref is None:
                u_ref = np.zeros((self.N, self._nu))
            u_ref_arr = np.asarray(u_ref, dtype=float)
            if u_ref_arr.shape != (self.N, self._nu):
                msg = f"u_ref must have shape ({self.N}, {self._nu})"
                raise ValueError(msg)
            self._opti.set_value(self._u_ref_param, u_ref_arr.T)

        if X_guess is not None:
            X_guess = np.asarray(X_guess)
            if X_guess.shape != (self.N + 1, self._nx):
                msg = f"X_guess must have shape ({self.N + 1}, {self._nx})"
                raise ValueError(msg)
            self._opti.set_initial(self._X, X_guess.T)

        if U_guess is not None:
            U_guess = np.asarray(U_guess)
            if U_guess.shape != (self.N, self._nu):
                msg = f"U_guess must have shape ({self.N}, {self._nu})"
                raise ValueError(msg)
            self._opti.set_initial(self._U, U_guess.T)

        try:
            sol = self._opti.solve()
            X_opt = sol.value(self._X)
            U_opt = sol.value(self._U)
            status: str = sol.stats()["return_status"]
        except Exception as e:  # noqa: BLE001
            # If solve fails, return the values at the last iteration
            X_opt = self._opti.debug.value(self._X)
            U_opt = self._opti.debug.value(self._U)
            status = f"Solve_Failed: {e!s}"

        # Ensure 2D arrays even if nx=1 or nu=1
        if isinstance(X_opt, np.ndarray) and X_opt.ndim == 1:
            X_opt = X_opt.reshape(1, -1)
        if isinstance(U_opt, np.ndarray) and U_opt.ndim == 1:
            U_opt = U_opt.reshape(1, -1)

        return X_opt.T, U_opt.T, status

    def calculate_trajectory_cost(  # noqa: C901, PLR0912, PLR0915
        self,
        X: ArrayLike,
        U: ArrayLike,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> float:
        """Calculate the total cost of a given state and control trajectory."""
        X_arr = np.asarray(X, dtype=float)
        U_arr = np.asarray(U, dtype=float)

        if X_arr.shape != (self.N + 1, self._nx):
            msg = f"X must have shape ({self.N + 1}, {self._nx})"
            raise ValueError(msg)
        if U_arr.shape != (self.N, self._nu):
            msg = f"U must have shape ({self.N}, {self._nu})"
            raise ValueError(msg)

        X_ref_arr = None
        U_ref_arr = None

        if self.objective.n_in() == 4:
            if x_ref is None:
                X_ref_arr = np.zeros((self.N + 1, self._nx))
            else:
                X_ref_arr = np.asarray(x_ref, dtype=float)
                if X_ref_arr.shape != (self.N + 1, self._nx):
                    msg = f"x_ref must have shape ({self.N + 1}, {self._nx})"
                    raise ValueError(msg)

            if u_ref is None:
                U_ref_arr = np.zeros((self.N, self._nu))
            else:
                U_ref_arr = np.asarray(u_ref, dtype=float)
                if U_ref_arr.shape != (self.N, self._nu):
                    msg = f"u_ref must have shape ({self.N}, {self._nu})"
                    raise ValueError(msg)
        elif (
            hasattr(self, "terminal_objective")
            and self.terminal_objective is not None
            and self.terminal_objective.n_in() == 2
        ):
            if x_ref is None:
                X_ref_arr = np.zeros((self.N + 1, self._nx))
            else:
                X_ref_arr = np.asarray(x_ref, dtype=float)
                if X_ref_arr.shape != (self.N + 1, self._nx):
                    msg = f"x_ref must have shape ({self.N + 1}, {self._nx})"
                    raise ValueError(msg)

        total_cost = 0.0

        for k in range(self.N):
            x_k = X_arr[k, :]
            u_k = U_arr[k, :]

            if self.objective.n_in() == 4:
                assert X_ref_arr is not None
                assert U_ref_arr is not None
                cost_k = self.objective(x_k, u_k, X_ref_arr[k, :], U_ref_arr[k, :])
            else:
                cost_k = self.objective(x_k, u_k)
            total_cost += float(cost_k)

        x_N = X_arr[self.N, :]
        if self.terminal_objective is not None:
            if self.terminal_objective.n_in() == 2:
                assert X_ref_arr is not None
                cost_N = self.terminal_objective(x_N, X_ref_arr[self.N, :])
            else:
                cost_N = self.terminal_objective(x_N)
            total_cost += float(cost_N)

        return total_cost


def quadratic_objective(  # noqa: D103, TODO: fix D103
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
    if N.shape[0] != nx or N.shape[1] != nu:
        msg = "Matrix N must match state and control dimensions."
        raise ValueError(msg)

    return ca.Function(
        "quadr_obj", [x, u], [x.T @ Q @ x + x.T @ q + u.T @ R @ u + u.T @ r + x.T @ N @ u], ["x", "u"], ["f"]
    )


def lqr_objective(  # noqa: D103, TODO: fix D103
    Q: np.ndarray, R: np.ndarray, N: np.ndarray | None = None
) -> Function:
    nx = Q.shape[0]
    nu = R.shape[0]
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    x_ref = ca.MX.sym("x_ref", nx)
    u_ref = ca.MX.sym("u_ref", nu)

    if N is None:
        N = np.zeros((nx, nu))

    if Q.shape[0] != Q.shape[1] or Q.shape[0] != nx:
        msg = "Matrix Q must be square and match state dimension."
        raise ValueError(msg)
    if R.shape[0] != R.shape[1] or R.shape[0] != nu:
        msg = "Matrix R must be square and match control dimension."
        raise ValueError(msg)
    if N.shape[0] != nx or N.shape[1] != nu:
        msg = "Matrix N must match state and control dimensions."
        raise ValueError(msg)

    dx = x - x_ref
    du = u - u_ref

    return ca.Function(
        "lqr_obj",
        [x, u, x_ref, u_ref],
        [dx.T @ Q @ dx + du.T @ R @ du + dx.T @ N @ du],
        ["x", "u", "x_ref", "u_ref"],
        ["f"],
    )


def linear_constraints(F: np.ndarray, G: np.ndarray, h: np.ndarray) -> Function:  # noqa: D103, TODO: fix D103
    nx = F.shape[1]
    nu = G.shape[1]

    if F.shape[0] != G.shape[0] or F.shape[0] != h.shape[0]:
        msg = "The number of rows in F, G, and h must be equal."
        raise ValueError(msg)

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    return ca.Function("lin_con", [x, u], [F @ x + G @ u - h], ["x", "u"], ["f"])


def linear_dynamics(A: np.ndarray, B: np.ndarray) -> Function:  # noqa: D103, TODO: fix D103
    nx = A.shape[1]
    nu = B.shape[1]

    if A.shape[0] != nx:
        msg = "Matrix A must be square."
        raise ValueError(msg)
    if B.shape[0] != nx:
        msg = "Matrix B must have the same number of rows as A."
        raise ValueError(msg)

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    return ca.Function("lin_dyn", [x, u], [A @ x + B @ u], ["x", "u"], ["f"])


def state_bounds_constraints(x_min: np.ndarray, x_max: np.ndarray, nu: int) -> Function:  # noqa: D103, TODO: fix D103
    nx = x_min.shape[0]
    if x_max.shape[0] != nx:
        msg = "x_min and x_max must have the same length."
        raise ValueError(msg)

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    return ca.Function("state_bounds", [x, u], [ca.vertcat(x_min - x, x - x_max)], ["x", "u"], ["f"])


def control_bounds_constraints(u_min: np.ndarray, u_max: np.ndarray, nx: int) -> Function:  # noqa: D103, TODO: fix D103
    nu = u_min.shape[0]
    if u_max.shape[0] != nu:
        msg = "u_min and u_max must have the same length."
        raise ValueError(msg)

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    return ca.Function("control_bounds", [x, u], [ca.vertcat(u_min - u, u - u_max)], ["x", "u"], ["f"])


def terminal_quadratic_objective(Q: np.ndarray, q: np.ndarray) -> Function:  # noqa: D103, TODO: fix D103
    nx = Q.shape[0]

    if Q.shape[1] != nx:
        msg = "Matrix Q must be square."
        raise ValueError(msg)
    if q.shape[0] != nx:
        msg = "Vector q must have the same length as Q."
        raise ValueError(msg)

    x = ca.MX.sym("x", nx)

    return ca.Function("term_quadr_obj", [x], [x.T @ Q @ x + x.T @ q], ["x"], ["f"])


def terminal_lqr_objective(Q: np.ndarray) -> Function:  # noqa: D103   TODO: fix
    nx = Q.shape[0]

    if Q.shape[1] != nx:
        msg = "Matrix Q must be square."
        raise ValueError(msg)

    x = ca.MX.sym("x", nx)
    x_ref = ca.MX.sym("x_ref", nx)

    dx = x - x_ref

    return ca.Function("term_lqr_obj", [x, x_ref], [dx.T @ Q @ dx], ["x", "x_ref"], ["f"])


def terminal_linear_constraints(F: np.ndarray, h: np.ndarray) -> Function:  # noqa: D103, TODO: fix D103
    nx = F.shape[1]

    if F.shape[0] != h.shape[0]:
        msg = "The number of rows in F and h must be equal."
        raise ValueError(msg)

    x = ca.MX.sym("x", nx)

    return ca.Function("term_lin_con", [x], [F @ x - h], ["x"], ["f"])


class LinearOCP:  # noqa: D101
    def __init__(  # noqa: PLR0913  # TODO: fix D101
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
        constraints: ConstraintList | None = None,
    ) -> None:
        """
        Initialize Linear Optimal Control Problem.

        Cost stage: 0.5 * (x^T Q x + u^T R u) + x^T N_cross u + q^T x + r^T u
        Terminal cost: 0.5 * x_N^T Qf x_N + qf^T x_N
        Dynamics: x_{k+1} = A x_k + B u_k
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

        self.constraints = constraints if constraints is not None else ConstraintList()

        self.validate_dimensions()

        self._method: str = ""
        self._solver_obj: ca.Function | None = None
        self._solver_args: dict[str, Any] = {}

        # We will save the parametric pieces of the QP to form arguments quickly in solve()
        self._qp_setup: dict[str, Any] = {}

    def _is_time_varying(self, arr: np.ndarray, expected_dims: int) -> bool:
        return bool(arr.ndim == expected_dims + 1 and arr.shape[0] == self.N)

    def validate_dimensions(self) -> None:
        """Validate all casadi functions and returns nx and nu."""
        nx = self.nx
        nu = self.nu
        N = self.N

        def check_shape(arr: np.ndarray, base_shape: tuple[int, ...], name: str) -> None:
            if arr.shape not in (base_shape, (N, *base_shape)):
                msg = f"Array {name} must be {base_shape} or {(N, *base_shape)}"
                raise ValueError(msg)

        check_shape(self.A, (nx, nx), "A")
        check_shape(self.B, (nx, nu), "B")
        check_shape(self.Q, (nx, nx), "Q")
        check_shape(self.R, (nu, nu), "R")
        check_shape(self.q, (nx,), "q")
        check_shape(self.r, (nu,), "r")
        check_shape(self.N_cross, (nx, nu), "N_cross")

        if self.Qf.shape != (nx, nx):
            msg = f"Matrix Qf must be ({nx}, {nx})"
            raise ValueError(msg)
        if self.qf.shape != (nx,):
            msg = f"Vector qf must be ({nx},)"
            raise ValueError(msg)

        for constraint, _ in self.constraints:
            constraint.validate_dimensions(nx, nu)

    def setup(  # noqa: PLR0915, PLR0912, C901   # TODO: refactor
        self,
        method: str = "multiple_shooting",
        dynamics_type: str = "discrete",
        solver: str = "qrqp",
        plugin_opts: dict[str, Any] | None = None,
        solver_opts: dict[str, Any] | None = None,
    ) -> None:
        """
        Set up the QP solver for the given method and solver backend.

        method: "multiple_shooting" (sparse) or "single_shooting" (condensed)
        dynamics_type: "discrete" or "continuous" (will be exactly discretized using ZOH)
        solver: The backend solver for ca.qpsol (e.g. 'qrqp', 'osqp').
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
                msg = f"Unknown dynamics_type: {dynamics_type}"
                raise ValueError(msg)

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
            # Pre-calculate total number of constraints
            for constraint, time_indices in self.constraints:
                if not isinstance(constraint, LinearConstraint):
                    continue
                resolved_indices = self.constraints.resolve_indices(time_indices, N)
                # constraint.h could be (nc,) or (len(resolved), nc)
                # if it's (nc,), it applies nc constraints per time index
                # if it's (len(resolved), nc), it also applies nc constraints per time index
                nc = constraint.nc
                n_ineq += len(resolved_indices) * nc

            if n_ineq > 0:
                A_ineq = np.zeros((n_ineq, n_vars))
                uba_ineq = np.zeros(n_ineq)
                lba_ineq = np.zeros(n_ineq)

                curr_row = 0
                for constraint, time_indices in self.constraints:
                    if not isinstance(constraint, LinearConstraint):
                        continue

                    resolved_indices = self.constraints.resolve_indices(time_indices, N)
                    nc = constraint.nc

                    is_time_varying = constraint.h.ndim > 1

                    for i, k in enumerate(resolved_indices):
                        if k == N:
                            idx_x = N * (nx + nu)
                            if constraint.F is not None:
                                F_k = constraint.F[i] if is_time_varying and constraint.F.ndim > 2 else constraint.F
                                A_ineq[curr_row : curr_row + nc, idx_x : idx_x + nx] = F_k
                        else:
                            idx_x = k * (nx + nu)
                            idx_u = idx_x + nx
                            if constraint.F is not None:
                                F_k = constraint.F[i] if is_time_varying and constraint.F.ndim > 2 else constraint.F
                                A_ineq[curr_row : curr_row + nc, idx_x : idx_x + nx] = F_k
                            if constraint.G is not None:
                                G_k = constraint.G[i] if is_time_varying and constraint.G.ndim > 2 else constraint.G
                                A_ineq[curr_row : curr_row + nc, idx_x + nx : idx_x + nx + nu] = G_k

                        h_k = constraint.h[i] if is_time_varying else constraint.h
                        uba_ineq[curr_row : curr_row + nc] = h_k
                        if constraint.is_equality:
                            lba_ineq[curr_row : curr_row + nc] = h_k
                        else:
                            lba_ineq[curr_row : curr_row + nc] = -1e9

                        curr_row += nc

                A_c = np.vstack([A_eq, A_ineq])
                uba = np.concatenate([np.zeros(n_eq), uba_ineq])
                lba = np.concatenate([np.zeros(n_eq), lba_ineq])
            else:
                A_c = A_eq
                uba = np.zeros(n_eq)
                lba = np.zeros(n_eq)

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
            for constraint, time_indices in self.constraints:
                if not isinstance(constraint, LinearConstraint):
                    continue
                resolved_indices = self.constraints.resolve_indices(time_indices, N)
                nc = constraint.nc
                n_ineq += len(resolved_indices) * nc

            if n_ineq > 0:
                F_bar = np.zeros((n_ineq, (N + 1) * nx))
                G_bar = np.zeros((n_ineq, N * nu))
                h_bar = np.zeros(n_ineq)
                lba_ineq = np.zeros(n_ineq)

                curr_row = 0
                for constraint, time_indices in self.constraints:
                    if not isinstance(constraint, LinearConstraint):
                        continue

                    resolved_indices = self.constraints.resolve_indices(time_indices, N)
                    nc = constraint.nc
                    is_time_varying = constraint.h.ndim > 1

                    for i, k in enumerate(resolved_indices):
                        if constraint.F is not None:
                            F_k = constraint.F[i] if is_time_varying and constraint.F.ndim > 2 else constraint.F
                            F_bar[curr_row : curr_row + nc, k * nx : (k + 1) * nx] = F_k

                        if k < N and constraint.G is not None:
                            G_k = constraint.G[i] if is_time_varying and constraint.G.ndim > 2 else constraint.G
                            G_bar[curr_row : curr_row + nc, k * nu : (k + 1) * nu] = G_k

                        h_k = constraint.h[i] if is_time_varying else constraint.h
                        h_bar[curr_row : curr_row + nc] = h_k

                        if constraint.is_equality:
                            lba_ineq[curr_row : curr_row + nc] = h_k
                        else:
                            lba_ineq[curr_row : curr_row + nc] = -1e9

                        curr_row += nc

                A_ineq_u = F_bar @ S_u + G_bar
                A_c_ca = ca.DM(A_ineq_u)
            else:
                A_c_ca = ca.DM.zeros(0, n_vars)
                F_bar = np.zeros((0, (N + 1) * nx))
                h_bar = np.zeros(0)
                lba_ineq = np.zeros(0)

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
                "lba_ineq": lba_ineq if n_ineq > 0 else None,
                "n_ineq": n_ineq,
                "n_vars": n_vars,
            }

        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

    def solve(  # noqa: PLR0915, PLR0912, C901  # TODO: refactor to fix PLR0915, PLR0912
        self,
        x0: ArrayLike,
        X_guess: ArrayLike | None = None,
        U_guess: ArrayLike | None = None,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str]:
        """
        Solves the Linear OCP for a given initial state.

        Args:
            x0: Initial state as a numpy array or list.
            X_guess: Optional initial guess for state trajectory of shape (N + 1, nx).
            U_guess: Optional initial guess for control trajectory of shape (N, nu).
            x_ref: Optional time-varying state reference of shape (N + 1, nx).
            u_ref: Optional time-varying control reference of shape (N, nu).

        Returns
        -------
            Tuple of (X_opt, U_opt, status)
            - X_opt: numpy array of optimal state trajectory of shape (N + 1, nx)
            - U_opt: numpy array of optimal control trajectory of shape (N, nu)
            - status: string indicating solver status
        """
        if self._solver_obj is None:
            msg = "LinearOCP has not been set up. Call setup() first."
            raise RuntimeError(msg)

        x0_arr = np.asarray(x0, dtype=float).flatten()
        if x0_arr.shape != (self.nx,):
            msg = f"Initial state must have length {self.nx}"
            raise ValueError(msg)

        if X_guess is not None:
            X_guess = np.asarray(X_guess)
            if X_guess.shape != (self.N + 1, self.nx):
                msg = f"X_guess must have shape ({self.N + 1}, {self.nx})"
                raise ValueError(msg)
        if U_guess is not None:
            U_guess = np.asarray(U_guess)
            if U_guess.shape != (self.N, self.nu):
                msg = f"U_guess must have shape ({self.N}, {self.nu})"
                raise ValueError(msg)

        X_ref_arr = None
        if x_ref is not None:
            X_ref_arr = np.asarray(x_ref, dtype=float)
            if X_ref_arr.ndim == 1 and X_ref_arr.shape[0] == self.nx:
                X_ref_arr = np.tile(X_ref_arr, (self.N + 1, 1))
            if X_ref_arr.shape != (self.N + 1, self.nx):
                msg = f"x_ref must have shape ({self.N + 1}, {self.nx}) or ({self.nx},)"
                raise ValueError(msg)

        U_ref_arr = None
        if u_ref is not None:
            U_ref_arr = np.asarray(u_ref, dtype=float)
            if U_ref_arr.ndim == 1 and U_ref_arr.shape[0] == self.nu:
                U_ref_arr = np.tile(U_ref_arr, (self.N, 1))
            if U_ref_arr.shape != (self.N, self.nu):
                msg = f"u_ref must have shape ({self.N}, {self.nu}) or ({self.nu},)"
                raise ValueError(msg)

        if self._method == "multiple_shooting":
            lba = self._qp_setup["lba"].copy()
            uba = self._qp_setup["uba"].copy()
            g_vec = self._qp_setup["g"].copy()

            # Apply tracking reference shifts to the linear cost term g_vec
            if X_ref_arr is not None or U_ref_arr is not None:
                for k in range(self.N):
                    Qk = self.Q[k] if self.Q.ndim == 3 else self.Q
                    Rk = self.R[k] if self.R.ndim == 3 else self.R
                    N_cross_k = self.N_cross[k] if self.N_cross.ndim == 3 else self.N_cross

                    idx_x = k * (self.nx + self.nu)
                    idx_u = idx_x + self.nx

                    if X_ref_arr is not None:
                        g_vec[idx_x : idx_x + self.nx] -= Qk @ X_ref_arr[k]
                        if np.any(N_cross_k):
                            g_vec[idx_u : idx_u + self.nu] -= N_cross_k.T @ X_ref_arr[k]

                    if U_ref_arr is not None:
                        g_vec[idx_u : idx_u + self.nu] -= Rk @ U_ref_arr[k]
                        if np.any(N_cross_k):
                            g_vec[idx_x : idx_x + self.nx] -= N_cross_k @ U_ref_arr[k]

                if X_ref_arr is not None:
                    idx_xN = self.N * (self.nx + self.nu)
                    g_vec[idx_xN : idx_xN + self.nx] -= self.Qf @ X_ref_arr[self.N]

            # Update initial condition constraint: I * x_0 = x0
            lba[: self.nx] = x0_arr
            uba[: self.nx] = x0_arr

            kwargs = {"h": self._qp_setup["h"], "g": g_vec, "a": self._qp_setup["a"], "lba": lba, "uba": uba}

            if X_guess is not None or U_guess is not None:
                x0_guess = np.zeros(self._qp_setup["n_vars"])
                X_guess_used = X_guess if X_guess is not None else np.zeros((self.N + 1, self.nx))
                U_guess_used = U_guess if U_guess is not None else np.zeros((self.N, self.nu))

                for k in range(self.N):
                    idx_x = k * (self.nx + self.nu)
                    idx_u = idx_x + self.nx
                    x0_guess[idx_x : idx_x + self.nx] = X_guess_used[k]
                    x0_guess[idx_u : idx_u + self.nu] = U_guess_used[k]

                idx_xN = self.N * (self.nx + self.nu)
                x0_guess[idx_xN : idx_xN + self.nx] = X_guess_used[self.N]
                kwargs["x0"] = x0_guess

            res = self._solver_obj(**kwargs)

            z_opt = np.array(res["x"]).flatten()
            status = self._solver_obj.stats()["return_status"]

            # Unpack z = [x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N]
            X_opt = np.zeros((self.N + 1, self.nx))
            U_opt = np.zeros((self.N, self.nu))

            for k in range(self.N):
                idx_x = k * (self.nx + self.nu)
                idx_u = idx_x + self.nx
                X_opt[k, :] = z_opt[idx_x : idx_x + self.nx]
                U_opt[k, :] = z_opt[idx_u : idx_u + self.nu]

            idx_xN = self.N * (self.nx + self.nu)
            X_opt[self.N, :] = z_opt[idx_xN : idx_xN + self.nx]

        elif self._method == "single_shooting":
            # Calculate updated q_bar and r_bar based on x_ref and u_ref
            q_bar = np.zeros((self.N + 1) * self.nx)
            r_bar = np.zeros(self.N * self.nu)

            for k in range(self.N):
                Qk = self.Q[k] if self.Q.ndim == 3 else self.Q
                Rk = self.R[k] if self.R.ndim == 3 else self.R
                N_cross_k = self.N_cross[k] if self.N_cross.ndim == 3 else self.N_cross
                qk = self.q[k] if self.q.ndim == 2 else self.q
                rk = self.r[k] if self.r.ndim == 2 else self.r

                if X_ref_arr is not None:
                    qk = qk - Qk @ X_ref_arr[k]
                    if np.any(N_cross_k):
                        rk = rk - N_cross_k.T @ X_ref_arr[k]

                if U_ref_arr is not None:
                    rk = rk - Rk @ U_ref_arr[k]
                    if np.any(N_cross_k):
                        qk = qk - N_cross_k @ U_ref_arr[k]

                q_bar[k * self.nx : (k + 1) * self.nx] = qk
                r_bar[k * self.nu : (k + 1) * self.nu] = rk

            qf = self.qf.copy()
            if X_ref_arr is not None:
                qf = qf - self.Qf @ X_ref_arr[self.N]
            q_bar[self.N * self.nx : (self.N + 1) * self.nx] = qf

            # g_u(x0) = (x_0^T S_x^T Q_bar S_u + x_0^T S_x^T N_bar + q_bar^T S_u + r_bar^T)^T
            q_barT_Su_plus_r_barT = q_bar.T @ self._qp_setup["S_u"] + r_bar.T

            g_u_T = (
                x0_arr.T @ self._qp_setup["S_xT_Q_bar_Su"]
                + x0_arr.T @ self._qp_setup["S_xT_N_bar"]
                + q_barT_Su_plus_r_barT
            )
            g_u = g_u_T.T

            kwargs = {"h": self._qp_setup["h"], "g": g_u, "a": self._qp_setup["a"]}

            if self._qp_setup["n_ineq"] > 0:
                # lba <= A U <= uba
                # A_ineq_u U <= h_bar - F_bar S_x x_0
                # Be careful: for equality constraints we need lba to match uba dynamically
                uba = self._qp_setup["h_bar"] - self._qp_setup["F_bar_S_x"] @ x0_arr
                lba = self._qp_setup["lba_ineq"].copy()

                # Where lba != -1e9 (equality constraints), set lba = uba
                eq_mask = lba != -1e9
                lba[eq_mask] = uba[eq_mask]

                kwargs["lba"] = lba
                kwargs["uba"] = uba

            if U_guess is not None:
                kwargs["x0"] = U_guess.flatten()
            elif X_guess is not None:
                # Provide zeros for control guess if only X_guess is given,
                # since single shooting only takes U as primal variables.
                kwargs["x0"] = np.zeros(self.N * self.nu)

            res = self._solver_obj(**kwargs)

            U_vec = np.array(res["x"]).flatten()
            status = self._solver_obj.stats()["return_status"]

            # Reconstruct X_opt and U_opt
            U_opt = U_vec.reshape((self.N, self.nu))
            X_vec = self._qp_setup["S_x"] @ x0_arr + self._qp_setup["S_u"] @ U_vec
            X_opt = X_vec.reshape((self.N + 1, self.nx))

        else:
            msg = f"Unknown method: {self._method}"
            raise ValueError(msg)

        return X_opt, U_opt, status

    def calculate_trajectory_cost(
        self,
        X: ArrayLike,
        U: ArrayLike,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> float:
        """Calculate the total numerical cost of a given state and control trajectory."""
        X_arr = np.asarray(X, dtype=float)
        U_arr = np.asarray(U, dtype=float)

        if X_arr.shape != (self.N + 1, self.nx):
            msg = f"X must have shape ({self.N + 1}, {self.nx})"
            raise ValueError(msg)
        if U_arr.shape != (self.N, self.nu):
            msg = f"U must have shape ({self.N}, {self.nu})"
            raise ValueError(msg)

        X_ref_arr = None
        if x_ref is not None:
            X_ref_arr = np.asarray(x_ref, dtype=float)
            if X_ref_arr.ndim == 1 and X_ref_arr.shape[0] == self.nx:
                X_ref_arr = np.tile(X_ref_arr, (self.N + 1, 1))
            if X_ref_arr.shape != (self.N + 1, self.nx):
                msg = f"x_ref must have shape ({self.N + 1}, {self.nx}) or ({self.nx},)"
                raise ValueError(msg)

        U_ref_arr = None
        if u_ref is not None:
            U_ref_arr = np.asarray(u_ref, dtype=float)
            if U_ref_arr.ndim == 1 and U_ref_arr.shape[0] == self.nu:
                U_ref_arr = np.tile(U_ref_arr, (self.N, 1))
            if U_ref_arr.shape != (self.N, self.nu):
                msg = f"u_ref must have shape ({self.N}, {self.nu}) or ({self.nu},)"
                raise ValueError(msg)

        total_cost = 0.0

        for k in range(self.N):
            Qk = self.Q[k] if self.Q.ndim == 3 else self.Q
            Rk = self.R[k] if self.R.ndim == 3 else self.R
            N_cross_k = self.N_cross[k] if self.N_cross.ndim == 3 else self.N_cross
            qk = self.q[k] if self.q.ndim == 2 else self.q
            rk = self.r[k] if self.r.ndim == 2 else self.r

            x_k = X_arr[k]
            u_k = U_arr[k]

            dx = x_k if X_ref_arr is None else x_k - X_ref_arr[k]
            du = u_k if U_ref_arr is None else u_k - U_ref_arr[k]

            cost_k = 0.5 * (dx.T @ Qk @ dx + du.T @ Rk @ du) + dx.T @ N_cross_k @ du + qk.T @ dx + rk.T @ du
            total_cost += float(cost_k)

        x_N = X_arr[self.N]
        dx_N = x_N if X_ref_arr is None else x_N - X_ref_arr[self.N]
        cost_N = 0.5 * (dx_N.T @ self.Qf @ dx_N) + self.qf.T @ dx_N
        total_cost += float(cost_N)

        return total_cost
