from typing import Any

import casadi as ca
import numpy as np
from numpy._typing import ArrayLike
import scipy.linalg

class LinearOCP:
    def __init__(
        self,
        N: int,
        dt: float,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        q_term: np.ndarray | None = None,
        r_term: np.ndarray | None = None,
        N_cross: np.ndarray | None = None,
        Qf: np.ndarray | None = None,
        qf_term: np.ndarray | None = None,
        F: np.ndarray | None = None,
        G: np.ndarray | None = None,
        h: np.ndarray | None = None,
        F_term: np.ndarray | None = None,
        h_term: np.ndarray | None = None,
    ) -> None:
        """
        Initializes a Linear Optimal Control Problem.

        Cost stage: 0.5 * (x^T Q x + u^T R u) + x^T N_cross u + q_term^T x + r_term^T u
        Terminal cost: 0.5 * x_N^T Qf x_N + qf_term^T x_N
        Dynamics: x_{k+1} = A x_k + B u_k
        Constraints: F x_k + G u_k <= h
        Terminal constraints: F_term x_N <= h_term
        """
        self.N = N
        self.dt = dt

        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.q_term = np.zeros(self.nx) if q_term is None else np.asarray(q_term, dtype=float)
        self.r_term = np.zeros(self.nu) if r_term is None else np.asarray(r_term, dtype=float)
        self.N_cross = np.zeros((self.nx, self.nu)) if N_cross is None else np.asarray(N_cross, dtype=float)

        self.Qf = self.Q.copy() if Qf is None else np.asarray(Qf, dtype=float)
        self.qf_term = self.q_term.copy() if qf_term is None else np.asarray(qf_term, dtype=float)

        self.F = None if F is None else np.asarray(F, dtype=float)
        self.G = None if G is None else np.asarray(G, dtype=float)
        self.h = None if h is None else np.asarray(h, dtype=float).flatten()

        self.F_term = None if F_term is None else np.asarray(F_term, dtype=float)
        self.h_term = None if h_term is None else np.asarray(h_term, dtype=float).flatten()

        self._validate_dimensions()

        self._method: str = ""
        self._solver_obj: ca.Function | None = None
        self._solver_args: dict[str, Any] = {}

        # We will save the parametric pieces of the QP to form arguments quickly in solve()
        self._qp_setup: dict[str, Any] = {}

    def _validate_dimensions(self) -> None:
        nx = self.nx
        nu = self.nu

        if self.A.shape != (nx, nx):
            raise ValueError(f"Matrix A must be ({nx}, {nx})")
        if self.B.shape != (nx, nu):
            raise ValueError(f"Matrix B must be ({nx}, {nu})")

        if self.Q.shape != (nx, nx):
            raise ValueError(f"Matrix Q must be ({nx}, {nx})")
        if self.R.shape != (nu, nu):
            raise ValueError(f"Matrix R must be ({nu}, {nu})")

        if self.q_term.shape != (nx,):
            raise ValueError(f"Vector q_term must be ({nx},)")
        if self.r_term.shape != (nu,):
            raise ValueError(f"Vector r_term must be ({nu},)")
        if self.N_cross.shape != (nx, nu):
            raise ValueError(f"Matrix N_cross must be ({nx}, {nu})")

        if self.Qf.shape != (nx, nx):
            raise ValueError(f"Matrix Qf must be ({nx}, {nx})")
        if self.qf_term.shape != (nx,):
            raise ValueError(f"Vector qf_term must be ({nx},)")

        if self.F is not None or self.G is not None or self.h is not None:
            if self.F is None or self.G is None or self.h is None:
                raise ValueError("If any of F, G, h are provided, all three must be provided.")
            nc = self.F.shape[0]
            if self.F.shape != (nc, nx):
                raise ValueError(f"Matrix F must be ({nc}, {nx})")
            if self.G.shape != (nc, nu):
                raise ValueError(f"Matrix G must be ({nc}, {nu})")
            if self.h.shape != (nc,):
                raise ValueError(f"Vector h must be ({nc},)")

        if self.F_term is not None or self.h_term is not None:
            if self.F_term is None or self.h_term is None:
                raise ValueError("If either F_term or h_term is provided, both must be provided.")
            nc_term = self.F_term.shape[0]
            if self.F_term.shape != (nc_term, nx):
                raise ValueError(f"Matrix F_term must be ({nc_term}, {nx})")
            if self.h_term.shape != (nc_term,):
                raise ValueError(f"Vector h_term must be ({nc_term},)")
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

        if dynamics_type == "continuous":
            M = np.zeros((nx + nu, nx + nu))
            M[:nx, :nx] = self.A
            M[:nx, nx:] = self.B
            M_d = scipy.linalg.expm(M * self.dt)
            A_d = M_d[:nx, :nx]
            B_d = M_d[:nx, nx:]
        elif dynamics_type == "discrete":
            A_d = self.A
            B_d = self.B
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
                idx_x = k * (nx + nu)
                idx_u = idx_x + nx
                H_sp[idx_x:idx_x+nx, idx_x:idx_x+nx] = self.Q
                H_sp[idx_u:idx_u+nu, idx_u:idx_u+nu] = self.R
                if np.any(self.N_cross):
                    H_sp[idx_x:idx_x+nx, idx_u:idx_u+nu] = self.N_cross
                    H_sp[idx_u:idx_u+nu, idx_x:idx_x+nx] = self.N_cross.T

                g_vec[idx_x:idx_x+nx] = self.q_term
                g_vec[idx_u:idx_u+nu] = self.r_term

            idx_xN = N * (nx + nu)
            H_sp[idx_xN:idx_xN+nx, idx_xN:idx_xN+nx] = self.Qf
            g_vec[idx_xN:idx_xN+nx] = self.qf_term

            n_eq = (N + 1) * nx
            A_eq = np.zeros((n_eq, n_vars))

            A_eq[:nx, :nx] = np.eye(nx)

            for k in range(N):
                row_idx = (k + 1) * nx
                idx_x = k * (nx + nu)
                idx_u = idx_x + nx
                idx_x_next = (k + 1) * (nx + nu)

                A_eq[row_idx:row_idx+nx, idx_x:idx_x+nx] = -A_d
                A_eq[row_idx:row_idx+nx, idx_u:idx_u+nu] = -B_d
                A_eq[row_idx:row_idx+nx, idx_x_next:idx_x_next+nx] = np.eye(nx)

            n_ineq = 0
            nc = 0
            nc_term = 0
            if self.F is not None:
                nc = self.F.shape[0]
                n_ineq += N * nc
            if self.F_term is not None:
                nc_term = self.F_term.shape[0]
                n_ineq += nc_term

            if n_ineq > 0:
                A_ineq = np.zeros((n_ineq, n_vars))
                uba = np.zeros(n_eq + n_ineq)

                curr_row = 0
                for k in range(N):
                    if self.F is not None and self.G is not None and self.h is not None:
                        idx_x = k * (nx + nu)
                        idx_u = idx_x + nx
                        A_ineq[curr_row:curr_row+nc, idx_x:idx_x+nx] = self.F
                        A_ineq[curr_row:curr_row+nc, idx_u:idx_u+nu] = self.G
                        uba[n_eq + curr_row:n_eq + curr_row + nc] = self.h
                        curr_row += nc

                if self.F_term is not None and self.h_term is not None:
                    A_ineq[curr_row:curr_row+nc_term, idx_xN:idx_xN+nx] = self.F_term
                    uba[n_eq + curr_row:n_eq + curr_row + nc_term] = self.h_term

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
                S_x[k*nx:(k+1)*nx, :] = A_d @ S_x[(k-1)*nx:k*nx, :]

                for i in range(k):
                    if i == k - 1:
                        S_u[k*nx:(k+1)*nx, i*nu:(i+1)*nu] = B_d
                    else:
                        S_u[k*nx:(k+1)*nx, i*nu:(i+1)*nu] = A_d @ S_u[(k-1)*nx:k*nx, i*nu:(i+1)*nu]

            Q_bar = np.zeros(((N + 1) * nx, (N + 1) * nx))
            R_bar = np.zeros((N * nu, N * nu))
            N_bar = np.zeros(((N + 1) * nx, N * nu))
            q_bar = np.zeros((N + 1) * nx)
            r_bar = np.zeros(N * nu)

            for k in range(N):
                Q_bar[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = self.Q
                R_bar[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = self.R
                N_bar[k*nx:(k+1)*nx, k*nu:(k+1)*nu] = self.N_cross
                q_bar[k*nx:(k+1)*nx] = self.q_term
                r_bar[k*nu:(k+1)*nu] = self.r_term

            Q_bar[N*nx:(N+1)*nx, N*nx:(N+1)*nx] = self.Qf
            q_bar[N*nx:(N+1)*nx] = self.qf_term

            H_u = S_u.T @ Q_bar @ S_u + R_bar + S_u.T @ N_bar + N_bar.T @ S_u
            H_sp_ca = ca.DM(H_u)

            n_ineq = 0
            nc = 0
            nc_term = 0
            if self.F is not None:
                nc = self.F.shape[0]
                n_ineq += N * nc
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
                        F_bar[curr_row:curr_row+nc, k*nx:(k+1)*nx] = self.F
                        G_bar[curr_row:curr_row+nc, k*nu:(k+1)*nu] = self.G
                        h_bar[curr_row:curr_row+nc] = self.h
                        curr_row += nc

                if self.F_term is not None and self.h_term is not None:
                    F_bar[curr_row:curr_row+nc_term, N*nx:(N+1)*nx] = self.F_term
                    h_bar[curr_row:curr_row+nc_term] = self.h_term

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
            lba[:self.nx] = x0_arr
            uba[:self.nx] = x0_arr

            res = self._solver_obj(
                h=self._qp_setup["h"],
                g=self._qp_setup["g"],
                a=self._qp_setup["a"],
                lba=lba,
                uba=uba
            )

            z_opt = np.array(res["x"]).flatten()
            status = self._solver_obj.stats()["return_status"]

            # Unpack z = [x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N]
            X_opt = np.zeros((self.nx, self.N + 1))
            U_opt = np.zeros((self.nu, self.N))

            for k in range(self.N):
                idx_x = k * (self.nx + self.nu)
                idx_u = idx_x + self.nx
                X_opt[:, k] = z_opt[idx_x:idx_x+self.nx]
                U_opt[:, k] = z_opt[idx_u:idx_u+self.nu]

            idx_xN = self.N * (self.nx + self.nu)
            X_opt[:, self.N] = z_opt[idx_xN:idx_xN+self.nx]

        elif self._method == "single_shooting":
            # g_u(x0) = (x_0^T S_x^T Q_bar S_u + x_0^T S_x^T N_bar + q_bar^T S_u + r_bar^T)^T
            g_u_T = (
                x0_arr.T @ self._qp_setup["S_xT_Q_bar_Su"] +
                x0_arr.T @ self._qp_setup["S_xT_N_bar"] +
                self._qp_setup["q_barT_Su_plus_r_barT"]
            )
            g_u = g_u_T.T

            if self._qp_setup["n_ineq"] > 0:
                # lba <= A U <= uba
                # A_ineq_u U <= h_bar - F_bar S_x x_0
                uba = self._qp_setup["h_bar"] - self._qp_setup["F_bar_S_x"] @ x0_arr
                lba = np.full(self._qp_setup["n_ineq"], -1e9)
                res = self._solver_obj(
                    h=self._qp_setup["h"],
                    g=g_u,
                    a=self._qp_setup["a"],
                    lba=lba,
                    uba=uba
                )
            else:
                res = self._solver_obj(
                    h=self._qp_setup["h"],
                    g=g_u,
                    a=self._qp_setup["a"]
                )

            U_vec = np.array(res["x"]).flatten()
            status = self._solver_obj.stats()["return_status"]

            # Reconstruct X_opt and U_opt
            U_opt = U_vec.reshape((self.N, self.nu)).T
            X_vec = self._qp_setup["S_x"] @ x0_arr + self._qp_setup["S_u"] @ U_vec
            X_opt = X_vec.reshape((self.N + 1, self.nx)).T

        else:
            raise ValueError(f"Unknown method: {self._method}")

        return X_opt, U_opt, status
