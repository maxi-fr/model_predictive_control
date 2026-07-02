"""Benchmark for non-linear OCP without abstractions (pure CasADi)."""

import casadi as ca
import numpy as np


def setup_raw_inverted_pendulum_ocp():
    """Set up the inverted pendulum OCP benchmark using pure CasADi Opti."""
    m_cart = 1.0
    m_pend = 0.1
    length = 0.5
    g = 9.81

    nx = 4
    nu = 1
    N = 100
    dt = 0.05

    opti = ca.Opti()

    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    x0_param = opti.parameter(nx)

    # Dynamics function
    x_sym = ca.MX.sym("x", nx)
    u_sym = ca.MX.sym("u", nu)

    _p, v, theta, omega = x_sym[0], x_sym[1], x_sym[2], x_sym[3]
    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)
    denominator = m_cart + m_pend - m_pend * cos_theta**2

    p_ddot = (u_sym[0] + m_pend * length * omega**2 * sin_theta - m_pend * g * sin_theta * cos_theta) / denominator
    theta_ddot = (
        -u_sym[0] * cos_theta - m_pend * length * omega**2 * sin_theta * cos_theta + (m_cart + m_pend) * g * sin_theta
    ) / (length * denominator)

    x_dot = ca.vertcat(v, p_ddot, omega, theta_ddot)
    f_dyn = ca.Function("dynamics", [x_sym, u_sym], [x_dot])

    Q = np.diag([10.0, 1.0, 10.0, 1.0])
    R = np.array([[0.1]])
    Qf = np.diag([100.0, 10.0, 100.0, 10.0])

    u_max_val = 20.0
    p_max_val = 2.0

    opti.subject_to(X[:, 0] == x0_param)

    cost = 0.0
    for k in range(N):
        x_k = X[:, k]
        x_k_next = X[:, k + 1]
        u_k = U[:, k]

        # Stage cost
        cost += ca.mtimes([x_k.T, Q, x_k]) + ca.mtimes([u_k.T, R, u_k])

        # Stage constraints
        opti.subject_to(u_k >= -u_max_val)
        opti.subject_to(u_k <= u_max_val)
        opti.subject_to(x_k[0] >= -p_max_val)
        opti.subject_to(x_k[0] <= p_max_val)

        # Hermite-Simpson direct collocation
        f_k = f_dyn(x_k, u_k)
        f_k_next = f_dyn(x_k_next, u_k)
        x_c = 0.5 * (x_k + x_k_next) + (dt / 8.0) * (f_k - f_k_next)
        f_c = f_dyn(x_c, u_k)
        opti.subject_to(x_k_next == x_k + (dt / 6.0) * (f_k + 4 * f_c + f_k_next))

    # Terminal cost
    x_N = X[:, N]
    cost += ca.mtimes([x_N.T, Qf, x_N])

    # Terminal constraints
    opti.subject_to(x_N[0] >= -p_max_val)
    opti.subject_to(x_N[0] <= p_max_val)

    opti.minimize(cost)

    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)

    return opti, x0_param, X, U


def test_nonlinear_raw_ocp_solve(benchmark) -> None:
    """Benchmark solving the raw non-linear OCP with CasADi Opti."""
    opti, x0_param, _X, _U = setup_raw_inverted_pendulum_ocp()
    x0_val = np.array([0.0, 0.0, 0.5, 0.0])

    def solve():
        opti.set_value(x0_param, x0_val)
        return opti.solve()

    sol = benchmark(solve)

    status = sol.stats()["return_status"]
    assert "Solve_Succeeded" in status

    benchmark.extra_info["ipopt_iterations"] = sol.stats()["iter_count"]


def test_nonlinear_raw_mpc_step(benchmark) -> None:
    """Benchmark stepping the raw non-linear MPC loop with CasADi Opti."""
    opti, x0_param, X, U = setup_raw_inverted_pendulum_ocp()

    x_current = np.array([0.0, 0.0, 0.5, 0.0])
    num_steps = 10

    m_cart = 1.0
    m_pend = 0.1
    length = 0.5
    g = 9.81
    dt = 0.05

    def simulate_step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        p, v, theta, omega = x[0], x[1], x[2], x[3]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        denominator = m_cart + m_pend - m_pend * cos_theta**2

        p_ddot = (u[0] + m_pend * length * omega**2 * sin_theta - m_pend * g * sin_theta * cos_theta) / denominator
        theta_ddot = (
            -u[0] * cos_theta - m_pend * length * omega**2 * sin_theta * cos_theta + (m_cart + m_pend) * g * sin_theta
        ) / (length * denominator)

        x_next = np.zeros_like(x)
        x_next[0] = p + v * dt
        x_next[1] = v + p_ddot * dt
        x_next[2] = theta + omega * dt
        x_next[3] = omega + theta_ddot * dt
        return x_next

    def run_mpc_loop() -> list[int]:
        iterations_list = []
        x = x_current.copy()
        for _k in range(num_steps):
            opti.set_value(x0_param, x)
            sol = opti.solve()

            iterations = sol.stats()["iter_count"]
            iterations_list.append(iterations)

            u_opt = np.atleast_1d(sol.value(U[:, 0]))

            x = simulate_step(x, u_opt)

            # Warm-start
            opti.set_initial(X, sol.value(X))
            opti.set_initial(U, sol.value(U))

        return iterations_list

    iterations_list = benchmark(run_mpc_loop)

    valid_iters = [it for it in iterations_list if it >= 0]
    if valid_iters:
        benchmark.extra_info["avg_solver_iterations"] = sum(valid_iters) / len(valid_iters)
        benchmark.extra_info["total_solver_iterations"] = sum(valid_iters)
    else:
        benchmark.extra_info["avg_solver_iterations"] = -1
        benchmark.extra_info["total_solver_iterations"] = -1
