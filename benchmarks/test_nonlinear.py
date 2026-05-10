"""Benchmark for non-linear OCP."""

import casadi as ca
import numpy as np

from model_predictive_control.constraints import ConstraintList, ControlBoundConstraint, StateBoundConstraint
from model_predictive_control.dynamics import Dynamics
from model_predictive_control.mpc import MPC
from model_predictive_control.objective import QuadraticObjective
from model_predictive_control.ocp import OCP


def setup_inverted_pendulum_ocp() -> OCP:
    """Set up the inverted pendulum OCP benchmark."""
    # Physical parameters
    m_cart = 1.0
    m_pend = 0.1
    length = 0.5
    g = 9.81

    nx = 4
    nu = 1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    _, v, theta, omega = x[0], x[1], x[2], x[3]

    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)
    denominator = m_cart + m_pend - m_pend * cos_theta**2

    p_ddot = (u[0] + m_pend * length * omega**2 * sin_theta - m_pend * g * sin_theta * cos_theta) / denominator
    theta_ddot = (
        -u[0] * cos_theta - m_pend * length * omega**2 * sin_theta * cos_theta + (m_cart + m_pend) * g * sin_theta
    ) / (length * denominator)

    x_dot = ca.vertcat(v, p_ddot, omega, theta_ddot)
    dynamics = Dynamics(ca.Function("dynamics", [x, u], [x_dot]))

    Q = np.diag([10.0, 1.0, 10.0, 1.0])
    R = np.array([[0.1]])
    q_term = np.zeros(nx)
    r_term = np.zeros(nu)
    N_cross = np.zeros((nx, nu))
    Qf = np.diag([100.0, 10.0, 100.0, 10.0])

    N = 100
    dt = 0.05

    objective = QuadraticObjective(Q, R, Qf, q_term, N, q_term, r_term, N_cross)

    u_max_val = 20.0
    u_min = np.array([-u_max_val])
    u_max = np.array([u_max_val])

    p_max_val = 2.0
    inf = 1e9
    x_min = np.array([-p_max_val, -inf, -inf, -inf])
    x_max = np.array([p_max_val, inf, inf, inf])

    state_bounds = StateBoundConstraint(x_min, x_max)
    control_bounds = ControlBoundConstraint(u_min, u_max)

    cl = ConstraintList()
    cl.add(state_bounds, slice(None))
    cl.add(control_bounds, slice(0, N))

    ocp = OCP(N=N, dt=dt, objective=objective, dynamics=dynamics, constraints=cl)
    ocp.setup(
        method="collocation",
        dynamics_type="continuous",
        solver="ipopt",
        solver_opts={"print_level": 0, "max_iter": 1000},
    )

    return ocp


def test_nonlinear_ocp_solve(benchmark) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
    """Benchmark solving the non-linear OCP."""
    ocp = setup_inverted_pendulum_ocp()
    x0_val = np.array([0.0, 0.0, 0.5, 0.0])

    result = benchmark(ocp.solve, x0_val)

    # Result is a tuple (X_opt, U_opt, status)
    _, _, status = result
    assert (
        "Solve_Succeeded" in status
        or "Optimal" in status
        or "success" in status.lower()
        or "succeeded" in status.lower()
    )

    # Get the iterations from Opti's stats if possible
    # We access the internal solver object to get stats for the last run
    try:
        if getattr(ocp, "_opti", None) is not None:
            stats = ocp._opti.stats()  # type: ignore[union-attr] # noqa: SLF001
            iterations = stats.get("iter_count", -1)
        else:
            iterations = -1
    except Exception:  # noqa: BLE001
        iterations = -1

    benchmark.extra_info["ipopt_iterations"] = iterations


def test_nonlinear_mpc_step(benchmark) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
    """Benchmark stepping the non-linear MPC loop."""
    ocp = setup_inverted_pendulum_ocp()
    setup_args = {
        "method": "collocation",
        "dynamics_type": "continuous",
        "solver": "ipopt",
        "solver_opts": {"print_level": 0, "max_iter": 1000},
    }
    mpc = MPC(ocp=ocp, setup_args=setup_args)

    # Initial offset
    x_current = np.array([0.0, 0.0, 0.5, 0.0])
    num_steps = 10

    # We need to simulate the system, using a simple euler step for demonstration
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
        for _ in range(num_steps):
            u, _status = mpc.step(x)

            try:
                opti = getattr(mpc.ocp, "_opti", None)
                if opti is not None:
                    stats = opti.stats()
                    iterations = stats.get("iter_count", -1)
                else:
                    iterations = -1
            except Exception:  # noqa: BLE001
                iterations = -1
            iterations_list.append(iterations)

            x = simulate_step(x, u)

        return iterations_list

    iterations_list = benchmark(run_mpc_loop)

    valid_iters = [it for it in iterations_list if it >= 0]
    if valid_iters:
        benchmark.extra_info["avg_solver_iterations"] = sum(valid_iters) / len(valid_iters)
        benchmark.extra_info["total_solver_iterations"] = sum(valid_iters)
    else:
        benchmark.extra_info["avg_solver_iterations"] = -1
        benchmark.extra_info["total_solver_iterations"] = -1
