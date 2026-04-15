"""Benchmark for non-linear OCP."""

from typing import Any

import casadi as ca
import numpy as np

from model_predictive_control.constraints import ConstraintList, ControlBoundConstraint, StateBoundConstraint
from model_predictive_control.dynamics import Dynamics
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

    def solve_ocp() -> tuple[Any, Any, str]:
        return ocp.solve(x0_val)

    # We use benchmark to run solve_ocp repeatedly and measure time
    result = benchmark(solve_ocp)

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
