"""Benchmark for linear MPC."""

import numpy as np
import numpy.typing as npt

from model_predictive_control.constraints import ConstraintList, LinearConstraint
from model_predictive_control.dynamics import LinearDynamics
from model_predictive_control.mpc import LinearMPC
from model_predictive_control.ocp import LinearOCP


def setup_linearized_inverted_pendulum_mpc() -> tuple[LinearMPC, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Set up the linearized inverted pendulum MPC benchmark."""
    # Linearized continuous-time dynamics of inverted pendulum around upright position (theta = 0)
    # x = [p, v, theta, omega], u = [F]
    m_cart = 1.0
    m_pend = 0.1
    length = 0.5
    g = 9.81

    denominator = m_cart

    A_cont = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -m_pend * g / denominator, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (m_cart + m_pend) * g / (length * denominator), 0.0],
        ]
    )

    B_cont = np.array([[0.0], [1.0 / denominator], [0.0], [-1.0 / (length * denominator)]])

    # Simple discrete approximation for demo
    dt = 0.05
    A = np.eye(4) + A_cont * dt
    B = B_cont * dt

    N = 20
    nu = 1

    Q = np.diag([10.0, 1.0, 10.0, 1.0])
    R = np.array([[0.1]])

    Qf = np.diag([100.0, 10.0, 100.0, 10.0])

    # Constraints
    u_max_val = 20.0
    p_max_val = 2.0

    F = np.zeros((4, 4))
    F[0, 0] = 1.0
    F[1, 0] = -1.0

    G = np.zeros((4, 1))
    G[2, 0] = 1.0
    G[3, 0] = -1.0

    h = np.array([p_max_val, p_max_val, u_max_val, u_max_val])

    F_term = np.zeros((2, 4))
    F_term[0, 0] = 1.0
    F_term[1, 0] = -1.0
    h_term = np.array([p_max_val, p_max_val])

    cl = ConstraintList()
    cl.add(LinearConstraint(F=F, G=G, h=h), range(N))
    cl.add(LinearConstraint(F=F_term, h=h_term, nu=nu), [N])

    ocp = LinearOCP(N=N, dt=dt, dynamics=LinearDynamics(A, B), Q=Q, R=R, Qf=Qf, constraints=cl)

    setup_args = {"method": "multiple_shooting", "solver": "osqp", "solver_opts": {"verbose": False}}

    return LinearMPC(linear_ocp=ocp, setup_args=setup_args), A, B


def test_linear_mpc_step(benchmark) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
    """Benchmark stepping the linear MPC loop."""
    mpc, A, B = setup_linearized_inverted_pendulum_mpc()
    # Initial offset
    x_current = np.array([0.0, 0.0, 0.1, 0.0])
    num_steps = 50

    def run_mpc_loop() -> list[int]:
        # Using lists to collect solver iterations over steps
        iterations_list = []

        x = x_current.copy()
        for _ in range(num_steps):
            u, _status = mpc.step(x)

            # Record iterations if available (depends on OSQP solver wrapper in casadi)
            try:
                solver_obj = getattr(mpc.ocp, "_solver_obj", None)
                if solver_obj is not None:
                    stats = solver_obj.stats()
                    iterations = stats.get("iter_count", -1)
                else:
                    iterations = -1
            except Exception:  # noqa: BLE001
                iterations = -1
            iterations_list.append(iterations)

            x = A @ x + B @ u

        return iterations_list

    iterations_list = benchmark(run_mpc_loop)

    # Filter valid iterations and store average/total in extra_info
    valid_iters = [it for it in iterations_list if it >= 0]
    if valid_iters:
        benchmark.extra_info["avg_solver_iterations"] = sum(valid_iters) / len(valid_iters)
        benchmark.extra_info["total_solver_iterations"] = sum(valid_iters)
    else:
        benchmark.extra_info["avg_solver_iterations"] = -1
        benchmark.extra_info["total_solver_iterations"] = -1


def test_linear_ocp_solve(benchmark) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
    """Benchmark solving the linear OCP."""
    mpc, _, _ = setup_linearized_inverted_pendulum_mpc()
    ocp = mpc.ocp
    x0_val = np.array([0.0, 0.0, 0.5, 0.0])

    result = benchmark(ocp.solve, x0_val)
    _, _, status = result
    assert "solved" in status.lower() or "success" in status.lower()

    try:
        solver_obj = getattr(ocp, "_solver_obj", None)
        if solver_obj is not None:
            stats = solver_obj.stats()
            iterations = stats.get("iter_count", -1)
        else:
            iterations = -1
    except Exception:  # noqa: BLE001
        iterations = -1

    benchmark.extra_info["osqp_iterations"] = iterations
