import numpy as np
import pytest

from model_predictive_control.constraints import ConstraintList, LinearConstraint
from model_predictive_control.dynamics import Dynamics, LinearDynamics
from model_predictive_control.ocp import LinearOCP


def test_linear_ocp_validation() -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2)
    R = np.eye(1)

    with pytest.raises(ValueError):
        LinearOCP(N=10, dt=0.1, dynamics=LinearDynamics(np.eye(3), B), Q=Q, R=R)

    ocp = LinearOCP(N=10, dt=0.1, dynamics=LinearDynamics(A, B), Q=Q, R=R)
    assert ocp.nx == 2
    assert ocp.nu == 1


def test_linear_ocp_solve_multiple_shooting() -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2) * 10
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, dynamics=LinearDynamics(A, B), Q=Q, R=R)
    ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X, U, status = ocp.solve(np.array([1.0, 0.0]))
    assert status["solved_successfully"]
    assert X.shape == (6, 2)
    assert U.shape == (5, 1)

    # Test warm start functionality
    X_warm, U_warm, status_warm = ocp.solve(np.array([1.0, 0.0]), X_guess=X, U_guess=U)
    assert status_warm["solved_successfully"]
    np.testing.assert_allclose(X, X_warm, atol=1e-5)
    np.testing.assert_allclose(U, U_warm, atol=1e-5)

    # Test bad warm start shapes
    with pytest.raises(ValueError):
        ocp.solve(np.array([1.0, 0.0]), X_guess=np.zeros((6, 3)))
    with pytest.raises(ValueError):
        ocp.solve(np.array([1.0, 0.0]), U_guess=np.zeros((4, 1)))


def test_linear_ocp_solve_single_shooting() -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2) * 10
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, dynamics=LinearDynamics(A, B), Q=Q, R=R)
    ocp.setup(
        method="single_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X, U, status = ocp.solve(np.array([1.0, 0.0]))
    assert status["solved_successfully"]
    assert X.shape == (6, 2)
    assert U.shape == (5, 1)

    # Test warm start functionality
    X_warm, U_warm, status_warm = ocp.solve(np.array([1.0, 0.0]), X_guess=X, U_guess=U)
    assert status_warm["solved_successfully"]
    np.testing.assert_allclose(X, X_warm, atol=1e-5)
    np.testing.assert_allclose(U, U_warm, atol=1e-5)


@pytest.mark.parametrize("method", ["multiple_shooting", "single_shooting"])
def test_linear_ocp_continuous(method: str) -> None:
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, dynamics=LinearDynamics(A, B), Q=Q, R=R)
    ocp.setup(
        method=method,
        dynamics_type="continuous",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    _X, _U, status = ocp.solve(np.array([1.0, 0.0]))
    assert status["solved_successfully"]


@pytest.mark.parametrize("solver", ["qrqp", "osqp"])
def test_linear_ocp_solvers(solver: str) -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2) * 10
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, dynamics=LinearDynamics(A, B), Q=Q, R=R)
    ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver=solver,
        solver_opts={"print_iter": False, "print_header": False} if solver == "qrqp" else {},
    )

    _X, _U, status = ocp.solve(np.array([1.0, 0.0]))
    assert status["solved_successfully"]  # osqp returns "solved" instead of "success"


def test_linear_ocp_constraints() -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2)
    R = np.eye(1)

    F = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    G = np.array([[0.0], [0.0], [1.0], [-1.0]])
    h = np.array([2.0, 2.0, 50.0, 50.0])

    cl = ConstraintList()
    cl.add(LinearConstraint(F=F, G=G, h=h), range(5))
    ocp = LinearOCP(N=5, dt=0.1, dynamics=LinearDynamics(A, B), Q=Q, R=R, constraints=cl)
    ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X, U, status = ocp.solve(np.array([1.5, 0.0]))
    assert status["solved_successfully"]

    ocp_ss = LinearOCP(N=5, dt=0.1, dynamics=LinearDynamics(A, B), Q=Q, R=R, constraints=cl)
    ocp_ss.setup(
        method="single_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X_ss, U_ss, status_ss = ocp_ss.solve(np.array([1.5, 0.0]))
    assert status_ss["solved_successfully"]
    np.testing.assert_allclose(X, X_ss, atol=1e-5)
    np.testing.assert_allclose(U, U_ss, atol=1e-5)


def test_linear_ocp_time_varying() -> None:
    A1 = np.array([[1.0, 0.1], [0.0, 1.0]])
    B1 = np.array([[0.0], [0.1]])

    A2 = np.array([[1.0, 0.2], [0.0, 1.0]])
    B2 = np.array([[0.0], [0.2]])

    A_tv = np.stack([A1, A1, A2, A2, A2])
    B_tv = np.stack([B1, B1, B2, B2, B2])

    Q = np.eye(2) * 10
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, dynamics=LinearDynamics(A_tv, B_tv), Q=Q, R=R)
    ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X_ms, U_ms, status_ms = ocp.solve(np.array([1.0, 0.0]))
    assert status_ms["solved_successfully"]

    ocp_ss = LinearOCP(N=5, dt=0.1, dynamics=LinearDynamics(A_tv, B_tv), Q=Q, R=R)
    ocp_ss.setup(
        method="single_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X_ss, U_ss, status_ss = ocp_ss.solve(np.array([1.0, 0.0]))
    assert status_ss["solved_successfully"]

    np.testing.assert_allclose(X_ms, X_ss, atol=1e-5)
    np.testing.assert_allclose(U_ms, U_ss, atol=1e-5)
