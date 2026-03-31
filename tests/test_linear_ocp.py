import numpy as np
import pytest

from model_predictive_control.ocp import LinearOCP


def test_linear_ocp_validation() -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2)
    R = np.eye(1)

    with pytest.raises(ValueError):
        LinearOCP(N=10, dt=0.1, A=np.eye(3), B=B, Q=Q, R=R)

    ocp = LinearOCP(N=10, dt=0.1, A=A, B=B, Q=Q, R=R)
    assert ocp.nx == 2
    assert ocp.nu == 1


def test_linear_ocp_solve_multiple_shooting() -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2) * 10
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, A=A, B=B, Q=Q, R=R)
    ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X, U, status = ocp.solve(np.array([1.0, 0.0]))
    assert status == "success"
    assert X.shape == (2, 6)
    assert U.shape == (1, 5)


def test_linear_ocp_solve_single_shooting() -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2) * 10
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, A=A, B=B, Q=Q, R=R)
    ocp.setup(
        method="single_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X, U, status = ocp.solve(np.array([1.0, 0.0]))
    assert status == "success"
    assert X.shape == (2, 6)
    assert U.shape == (1, 5)


def test_linear_ocp_continuous() -> None:
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, A=A, B=B, Q=Q, R=R)
    ocp.setup(
        method="multiple_shooting",
        dynamics_type="continuous",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X, U, status = ocp.solve(np.array([1.0, 0.0]))
    assert status == "success"


def test_linear_ocp_constraints() -> None:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2)
    R = np.eye(1)

    F = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    G = np.array([[0.0], [0.0], [1.0], [-1.0]])
    h = np.array([2.0, 2.0, 50.0, 50.0])

    ocp = LinearOCP(N=5, dt=0.1, A=A, B=B, Q=Q, R=R, F=F, G=G, h=h)
    ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X, U, status = ocp.solve(np.array([1.5, 0.0]))
    assert status == "success"

    ocp_ss = LinearOCP(N=5, dt=0.1, A=A, B=B, Q=Q, R=R, F=F, G=G, h=h)
    ocp_ss.setup(
        method="single_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X_ss, U_ss, status_ss = ocp_ss.solve(np.array([1.5, 0.0]))
    assert status_ss == "success"
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

    ocp = LinearOCP(N=5, dt=0.1, A=A_tv, B=B_tv, Q=Q, R=R)
    ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X_ms, U_ms, status_ms = ocp.solve(np.array([1.0, 0.0]))
    assert status_ms == "success"

    ocp_ss = LinearOCP(N=5, dt=0.1, A=A_tv, B=B_tv, Q=Q, R=R)
    ocp_ss.setup(
        method="single_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    X_ss, U_ss, status_ss = ocp_ss.solve(np.array([1.0, 0.0]))
    assert status_ss == "success"

    np.testing.assert_allclose(X_ms, X_ss, atol=1e-5)
    np.testing.assert_allclose(U_ms, U_ss, atol=1e-5)
