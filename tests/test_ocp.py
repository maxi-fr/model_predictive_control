from typing import Any

import casadi as ca
import numpy as np
import pytest

from model_predictive_control.constraints import (
    Constraint,
    ConstraintList,
    ControlConstraint,
    LinearConstraint,
    StateConstraint,
)
from model_predictive_control.dynamics import Dynamics
from model_predictive_control.objective import CostFunction, Objective, QuadraticObjective
from model_predictive_control.ocp import OCP, rk4_integrator


def setup_simple_ocp(dynamics: Any = None, objective: Objective | None = None, **kwargs: Any) -> OCP:  # noqa: ANN401
    # Simple double integrator system
    # x = [p, v], u = [a]
    nx = 2
    nu = 1
    N = 10
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    if dynamics is None:
        dyn = ca.vertcat(x[1], u[0])
        dynamics = Dynamics(ca.Function("dyn", [x, u], [dyn]))

    if objective is None:
        obj = x[0] ** 2 + x[1] ** 2 + u[0] ** 2
        term_obj = 10 * (x[0] ** 2 + x[1] ** 2)
        objective = Objective(
            CostFunction(ca.Function("obj", [x, u], [obj])), CostFunction(ca.Function("term_obj", [x], [term_obj])), N
        )

    if "constraints" not in kwargs:
        ineq = u[0] ** 2 - 1.0
        cl = ConstraintList()
        cl.add(Constraint(ca.Function("ineq", [x, u], [ineq])), range(N))
        kwargs["constraints"] = cl

    return OCP(N, dt, objective, dynamics, **kwargs)


def test_ocp_validation_missing_attrs() -> None:
    # Test missing arguments explicitly
    with pytest.raises(TypeError, match="missing 2 required positional arguments: 'objective' and 'dynamics'"):
        OCP(10, 0.1)  # type: ignore[call-arg]


def test_ocp_validation_wrong_dims() -> None:
    nx, nu = 2, 1

    # Break objective input size
    x_wrong = ca.MX.sym("x", nx + 1)
    u_wrong = ca.MX.sym("u", nu)
    obj_wrong = ca.Function("obj", [x_wrong, u_wrong], [x_wrong[0] ** 2 + u_wrong[0] ** 2])

    with pytest.raises(ValueError, match="Cost function state input size"):
        setup_simple_ocp(objective=Objective(CostFunction(obj_wrong), 10))

    # Break dynamics output size
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    dyn_wrong = ca.Function("dyn", [x, u], [x[0] + u[0]])  # Returns scalar instead of nx
    with pytest.raises(ValueError, match=r"Dynamics function output size .* must match state size"):
        setup_simple_ocp(dynamics=Dynamics(dyn_wrong))

    # Break dynamics missing argument
    dyn_missing_arg = ca.Function("dyn", [x], [x])
    with pytest.raises(ValueError, match="Dynamics function must take at least two arguments"):
        setup_simple_ocp(dynamics=Dynamics(dyn_missing_arg))

    # Break eq_constraints input size
    eq_wrong = ca.Function("eq", [x_wrong, u], [x_wrong[0]])
    cl = ConstraintList()
    cl.add(Constraint(eq_wrong, is_equality=True), range(10))
    with pytest.raises(ValueError, match="Constraint function inputs must match state"):
        setup_simple_ocp(constraints=cl)

    # Break in_eq_constraints input size
    ineq_wrong = ca.Function("ineq", [ca.MX.sym("x", 3), u], [u[0]])
    cl = ConstraintList()
    cl.add(Constraint(ineq_wrong, is_equality=False), range(10))
    with pytest.raises(ValueError, match="Constraint function inputs must match state"):
        setup_simple_ocp(constraints=cl)

    # Break terminal objective
    term_obj_wrong = ca.Function("term_obj_wrong", [x_wrong], [x_wrong[0]])
    with pytest.raises(ValueError, match="Cost function state input size"):
        setup_simple_ocp(
            objective=Objective(CostFunction(ca.Function("obj", [x, u], [x[0] ** 2])), CostFunction(term_obj_wrong), 10)
        )

    # Break terminal eq_constraints
    term_eq_wrong = ca.Function("term_eq_wrong", [x_wrong], [x_wrong[0]])
    cl = ConstraintList()
    cl.add(StateConstraint(term_eq_wrong, is_equality=True), [10])
    with pytest.raises(ValueError, match="StateConstraint function input must match state"):
        setup_simple_ocp(constraints=cl)


@pytest.mark.parametrize(
    ("method", "dynamics_type"),
    [
        ("multiple_shooting", "continuous"),
        ("single_shooting", "continuous"),
        ("collocation", "continuous"),
        ("multiple_shooting", "discrete"),
        ("single_shooting", "discrete"),
    ],
)
@pytest.mark.filterwarnings("ignore:integrator argument is ignored when method is 'collocation'")
def test_ocp_setup_and_solve(method: str, dynamics_type: str) -> None:
    if dynamics_type == "discrete":
        # Discretize it manually for the discrete test
        x = ca.MX.sym("x", 2)
        u = ca.MX.sym("u", 1)
        x_next = ca.vertcat(x[0] + 0.1 * x[1], x[1] + 0.1 * u[0])
        dyn = ca.Function("dyn", [x, u], [x_next])
        ocp = setup_simple_ocp(dynamics=Dynamics(dyn))
    else:
        ocp = setup_simple_ocp()

    ocp.setup(
        method=method, dynamics_type=dynamics_type, integrator=rk4_integrator if dynamics_type == "continuous" else None
    )

    x0 = np.array([1.0, 0.0])
    X_opt, U_opt, status = ocp.solve(x0)

    assert status["solved_successfully"]
    assert X_opt.shape == (11, 2)  # N+1 points, nx=2
    assert U_opt.shape == (10, 1)  # N points, nu=1

    # State should be driven towards 0
    assert abs(X_opt[-1, 0]) < 0.9  # Loose bounds since it's a very short horizon N=10
    assert abs(X_opt[-1, 1]) < 0.9

    # Test warm start functionality
    X_warm, U_warm, status_warm = ocp.solve(x0, X_guess=X_opt, U_guess=U_opt)
    assert status_warm["solved_successfully"]
    np.testing.assert_allclose(X_opt, X_warm, atol=1e-5)
    np.testing.assert_allclose(U_opt, U_warm, atol=1e-5)

    # Check constraints
    assert np.all(U_opt >= -1.0001)
    assert np.all(U_opt <= 1.0001)


def test_ocp_collocation_discrete_fails() -> None:
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)
    x_next = ca.vertcat(x[0] + 0.1 * x[1], x[1] + 0.1 * u[0])
    dyn = ca.Function("dyn", [x, u], [x_next])
    ocp = setup_simple_ocp(dynamics=Dynamics(dyn))

    with pytest.raises(ValueError, match="Collocation method is not applicable to discrete dynamics"):
        ocp.setup(method="collocation", dynamics_type="discrete")


def test_ocp_solve_without_setup() -> None:
    ocp = setup_simple_ocp()
    with pytest.raises(RuntimeError, match="OCP has not been set up"):
        ocp.solve(np.array([1.0, 0.0]))


@pytest.mark.parametrize("solver", ["ipopt"])
def test_ocp_custom_solver_opts(solver: str) -> None:
    ocp = setup_simple_ocp()
    # Use max_iter=2 to force a premature exit for solvers that support it (like ipopt)
    plugin_opts = {"expand": False}
    solver_opts = {"max_iter": 2}

    ocp.setup(solver=solver, plugin_opts=plugin_opts, solver_opts=solver_opts, integrator=rk4_integrator)

    x0 = np.array([10.0, 10.0])  # Hard state to solve in 2 iters
    _, _, status = ocp.solve(x0)

    if solver == "ipopt":
        # IPOPT should reach max iter and fail gracefully
        assert not status["solved_successfully"]


def test_linearize_method() -> None:
    # Let's create a simple nonlinear OCP
    nx = 2
    nu = 1
    N = 5
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    # Dynamics: x1_dot = x2, x2_dot = sin(x1) + u
    dyn = ca.Function("dyn", [x, u], [ca.vertcat(x[1], ca.sin(x[0]) + u)], ["x", "u"], ["f"])

    # Objective: L(x,u) = x1^2 + x2^2 + u^2 + x1*u
    obj_func = ca.Function("obj", [x, u], [x[0] ** 2 + x[1] ** 2 + u**2 + x[0] * u], ["x", "u"], ["f"])
    obj = Objective(CostFunction(obj_func), N)

    # Constraints: u <= 1 -> u - 1 <= 0
    in_eq = ca.Function("in_eq", [x, u], [u - 1], ["x", "u"], ["f"])

    cl = ConstraintList()
    cl.add(Constraint(in_eq), range(N))
    ocp = OCP(N=N, dt=dt, objective=obj, dynamics=Dynamics(dyn), constraints=cl)

    x_bar = np.array([0.0, 0.0])
    u_bar = np.array([0.0])

    lin_ocp = ocp.linearize(x_bar, u_bar, dynamics_type="discrete")

    # The jacobian of [x2, sin(x1)+u] at [0,0] is A = [[0, 1], [1, 0]], B = [[0], [1]]
    A_expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    B_expected = np.array([[0.0], [1.0]])

    # Objective hessians: Q = [[2, 0], [0, 2]], R = [[2]], N_cross = [[1], [0]]
    # Since cost is L = x1^2 + x2^2 + u^2 + x1*u
    # dL/dx = [2x1 + u, 2x2], d2L/dx2 = [[2, 0], [0, 2]]
    # dL/du = 2u + x1, d2L/du2 = [[2]]
    # d2L/dxdu = [[1], [0]]
    Q_expected = np.array([[2.0, 0.0], [0.0, 2.0]])
    R_expected = np.array([[2.0]])
    N_cross_expected = np.array([[1.0], [0.0]])

    np.testing.assert_allclose(lin_ocp.A[0], A_expected, atol=1e-10)
    np.testing.assert_allclose(lin_ocp.B[0], B_expected, atol=1e-10)
    np.testing.assert_allclose(lin_ocp.Q[0], Q_expected, atol=1e-10)
    np.testing.assert_allclose(lin_ocp.R[0], R_expected, atol=1e-10)
    np.testing.assert_allclose(lin_ocp.N_cross[0], N_cross_expected, atol=1e-10)

    lin_c = lin_ocp.constraints.constraints[0][0]
    assert isinstance(lin_c, LinearConstraint)
    assert lin_c.G is not None
    assert lin_c.h is not None
    np.testing.assert_allclose(lin_c.G, np.array([[1.0]]), atol=1e-10)
    np.testing.assert_allclose(lin_c.h, np.array([1.0]), atol=1e-10)

    lin_c = lin_ocp.constraints.constraints[0][0]
    assert isinstance(lin_c, LinearConstraint)
    assert lin_c.G is not None
    assert lin_c.h is not None
    np.testing.assert_allclose(lin_c.G, np.array([[1.0]]), atol=1e-10)
    np.testing.assert_allclose(lin_c.h, np.array([1.0]), atol=1e-10)


def test_linearize_equivalence() -> None:
    # Create a linear-quadratic OCP and solve it with non-linear solver
    # Then linearize it around 0 and solve it with QP solver
    # The result should be exactly the same

    nx = 2
    nu = 1
    N = 10
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    # Dynamics: x_next = A_d x + B_d u
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    dyn = ca.Function("dyn", [x, u], [A @ x + B @ u], ["x", "u"], ["f"])

    # Objective: 0.5*(x^T Q x + u^T R u)
    Q = np.eye(2) * 10
    R = np.eye(1)
    obj_func = ca.Function("obj", [x, u], [0.5 * (x.T @ Q @ x + u.T @ R @ u)], ["x", "u"], ["f"])
    term_obj_func = ca.Function("term_obj", [x], [0.5 * (x.T @ Q @ x)], ["x"], ["f"])
    obj = Objective(CostFunction(obj_func), CostFunction(term_obj_func), N)

    ocp = OCP(N=N, dt=dt, objective=obj, dynamics=Dynamics(dyn))

    ocp.setup(method="multiple_shooting", dynamics_type="discrete", solver="ipopt", solver_opts={"print_level": 0})
    X_nl, U_nl, status_nl = ocp.solve(np.array([1.0, 0.0]))
    assert status_nl["solved_successfully"]

    # Linearize around 0
    x_bar = np.array([0.0, 0.0])
    u_bar = np.array([0.0])
    lin_ocp = ocp.linearize(x_bar, u_bar, dynamics_type="discrete")

    lin_ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )
    X_lin, U_lin, status_lin = lin_ocp.solve(np.array([1.0, 0.0]))
    assert status_lin["solved_successfully"]

    np.testing.assert_allclose(X_nl, X_lin, atol=1e-10)
    np.testing.assert_allclose(U_nl, U_lin, atol=1e-10)


def solve_riccati(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, Qf: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # Solves finite-horizon LQR backwards
    P: list[np.ndarray] = [Qf]
    K: list[np.ndarray] = []

    for _k in range(N):
        Pk = P[0]

        # K_k = (R + B^T P_{k+1} B)^-1 B^T P_{k+1} A
        temp = np.linalg.inv(R + B.T @ Pk @ B)
        K_k = temp @ B.T @ Pk @ A

        # P_k = Q + A^T P_{k+1} (A - B K_k)
        P_prev = Q + A.T @ Pk @ (A - B @ K_k)

        K.insert(0, K_k)
        P.insert(0, P_prev)

    return K, P


def test_riccati_equivalence() -> None:
    nx = 2
    nu = 1
    N = 15
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    dyn = ca.Function("dyn", [x, u], [A @ x + B @ u], ["x", "u"], ["f"])

    Q = np.eye(2) * 10
    R = np.eye(1)
    obj_func = ca.Function("obj", [x, u], [0.5 * (x.T @ Q @ x + u.T @ R @ u)], ["x", "u"], ["f"])
    term_obj_func = ca.Function("term_obj", [x], [0.5 * (x.T @ Q @ x)], ["x"], ["f"])
    obj = Objective(CostFunction(obj_func), CostFunction(term_obj_func), N)

    ocp = OCP(N=N, dt=dt, objective=obj, dynamics=Dynamics(dyn))

    # Linearize around 0
    x_bar = np.array([0.0, 0.0])
    u_bar = np.array([0.0])
    lin_ocp = ocp.linearize(x_bar, u_bar, dynamics_type="discrete")

    lin_ocp.setup(
        method="multiple_shooting",
        dynamics_type="discrete",
        solver="qrqp",
        solver_opts={"print_iter": False, "print_header": False},
    )

    x0 = np.array([2.0, -1.0])
    X_lin, U_lin, status_lin = lin_ocp.solve(x0)
    assert status_lin["solved_successfully"]

    # Riccati solution
    K_gains, _ = solve_riccati(A, B, Q, R, N, Q)

    X_ric = np.zeros((N + 1, nx))
    U_ric = np.zeros((N, nu))

    X_ric[0, :] = x0
    for k in range(N):
        U_ric[k, :] = -K_gains[k] @ X_ric[k, :]
        X_ric[k + 1, :] = A @ X_ric[k, :] + B @ U_ric[k, :]

    np.testing.assert_allclose(X_lin, X_ric, atol=1e-10)
    np.testing.assert_allclose(U_lin, U_ric, atol=1e-10)


def test_ocp_calculate_trajectory_cost() -> None:
    nx = 2
    nu = 1
    N = 5
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    dyn = ca.Function("dyn", [x, u], [ca.vertcat(x[0] + u, x[1])])
    obj_func = ca.Function("obj", [x, u], [x[0] ** 2 + u[0] ** 2])
    term_obj_func = ca.Function("term", [x], [2 * x[0] ** 2])
    obj = Objective(CostFunction(obj_func), CostFunction(term_obj_func), N)

    ocp = OCP(N=N, dt=dt, dynamics=Dynamics(dyn), objective=obj)

    X_test = np.ones((N + 1, nx))
    U_test = np.ones((N, nu)) * 2

    # expected stage cost: 1^2 + 2^2 = 5
    # total stage cost: 5 * 5 = 25
    # terminal cost: 2 * 1^2 = 2
    # total expected = 27
    cost = ocp.calculate_trajectory_cost(X_test, U_test)
    assert np.isclose(cost, 27.0)

    # Test error cases
    with pytest.raises(ValueError, match="X must have shape"):
        ocp.calculate_trajectory_cost(np.ones((N, nx)), U_test)
    with pytest.raises(ValueError, match="U must have shape"):
        ocp.calculate_trajectory_cost(X_test, np.ones((N + 1, nu)))


def test_linear_ocp_calculate_trajectory_cost() -> None:
    nx = 2
    nu = 1
    N = 5
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    A = np.eye(2)
    B = np.array([[1.0], [0.0]])
    dyn = ca.Function("dyn", [x, u], [A @ x + B @ u])

    # Obj is 0.5 * (x^T Q x + u^T R u) for linear OCP setup -> x^T x + u^T u
    # so we define nonlinear cost identically for equivalence
    obj_func = ca.Function("obj", [x, u], [x.T @ x + u.T @ u])

    term_obj_func = ca.Function("term_obj", [x], [2 * (x.T @ x)])
    obj = Objective(CostFunction(obj_func), CostFunction(term_obj_func), N)

    ocp = OCP(N=N, dt=dt, objective=obj, dynamics=Dynamics(dyn))
    lin_ocp = ocp.linearize(np.zeros(2), np.zeros(1), dynamics_type="discrete")

    X_test = np.ones((N + 1, nx))
    U_test = np.ones((N, nu)) * 2

    # Same calculation:
    # 0.5*(dx^T Q dx + du^T R du)
    # Q = [[2,0],[0,2]], R=[[2]] -> stage cost = 1*(1^2+1^2) + 1*(2^2) = 2 + 4 = 6
    # N=5 stages -> 30
    # term: 0.5*(x^T Qf x) -> Qf is Hessian of 2*(x1^2+x2^2) -> 4*I
    # term cost = 0.5 * x^T (4I) x = 2 * (1^2 + 1^2) = 4
    # Total = 34

    cost = lin_ocp.calculate_trajectory_cost(X_test, U_test)
    assert np.isclose(cost, 34.0)

    # Test tracking references
    x_ref = np.ones((N + 1, nx)) * 0.5
    u_ref = np.ones((N, nu))

    # stage cost with ref:
    # dx = [0.5, 0.5], du = [1]
    # 1*(0.5^2+0.5^2) + 1*(1^2) = 0.5 + 1 = 1.5
    # total stage = 5 * 1.5 = 7.5
    # term: 2*(0.5^2+0.5^2) = 1.0
    # Total = 8.5
    cost_with_ref = lin_ocp.calculate_trajectory_cost(X_test, U_test, x_ref=x_ref, u_ref=u_ref)
    assert np.isclose(cost_with_ref, 8.5)
