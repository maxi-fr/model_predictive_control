import casadi as ca
import numpy as np
import pytest

from model_predictive_control.objective import (
    CostFunction,
    LQRCost,
    LQRObjective,
    Objective,
    QuadraticCost,
    QuadraticObjective,
    TerminalLQRCost,
    TerminalQuadraticCost,
)


def test_lqr_cost() -> None:
    Q = np.eye(2)
    R = np.eye(1)
    N_cross = np.zeros((2, 1))

    # Valid init
    cost = LQRCost(Q, R, N_cross)
    assert cost.f.n_in() == 4
    assert cost.has_reference is True

    # Test Q must be square
    with pytest.raises(ValueError, match=r"Matrix Q must be square and match state dimension."):
        LQRCost(np.ones((2, 3)), R)

    # R not square
    with pytest.raises(ValueError, match=r"Matrix R must be square and match control dimension."):
        LQRCost(Q, np.ones((1, 2)))

    # N_cross wrong shape
    with pytest.raises(ValueError, match=r"Matrix N_cross must match state and control dimensions."):
        LQRCost(Q, R, np.ones((3, 1)))


def test_terminal_lqr_cost() -> None:
    Qf = np.eye(2)

    # Valid init
    cost = TerminalLQRCost(Qf)
    assert cost.f.n_in() == 2
    assert cost.has_reference is True

    # Test Q must be square
    with pytest.raises(ValueError, match=r"Matrix Q must be square."):
        TerminalLQRCost(np.ones((2, 3)))


def test_quadratic_cost() -> None:
    Q = np.eye(2)
    R = np.eye(1)
    q = np.ones((2, 1))
    r = np.ones((1, 1))
    N_cross = np.zeros((2, 1))

    # Valid init
    cost = QuadraticCost(Q, R, q, r, N_cross)
    assert cost.f.n_in() == 2
    assert cost.has_reference is False

    # Q not square
    with pytest.raises(ValueError, match=r"Matrix Q must be square and match state dimension."):
        QuadraticCost(np.ones((2, 3)), R)

    # R not square
    with pytest.raises(ValueError, match=r"Matrix R must be square and match control dimension."):
        QuadraticCost(Q, np.ones((1, 2)))

    # q wrong size
    with pytest.raises(ValueError, match=r"Vector q must match state dimension."):
        QuadraticCost(Q, R, q=np.ones((3, 1)))

    # r wrong size
    with pytest.raises(ValueError, match=r"Vector r must match control dimension."):
        QuadraticCost(Q, R, r=np.ones((2, 1)))

    # N_cross wrong size
    with pytest.raises(ValueError, match=r"Matrix N_cross must match state and control dimensions."):
        QuadraticCost(Q, R, N_cross=np.ones((3, 1)))


def test_terminal_quadratic_cost() -> None:
    Qf = np.eye(2)
    qf = np.ones((2, 1))

    # Valid init
    cost = TerminalQuadraticCost(Qf, qf)
    assert cost.f.n_in() == 1
    assert cost.has_reference is False

    # Q not square
    with pytest.raises(ValueError, match=r"Matrix Q must be square."):
        TerminalQuadraticCost(np.ones((2, 3)), qf)

    # qf wrong size
    with pytest.raises(ValueError, match=r"Vector q must have the same length as Q."):
        TerminalQuadraticCost(Qf, np.ones((3, 1)))


def test_cost_function_validate_dimensions() -> None:
    # Test LQR Cost (has reference)
    Q = np.eye(2)
    R = np.eye(1)
    cost = LQRCost(Q, R)

    # State input wrong size
    with pytest.raises(ValueError, match=r"Cost function state input size \(2\) must match state size \(3\)."):
        cost.validate_dimensions(nx=3, nu=1)

    # Control input wrong size
    with pytest.raises(ValueError, match=r"Cost function control input size \(1\) must match control size \(2\)."):
        cost.validate_dimensions(nx=2, nu=2)

    # Dummy cost function for ref mismatch
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)
    x_ref = ca.MX.sym("x_ref", 3)
    u_ref = ca.MX.sym("u_ref", 2)

    f_bad_ref = ca.Function("f", [x, u, x_ref, u_ref], [x.T @ x], ["x", "u", "x_ref", "u_ref"], ["f"])
    bad_cost = CostFunction(f_bad_ref)

    with pytest.raises(
        ValueError, match=r"Cost function reference inputs must match state \(2\) and control \(1\) sizes."
    ):
        bad_cost.validate_dimensions(nx=2, nu=1)

    # Test Terminal LQR Cost (has reference, 2 inputs)
    term_cost = TerminalLQRCost(Q)

    # State input wrong size
    with pytest.raises(ValueError, match=r"Cost function state input size \(2\) must match state size \(3\)."):
        term_cost.validate_dimensions(nx=3)

    x2 = ca.MX.sym("x", 2)
    x_ref2 = ca.MX.sym("x_ref", 3)
    f_bad_term = ca.Function("f", [x2, x_ref2], [x2.T @ x2], ["x", "x_ref"], ["f"])
    bad_term_cost = CostFunction(f_bad_term)
    bad_term_cost._has_reference = True

    with pytest.raises(ValueError, match=r"Cost function reference input must match state \(2\) size."):
        bad_term_cost.validate_dimensions(nx=2)

    # Test wrong output size
    f_bad_out = ca.Function("f", [x2], [ca.vertcat(x2.T @ x2, 0)], ["x"], ["f"])
    bad_out_cost = CostFunction(f_bad_out)

    with pytest.raises(ValueError, match=r"Cost function must return a scalar."):
        bad_out_cost.validate_dimensions(nx=2)


def test_objective_initialization() -> None:
    Q = np.eye(2)
    R = np.eye(1)
    stage_cost = LQRCost(Q, R)
    term_cost = TerminalLQRCost(Q)

    # test init(cost, N)
    obj1 = Objective(stage_cost, 5)
    assert len(obj1.stage_costs) == 5
    assert obj1.terminal_cost is None
    assert obj1.has_reference is True

    # test init(cost, cost_term, N)
    obj2 = Objective(stage_cost, term_cost, 5)
    assert len(obj2.stage_costs) == 5
    assert obj2.terminal_cost is term_cost
    assert obj2.has_reference is True

    # test init(stage_costs)
    obj3 = Objective([stage_cost] * 3)
    assert len(obj3.stage_costs) == 3
    assert obj3.terminal_cost is None
    assert obj3.has_reference is True

    # test init(stage_costs, cost_term)
    obj4 = Objective([stage_cost] * 3, term_cost)
    assert len(obj4.stage_costs) == 3
    assert obj4.terminal_cost is term_cost
    assert obj4.has_reference is True

    # invalid arguments
    with pytest.raises(ValueError, match=r"Invalid arguments for Objective constructor."):
        Objective(stage_cost, term_cost, 5, "extra")  # ty:ignore[no-matching-overload]

    with pytest.raises(ValueError, match=r"Invalid arguments for Objective constructor."):
        Objective()  # ty:ignore[no-matching-overload]


def test_objective_validate_dimensions() -> None:
    Q = np.eye(2)
    R = np.eye(1)
    stage_cost = LQRCost(Q, R)
    term_cost = TerminalLQRCost(Q)

    obj = Objective(stage_cost, term_cost, 5)

    # Valid call
    obj.validate_dimensions(nx=2, nu=1)

    # Invalid stage
    with pytest.raises(ValueError, match=r"Cost function state input size \(2\) must match state size \(3\)."):
        obj.validate_dimensions(nx=3, nu=1)


def test_lqr_objective() -> None:
    Q = np.eye(2)
    R = np.eye(1)
    Qf = np.eye(2)
    N = 5

    obj = LQRObjective(Q, R, Qf, N)
    assert len(obj.stage_costs) == 5
    assert obj.terminal_cost is not None
    assert obj.has_reference is True


def test_quadratic_objective() -> None:
    Q = np.eye(2)
    R = np.eye(1)
    Qf = np.eye(2)
    qf = np.ones((2, 1))
    N = 5

    obj = QuadraticObjective(Q, R, Qf, qf, N)
    assert len(obj.stage_costs) == 5
    assert obj.terminal_cost is not None
    assert obj.has_reference is False


def test_cost_function_call() -> None:
    Q = np.eye(2)
    R = np.eye(1)
    cost = LQRCost(Q, R)
    x_val = np.array([1, 1])
    u_val = np.array([1])
    x_ref = np.array([0, 0])
    u_ref = np.array([0])

    # Evaluate cost
    val = cost(x_val, u_val, x_ref, u_ref)
    assert val.shape == (1, 1)


def test_lqr_cost_math() -> None:
    Q = np.array([[2, 0], [0, 3]])
    R = np.array([[4]])
    N_cross = np.array([[1], [0.5]])
    cost = LQRCost(Q, R, N_cross)

    x_val = np.array([1.5, -2.0])
    u_val = np.array([3.0])
    x_ref = np.array([0.5, -1.0])
    u_ref = np.array([1.0])

    # dx = [1.0, -1.0]
    # du = [2.0]
    # dx.T * Q * dx = 1*2*1 + (-1)*3*(-1) = 2 + 3 = 5
    # du.T * R * du = 2*4*2 = 16
    # dx.T * N_cross * du = [1.0, -1.0] * [[1], [0.5]] * 2 = (1*1 + -1*0.5) * 2 = 0.5 * 2 = 1
    # total = 5 + 16 + 1 = 22

    val = cost(x_val, u_val, x_ref, u_ref)
    assert np.isclose(float(val), 22.0)


def test_terminal_lqr_cost_math() -> None:
    Qf = np.array([[2, 0], [0, 3]])
    cost = TerminalLQRCost(Qf)

    x_val = np.array([1.5, -2.0])
    x_ref = np.array([0.5, -1.0])

    # dx = [1.0, -1.0]
    # dx.T * Qf * dx = 1*2*1 + (-1)*3*(-1) = 5

    val = cost(x_val, x_ref)
    assert np.isclose(float(val), 5.0)


def test_quadratic_cost_math() -> None:
    Q = np.array([[2, 0], [0, 3]])
    R = np.array([[4]])
    q = np.array([[0.5], [1.5]])
    r = np.array([[-1.0]])
    N_cross = np.array([[1], [0.5]])

    cost = QuadraticCost(Q, R, q, r, N_cross)

    x_val = np.array([1.0, -1.0])
    u_val = np.array([2.0])

    # x.T * Q * x = 1*2*1 + (-1)*3*(-1) = 5
    # u.T * R * u = 2*4*2 = 16
    # x.T * N_cross * u = [1, -1] * [[1], [0.5]] * 2 = (1 - 0.5) * 2 = 1
    # x.T * q = [1, -1] * [0.5, 1.5] = 0.5 - 1.5 = -1.0
    # u.T * r = 2 * -1 = -2.0
    # Total = 5 + 16 + 1 - 1 - 2 = 19.0

    val = cost(x_val, u_val)
    assert np.isclose(float(val), 19.0)


def test_terminal_quadratic_cost_math() -> None:
    Qf = np.array([[2, 0], [0, 3]])
    qf = np.array([[0.5], [1.5]])

    cost = TerminalQuadraticCost(Qf, qf)

    x_val = np.array([1.0, -1.0])

    # x.T * Qf * x = 5
    # x.T * qf = -1.0
    # Total = 4.0

    val = cost(x_val)
    assert np.isclose(float(val), 4.0)
