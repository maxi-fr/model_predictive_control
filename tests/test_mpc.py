import casadi as ca
import numpy as np
import pytest

from model_predictive_control.constraints import ConstraintList, ControlConstraint
from model_predictive_control.mpc import MPC, LinearMPC
from model_predictive_control.objective import CostFunction, Objective
from model_predictive_control.ocp import OCP, LinearOCP, rk4_integrator


def setup_simple_ocp() -> OCP:
    # Double integrator system
    # x = [p, v], u = [a]
    nx = 2
    nu = 1
    N = 10
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    dyn = ca.vertcat(x[1], u[0])
    dynamics = ca.Function("dyn", [x, u], [dyn])

    obj_func = x[0] ** 2 + x[1] ** 2 + u[0] ** 2
    term_obj_func = 10 * (x[0] ** 2 + x[1] ** 2)
    objective = Objective(
        CostFunction(ca.Function("obj", [x, u], [obj_func])),
        CostFunction(ca.Function("term_obj", [x], [term_obj_func])),
        N,
    )

    ineq = u[0] ** 2 - 1.0
    in_eq_constraints = ca.Function("ineq", [u], [ineq])
    cl = ConstraintList()
    cl.add(ControlConstraint(in_eq_constraints), range(N))

    return OCP(
        N=N,
        dt=dt,
        objective=objective,
        dynamics=dynamics,
        constraints=cl,
    )


def test_mpc_initialization() -> None:
    ocp = setup_simple_ocp()
    setup_args = {
        "method": "multiple_shooting",
        "dynamics_type": "continuous",
        "integrator": rk4_integrator,
        "solver_opts": {"print_level": 0},
    }
    mpc = MPC(ocp=ocp, setup_args=setup_args)

    assert mpc.N == 10
    assert mpc.nx == 2
    assert mpc.nu == 1
    assert mpc._X_guess.shape == (11, 2)
    assert mpc._U_guess.shape == (10, 1)


def test_mpc_step_and_shifting() -> None:
    ocp = setup_simple_ocp()
    setup_args = {
        "method": "multiple_shooting",
        "dynamics_type": "continuous",
        "integrator": rk4_integrator,
        "solver_opts": {"print_level": 0},
    }
    mpc = MPC(ocp=ocp, setup_args=setup_args)

    x0 = np.array([1.0, 0.0])
    u0 = mpc.step(x0)

    assert u0.shape == (1,)
    # After step, guesses should be shifted
    # Check that last two elements of X_guess and U_guess are the same due to replication
    np.testing.assert_array_equal(mpc._X_guess[-1], mpc._X_guess[-2])
    np.testing.assert_array_equal(mpc._U_guess[-1], mpc._U_guess[-2])

    # Second step should run fine with warm-started guesses
    x1 = np.array([0.9, -0.1])
    u1 = mpc.step(x1)
    assert u1.shape == (1,)


def test_mpc_solve_failure() -> None:
    ocp = setup_simple_ocp()
    # Force failure with max_iter=0 and hard initial state
    setup_args = {
        "method": "multiple_shooting",
        "dynamics_type": "continuous",
        "integrator": rk4_integrator,
        "solver_opts": {"max_iter": 0, "print_level": 0},
    }
    mpc = MPC(ocp=ocp, setup_args=setup_args)

    x0 = np.array([100.0, 100.0])
    with pytest.raises(RuntimeError, match="solve failed with status"):
        mpc.step(x0)


def setup_simple_linear_ocp() -> LinearOCP:
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2) * 10
    R = np.eye(1)

    return LinearOCP(N=5, dt=0.1, A=A, B=B, Q=Q, R=R)


def test_linear_mpc_initialization() -> None:
    ocp = setup_simple_linear_ocp()
    setup_args = {
        "method": "multiple_shooting",
        "dynamics_type": "discrete",
        "solver_opts": {"print_iter": False, "print_header": False},
    }
    mpc = LinearMPC(linear_ocp=ocp, setup_args=setup_args)

    assert mpc.N == 5
    assert mpc.nx == 2
    assert mpc.nu == 1
    assert mpc._X_guess.shape == (6, 2)
    assert mpc._U_guess.shape == (5, 1)


def test_linear_mpc_step_and_shifting() -> None:
    ocp = setup_simple_linear_ocp()
    setup_args = {
        "method": "multiple_shooting",
        "dynamics_type": "discrete",
        "solver_opts": {"print_iter": False, "print_header": False},
    }
    mpc = LinearMPC(linear_ocp=ocp, setup_args=setup_args)

    x0 = np.array([1.0, 0.0])
    u0 = mpc.step(x0)

    assert u0.shape == (1,)
    np.testing.assert_array_equal(mpc._X_guess[-1], mpc._X_guess[-2])
    np.testing.assert_array_equal(mpc._U_guess[-1], mpc._U_guess[-2])

    x1 = np.array([[1.0, 0.1], [0.0, 1.0]]) @ x0 + np.array([[0.0], [0.1]]) @ u0
    u1 = mpc.step(x1)
    assert u1.shape == (1,)


def test_linear_mpc_tracking() -> None:
    # Setup linear tracking OCP
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(2) * 10
    R = np.eye(1)

    ocp = LinearOCP(N=5, dt=0.1, A=A, B=B, Q=Q, R=R)
    setup_args = {
        "method": "multiple_shooting",
        "dynamics_type": "discrete",
        "solver_opts": {"print_iter": False, "print_header": False},
    }
    mpc = LinearMPC(linear_ocp=ocp, setup_args=setup_args)

    x0 = np.array([0.0, 0.0])
    x_ref_1d = np.array([1.0, 0.0])
    u_ref_1d = np.array([0.0])

    # Pass 1D references, the wrapper should pass them down, and LinearOCP should tile them
    u0 = mpc.step(x0, x_ref=x_ref_1d, u_ref=u_ref_1d)
    assert u0.shape == (1,)
