import pytest
import casadi as ca
import numpy as np

from model_predictive_control.ocp import OCP, Quadratic_Objective

def setup_simple_ocp():
    # Simple double integrator system
    # x = [p, v], u = [a]
    nx = 2
    nu = 1
    N = 10
    dt = 0.1

    ocp = OCP(N, dt)

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    # Dynamics: p_dot = v, v_dot = a
    dyn = ca.vertcat(x[1], u[0])
    ocp.dynamics = ca.Function('dyn', [x, u], [dyn])

    # Objective: minimize p^2 + v^2 + a^2
    obj = x[0]**2 + x[1]**2 + u[0]**2
    ocp.objective = ca.Function('obj', [x, u], [obj])

    # Constraints: -1 <= a <= 1
    ineq = u[0]**2 - 1.0
    ocp.in_eq_constraints = ca.Function('ineq', [x, u], [ineq])

    # Terminal conditions
    term_obj = 10 * (x[0]**2 + x[1]**2)
    ocp.terminal_objective = ca.Function('term_obj', [x], [term_obj])

    return ocp

def test_ocp_validation_missing_attrs():
    ocp = OCP(10, 0.1)
    with pytest.raises(ValueError, match="OCP must have a 'dynamics' attribute"):
        ocp.setup()

def test_ocp_validation_wrong_dims():
    ocp = setup_simple_ocp()
    # Break objective input size
    nx, nu = 2, 1
    x_wrong = ca.MX.sym("x", nx+1)
    u_wrong = ca.MX.sym("u", nu)
    ocp.objective = ca.Function('obj', [x_wrong, u_wrong], [x_wrong[0]**2 + u_wrong[0]**2])

    with pytest.raises(ValueError, match="Objective function inputs must match"):
        ocp.setup()

@pytest.mark.parametrize("method,dynamics_type", [
    ("multiple_shooting", "continuous"),
    ("single_shooting", "continuous"),
    ("collocation", "continuous"),
    ("multiple_shooting", "discrete"),
])
def test_ocp_setup_and_solve(method, dynamics_type):
    ocp = setup_simple_ocp()

    if dynamics_type == "discrete":
        # Discretize it manually for the discrete test
        x = ca.MX.sym("x", 2)
        u = ca.MX.sym("u", 1)
        x_next = ca.vertcat(x[0] + 0.1 * x[1], x[1] + 0.1 * u[0])
        ocp.dynamics = ca.Function('dyn', [x, u], [x_next])

    ocp.setup(method=method, dynamics_type=dynamics_type)

    x0 = np.array([1.0, 0.0])
    X_opt, U_opt, status = ocp.solve(x0)

    assert status == "Solve_Succeeded"
    assert X_opt.shape == (2, 11) # nx=2, N=10 -> N+1 points
    assert U_opt.shape == (1, 10) # nu=1, N=10 points

    # State should be driven towards 0
    assert abs(X_opt[0, -1]) < 0.9  # Loose bounds since it's a very short horizon N=10
    assert abs(X_opt[1, -1]) < 0.9

    # Check constraints
    assert np.all(U_opt >= -1.0001)
    assert np.all(U_opt <= 1.0001)

def test_ocp_solve_without_setup():
    ocp = setup_simple_ocp()
    with pytest.raises(RuntimeError, match="OCP has not been set up"):
        ocp.solve(np.array([1.0, 0.0]))
