import pytest
import casadi as ca
import numpy as np

from model_predictive_control.ocp import OCP, Quadratic_Objective

def setup_simple_ocp(dynamics=None, objective=None, **kwargs):
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
        dynamics = ca.Function('dyn', [x, u], [dyn])

    if objective is None:
        obj = x[0]**2 + x[1]**2 + u[0]**2
        objective = ca.Function('obj', [x, u], [obj])

    if 'in_eq_constraints' not in kwargs:
        ineq = u[0]**2 - 1.0
        kwargs['in_eq_constraints'] = ca.Function('ineq', [x, u], [ineq])

    if 'terminal_objective' not in kwargs:
        term_obj = 10 * (x[0]**2 + x[1]**2)
        kwargs['terminal_objective'] = ca.Function('term_obj', [x], [term_obj])

    return OCP(N, dt, objective, dynamics, **kwargs)

def test_ocp_validation_missing_attrs():
    # Test missing arguments explicitly
    with pytest.raises(TypeError, match="missing 2 required positional arguments: 'objective' and 'dynamics'"):
        ocp = OCP(10, 0.1)

def test_ocp_validation_wrong_dims():
    nx, nu = 2, 1

    # Break objective input size
    x_wrong = ca.MX.sym("x", nx+1)
    u_wrong = ca.MX.sym("u", nu)
    obj_wrong = ca.Function('obj', [x_wrong, u_wrong], [x_wrong[0]**2 + u_wrong[0]**2])

    with pytest.raises(ValueError, match="Objective function inputs must match"):
        setup_simple_ocp(objective=obj_wrong)

    # Break dynamics output size
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    dyn_wrong = ca.Function('dyn', [x, u], [x[0] + u[0]]) # Returns scalar instead of nx
    with pytest.raises(ValueError, match="Dynamics function output size .* must match state size"):
        setup_simple_ocp(dynamics=dyn_wrong)

    # Break dynamics missing argument
    dyn_missing_arg = ca.Function('dyn', [x], [x])
    with pytest.raises(ValueError, match="Dynamics function must take at least two arguments"):
        setup_simple_ocp(dynamics=dyn_missing_arg)

    # Break eq_constraints input size
    eq_wrong = ca.Function('eq', [x_wrong, u], [x_wrong[0]])
    with pytest.raises(ValueError, match="eq_constraints function inputs must match"):
        setup_simple_ocp(eq_constraints=eq_wrong)

    # Break in_eq_constraints input size
    ineq_wrong = ca.Function('ineq', [x_wrong, u_wrong], [u_wrong[0]])
    with pytest.raises(ValueError, match="in_eq_constraints function inputs must match"):
        setup_simple_ocp(in_eq_constraints=ineq_wrong)

    # Break terminal conditions
    term_obj_wrong = ca.Function('term_obj', [x_wrong], [x_wrong[0]**2])
    with pytest.raises(ValueError, match="terminal_objective function input must match"):
        setup_simple_ocp(terminal_objective=term_obj_wrong)

    term_eq_wrong = ca.Function('term_eq', [x_wrong], [x_wrong[0]])
    with pytest.raises(ValueError, match="terminal_eq_constraints function input must match"):
        setup_simple_ocp(terminal_eq_constraints=term_eq_wrong)

@pytest.mark.parametrize("method,dynamics_type", [
    ("multiple_shooting", "continuous"),
    ("single_shooting", "continuous"),
    ("collocation", "continuous"),
    ("multiple_shooting", "discrete"),
])
def test_ocp_setup_and_solve(method, dynamics_type):
    if dynamics_type == "discrete":
        # Discretize it manually for the discrete test
        x = ca.MX.sym("x", 2)
        u = ca.MX.sym("u", 1)
        x_next = ca.vertcat(x[0] + 0.1 * x[1], x[1] + 0.1 * u[0])
        dyn = ca.Function('dyn', [x, u], [x_next])
        ocp = setup_simple_ocp(dynamics=dyn)
    else:
        ocp = setup_simple_ocp()

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
