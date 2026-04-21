import casadi as ca
import numpy as np
import pytest

from model_predictive_control.constraints import Constraint, ConstraintList
from model_predictive_control.dynamics import Dynamics
from model_predictive_control.objective import Objective, QuadraticCost
from model_predictive_control.ocp import OCP, euler_integrator


def test_terminal_constraint_with_control_raises_valueerror() -> None:
    nx = 2
    nu = 1
    N = 10
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    dyn_f = ca.Function("f", [x, u], [x + ca.vertcat(u, u)])
    dynamics = Dynamics(dyn_f)

    Q = np.eye(nx)
    R = np.eye(nu)
    stage_cost = QuadraticCost(Q, R)
    objective = Objective([stage_cost] * N)

    # Base Constraint uses f(x, u)
    c = Constraint(ca.Function("c_f", [x, u], [x[0] + u[0]]))

    constraints = ConstraintList()
    # Apply to terminal step (N)
    constraints.add(c, N)

    ocp = OCP(N=N, dt=dt, objective=objective, dynamics=dynamics, constraints=constraints)

    # Should raise ValueError during setup
    with pytest.raises(
        ValueError, match=r"Terminal constraints cannot depend on control u\. Use StateConstraint instead\."
    ):
        ocp.setup(integrator=euler_integrator)
