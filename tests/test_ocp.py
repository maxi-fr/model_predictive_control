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
from model_predictive_control.ocp import OCP


def setup_simple_ocp(dynamics: ca.Function | None = None, objective: ca.Function | None = None, **kwargs: Any) -> OCP:
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
        dynamics = ca.Function("dyn", [x, u], [dyn])

    if objective is None:
        obj = x[0] ** 2 + x[1] ** 2 + u[0] ** 2
        objective = ca.Function("obj", [x, u], [obj])

    if "constraints" not in kwargs:
        ineq = u[0] ** 2 - 1.0
        cl = ConstraintList()
        cl.add(Constraint(ca.Function("ineq", [x, u], [ineq])), range(N))
        kwargs["constraints"] = cl

    if "terminal_objective" not in kwargs:
        term_obj = 10 * (x[0] ** 2 + x[1] ** 2)
        kwargs["terminal_objective"] = ca.Function("term_obj", [x], [term_obj])

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

    with pytest.raises(ValueError, match="Objective function inputs must match"):
        setup_simple_ocp(objective=obj_wrong)

    # Break dynamics output size
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    dyn_wrong = ca.Function("dyn", [x, u], [x[0] + u[0]])  # Returns scalar instead of nx
    with pytest.raises(ValueError, match=r"Dynamics function output size .* must match state size"):
        setup_simple_ocp(dynamics=dyn_wrong)

    # Break dynamics missing argument
    dyn_missing_arg = ca.Function("dyn", [x], [x])
    with pytest.raises(ValueError, match="Dynamics function must take at least two arguments"):
        setup_simple_ocp(dynamics=dyn_missing_arg)

    # Break eq_constraints input size
    eq_wrong = ca.Function("eq", [x_wrong, u], [x_wrong[0]])
    with pytest.raises(ValueError, match="Constraint function inputs must match state"):
        cl = ConstraintList()
        cl.add(Constraint(eq_wrong, is_equality=True), range(10))
        setup_simple_ocp(constraints=cl)

    # Break in_eq_constraints input size
    ineq_wrong = ca.Function("ineq", [ca.MX.sym("x", 3), u], [u[0]])
    with pytest.raises(ValueError, match="Constraint function inputs must match state"):
        cl = ConstraintList()
        cl.add(Constraint(ineq_wrong, is_equality=False), range(10))
        setup_simple_ocp(constraints=cl)

    # Break terminal objective
    term_obj_wrong = ca.Function("term_obj_wrong", [x_wrong], [x_wrong[0]])
    with pytest.raises(ValueError, match="terminal_objective function input must match"):
        setup_simple_ocp(terminal_objective=term_obj_wrong)

    # Break terminal eq_constraints
    term_eq_wrong = ca.Function("term_eq_wrong", [x_wrong], [x_wrong[0]])
    with pytest.raises(ValueError, match="StateConstraint function input must match state"):
        cl = ConstraintList()
        cl.add(StateConstraint(term_eq_wrong, is_equality=True), [10])
        setup_simple_ocp(constraints=cl)
