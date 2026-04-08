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


def test_constraint_validation() -> None:
    nx = 2
    nu = 1
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    x_wrong = ca.MX.sym("x_wrong", nx + 1)
    u_wrong = ca.MX.sym("u_wrong", nu + 1)

    # Base Constraint
    c = Constraint(ca.Function("f", [x, u], [x[0] + u[0]]))
    c.validate_dimensions(nx, nu)  # Should pass

    c_wrong_args = Constraint(ca.Function("f", [x], [x[0]]))
    with pytest.raises(ValueError, match="Constraint must take exactly two arguments"):
        c_wrong_args.validate_dimensions(nx, nu)

    c_wrong_x = Constraint(ca.Function("f", [x_wrong, u], [x_wrong[0] + u[0]]))
    with pytest.raises(ValueError, match="Constraint function inputs must match state"):
        c_wrong_x.validate_dimensions(nx, nu)

    # StateConstraint
    sc = StateConstraint(ca.Function("f", [x], [x[0]]))
    sc.validate_dimensions(nx, nu)  # Should pass

    sc_wrong_args = StateConstraint(ca.Function("f", [x, u], [x[0]]))
    with pytest.raises(ValueError, match="StateConstraint must take exactly one argument"):
        sc_wrong_args.validate_dimensions(nx, nu)

    sc_wrong_x = StateConstraint(ca.Function("f", [x_wrong], [x_wrong[0]]))
    with pytest.raises(ValueError, match="StateConstraint function input must match state"):
        sc_wrong_x.validate_dimensions(nx, nu)

    # ControlConstraint
    cc = ControlConstraint(ca.Function("f", [u], [u[0]]))
    cc.validate_dimensions(nx, nu)  # Should pass

    cc_wrong_args = ControlConstraint(ca.Function("f", [u, x], [u[0]]))
    with pytest.raises(ValueError, match="ControlConstraint must take exactly one argument"):
        cc_wrong_args.validate_dimensions(nx, nu)

    cc_wrong_u = ControlConstraint(ca.Function("f", [u_wrong], [u_wrong[0]]))
    with pytest.raises(ValueError, match="ControlConstraint function input must match control"):
        cc_wrong_u.validate_dimensions(nx, nu)


def test_linear_constraint_validation() -> None:
    nx = 2
    nu = 1

    h = np.array([1.0, 2.0])
    F = np.array([[1.0, 0.0], [0.0, 1.0]])
    G = np.array([[1.0], [1.0]])

    lc = LinearConstraint(h=h, F=F, G=G)
    lc.validate_dimensions(nx, nu)  # Should pass

    # Wrong F dims
    F_wrong_nx = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="Constraint function inputs must match state"):
        LinearConstraint(h=h, F=F_wrong_nx, G=G).validate_dimensions(nx, nu)

    F_wrong_nc = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="LinearConstraint F matrix must have 2 rows"):
        LinearConstraint(h=h, F=F_wrong_nc, G=G).validate_dimensions(nx, nu)

    # Wrong G dims
    G_wrong_nu = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="Constraint function inputs must match state"):
        LinearConstraint(h=h, F=F, G=G_wrong_nu).validate_dimensions(nx, nu)

    G_wrong_nc = np.array([[1.0]])
    with pytest.raises(ValueError, match="LinearConstraint G matrix must have 2 rows"):
        LinearConstraint(h=h, F=F, G=G_wrong_nc).validate_dimensions(nx, nu)


def test_constraint_list_resolve_indices() -> None:
    cl = ConstraintList()
    N = 10

    # Int index
    assert cl.resolve_indices(0, N) == [0]
    assert cl.resolve_indices(5, N) == [5]
    assert cl.resolve_indices(N, N) == [10]
    assert cl.resolve_indices(-1, N) == [10]
    assert cl.resolve_indices(-2, N) == [9]

    # Iterable index
    assert cl.resolve_indices([0, 1, 2], N) == [0, 1, 2]
    assert cl.resolve_indices([0, -1], N) == [0, 10]
    assert cl.resolve_indices(range(3), N) == [0, 1, 2]

    # Slice index
    assert cl.resolve_indices(slice(0, 3), N) == [0, 1, 2]
    assert cl.resolve_indices(slice(None), N) == list(range(11))
    assert cl.resolve_indices(slice(0, -1), N) == list(range(10))


def test_constraint_list_add_and_iter() -> None:
    cl = ConstraintList()

    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 1)
    c1 = Constraint(ca.Function("f1", [x, u], [x[0] + u[0]]))
    c2 = StateConstraint(ca.Function("f2", [x], [x[1]]))

    cl.add(c1, range(5))
    cl.add(c2, -1)

    assert len(cl) == 2

    c_list = list(cl)
    assert c_list[0][0] is c1
    assert list(c_list[0][1]) == list(range(5))  # type: ignore[arg-type]

    assert c_list[1][0] is c2
    assert c_list[1][1] == -1
