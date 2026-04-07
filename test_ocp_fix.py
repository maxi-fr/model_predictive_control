with open("tests/test_ocp.py") as f:
    content = f.read()

# I see it failed because `cl = ConstraintList()` is not defined and `in_eq_constraints` was not removed from setup_simple_ocp!
# This means my string replaces failed because the source string did not exactly match what was in the file.
# Let's fix test_ocp.py using AST or manual regex.

import re

# Insert imports
if "ConstraintList" not in content[:200]:
    content = content.replace(
        "from model_predictive_control.ocp import OCP, lqr_objective",
        "from model_predictive_control.ocp import OCP, lqr_objective\\nfrom model_predictive_control.constraints import ConstraintList, Constraint, StateConstraint, ControlConstraint, LinearConstraint",
    )

# Fix setup_simple_ocp
content = re.sub(
    r'if "in_eq_constraints" not in kwargs:\n\s*ineq = u\[0\] \*\* 2 - 1\.0\n\s*kwargs\["in_eq_constraints"\] = ca\.Function\("ineq", \[x, u\], \[ineq\]\)',
    'if "constraints" not in kwargs:\n        ineq = u[0] ** 2 - 1.0\n        cl = ConstraintList()\n        cl.add(Constraint(ca.Function("ineq", [x, u], [ineq])), range(N))\n        kwargs["constraints"] = cl',
    content,
)

content = re.sub(
    r"def setup_simple_ocp\(\n\s*dynamics: ca\.Function \| None = None, objective: ca\.Function \| None = None, \*\*kwargs: dict\[str, Any\]\n\) -> OCP:",
    "def setup_simple_ocp(\n    dynamics: ca.Function | None = None, objective: ca.Function | None = None, **kwargs: Any\n) -> OCP:",
    content,
)


# Fix validation dims
bad_val = """    # Break eq_constraints input size
    eq_wrong = ca.Function("eq", [x_wrong, u], [x_wrong[0]])
    with pytest.raises(ValueError, match="eq_constraints function inputs must match"):
        setup_simple_ocp(eq_constraints=eq_wrong)

    # Break in_eq_constraints input size
    ineq_wrong = ca.Function("ineq", [ca.MX.sym("x", 3), u], [u[0]])
    with pytest.raises(ValueError, match="in_eq_constraints function inputs must match"):
        setup_simple_ocp(in_eq_constraints=ineq_wrong)

    # Break terminal objective
    term_obj_wrong = ca.Function("term_obj", [x_wrong], [x_wrong[0] ** 2])
    with pytest.raises(ValueError, match="terminal_objective function input must match"):
        setup_simple_ocp(terminal_objective=term_obj_wrong)

    term_eq_wrong = ca.Function("term_eq", [x_wrong], [x_wrong[0]])
    with pytest.raises(ValueError, match="terminal_eq_constraints function input must match"):
        setup_simple_ocp(terminal_eq_constraints=term_eq_wrong)"""

new_val = """    # Break eq_constraints input size
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
    term_obj_wrong = ca.Function("term_obj", [x_wrong], [x_wrong[0] ** 2])
    with pytest.raises(ValueError, match="terminal_objective function input must match"):
        setup_simple_ocp(terminal_objective=term_obj_wrong)

    term_eq_wrong = ca.Function("term_eq", [x_wrong], [x_wrong[0]])
    with pytest.raises(ValueError, match="StateConstraint function input must match state"):
        cl = ConstraintList()
        cl.add(StateConstraint(term_eq_wrong, is_equality=True), [10])
        setup_simple_ocp(constraints=cl)"""
content = content.replace(bad_val, new_val)

bad_lin = """    # Constraints: u <= 1 -> u - 1 <= 0
    in_eq = ca.Function("in_eq", [x, u], [u - 1], ["x", "u"], ["f"])

    ocp = OCP(N=N, dt=dt, objective=obj, dynamics=dyn, in_eq_constraints=in_eq)"""

new_lin = """    # Constraints: u <= 1 -> u - 1 <= 0
    in_eq = ca.Function("in_eq", [x, u], [u - 1], ["x", "u"], ["f"])

    cl = ConstraintList()
    cl.add(Constraint(in_eq), range(N))
    ocp = OCP(N=N, dt=dt, objective=obj, dynamics=dyn, constraints=cl)"""
content = content.replace(bad_lin, new_lin)

bad_lin_as = """    np.testing.assert_allclose(lin_ocp.Q[0], Q_expected, atol=1e-10)
    np.testing.assert_allclose(lin_ocp.R[0], R_expected, atol=1e-10)
    np.testing.assert_allclose(lin_ocp.N_cross[0], N_cross_expected, atol=1e-10)"""

new_lin_as = """    np.testing.assert_allclose(lin_ocp.Q[0], Q_expected, atol=1e-10)
    np.testing.assert_allclose(lin_ocp.R[0], R_expected, atol=1e-10)
    np.testing.assert_allclose(lin_ocp.N_cross[0], N_cross_expected, atol=1e-10)

    lin_c = lin_ocp.constraints.constraints[0][0]
    assert isinstance(lin_c, LinearConstraint)
    assert lin_c.G is not None
    assert lin_c.h is not None
    np.testing.assert_allclose(lin_c.G[0], np.array([[1.0]]), atol=1e-10)
    np.testing.assert_allclose(lin_c.h[0], np.array([1.0]), atol=1e-10)"""
content = content.replace(bad_lin_as, new_lin_as)


with open("tests/test_ocp.py", "w") as f:
    f.write(content)
