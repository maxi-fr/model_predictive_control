with open("tests/test_ocp.py") as f:
    content = f.read()

content = content.replace(
    "from model_predictive_control.ocp import OCP",
    "from model_predictive_control.ocp import OCP\nfrom model_predictive_control.constraints import ConstraintList, Constraint, StateConstraint, ControlConstraint, LinearConstraint",
)

with open("tests/test_ocp.py", "w") as f:
    f.write(content)
