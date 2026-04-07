with open("tests/test_ocp.py") as f:
    content = f.read()

# Add imports
if "ConstraintList" not in content[:200]:
    content = content.replace(
        "from model_predictive_control.ocp import OCP, lqr_objective",
        "from model_predictive_control.ocp import OCP, lqr_objective\nfrom model_predictive_control.constraints import ConstraintList, Constraint, StateConstraint, ControlConstraint, LinearConstraint",
    )

with open("tests/test_ocp.py", "w") as f:
    f.write(content)
