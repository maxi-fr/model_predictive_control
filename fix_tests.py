with open("tests/test_ocp.py") as f:
    content = f.read()

# Make sure we import rk4_integrator
if "rk4_integrator" not in content:
    content = content.replace(
        "from model_predictive_control.ocp import OCP, linear_constraints",
        "from model_predictive_control.ocp import OCP, linear_constraints, rk4_integrator",
    )
    content = content.replace(
        "from model_predictive_control.ocp import OCP, tracking_objective",
        "from model_predictive_control.ocp import OCP, tracking_objective, rk4_integrator",
    )
    # Just to be safe, add it to general import if it wasn't caught
    if "rk4_integrator" not in content:
        content = content.replace(
            "from model_predictive_control.ocp import OCP",
            "from model_predictive_control.ocp import OCP, rk4_integrator",
        )

# Since the previous parameter `integrator="rk4"` was removed, tests relying on continuous dynamics default need it explicitly.
content = content.replace(
    "ocp.setup(method=method, dynamics_type=dynamics_type)",
    "ocp.setup(method=method, dynamics_type=dynamics_type, integrator=rk4_integrator if dynamics_type == 'continuous' else None)",
)

content = content.replace(
    "ocp.setup(solver=solver, plugin_opts=plugin_opts, solver_opts=solver_opts)",
    "ocp.setup(solver=solver, plugin_opts=plugin_opts, solver_opts=solver_opts, integrator=rk4_integrator)",
)

with open("tests/test_ocp.py", "w") as f:
    f.write(content)
