import re

with open("src/model_predictive_control/ocp.py") as f:
    content = f.read()

# Add missing import for Callable, UserWarning
content = content.replace("from typing import Any", "from typing import Any, Callable\nimport warnings")

# Replace linearize signature and implementation
linearize_pattern = r"    def linearize\(\s*self,\s*x_bar: ArrayLike,\s*u_bar: ArrayLike,\s*dynamics_type: str = \"continuous\",\s*integrator: str = \"rk4\",\s*x_ref: ArrayLike \| None = None,\s*u_ref: ArrayLike \| None = None,\s*\) -> \"LinearOCP\":"
linearize_replacement = r"""    def linearize(
        self,
        x_bar: ArrayLike,
        u_bar: ArrayLike,
        dynamics_type: str = "continuous",
        integrator: Callable[[ca.Function, float], ca.Function] | None = None,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> "LinearOCP":"""
content = re.sub(linearize_pattern, linearize_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Replace linearize logic
lin_logic_old = r"""        if dynamics_type == "continuous":
            if integrator == "rk4":
                # Runge-Kutta 4 integration
                X0 = ca.MX.sym("X0", nx)
                U0 = ca.MX.sym("U0", nu)
                k1 = self.dynamics(X0, U0)
                k2 = self.dynamics(X0 + self.dt / 2.0 * k1, U0)
                k3 = self.dynamics(X0 + self.dt / 2.0 * k2, U0)
                k4 = self.dynamics(X0 + self.dt * k3, U0)
                X_next = X0 + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                dyn_func = ca.Function("dyn_rk4", \[X0, U0\], \[X_next\])
            else:
                # Forward Euler
                X0 = ca.MX.sym("X0", nx)
                U0 = ca.MX.sym("U0", nu)
                X_next = X0 + self.dt \* self.dynamics(X0, U0)
                dyn_func = ca.Function("dyn_euler", \[X0, U0\], \[X_next\])
        elif dynamics_type == "discrete":
            dyn_func = self.dynamics
        else:
            msg = f"Unknown dynamics_type: {dynamics_type}"
            raise ValueError(msg)"""

lin_logic_new = r"""        if dynamics_type == "continuous":
            if integrator is None:
                msg = "integrator must be provided when dynamics_type is 'continuous'"
                raise ValueError(msg)
            dyn_func = integrator(self.dynamics, self.dt)
        elif dynamics_type == "discrete":
            if integrator is None:
                pass
            elif integrator is not None:
                warnings.warn("integrator argument is ignored when dynamics_type is 'discrete'", UserWarning, stacklevel=2)
            dyn_func = self.dynamics
        else:
            msg = f"Unknown dynamics_type: {dynamics_type}"
            raise ValueError(msg)"""
content = re.sub(lin_logic_old, lin_logic_new, content)

# Replace setup signature and implementation
setup_pattern = r"    def setup\(\s*self,\s*method: str = \"multiple_shooting\",\s*dynamics_type: str = \"continuous\",\s*integrator: str = \"rk4\",\s*solver: str = \"ipopt\",\s*plugin_opts: dict\[Any, Any\] \| None = None,\s*solver_opts: dict\[Any, Any\] \| None = None,\s*\) -> None:"
setup_replacement = r"""    def setup(  # noqa: D102, PLR0915, PLR0912, PLR0913, C901 TODO: fix issues
        self,
        method: str = "multiple_shooting",
        dynamics_type: str = "continuous",
        integrator: Callable[[ca.Function, float], ca.Function] | None = None,
        solver: str = "ipopt",
        plugin_opts: dict[Any, Any] | None = None,
        solver_opts: dict[Any, Any] | None = None,
    ) -> None:"""
content = re.sub(setup_pattern, setup_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Collocation fix in setup logic
setup_logic_old = r"""        if dynamics_type == "continuous":
            if integrator == "rk4":
                # Runge-Kutta 4 integration
                X0 = ca.MX.sym("X0", nx)
                U0 = ca.MX.sym("U0", nu)
                k1 = self.dynamics(X0, U0)
                k2 = self.dynamics(X0 + self.dt / 2.0 * k1, U0)
                k3 = self.dynamics(X0 + self.dt / 2.0 * k2, U0)
                k4 = self.dynamics(X0 + self.dt * k3, U0)
                X_next = X0 + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                dyn_func = ca.Function("dyn_rk4", \[X0, U0\], \[X_next\])
            else:
                # Forward Euler
                X0 = ca.MX.sym("X0", nx)
                U0 = ca.MX.sym("U0", nu)
                X_next = X0 + self.dt \* self.dynamics(X0, U0)
                dyn_func = ca.Function("dyn_euler", \[X0, U0\], \[X_next\])
        elif dynamics_type == "discrete":
            dyn_func = self.dynamics
        else:
            msg = f"Unknown dynamics_type: {dynamics_type}"
            raise ValueError(msg)"""
setup_logic_new = r"""        if method == "collocation":
            if integrator is not None:
                warnings.warn("integrator argument is ignored when method is 'collocation'", UserWarning, stacklevel=2)
            # dyn_func is not used in collocation, but we set it to self.dynamics to avoid UnboundLocalError just in case
            dyn_func = self.dynamics
            if dynamics_type == "discrete":
                pass
        elif dynamics_type == "continuous":
            if integrator is None:
                msg = "integrator must be provided when dynamics_type is 'continuous'"
                raise ValueError(msg)
            dyn_func = integrator(self.dynamics, self.dt)
        elif dynamics_type == "discrete":
            if integrator is None:
                pass
            elif integrator is not None:
                warnings.warn("integrator argument is ignored when dynamics_type is 'discrete'", UserWarning, stacklevel=2)
            dyn_func = self.dynamics
        else:
            msg = f"Unknown dynamics_type: {dynamics_type}"
            raise ValueError(msg)"""
content = re.sub(setup_logic_old, setup_logic_new, content)

with open("src/model_predictive_control/ocp.py", "w") as f:
    f.write(content)
