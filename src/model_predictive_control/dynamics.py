from typing import Any

import casadi as ca
import numpy as np
from pydantic import BaseModel, ConfigDict
from simulate.integrator import Integrator
from simulate.plant import Plant


class DynamicsLog(BaseModel):
    """Log for dynamics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: np.ndarray


class Dynamics(Plant[DynamicsLog]):
    """Base class for dynamics, compatible with the simulate framework."""

    def __init__(self, f: ca.Function, dt: float = 1.0, integrator: Integrator | None = None) -> None:
        """
        Initialize the dynamics.

        Args:
            f: CasADi function for the dynamics.
            dt: Simulation time step.
            integrator: Optional integrator for continuous-time dynamics.
        """
        super().__init__(dt, integrator)
        self.f = f
        self.nx = f.size_in(0)[0]
        self.nu = f.size_in(1)[0] if f.n_in() > 1 else 0
        self.x = np.zeros(self.nx)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Dynamics":
        """Instantiate from config (not fully implemented for CasADi functions)."""
        msg = "from_config not yet implemented for Dynamics."
        raise NotImplementedError(msg)

    def __call__(self, x: ca.MX | np.ndarray, u: ca.MX | np.ndarray) -> ca.MX | np.ndarray:
        """Evaluate the dynamics."""
        return self.f(x, u)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Continuous-time dynamics x_dot = f(t, x, u)."""
        return np.asarray(self.f(x, u)).flatten()

    def step(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, DynamicsLog]:
        """Execute the public step method to be called by the orchestrator."""
        return self._execute_zoh(t, self.update, u)

    def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, DynamicsLog]:
        """
        Advance the dynamics by one time step.

        Args:
            t: Simulation time.
            u: Control input vector.
        """
        u_vec = self.to_col_vec(u).flatten()

        if self.integrator is not None:
            # Integrator expects (f, t, dt, x, u)
            self.x = self.integrator(self.dynamics, t, self.dt, self.x, u_vec)
        else:
            # Assume discrete dynamics if no integrator
            self.x = np.asarray(self.f(self.x, u_vec)).flatten()

        return self.from_col_vec(self.x), DynamicsLog(x=self.x.copy())

    def validate_dimensions(self, nx: int, nu: int) -> None:
        """Validate dimensions of the casadi function."""
        if self.f.size_in(0)[0] != nx:
            msg = f"Dynamics state input size ({self.f.size_in(0)[0]}) must match state size ({nx})."
            raise ValueError(msg)

        if self.f.size_in(1)[1] if self.f.size_in(1)[0] == 0 else self.f.size_in(1)[0] != nu:
            # Casadi function dimensions can be tricky
            pass
        # simpler check
        if self.f.size_in(1)[0] != nu:
            msg = f"Dynamics control input size ({self.f.size_in(1)[0]}) must match control size ({nu})."
            raise ValueError(msg)

        if self.f.size_out(0)[0] != nx:
            msg = f"Dynamics output state size ({self.f.size_out(0)[0]}) must match state size ({nx})."
            raise ValueError(msg)


class LinearDynamics(Dynamics):
    """Linear dynamics constraint: x_{k+1} = A x_k + B u_k."""

    def __init__(self, A: np.ndarray, B: np.ndarray, dt: float = 1.0, integrator: Integrator | None = None) -> None:
        """
        Initialize linear dynamics.

        Args:
            A: State transition matrix.
            B: Control input matrix.
            dt: Simulation time step.
            integrator: Optional integrator.
        """
        A = np.asarray(A)
        B = np.asarray(B)

        # If time-varying (3D), we take the first step's dimensions for the casadi function
        if A.ndim == 3:
            A_k = A[0]
            B_k = B[0]
        else:
            A_k = A
            B_k = B

        nx = A_k.shape[1]
        nu = B_k.shape[1]

        if A_k.shape[0] != nx:
            msg = "Matrix A must be square."
            raise ValueError(msg)
        if B_k.shape[0] != nx:
            msg = "Matrix B must have the same number of rows as A."
            raise ValueError(msg)

        self.A = A
        self.B = B

        x = ca.MX.sym("x", nx)
        u = ca.MX.sym("u", nu)

        # For casadi function, we use the 2D A_k and B_k
        f = ca.Function("lin_dyn", [x, u], [A_k @ x + B_k @ u], ["x", "u"], ["f"])
        super().__init__(f, dt, integrator)
