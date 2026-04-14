import casadi as ca
import numpy as np


class Dynamics:
    """Base class for dynamics."""

    def __init__(self, f: ca.Function) -> None:
        self.f = f

    def __call__(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Evaluate the dynamics."""
        return self.f(x, u)

    def validate_dimensions(self, nx: int, nu: int) -> None:
        """Validate dimensions of the casadi function."""
        if self.f.size_in(0)[0] != nx:
            msg = f"Dynamics state input size ({self.f.size_in(0)[0]}) must match state size ({nx})."
            raise ValueError(msg)

        if self.f.size_in(1)[0] != nu:
            msg = f"Dynamics control input size ({self.f.size_in(1)[0]}) must match control size ({nu})."
            raise ValueError(msg)

        if self.f.size_out(0)[0] != nx:
            msg = f"Dynamics output state size ({self.f.size_out(0)[0]}) must match state size ({nx})."
            raise ValueError(msg)


class LinearDynamics(Dynamics):
    """Linear dynamics constraint: x_{k+1} = A x_k + B u_k."""

    def __init__(self, A: np.ndarray, B: np.ndarray) -> None:
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

        # For casadi function, we use the 2D A_k and B_k (if time-varying, it's just a placeholder or we can't create a simple single-step function, but LinearOCP handles time-varying QPs directly anyway)
        f = ca.Function("lin_dyn", [x, u], [A_k @ x + B_k @ u], ["x", "u"], ["f"])
        super().__init__(f)
