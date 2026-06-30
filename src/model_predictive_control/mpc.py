from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict
from simulate.controller import Controller

from model_predictive_control.ocp import OCP, LinearOCP


class MPCLog(BaseModel):
    """Internal log for MPC execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X_opt: np.ndarray
    U_opt: np.ndarray
    status: str
    traj_cost: float
    stage_cost: float


class MPC(Controller[MPCLog]):
    """
    Wrapper for executing nonlinear Model Predictive Control (MPC).

    Compatible with the simulate framework.
    """

    def __init__(
        self,
        ocp: OCP | LinearOCP,
        dt: float | None = None,
        setup_args: dict[str, Any] | None = None,
        X_guess: ArrayLike | None = None,
        U_guess: ArrayLike | None = None,
    ) -> None:
        """
        Initialize the Model Predictive Control wrapper.

        Args:
            ocp: The Optimal Control Problem or Linear Optimal Control Problem to solve.
            dt: Control sampling time. If None, uses ocp.dt.
            setup_args: Dictionary of arguments to pass to ocp.setup().
            X_guess: Optional initial guess for state trajectory of shape (N + 1, nx).
            U_guess: Optional initial guess for control trajectory of shape (N, nu).
        """
        if dt is None:
            dt = ocp.dt
        super().__init__(dt)
        self.ocp = ocp

        if setup_args is None:
            setup_args = {}
        self.ocp.setup(**setup_args)

        self.N = self.ocp.N
        self.nx, self.nu = self.ocp.validate_dimensions()

        if X_guess is not None:
            self._X_guess = np.asarray(X_guess, dtype=float)
            if self._X_guess.shape != (self.N + 1, self.nx):
                msg = f"X_guess must have shape ({self.N + 1}, {self.nx})"
                raise ValueError(msg)
        else:
            self._X_guess = np.zeros((self.N + 1, self.nx))

        if U_guess is not None:
            self._U_guess = np.asarray(U_guess, dtype=float)
            if self._U_guess.shape != (self.N, self.nu):
                msg = f"U_guess must have shape ({self.N}, {self.nu})"
                raise ValueError(msg)
        else:
            self._U_guess = np.zeros((self.N, self.nu))

        self.last_X_opt: np.ndarray | None = None
        self.last_U_opt: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MPC":
        """Instantiate from config (not fully implemented for OCP objects)."""
        msg = "from_config not yet implemented for MPC."
        raise NotImplementedError(msg)

    def get_last_open_loop_predictions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return the open-loop state and control trajectory predictions from the last solve.

        Returns
        -------
            Tuple of (X_opt, U_opt), where X_opt has shape (N + 1, nx) and U_opt has shape (N, nu).
            Returns (None, None) if step() has not been called yet.
        """
        return self.last_X_opt, self.last_U_opt

    def step(
        self,
        t: float,
        ref: float | np.ndarray | tuple[np.ndarray, np.ndarray] | None,
        x_hat: float | np.ndarray,
    ) -> tuple[np.ndarray, MPCLog]:
        """Execute the public step method to be called by the orchestrator."""
        res, log = self._execute_zoh(t, self.update, ref, x_hat)
        return np.atleast_1d(res), log

    def update(
        self,
        t: float,
        ref: float | np.ndarray | tuple[np.ndarray, np.ndarray] | None,
        x_hat: float | np.ndarray,
    ) -> tuple[np.ndarray, MPCLog]:
        """
        Compute control action based on reference and measurement.

        Args:
            t: Simulation time.
            ref: Reference signal (state reference or tuple of (state_ref, control_ref)).
            x_hat: Estimated state vector.
        """
        # Handle scalar zero from simulate.Simulation.run() initial state
        x_hat_vec = self.to_col_vec(x_hat)
        if x_hat_vec.size != self.nx:
            if np.all(x_hat_vec == 0):
                x_current = np.zeros(self.nx)
            else:
                msg = f"Estimated state size ({x_hat_vec.size}) does not match MPC state size ({self.nx})."
                raise ValueError(msg)
        else:
            x_current = x_hat_vec.flatten()

        # Handle reference: either state reference or (state_ref, control_ref)
        x_ref: np.ndarray | None = None
        u_ref: np.ndarray | None = None
        if isinstance(ref, tuple) and len(ref) == 2:
            x_ref, u_ref = ref
        elif isinstance(ref, float | int | np.floating | np.integer):
            x_ref = np.atleast_1d(float(ref))
        elif isinstance(ref, np.ndarray):
            x_ref = ref

        X_opt, U_opt, status = self.ocp.solve(
            x0=x_current,
            X_guess=self._X_guess,
            U_guess=self._U_guess,
            x_ref=x_ref,
            u_ref=u_ref,
        )

        if "solve_failed" in status.lower() or (
            "success" not in status.lower()
            and "succeeded" not in status.lower()
            and "optimal" not in status.lower()
            and "solved" not in status.lower()
        ):
            msg = f"{self.ocp.__class__.__name__} solve failed at t={t} with status: {status}"
            raise RuntimeError(msg)

        self.last_X_opt = X_opt
        self.last_U_opt = U_opt

        # Shift guesses for the next step in-place to avoid allocation
        self._X_guess[:-1, :] = X_opt[1:, :]
        self._X_guess[-1, :] = X_opt[-1, :]

        self._U_guess[:-1, :] = U_opt[1:, :]
        self._U_guess[-1, :] = U_opt[-1, :]

        u_opt_0 = U_opt[0]

        # Calculate costs for logging
        traj_cost = self.ocp.calculate_trajectory_cost(X_opt, U_opt, x_ref, u_ref)
        # Simplified stage cost calc for log
        stage_cost = 0.0

        log = MPCLog(
            X_opt=X_opt,
            U_opt=U_opt,
            status=status,
            traj_cost=float(traj_cost),
            stage_cost=float(stage_cost),
        )

        return np.atleast_1d(u_opt_0), log


class LinearMPC(MPC):
    """
    Wrapper for executing linear Model Predictive Control (MPC).

    Compatible with the simulate framework.
    """

    def __init__(
        self,
        linear_ocp: LinearOCP,
        dt: float | None = None,
        setup_args: dict[str, Any] | None = None,
        X_guess: ArrayLike | None = None,
        U_guess: ArrayLike | None = None,
    ) -> None:
        """
        Initialize the Linear Model Predictive Control wrapper.

        Args:
            linear_ocp: The Linear Optimal Control Problem to solve.
            dt: Control sampling time. If None, uses linear_ocp.dt.
            setup_args: Dictionary of arguments to pass to linear_ocp.setup().
            X_guess: Optional initial guess for state trajectory of shape (N + 1, nx).
            U_guess: Optional initial guess for control trajectory of shape (N, nu).
        """
        super().__init__(ocp=linear_ocp, dt=dt, setup_args=setup_args, X_guess=X_guess, U_guess=U_guess)
