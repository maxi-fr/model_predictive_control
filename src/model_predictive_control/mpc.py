from typing import Any

import numpy as np
from numpy._typing import ArrayLike

from model_predictive_control.ocp import OCP, LinearOCP


class MPC:
    def __init__(
        self,
        ocp: OCP,
        setup_args: dict[str, Any] | None = None,
        X_guess: ArrayLike | None = None,
        U_guess: ArrayLike | None = None,
    ) -> None:
        """
        Initializes the Model Predictive Control wrapper.

        Args:
            ocp: The Optimal Control Problem to solve.
            setup_args: Dictionary of arguments to pass to ocp.setup().
            X_guess: Optional initial guess for state trajectory of shape (N + 1, nx).
            U_guess: Optional initial guess for control trajectory of shape (N, nu).
        """
        self.ocp = ocp

        if setup_args is None:
            setup_args = {}
        self.ocp.setup(**setup_args)

        self.N = self.ocp.N
        self.nx = self.ocp._nx
        self.nu = self.ocp._nu

        if X_guess is not None:
            self._X_guess = np.asarray(X_guess, dtype=float)
            if self._X_guess.shape != (self.N + 1, self.nx):
                raise ValueError(f"X_guess must have shape ({self.N + 1}, {self.nx})")
        else:
            self._X_guess = np.zeros((self.N + 1, self.nx))

        if U_guess is not None:
            self._U_guess = np.asarray(U_guess, dtype=float)
            if self._U_guess.shape != (self.N, self.nu):
                raise ValueError(f"U_guess must have shape ({self.N}, {self.nu})")
        else:
            self._U_guess = np.zeros((self.N, self.nu))

    def step(self, x_current: ArrayLike, x_ref: ArrayLike | None = None, u_ref: ArrayLike | None = None) -> np.ndarray:
        """
        Solves the MPC problem for the current state and returns the control action.

        Args:
            x_current: Current state as a numpy array or list.
            x_ref: Optional time-varying state reference of shape (N + 1, nx) or constant reference of shape (nx,).
            u_ref: Optional time-varying control reference of shape (N, nu) or constant reference of shape (nu,).

        Returns:
            The control action to apply, as a numpy array of shape (nu,).
        """
        x_current_arr = np.asarray(x_current, dtype=float).flatten()
        if x_current_arr.shape != (self.nx,):
            raise ValueError(f"Current state must have length {self.nx}")

        X_opt, U_opt, status = self.ocp.solve(
            x0=x_current_arr, X_guess=self._X_guess, U_guess=self._U_guess, x_ref=x_ref, u_ref=u_ref
        )

        # Ipopt returns "Solve_Succeeded" on success, but different solvers might have different messages.
        # Check for success by looking for "success" or "succeeded" (case-insensitive) in the status string.
        # Note: sometimes IPOPT's failure message contains "return_success(accept_limit) failed", so
        # we also check if it starts with "solve_failed".
        # Fail loudly if solve fails.
        if "solve_failed" in status.lower() or (
            "success" not in status.lower() and "succeeded" not in status.lower() and "optimal" not in status.lower()
        ):
            raise RuntimeError(f"OCP solve failed with status: {status}")

        self.last_X_opt = X_opt
        self.last_U_opt = U_opt

        # Shift guesses for the next step
        self._X_guess = np.roll(X_opt, -1, axis=0)
        self._X_guess[-1, :] = self._X_guess[-2, :]

        self._U_guess = np.roll(U_opt, -1, axis=0)
        self._U_guess[-1, :] = self._U_guess[-2, :]

        return U_opt[0]


class LinearMPC:
    def __init__(
        self,
        linear_ocp: LinearOCP,
        setup_args: dict[str, Any] | None = None,
        X_guess: ArrayLike | None = None,
        U_guess: ArrayLike | None = None,
    ) -> None:
        """
        Initializes the Linear Model Predictive Control wrapper.

        Args:
            linear_ocp: The Linear Optimal Control Problem to solve.
            setup_args: Dictionary of arguments to pass to linear_ocp.setup().
            X_guess: Optional initial guess for state trajectory of shape (N + 1, nx).
            U_guess: Optional initial guess for control trajectory of shape (N, nu).
        """
        self.ocp = linear_ocp

        if setup_args is None:
            setup_args = {}
        self.ocp.setup(**setup_args)

        self.N = self.ocp.N
        self.nx = self.ocp.nx
        self.nu = self.ocp.nu

        if X_guess is not None:
            self._X_guess = np.asarray(X_guess, dtype=float)
            if self._X_guess.shape != (self.N + 1, self.nx):
                raise ValueError(f"X_guess must have shape ({self.N + 1}, {self.nx})")
        else:
            self._X_guess = np.zeros((self.N + 1, self.nx))

        if U_guess is not None:
            self._U_guess = np.asarray(U_guess, dtype=float)
            if self._U_guess.shape != (self.N, self.nu):
                raise ValueError(f"U_guess must have shape ({self.N}, {self.nu})")
        else:
            self._U_guess = np.zeros((self.N, self.nu))

    def step(self, x_current: ArrayLike, x_ref: ArrayLike | None = None, u_ref: ArrayLike | None = None) -> np.ndarray:
        """
        Solves the Linear MPC problem for the current state and returns the control action.

        Args:
            x_current: Current state as a numpy array or list.
            x_ref: Optional time-varying state reference of shape (N + 1, nx) or constant reference of shape (nx,).
            u_ref: Optional time-varying control reference of shape (N, nu) or constant reference of shape (nu,).

        Returns:
            The control action to apply, as a numpy array of shape (nu,).
        """
        x_current_arr = np.asarray(x_current, dtype=float).flatten()
        if x_current_arr.shape != (self.nx,):
            raise ValueError(f"Current state must have length {self.nx}")

        X_opt, U_opt, status = self.ocp.solve(
            x0=x_current_arr, X_guess=self._X_guess, U_guess=self._U_guess, x_ref=x_ref, u_ref=u_ref
        )

        if "solve_failed" in status.lower() or (
            "success" not in status.lower() and "succeeded" not in status.lower() and "optimal" not in status.lower()
        ):
            raise RuntimeError(f"LinearOCP solve failed with status: {status}")

        self.last_X_opt = X_opt
        self.last_U_opt = U_opt

        # Shift guesses for the next step
        self._X_guess = np.roll(X_opt, -1, axis=0)
        self._X_guess[-1, :] = self._X_guess[-2, :]

        self._U_guess = np.roll(U_opt, -1, axis=0)
        self._U_guess[-1, :] = self._U_guess[-2, :]

        return U_opt[0]
