from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from model_predictive_control.controller import Controller
from model_predictive_control.ocp import OCP, LinearOCP


class MPC(Controller):
    """
    Wrapper for executing nonlinear Model Predictive Control (MPC).

    This class provides a high-level interface for closed-loop MPC simulations by repeatedly
    solving an underlying nonlinear Optimal Control Problem (OCP) in a receding horizon fashion.
    It handles OCP setup, trajectory warm-starting, and dimension validation.
    """

    @property
    def nx(self) -> int:
        """Number of states."""
        return self.ocp.nx

    @property
    def nu(self) -> int:
        """Number of controls."""
        return self.ocp.nu

    def __init__(
        self,
        ocp: OCP | LinearOCP,
        setup_args: dict[str, Any] | None = None,
        X_guess: ArrayLike | None = None,
        U_guess: ArrayLike | None = None,
    ) -> None:
        """
        Initialize the Model Predictive Control wrapper.

        Args:
            ocp: The Optimal Control Problem or Linear Optimal Control Problem to solve.
            setup_args: Dictionary of arguments to pass to ocp.setup().
            X_guess: Optional initial guess for state trajectory of shape (N + 1, nx).
            U_guess: Optional initial guess for control trajectory of shape (N, nu).
        """
        self.ocp = ocp

        if setup_args is None:
            setup_args = {}
        self.ocp.setup(**setup_args)

        self.N = self.ocp.N
        self.ocp.validate_dimensions()

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
        x_current: ArrayLike,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
        k: int = 0,
    ) -> tuple[npt.NDArray[np.float64], str]:
        """
        Solves the MPC problem for the current state and returns the control action and solver status.

        Args:
            x_current: Current state as a numpy array or list.
            x_ref: Optional time-varying state reference of shape (N + 1, nx) or constant reference of shape (nx,).
            u_ref: Optional time-varying control reference of shape (N, nu) or constant reference of shape (nu,).
            k: Current time step index (used for slicing full reference trajectories).

        Returns
        -------
            A tuple (u_opt, status) where:
            - u_opt: The control action to apply, as a numpy array of shape (nu,).
            - status: The status string returned by the solver.
        """
        x_current_arr = np.asarray(x_current, dtype=float).flatten()
        if x_current_arr.shape != (self.nx,):
            msg = f"Current state must have length {self.nx}"
            raise ValueError(msg)

        x_ref_slice = None
        if x_ref is not None:
            x_ref_arr = np.asarray(x_ref, dtype=float)
            x_ref_slice = (
                x_ref_arr[k : k + self.N + 1] if x_ref_arr.ndim == 2 and x_ref_arr.shape[0] > self.N + 1 else x_ref_arr
            )

        u_ref_slice = None
        if u_ref is not None:
            u_ref_arr = np.asarray(u_ref, dtype=float)
            u_ref_slice = (
                u_ref_arr[k : k + self.N] if u_ref_arr.ndim == 2 and u_ref_arr.shape[0] > self.N else u_ref_arr
            )

        X_opt, U_opt, status = self.ocp.solve(
            x0=x_current_arr,
            X_guess=self._X_guess,
            U_guess=self._U_guess,
            x_ref=x_ref_slice,
            u_ref=u_ref_slice,
        )

        # Ipopt returns "Solve_Succeeded" on success, but different solvers might have different messages.
        # Check for success by looking for "success" or "succeeded" (case-insensitive) in the status string.
        # Note: sometimes IPOPT's failure message contains "return_success(accept_limit) failed", so
        # we also check if it starts with "solve_failed".
        # Fail loudly if solve fails.
        if "solve_failed" in status.lower() or (
            "success" not in status.lower()
            and "succeeded" not in status.lower()
            and "optimal" not in status.lower()
            and "solved" not in status.lower()
        ):
            msg = f"{self.ocp.__class__.__name__} solve failed with status: {status}"
            raise RuntimeError(msg)

        self.last_X_opt = X_opt
        self.last_U_opt = U_opt

        # Shift guesses for the next step
        self._X_guess = np.roll(X_opt, -1, axis=0)
        self._X_guess[-1, :] = self._X_guess[-2, :]

        self._U_guess = np.roll(U_opt, -1, axis=0)
        self._U_guess[-1, :] = self._U_guess[-2, :]

        u_opt_0: npt.NDArray[np.float64] = U_opt[0]

        # Save last current state and sliced references to calculate costs later if requested
        self._last_x_current = x_current_arr
        self._last_x_ref_slice = x_ref_slice
        self._last_u_ref_slice = u_ref_slice

        return u_opt_0, status

    def get_costs(self, **_kwargs: dict[str, Any]) -> tuple[float, float]:
        """
        Calculate the predicted trajectory cost and current stage cost based on the last solve.

        Returns
        -------
            Tuple of (total_trajectory_cost, stage_cost).
        """
        if self.last_X_opt is None or self.last_U_opt is None:
            return np.nan, np.nan

        traj_cost = self.ocp.calculate_trajectory_cost(
            self.last_X_opt, self.last_U_opt, self._last_x_ref_slice, self._last_u_ref_slice
        )

        s_cost = 0.0
        stage_cost_func = None
        if hasattr(self.ocp, "objective") and self.ocp.objective is not None and self.ocp.objective.stage_costs:
            stage_cost_func = self.ocp.objective.stage_costs[0]

        if stage_cost_func is not None:
            args = [self._last_x_current, self.last_U_opt[0]]
            if stage_cost_func.has_reference:
                if self._last_x_ref_slice is not None:
                    # Get the reference for the current step k
                    x_ref_k0 = (
                        self._last_x_ref_slice[0]
                        if np.asarray(self._last_x_ref_slice).ndim == 2
                        else self._last_x_ref_slice
                    )
                    args.append(x_ref_k0)
                else:
                    args.append(np.zeros(self.nx))

                if stage_cost_func.f.n_in() == 4:
                    if self._last_u_ref_slice is not None:
                        u_ref_k0 = (
                            self._last_u_ref_slice[0]
                            if np.asarray(self._last_u_ref_slice).ndim == 2
                            else self._last_u_ref_slice
                        )
                        args.append(u_ref_k0)
                    else:
                        args.append(np.zeros(self.nu))

            s_cost = float(stage_cost_func(*args))

        return float(traj_cost), s_cost


class LinearMPC(MPC):
    """
    Wrapper for executing linear Model Predictive Control (MPC).

    This class provides a high-level interface for closed-loop MPC simulations by repeatedly
    solving an underlying Linear Optimal Control Problem (LinearOCP) in a receding horizon fashion.
    It handles LinearOCP setup, trajectory warm-starting, and dimension validation.
    """

    def __init__(
        self,
        linear_ocp: LinearOCP,
        setup_args: dict[str, Any] | None = None,
        X_guess: ArrayLike | None = None,
        U_guess: ArrayLike | None = None,
    ) -> None:
        """
        Initialize the Linear Model Predictive Control wrapper.

        Args:
            linear_ocp: The Linear Optimal Control Problem to solve.
            setup_args: Dictionary of arguments to pass to linear_ocp.setup().
            X_guess: Optional initial guess for state trajectory of shape (N + 1, nx).
            U_guess: Optional initial guess for control trajectory of shape (N, nu).
        """
        super().__init__(ocp=linear_ocp, setup_args=setup_args, X_guess=X_guess, U_guess=U_guess)
