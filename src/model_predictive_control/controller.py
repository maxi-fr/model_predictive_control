import abc
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike


class Controller(abc.ABC):
    """Abstract base class for controllers."""

    @property
    @abc.abstractmethod
    def nx(self) -> int:
        """Number of states."""

    @property
    @abc.abstractmethod
    def nu(self) -> int:
        """Number of controls."""

    @abc.abstractmethod
    def step(
        self,
        x_current: ArrayLike,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
        k: int = 0,
    ) -> tuple[npt.NDArray[np.float64], str]:
        """
        Compute the control action for the current state.

        Args:
            x_current: Current state as an array-like.
            x_ref: Optional state reference trajectory (can be the full trajectory or a single point).
            u_ref: Optional control reference trajectory (can be the full trajectory or a single point).
            k: Current time step index (used for slicing full reference trajectories).

        Returns
        -------
            A tuple (u_opt, status) where:
            - u_opt: The control action to apply, as a numpy array of shape (nu,).
            - status: A status string indicating success or failure.
        """

    def get_last_open_loop_predictions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return the open-loop state and control trajectory predictions from the last solve.

        Returns
        -------
            Tuple of (X_opt, U_opt). Returns (None, None) for non-predictive controllers.
        """
        return None, None

    def get_costs(self, **_kwargs: dict[str, Any]) -> tuple[float, float]:
        """
        Return the total predicted cost and the one-step running stage cost from the last solve.

        Returns
        -------
            Tuple of (total_trajectory_cost, stage_cost). Returns (np.nan, np.nan) if costs are not computed.
        """
        return np.nan, np.nan
