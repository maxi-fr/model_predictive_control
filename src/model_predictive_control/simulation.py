# ruff: noqa: PLR0913
import datetime
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import casadi as ca
import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import ArrayLike

from .controller import Controller
from .dynamics import Dynamics


@dataclass
class SimulationResult:
    """Dataclass to store the results of an MPC simulation."""

    X: npt.NDArray[np.float64]
    """Predicted state trajectories from MPC, shape: (num_steps, N + 1, nx)"""
    U: npt.NDArray[np.float64]
    """Predicted control trajectories from MPC, shape: (num_steps, N, nu)"""
    cost: npt.NDArray[np.float64]
    """Total predicted cost over the horizon at each step, shape: (num_steps,)"""
    stage_cost: npt.NDArray[np.float64]
    """One-step running stage cost at each step, shape: (num_steps,)"""
    status: list[str]
    """Solver status at each step, length: num_steps"""
    solve_time: npt.NDArray[np.float64]
    """Computation time at each step in seconds, shape: (num_steps,)"""

    def save(self, filepath: str | Path) -> None:
        """
        Save the simulation results to a compressed numpy archive (.npz).

        Args:
            filepath: Path to save the file to. The '.npz' extension will be added if missing.
        """
        path = Path(filepath)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """
        Load simulation results from a compressed numpy archive (.npz).

        Args:
            filepath: Path to the .npz file.

        Returns
        -------
            A new SimulationResult instance.
        """
        path = Path(filepath)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")

        with np.load(path) as data:
            kwargs = {}
            for key in data.files:
                val = data[key]
                if key == "status":
                    kwargs[key] = val.tolist()
                else:
                    kwargs[key] = val

            return cls(**kwargs)


def simulate(
    controller: Controller,
    dynamics: Dynamics | Callable[[ca.MX | np.ndarray, ca.MX | np.ndarray], ca.MX | np.ndarray],
    x0: ArrayLike,
    num_steps: int,
    x_ref: ArrayLike | None = None,
    u_ref: ArrayLike | None = None,
) -> SimulationResult:
    """
    Run an MPC simulation for a single initial condition over a specified number of steps.

    Args:
        controller: The Controller object to use.
        dynamics: The real dynamics to simulate. Can be a Dynamics object or a callable f(x, u).
        x0: Initial state array.
        num_steps: Number of simulation steps to run.
        x_ref: Optional state reference trajectory.
        u_ref: Optional control reference trajectory.

    Returns
    -------
        SimulationResult containing logged trajectories and metrics.
    """
    nx = controller.nx
    nu = controller.nu

    x0_arr = np.asarray(x0, dtype=float).flatten()
    if x0_arr.shape != (nx,):
        msg = f"x0 must have length {nx}"
        raise ValueError(msg)

    # Initialize arrays
    X = np.zeros((num_steps + 1, nx))
    U = np.zeros((num_steps, nu))

    # We construct X_pred and U_pred lists and stack them later if applicable.
    X_pred_list = []
    U_pred_list = []
    cost = np.zeros(num_steps)
    stage_cost = np.zeros(num_steps)
    status_list: list[str] = []
    solve_time = np.zeros(num_steps)

    X[0] = x0_arr
    x_current = x0_arr

    x_ref_arr = np.asarray(x_ref, dtype=float) if x_ref is not None else None
    u_ref_arr = np.asarray(u_ref, dtype=float) if u_ref is not None else None

    for k in range(num_steps):
        start_time = time.perf_counter()
        u_opt, status = controller.step(x_current, x_ref=x_ref_arr, u_ref=u_ref_arr, k=k)
        end_time = time.perf_counter()

        # Extract predictions
        X_opt, U_opt = controller.get_last_open_loop_predictions()
        if X_opt is None:
            X_pred_list.append(np.full((1, nx), np.nan))
        else:
            X_pred_list.append(X_opt)

        if U_opt is None:
            U_pred_list.append(np.full((1, nu), np.nan))
        else:
            U_pred_list.append(U_opt)

        # Calculate costs
        traj_cost, s_cost = controller.get_costs()

        # Log
        U[k] = u_opt
        cost[k] = traj_cost
        stage_cost[k] = s_cost
        status_list.append(status)
        solve_time[k] = end_time - start_time

        x_next = np.asarray(dynamics(x_current, u_opt)).flatten()

        x_current = x_next
        X[k + 1] = x_current

    try:
        X_pred_arr = np.stack(X_pred_list)
        U_pred_arr = np.stack(U_pred_list)
    except ValueError:
        # If sizes vary (which shouldn't happen usually but just in case)
        X_pred_arr = np.array(X_pred_list, dtype=object)
        U_pred_arr = np.array(U_pred_list, dtype=object)

    return SimulationResult(
        X_pred_arr,
        U_pred_arr,
        cost,
        stage_cost,
        status_list,
        solve_time,
    )


def experiment(
    controller: Controller,
    dynamics: Dynamics | Callable[[ca.MX | np.ndarray, ca.MX | np.ndarray], ca.MX | np.ndarray],
    x0_list: list[npt.NDArray[np.float64]],
    num_steps: int,
    x_ref: ArrayLike | list[npt.NDArray[np.float64]] | None = None,
    u_ref: ArrayLike | list[npt.NDArray[np.float64]] | None = None,
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run an MPC simulation for a batch of initial conditions.

    Args:
        controller: The Controller object to use for control.
        dynamics: The real dynamics to simulate. Can be a Dynamics object or a callable f(x, u).
        x0_list: List of initial state arrays.
        num_steps: Number of simulation steps to run for each initial condition.
        x_ref: Optional state reference trajectory. Can be a single reference applied to all simulations,
            or a list of references, one for each initial condition.
        u_ref: Optional control reference trajectory. Can be a single reference applied to all simulations,
            or a list of references, one for each initial condition.
        save_dir: Directory where the results should be saved. If None, defaults to 'results/experiment_<datetime>'.

    Returns
    -------
        A pandas DataFrame containing summary metrics for each initial condition.
    """
    if save_dir is None:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        save_dir = Path("results") / f"experiment_{timestamp}"
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = []

    is_list_x_ref = isinstance(x_ref, list) and len(x_ref) == len(x0_list)
    x_ref_list = ca.vertcat() if not isinstance(x_ref, list) else x_ref
    u_ref_list = ca.vertcat() if not isinstance(u_ref, list) else u_ref
    is_list_u_ref = isinstance(u_ref, list) and len(u_ref) == len(x0_list)

    # TODO: Extend this to support multiprocessing for parallel execution
    for i, x0 in enumerate(x0_list):
        curr_x_ref = x_ref_list[i] if is_list_x_ref else x_ref
        curr_u_ref = u_ref_list[i] if is_list_u_ref else u_ref

        res = simulate(
            controller=controller,
            dynamics=dynamics,
            x0=x0,
            num_steps=num_steps,
            x_ref=curr_x_ref,
            u_ref=curr_u_ref,
        )

        res.save(save_dir / f"run_{i}.npz")

        metrics.append(
            {
                "run_id": i,
                "total_stage_cost": float(np.sum(res.stage_cost)),
                "mean_solve_time": float(np.mean(res.solve_time)),
                "max_solve_time": float(np.max(res.solve_time)),
                "final_status": res.status[-1] if len(res.status) > 0 else "unknown",
                "all_optimal": all(s == "optimal" for s in res.status),
            }
        )

    return pd.DataFrame(metrics)
