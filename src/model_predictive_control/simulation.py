# ruff: noqa: PLR0912, PLR0913, PLR0915, C901, TRY003, EM102, EM101, SIM102, SIM108, SLF001
import time
from collections.abc import Callable
from dataclasses import dataclass

import casadi as ca
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from .dynamics import Dynamics
from .mpc import MPC, LinearMPC


@dataclass
class SimulationResult:
    """Dataclass to store the results of an MPC simulation."""

    X: npt.NDArray[np.float64]
    """Actual state trajectory, shape: (num_steps + 1, nx)"""
    U: npt.NDArray[np.float64]
    """Actual control trajectory applied, shape: (num_steps, nu)"""
    X_pred: npt.NDArray[np.float64]
    """Predicted state trajectories from MPC, shape: (num_steps, N + 1, nx)"""
    U_pred: npt.NDArray[np.float64]
    """Predicted control trajectories from MPC, shape: (num_steps, N, nu)"""
    cost: npt.NDArray[np.float64]
    """Total predicted cost over the horizon at each step, shape: (num_steps,)"""
    stage_cost: npt.NDArray[np.float64]
    """One-step running stage cost at each step, shape: (num_steps,)"""
    status: list[str]
    """Solver status at each step, length: num_steps"""
    solve_time: npt.NDArray[np.float64]
    """Computation time at each step in seconds, shape: (num_steps,)"""


def simulate(
    mpc: MPC | LinearMPC,
    dynamics: Dynamics | Callable[[ca.MX | np.ndarray, ca.MX | np.ndarray], ca.MX | np.ndarray],
    x0: ArrayLike,
    num_steps: int,
    x_ref: ArrayLike | None = None,
    u_ref: ArrayLike | None = None,
) -> SimulationResult:
    """
    Run an MPC simulation for a single initial condition over a specified number of steps.

    Args:
        mpc: The MPC or LinearMPC object to use for control.
        dynamics: The real dynamics to simulate. Can be a Dynamics object or a callable f(x, u).
        x0: Initial state array.
        num_steps: Number of simulation steps to run.
        x_ref: Optional state reference trajectory. Can be shape (nx,), (N+1, nx) for constant reference over
            horizon, or (num_steps + N, nx) for a long reference trajectory that will be sliced per step.
        u_ref: Optional control reference trajectory. Can be shape (nu,), (N, nu) for constant reference over
            horizon, or (num_steps + N - 1, nu) for a long reference trajectory that will be sliced per step.

    Returns
    -------
        SimulationResult containing logged trajectories and metrics.
    """
    nx = mpc.nx
    nu = mpc.nu
    N = mpc.N

    x0_arr = np.asarray(x0, dtype=float).flatten()
    if x0_arr.shape != (nx,):
        raise ValueError(f"x0 must have length {nx}")

    X = np.zeros((num_steps + 1, nx))
    U = np.zeros((num_steps, nu))
    X_pred = np.zeros((num_steps, N + 1, nx))
    U_pred = np.zeros((num_steps, N, nu))
    cost = np.zeros(num_steps)
    stage_cost = np.zeros(num_steps)
    status_list: list[str] = []
    solve_time = np.zeros(num_steps)

    X[0] = x0_arr
    x_current = x0_arr

    # Pre-process x_ref
    x_ref_arr = None
    long_x_ref = False
    if x_ref is not None:
        x_ref_arr = np.asarray(x_ref, dtype=float)
        if x_ref_arr.ndim == 2 and x_ref_arr.shape[0] >= num_steps + N:
            long_x_ref = True
            if x_ref_arr.shape[1] != nx:
                raise ValueError(f"Long x_ref must have shape (>=num_steps+N, {nx})")

    # Pre-process u_ref
    u_ref_arr = None
    long_u_ref = False
    if u_ref is not None:
        u_ref_arr = np.asarray(u_ref, dtype=float)
        if u_ref_arr.ndim == 2 and u_ref_arr.shape[0] >= num_steps + N - 1:
            long_u_ref = True
            if u_ref_arr.shape[1] != nu:
                raise ValueError(f"Long u_ref must have shape (>=num_steps+N-1, {nu})")

    # Extract stage cost function if possible
    stage_cost_func = None
    if hasattr(mpc.ocp, "objective") and mpc.ocp.objective is not None:
        if mpc.ocp.objective.stage_costs:
            stage_cost_func = mpc.ocp.objective.stage_costs[0]

    for k in range(num_steps):
        # Slice references if long
        xk_ref = None
        if x_ref_arr is not None:
            if long_x_ref:
                xk_ref = x_ref_arr[k : k + N + 1]
            else:
                xk_ref = x_ref_arr

        uk_ref = None
        if u_ref_arr is not None:
            if long_u_ref:
                uk_ref = u_ref_arr[k : k + N]
            else:
                uk_ref = u_ref_arr

        start_time = time.perf_counter()
        u_opt = mpc.step(x_current, x_ref=xk_ref, u_ref=uk_ref)
        end_time = time.perf_counter()

        # Extract predictions
        X_opt, U_opt = mpc.get_last_open_loop_predictions()
        if X_opt is None or U_opt is None:
            raise RuntimeError("MPC did not return open-loop predictions.")

        status = (
            mpc.ocp._solver_obj.stats()["return_status"]
            if hasattr(mpc.ocp, "_solver_obj") and mpc.ocp._solver_obj is not None
            else "unknown"
        )

        # Calculate costs
        traj_cost = mpc.ocp.calculate_trajectory_cost(X_opt, U_opt, xk_ref, uk_ref)

        s_cost = 0.0
        if stage_cost_func is not None:
            # Stage cost evaluation
            args = [x_current, u_opt]
            if stage_cost_func.has_reference:
                if xk_ref is not None:
                    # Get the reference for the current step k
                    x_ref_k0 = xk_ref[0] if np.asarray(xk_ref).ndim == 2 else xk_ref
                    args.append(x_ref_k0)
                else:
                    args.append(np.zeros(nx))

                if stage_cost_func.f.n_in() == 4:
                    if uk_ref is not None:
                        u_ref_k0 = uk_ref[0] if np.asarray(uk_ref).ndim == 2 else uk_ref
                        args.append(u_ref_k0)
                    else:
                        args.append(np.zeros(nu))

            s_cost = float(stage_cost_func(*args))

        # Log
        U[k] = u_opt
        X_pred[k] = X_opt
        U_pred[k] = U_opt
        cost[k] = traj_cost
        stage_cost[k] = s_cost
        status_list.append(status)
        solve_time[k] = end_time - start_time

        # Step dynamics
        if isinstance(dynamics, Dynamics):
            x_next_ca = dynamics(x_current, u_opt)
            x_next = np.array(x_next_ca).flatten()
        else:
            x_next = np.array(dynamics(x_current, u_opt)).flatten()

        x_current = x_next
        X[k + 1] = x_current

    return SimulationResult(
        X=X,
        U=U,
        X_pred=X_pred,
        U_pred=U_pred,
        cost=cost,
        stage_cost=stage_cost,
        status=status_list,
        solve_time=solve_time,
    )


def experiment(
    mpc: MPC | LinearMPC,
    dynamics: Dynamics | Callable[[ca.MX | np.ndarray, ca.MX | np.ndarray], ca.MX | np.ndarray],
    x0_list: list[npt.NDArray[np.float64]],
    num_steps: int,
    x_ref: ArrayLike | list[npt.NDArray[np.float64]] | None = None,
    u_ref: ArrayLike | list[npt.NDArray[np.float64]] | None = None,
) -> list[SimulationResult]:
    """
    Run an MPC simulation for a batch of initial conditions.

    Args:
        mpc: The MPC or LinearMPC object to use for control.
        dynamics: The real dynamics to simulate. Can be a Dynamics object or a callable f(x, u).
        x0_list: List of initial state arrays.
        num_steps: Number of simulation steps to run for each initial condition.
        x_ref: Optional state reference trajectory. Can be a single reference applied to all simulations,
            or a list of references, one for each initial condition.
        u_ref: Optional control reference trajectory. Can be a single reference applied to all simulations,
            or a list of references, one for each initial condition.

    Returns
    -------
        A list of SimulationResult objects, one for each initial condition.
    """
    results = []

    is_list_x_ref = isinstance(x_ref, list) and len(x_ref) == len(x0_list)
    x_ref_list = ca.vertcat() if not isinstance(x_ref, list) else x_ref
    u_ref_list = ca.vertcat() if not isinstance(u_ref, list) else u_ref
    is_list_u_ref = isinstance(u_ref, list) and len(u_ref) == len(x0_list)

    # TODO: Extend this to support multiprocessing for parallel execution
    for i, x0 in enumerate(x0_list):
        curr_x_ref = x_ref_list[i] if is_list_x_ref else x_ref
        curr_u_ref = u_ref_list[i] if is_list_u_ref else u_ref

        res = simulate(
            mpc=mpc,
            dynamics=dynamics,
            x0=x0,
            num_steps=num_steps,
            x_ref=curr_x_ref,
            u_ref=curr_u_ref,
        )
        results.append(res)

    return results
