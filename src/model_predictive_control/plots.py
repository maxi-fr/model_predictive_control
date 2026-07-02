from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_states(  # noqa: PLR0913
    time: ArrayLike,
    X: ArrayLike,
    indices: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str = "States",
    bounds: Sequence[tuple[float | None, float | None] | None] | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the states of the system over time.

    Args:
        time (array-like): Time array.
        X (array-like): State array of shape (N+1, nx).
        indices (list, optional): List of indices of states to plot. Defaults to all.
        labels (list, optional): List of labels for the plotted states.
        fig (matplotlib.figure.Figure, optional): Figure to plot on.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        title (str, optional): Title of the plot.
        ylabel (str, optional): Y-axis label.
        bounds (list of tuples, optional): List of (min, max) bounds for the plotted states.

    Returns
    -------
        tuple: (fig, ax)
    """
    X = np.asarray(X)
    nx = X.shape[1] if X.ndim == 2 else 1

    if indices is None:
        indices = list(range(nx))

    if labels is None:
        labels = [f"$x_{i}$" for i in indices]

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    for i, idx in enumerate(indices):
        ax.plot(time, X[:, idx] if X.ndim == 2 else X, label=labels[i])

        if bounds is not None and i < len(bounds):
            bound = bounds[i]
            if bound is not None:
                min_val, max_val = bound
                if min_val is not None:
                    ax.axhline(min_val, color="red", linestyle=":", label="Min Bound" if i == 0 else "")
                if max_val is not None:
                    ax.axhline(max_val, color="red", linestyle=":", label="Max Bound" if i == 0 else "")

    if title:
        ax.set_title(title)

    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(visible=True)

    return fig, ax


def plot_mpc_trajectories(  # noqa: PLR0913, C901
    time: ArrayLike,
    X_closed_loop: ArrayLike,
    X_open_loop: ArrayLike,
    indices: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str = "States",
    bounds: Sequence[tuple[float | None, float | None] | None] | None = None,
    step_interval: int = 1,
) -> tuple[Figure, Axes]:
    """
    Plot the closed-loop state trajectories along with the open-loop predictions from MPC.

    Args:
        time (array-like): Time array of length (N_sim + 1).
        X_closed_loop (array-like): Closed-loop state array of shape (N_sim + 1, nx).
        X_open_loop (array-like): 3D array of open-loop predictions of shape (N_sim, N_horizon + 1, nx).
            X_open_loop[k, :, :] is the prediction made at time step k.
        indices (list, optional): List of indices of states to plot. Defaults to all.
        labels (list, optional): List of labels for the plotted states.
        fig (matplotlib.figure.Figure, optional): Figure to plot on.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        title (str, optional): Title of the plot.
        ylabel (str, optional): Y-axis label.
        bounds (list of tuples, optional): List of (min, max) bounds for the plotted states.
        step_interval (int, optional):  Interval of prediction horizons to plot
                                        (e.g., plot every 5th prediction to avoid clutter).

    Returns
    -------
        tuple: (fig, ax)
    """
    X_closed_loop = np.asarray(X_closed_loop)
    X_open_loop = np.asarray(X_open_loop)
    time = np.asarray(time)

    nx = X_closed_loop.shape[1] if X_closed_loop.ndim == 2 else 1

    N_sim = X_open_loop.shape[0]
    N_horizon = X_open_loop.shape[1] - 1

    # Estimate dt assuming uniform time steps
    dt = time[1] - time[0] if len(time) > 1 else 1.0

    if indices is None:
        indices = list(range(nx))

    if labels is None:
        labels = [f"$x_{i}$" for i in indices]

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Create a unified color cycle for states
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    relative_time = np.arange(N_horizon + 1) * dt

    for i, idx in enumerate(indices):
        color = colors[i % len(colors)]

        # Plot open-loop predictions first (so they are visually behind the closed-loop line)
        for k in range(0, N_sim, step_interval):
            # The prediction at step k corresponds to time range [time[k], time[k] + N_horizon * dt]
            pred_time = time[k] + relative_time

            # Plot the prediction trace. Add label only on the first iteration to avoid legend duplication
            pred_trace = X_open_loop[k, :, idx] if X_open_loop.ndim == 3 else X_open_loop[k, :]
            ax.plot(
                pred_time,
                pred_trace,
                color=color,
                alpha=0.3,
                linestyle="--",
                label=f"{labels[i]} (predictions)" if k == 0 else "",
            )

        # Plot closed-loop trajectory
        cl_trace = X_closed_loop[:, idx] if X_closed_loop.ndim == 2 else X_closed_loop
        ax.plot(time, cl_trace, color=color, linewidth=2, label=f"{labels[i]} (closed-loop)")

        # Plot bounds
        if bounds is not None and i < len(bounds):
            bound = bounds[i]
            if bound is not None:
                min_val, max_val = bound
                if min_val is not None:
                    ax.axhline(min_val, color=color, linestyle=":", label="Min Bound" if i == 0 else "")
                if max_val is not None:
                    ax.axhline(max_val, color=color, linestyle=":", label="Max Bound" if i == 0 else "")

    if title:
        ax.set_title(title)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(visible=True)

    return fig, ax


def plot_controls(  # noqa: PLR0913, C901
    time: ArrayLike,
    U: ArrayLike,
    indices: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str = "Control",
    bounds: Sequence[tuple[float | None, float | None] | None] | None = None,
    step: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot the controls of the system over time.

    Args:
        time (array-like): Time array.
        U (array-like): Control array of shape (N, nu).
        indices (list, optional): List of indices of controls to plot. Defaults to all.
        labels (list, optional): List of labels for the plotted controls.
        fig (matplotlib.figure.Figure, optional): Figure to plot on.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        title (str, optional): Title of the plot.
        ylabel (str, optional): Y-axis label.
        bounds (list of tuples, optional): List of (min, max) bounds for the plotted controls.
        step (bool, optional): If True, plots using step function (where='post').

    Returns
    -------
        tuple: (fig, ax)
    """
    U = np.asarray(U)
    nu = U.shape[1] if U.ndim == 2 else 1

    time = np.asarray(time)

    if indices is None:
        indices = list(range(nu))

    if labels is None:
        labels = [f"$u_{i}$" for i in indices]

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Handle time array length mismatch
    # U is typically (N, nu) and time is (N+1,)
    U_len = U.shape[0] if U.ndim == 2 else len(U)
    plot_time = time[:-1] if len(time) == U_len + 1 else time

    for i, idx in enumerate(indices):
        trace = U[:, idx] if U.ndim == 2 else U
        if step:
            ax.step(plot_time, trace, label=labels[i], where="post")
        else:
            ax.plot(plot_time, trace, label=labels[i])

        if bounds is not None and i < len(bounds):
            bound = bounds[i]
            if bound is not None:
                min_val, max_val = bound
                if min_val is not None:
                    ax.axhline(min_val, color="red", linestyle=":", label="Min Bound" if i == 0 else "")
                if max_val is not None:
                    ax.axhline(max_val, color="red", linestyle=":", label="Max Bound" if i == 0 else "")

    if title:
        ax.set_title(title)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(visible=True)

    return fig, ax
