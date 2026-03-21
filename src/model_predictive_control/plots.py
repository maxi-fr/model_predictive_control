from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_states(
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
    Plots the states of the system over time.

    Args:
        time (array-like): Time array.
        X (array-like): State array of shape (nx, N+1).
        indices (list, optional): List of indices of states to plot. Defaults to all.
        labels (list, optional): List of labels for the plotted states.
        fig (matplotlib.figure.Figure, optional): Figure to plot on.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        title (str, optional): Title of the plot.
        ylabel (str, optional): Y-axis label.
        bounds (list of tuples, optional): List of (min, max) bounds for the plotted states.

    Returns:
        tuple: (fig, ax)
    """
    X = np.asarray(X)
    nx = X.shape[0]

    if indices is None:
        indices = list(range(nx))

    if labels is None:
        labels = [f"$x_{i}$" for i in indices]

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    for i, idx in enumerate(indices):
        ax.plot(time, X[idx, :], label=labels[i])

        if bounds is not None and i < len(bounds) and bounds[i] is not None:
            min_val, max_val = bounds[i]  # type: ignore
            if min_val is not None:
                ax.axhline(min_val, color="red", linestyle=":", label="Min Bound" if i == 0 else "")
            if max_val is not None:
                ax.axhline(max_val, color="red", linestyle=":", label="Max Bound" if i == 0 else "")

    if title:
        ax.set_title(title)

    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    return fig, ax


def plot_controls(
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
    Plots the controls of the system over time.

    Args:
        time (array-like): Time array.
        U (array-like): Control array of shape (nu, N).
        indices (list, optional): List of indices of controls to plot. Defaults to all.
        labels (list, optional): List of labels for the plotted controls.
        fig (matplotlib.figure.Figure, optional): Figure to plot on.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        title (str, optional): Title of the plot.
        ylabel (str, optional): Y-axis label.
        bounds (list of tuples, optional): List of (min, max) bounds for the plotted controls.
        step (bool, optional): If True, plots using step function (where='post').

    Returns:
        tuple: (fig, ax)
    """
    U = np.asarray(U)
    nu = U.shape[0]
    time = np.asarray(time)

    if indices is None:
        indices = list(range(nu))

    if labels is None:
        labels = [f"$u_{i}$" for i in indices]

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Handle time array length mismatch
    # U is typically (nu, N) and time is (N+1,)
    plot_time = time[:-1] if len(time) == U.shape[1] + 1 else time

    for i, idx in enumerate(indices):
        if step:
            ax.step(plot_time, U[idx, :], label=labels[i], where="post")
        else:
            ax.plot(plot_time, U[idx, :], label=labels[i])

        if bounds is not None and i < len(bounds) and bounds[i] is not None:
            min_val, max_val = bounds[i]  # type: ignore
            if min_val is not None:
                ax.axhline(min_val, color="red", linestyle=":", label="Min Bound" if i == 0 else "")
            if max_val is not None:
                ax.axhline(max_val, color="red", linestyle=":", label="Max Bound" if i == 0 else "")

    if title:
        ax.set_title(title)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    return fig, ax
