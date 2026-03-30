import numpy as np
import pytest

from model_predictive_control.ocp import (
    control_bounds_constraints,
    linear_constraints,
    linear_dynamics,
    state_bounds_constraints,
    terminal_linear_constraints,
    terminal_quadratic_objective,
)


def test_linear_constraints() -> None:
    F = np.array([[1, 2], [3, 4]])
    G = np.array([[5], [6]])
    h = np.array([[7], [8]])

    func = linear_constraints(F, G, h)
    assert func.size_in(0) == (2, 1)  # nx
    assert func.size_in(1) == (1, 1)  # nu
    assert func.size_out(0) == (2, 1)  # out

    # Mismatch tests
    with pytest.raises(ValueError):
        linear_constraints(F, G, np.array([[1]]))


def test_linear_dynamics() -> None:
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5], [6]])

    func = linear_dynamics(A, B)
    assert func.size_in(0) == (2, 1)  # nx
    assert func.size_in(1) == (1, 1)  # nu
    assert func.size_out(0) == (2, 1)  # out

    # Mismatch tests
    with pytest.raises(ValueError):
        linear_dynamics(np.array([[1, 2]]), B)


def test_state_bounds_constraints() -> None:
    x_min = np.array([-1, -1])
    x_max = np.array([1, 1])
    nu = 2

    func = state_bounds_constraints(x_min, x_max, nu)
    assert func.size_in(0) == (2, 1)  # nx
    assert func.size_in(1) == (2, 1)  # nu
    assert func.size_out(0) == (4, 1)  # out

    with pytest.raises(ValueError):
        state_bounds_constraints(x_min, np.array([1]), nu)


def test_control_bounds_constraints() -> None:
    u_min = np.array([-1])
    u_max = np.array([1])
    nx = 2

    func = control_bounds_constraints(u_min, u_max, nx)
    assert func.size_in(0) == (2, 1)  # nx
    assert func.size_in(1) == (1, 1)  # nu
    assert func.size_out(0) == (2, 1)  # out

    with pytest.raises(ValueError):
        control_bounds_constraints(u_min, np.array([1, 2]), nx)


def test_terminal_quadratic_objective() -> None:
    Q = np.array([[1, 0], [0, 1]])
    q = np.array([[0], [0]])

    func = terminal_quadratic_objective(Q, q)
    assert func.size_in(0) == (2, 1)  # nx
    assert func.size_out(0) == (1, 1)  # out

    with pytest.raises(ValueError):
        terminal_quadratic_objective(np.array([[1]]), q)


def test_terminal_linear_constraints() -> None:
    F = np.array([[1, 2], [3, 4]])
    h = np.array([[5], [6]])

    func = terminal_linear_constraints(F, h)
    assert func.size_in(0) == (2, 1)  # nx
    assert func.size_out(0) == (2, 1)  # out

    with pytest.raises(ValueError):
        terminal_linear_constraints(F, np.array([[1]]))
