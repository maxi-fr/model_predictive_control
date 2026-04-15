from typing import Any

import numpy as np
import pytest

from model_predictive_control.dynamics import LinearDynamics
from model_predictive_control.mpc import LinearMPC
from model_predictive_control.ocp import LinearOCP
from model_predictive_control.simulation import experiment, simulate


@pytest.fixture
def simple_mpc_setup() -> tuple[LinearMPC, LinearDynamics, int, int, int]:
    nx = 2
    nu = 1
    N = 5
    dt = 0.1

    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])
    dynamics = LinearDynamics(A, B)

    Q = np.eye(nx)
    R = np.eye(nu)
    Qf = np.eye(nx) * 10

    # signature: N, dt, dynamics, Q, R, ..., Qf
    ocp = LinearOCP(N=N, dt=dt, dynamics=dynamics, Q=Q, R=R, Qf=Qf)
    mpc = LinearMPC(ocp)
    return mpc, dynamics, nx, nu, N


def test_simulate_constant_reference(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int]) -> None:
    mpc, dynamics, nx, _nu, _N = simple_mpc_setup
    x0 = np.array([1.0, 0.0])
    num_steps = 10

    x_ref = np.zeros(nx)
    u_ref = np.zeros(_nu)

    res = simulate(mpc, dynamics, x0, num_steps, x_ref, u_ref)

    assert res.X.shape == (num_steps + 1, nx)
    assert res.U.shape == (num_steps, _nu)
    assert res.X_pred.shape == (num_steps, _N + 1, nx)
    assert res.U_pred.shape == (num_steps, _N, _nu)
    assert res.cost.shape == (num_steps,)
    assert res.stage_cost.shape == (num_steps,)
    assert res.solve_time.shape == (num_steps,)
    assert len(res.status) == num_steps

    assert np.allclose(res.X[0], x0)
    assert res.status[0] in ["solve_succeeded", "optimal", "success"]


def test_simulate_long_reference(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int]) -> None:
    mpc, dynamics, nx, _nu, _N = simple_mpc_setup
    x0 = np.array([1.0, 0.0])
    num_steps = 5

    x_ref = np.zeros((num_steps + _N + 2, nx))  # Extra long
    u_ref = np.zeros((num_steps + _N, _nu))

    res = simulate(mpc, dynamics, x0, num_steps, x_ref, u_ref)

    assert res.X.shape == (num_steps + 1, nx)
    assert res.U.shape == (num_steps, _nu)


def test_experiment_batch(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int]) -> None:
    mpc, dynamics, nx, _nu, _N = simple_mpc_setup
    x0_list = [np.array([1.0, 0.0]), np.array([-1.0, 0.5])]
    num_steps = 3

    results = experiment(mpc, dynamics, x0_list, num_steps)

    assert len(results) == 2
    assert results[0].X.shape == (num_steps + 1, nx)
    assert results[1].X.shape == (num_steps + 1, nx)
    assert np.allclose(results[0].X[0], x0_list[0])
    assert np.allclose(results[1].X[0], x0_list[1])


def test_experiment_batch_with_ref_list(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int]) -> None:
    mpc, dynamics, nx, _nu, _N = simple_mpc_setup
    x0_list = [np.array([1.0, 0.0]), np.array([-1.0, 0.5])]
    num_steps = 3

    x_ref_1 = np.array([0.0, 0.0])
    x_ref_2 = np.array([1.0, 1.0])
    x_ref_list = [x_ref_1, x_ref_2]

    results = experiment(mpc, dynamics, x0_list, num_steps, x_ref=x_ref_list)

    assert len(results) == 2
    assert results[0].X.shape == (num_steps + 1, nx)
    assert results[1].X.shape == (num_steps + 1, nx)
