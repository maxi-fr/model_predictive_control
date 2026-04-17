from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_predictive_control.dynamics import LinearDynamics
from model_predictive_control.mpc import LinearMPC
from model_predictive_control.ocp import LinearOCP
from model_predictive_control.simulation import SimulationResult, experiment, simulate


@pytest.fixture
def simple_mpc_setup() -> tuple[LinearMPC, LinearDynamics, int, int, int]:
    nx = 2
    nu = 1
    N = 8
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
    num_steps = int(2 * _N)

    x_ref = np.zeros(nx)
    u_ref = np.zeros(_nu)

    res = simulate(mpc, dynamics, x0, num_steps, x_ref, u_ref)

    assert res.X.shape == (num_steps, _N + 1, nx)
    assert res.U.shape == (num_steps, _N, _nu)
    assert res.cost.shape == (num_steps,)
    assert res.stage_cost.shape == (num_steps,)
    assert res.solve_time.shape == (num_steps,)
    assert len(res.status) == num_steps

    assert np.allclose(res.X[0, 0], x0)
    assert res.status[0] in ["solve_succeeded", "optimal", "success"]


def test_simulate_long_reference(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int]) -> None:
    mpc, dynamics, nx, _nu, _N = simple_mpc_setup
    x0 = np.array([1.0, 0.0])
    num_steps = int(2 * _N)

    x_ref = np.zeros((num_steps + _N + 2, nx))  # Extra long
    u_ref = np.zeros((num_steps + _N, _nu))

    res = simulate(mpc, dynamics, x0, num_steps, x_ref, u_ref)

    assert res.X.shape == (num_steps, _N + 1, nx)
    assert res.U.shape == (num_steps, _N, _nu)


def test_simulation_result_save_load(tmp_path: Path) -> None:
    res = SimulationResult(
        X=np.zeros((9, 5, 2)),
        U=np.zeros((9, 4, 1)),
        cost=np.zeros(9),
        stage_cost=np.zeros(9),
        status=["optimal"] * 9,
        solve_time=np.zeros(9),
    )

    file_path = tmp_path / "test_result"
    res.save(file_path)

    loaded = SimulationResult.load(file_path)

    assert type(loaded.status) is list
    assert loaded.status == res.status
    np.testing.assert_array_equal(loaded.X, res.X)
    np.testing.assert_array_equal(loaded.U, res.U)
    np.testing.assert_array_equal(loaded.stage_cost, res.stage_cost)


def test_experiment_batch(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int], tmp_path: Path) -> None:
    mpc, dynamics, _nx, _nu, _N = simple_mpc_setup
    x0_list = [np.array([1.0, 0.0]), np.array([-1.0, 0.5])]
    num_steps = 3

    save_dir = tmp_path / "test_experiment"
    df = experiment(mpc, dynamics, x0_list, num_steps, save_dir=save_dir)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "run_id" in df.columns
    assert "total_stage_cost" in df.columns

    # Verify that files were saved
    assert (save_dir / "run_0.npz").exists()
    assert (save_dir / "run_1.npz").exists()


def test_experiment_batch_with_ref_list(
    simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int], tmp_path: Path
) -> None:
    mpc, dynamics, _nx, _nu, _N = simple_mpc_setup
    x0_list = [np.array([1.0, 0.0]), np.array([-1.0, 0.5])]
    num_steps = 3

    x_ref_1 = np.array([0.0, 0.0])
    x_ref_2 = np.array([1.0, 1.0])
    x_ref_list = [x_ref_1, x_ref_2]

    save_dir = tmp_path / "test_experiment_ref"
    df = experiment(mpc, dynamics, x0_list, num_steps, x_ref=x_ref_list, save_dir=save_dir)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_simulate_invalid_x0(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int]) -> None:
    mpc, dynamics, nx, _nu, _N = simple_mpc_setup
    # Wrong dimension for x0
    x0 = np.zeros(nx + 1)
    num_steps = 5

    with pytest.raises(ValueError, match=f"x0 must have length {nx}"):
        simulate(mpc, dynamics, x0, num_steps)


def test_simulate_invalid_long_x_ref(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int]) -> None:
    mpc, dynamics, nx, _nu, _N = simple_mpc_setup
    x0 = np.zeros(nx)
    num_steps = 5

    # Correct length but wrong inner dimension
    x_ref = np.zeros((num_steps + _N, nx + 1))

    with pytest.raises(ValueError, match=rf"Long x_ref must have shape \(>=num_steps\+N, {nx}\)"):
        simulate(mpc, dynamics, x0, num_steps, x_ref=x_ref)


def test_simulate_invalid_long_u_ref(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int]) -> None:
    mpc, dynamics, nx, nu, _N = simple_mpc_setup
    x0 = np.zeros(nx)
    num_steps = 5

    # Correct length but wrong inner dimension
    u_ref = np.zeros((num_steps + _N - 1, nu + 1))

    with pytest.raises(ValueError, match=rf"Long u_ref must have shape \(>=num_steps\+N-1, {nu}\)"):
        simulate(mpc, dynamics, x0, num_steps, u_ref=u_ref)
