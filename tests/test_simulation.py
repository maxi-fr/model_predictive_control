import numpy as np
import pandas as pd
import pytest
from simulate.estimator import IdentityEstimator
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor
from simulate.simulation import Simulation

from model_predictive_control.dynamics import LinearDynamics
from model_predictive_control.mpc import LinearMPC
from model_predictive_control.ocp import LinearOCP


@pytest.fixture
def simple_mpc_setup() -> tuple[LinearMPC, LinearDynamics, int, int, int, float]:
    nx = 2
    nu = 1
    N = 8
    dt = 0.1

    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.005], [0.1]])
    dynamics = LinearDynamics(A, B, dt=dt)

    Q = np.eye(nx)
    R = np.eye(nu)
    Qf = np.eye(nx) * 10

    ocp = LinearOCP(N=N, dt=dt, dynamics=dynamics, Q=Q, R=R, Qf=Qf)
    mpc = LinearMPC(ocp, dt=dt)
    return mpc, dynamics, nx, nu, N, dt


def test_simulation_run(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int, float]) -> None:
    mpc, dynamics, nx, _nu, _N, dt = simple_mpc_setup
    x0 = np.array([1.0, 0.0])
    dynamics.x = x0  # Set initial state

    t_end = 1.0

    ref = StepReference(dt=dt, step_value=np.zeros(nx))
    sensor = GaussianSensor(dt=dt, std_dev=0.0)
    estimator = IdentityEstimator(dt=dt)

    sim = Simulation(t_end=t_end, plant=dynamics, reference=ref, sensor=sensor, estimator=estimator, controller=mpc)

    sim.run()

    # Check that logging worked
    results = pd.DataFrame(sim.logger.universal_logs)
    assert len(results) > 0
    assert "t" in results.columns
    assert "u" in results.columns

    # Final state should be close to zero because of MPC
    final_y = sim.logger.universal_logs[-1]["y"]
    assert np.linalg.norm(final_y) < np.linalg.norm(x0)


def test_simulate_invalid_dt_multiple(simple_mpc_setup: tuple[LinearMPC, LinearDynamics, int, int, int, float]) -> None:
    mpc, dynamics, nx, _nu, _N, dt = simple_mpc_setup

    # Reference with incompatible dt
    ref = StepReference(dt=dt * 1.5, step_value=np.zeros(nx))
    sensor = GaussianSensor(dt=dt, std_dev=0.0)
    estimator = IdentityEstimator(dt=dt)

    with pytest.raises(ValueError, match=r"Reference dt .* must be an integer multiple of plant dt"):
        Simulation(t_end=1.0, plant=dynamics, reference=ref, sensor=sensor, estimator=estimator, controller=mpc)
