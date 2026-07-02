import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chance Constrained Model Predictive Control
    This notebook demonstrates a Chance Constrained MPC formulation for a 1D mass-spring-damper system subject to additive Gaussian noise. We compare a nominal MPC (which ignores noise) with a chance-constrained MPC (which tightens constraints to account for the noise variance).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    from simulate.estimator import IdentityEstimator
    from simulate.reference import StepReference
    from simulate.sensor import GaussianSensor
    from simulate.simulation import Simulation

    from model_predictive_control.constraints import ConstraintList, LinearConstraint
    from model_predictive_control.dynamics import DynamicsLog, LinearDynamics
    from model_predictive_control.mpc import LinearMPC
    from model_predictive_control.ocp import LinearOCP
    from model_predictive_control.plots import plot_controls

    return (
        ConstraintList,
        DynamicsLog,
        GaussianSensor,
        IdentityEstimator,
        LinearConstraint,
        LinearDynamics,
        LinearMPC,
        LinearOCP,
        Simulation,
        StepReference,
        np,
        pd,
        plot_controls,
        plt,
        st,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. System Dynamics
    We define a 1D mass-spring-damper system:
    $m \ddot{p} + c \dot{p} + k p = u$

    where $p$ is position, $v$ is velocity, and $u$ is the control force.
    """)
    return


@app.cell
def _(np):
    # Parameters
    m = 1.0  # mass
    _k = 0.5  # spring constant
    c = 0.1  # damping coefficient
    dt = 0.1  # sampling time
    Ac = np.array([[0, 1], [-_k / m, -c / m]])
    # Continuous-time matrices: x = [p, v]^T
    Bc = np.array([[0], [1 / m]])
    A = np.eye(2) + Ac * dt
    B = Bc * dt
    # Discrete-time matrices (Euler approximation)
    nx = A.shape[1]
    nu = B.shape[1]
    return A, B, dt, nu, nx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Stochastic Dynamics Implementation
    To simulate the system with noise using the `simulate` framework, we subclass `LinearDynamics` to include additive Gaussian noise in the `update` step.
    """)
    return


@app.cell
def _(DynamicsLog, LinearDynamics, np):
    class StochasticLinearDynamics(LinearDynamics):
        """Linear dynamics with additive Gaussian noise."""

        def __init__(self, A: np.ndarray, B: np.ndarray, Sigma_w: np.ndarray, dt: float = 0.1, seed: int = 42) -> None:
            super().__init__(A, B, dt=dt)
            self.Sigma_w = Sigma_w
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        def reset_rng(self) -> None:
            """Reset the random number generator to the initial seed."""
            self.rng = np.random.default_rng(self.seed)

        def update(self, t: float, u: float | np.ndarray) -> tuple[float | np.ndarray, DynamicsLog]:  # noqa: ARG002
            """Advance the dynamics and add noise."""
            u_vec = self.to_col_vec(u).flatten()
            self.x = np.asarray(self.f(self.x, u_vec)).flatten()  # Nominal discrete step
            noise = self.rng.multivariate_normal(np.zeros(self.nx), self.Sigma_w)
            self.x = self.x + noise  # Add additive noise: w ~ N(0, Sigma_w)
            return (self.from_col_vec(self.x), DynamicsLog(x=self.x.copy()))

    return (StochasticLinearDynamics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Noise Characteristics
    We assume additive zero-mean Gaussian noise on the states:
    $x_{k+1} = A x_k + B u_k + w_k$ where $w_k \sim \mathcal{N}(0, \Sigma_w)$
    """)
    return


@app.cell
def _(A, B, StochasticLinearDynamics, dt, np):
    # Noise covariance
    sigma_w_pos = 0.05
    sigma_w_vel = 0.02
    Sigma_w = np.diag([sigma_w_pos**2, sigma_w_vel**2])

    # Create the plant
    plant = StochasticLinearDynamics(A=A, B=B, Sigma_w=Sigma_w, dt=dt, seed=42)
    return Sigma_w, plant


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Formulate Nominal MPC
    Standard MPC controller aiming to regulate the system to the origin, subject to state and control bounds.
    """)
    return


@app.cell
def _(
    ConstraintList,
    LinearConstraint,
    LinearMPC,
    LinearOCP,
    dt,
    np,
    nu,
    nx,
    plant,
):
    # Objective
    Q = np.diag([3000.0, 0.0])
    R = np.array([[0.1]])

    # Constraints
    p_max = 1.5
    u_max = 2.0
    u_min = -2.0

    N = 10

    nominal_state_constraints = ConstraintList()
    # Position limits: p <= p_max
    nominal_state_constraints.add(
        LinearConstraint(h=np.array([p_max]), F=np.array([[1.0, 0.0]]), nu=nu), slice(1, None)
    )

    # Control limits: u_min <= u <= u_max
    nominal_state_constraints.add(LinearConstraint(h=np.array([u_max]), G=np.array([[1.0]]), nx=nx), slice(0, N))
    nominal_state_constraints.add(LinearConstraint(h=np.array([-u_min]), G=np.array([[-1.0]]), nx=nx), slice(0, N))

    nominal_ocp = LinearOCP(dynamics=plant, Q=Q, R=R, N=N, dt=dt, constraints=nominal_state_constraints)
    nominal_mpc = LinearMPC(linear_ocp=nominal_ocp, dt=dt, setup_args={"solver": "osqp"})
    return N, Q, R, nominal_mpc, p_max, u_max, u_min


@app.cell
def _(
    GaussianSensor,
    IdentityEstimator,
    Simulation,
    StepReference,
    dt,
    nominal_mpc,
    np,
    plant,
):
    n_steps = 50
    t_end = n_steps * dt
    x0 = np.array([0.0, 0.0])  # Start near the boundary
    ref = StepReference(dt=dt, step_value=np.array([1.0, 0.0]))
    # Simulation components
    _sensor = GaussianSensor(dt=dt, std_dev=0.0)
    _estimator = IdentityEstimator(dt=dt)  # Perfect sensing
    plant.x = x0.copy()
    plant.reset_rng()
    # 1. Run Nominal MPC
    sim_nom = Simulation(
        t_end=t_end, plant=plant, reference=ref, sensor=_sensor, estimator=_estimator, controller=nominal_mpc
    )
    sim_nom.run()
    return sim_nom, t_end, x0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Formulate Chance Constrained MPC
    For chance constraints $P(p_k \le p_{max}) \ge 1 - \epsilon$, we tighten the constraints using the propagated variance.
    We use time-varying tightening to ensure feasibility near the initial condition.
    """)
    return


@app.cell
def _(
    A,
    ConstraintList,
    LinearConstraint,
    LinearMPC,
    LinearOCP,
    N,
    Q,
    R,
    Sigma_w,
    dt,
    np,
    nu,
    nx,
    p_max,
    plant,
    st,
    u_max,
    u_min,
):
    epsilon = 0.05  # 5% violation probability
    z_val = st.norm.ppf(1 - epsilon)
    Sigma_k = np.zeros((2, 2))
    # Compute variance propagation over horizon N
    cc_state_constraints = ConstraintList()
    for _k in range(1, N + 1):
        std_dev_pos = np.sqrt(Sigma_k[0, 0])
        tightening = z_val * std_dev_pos
        cc_p_max = p_max - tightening  # Propagate covariance: Sigma_{k+1} = A Sigma_k A^T + Sigma_w
        cc_state_constraints.add(LinearConstraint(h=np.array([cc_p_max]), F=np.array([[1.0, 0.0]]), nu=nu), _k)
        Sigma_k = A @ Sigma_k @ A.T + Sigma_w
    cc_state_constraints.add(LinearConstraint(h=np.array([u_max]), G=np.array([[1.0]]), nx=nx), slice(0, N))
    cc_state_constraints.add(
        LinearConstraint(h=np.array([-u_min]), G=np.array([[-1.0]]), nx=nx), slice(0, N)
    )  # Tighten state constraints
    cc_ocp = LinearOCP(dynamics=plant, Q=Q, R=R, N=N, dt=dt, constraints=cc_state_constraints)
    # Control limits (unchanged)
    cc_mpc = LinearMPC(linear_ocp=cc_ocp, dt=dt, setup_args={"solver": "osqp"})  # Update Sigma for next step
    return (cc_mpc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Closed-Loop Simulation
    We simulate both controllers using the `simulate.Simulation` class. We reset the plant's RNG before each run to ensure they experience the same noise sequence.
    """)
    return


@app.cell
def _(
    A,
    B,
    GaussianSensor,
    IdentityEstimator,
    Sigma_w,
    Simulation,
    StepReference,
    StochasticLinearDynamics,
    cc_mpc,
    dt,
    np,
    t_end,
    x0,
):
    ref_1 = StepReference(dt=dt, step_value=np.array([1.0, 0.0]))
    _sensor = GaussianSensor(dt=dt, std_dev=0.0)
    _estimator = IdentityEstimator(dt=dt)
    plant_1 = StochasticLinearDynamics(A=A, B=B, Sigma_w=Sigma_w, dt=dt, seed=42)
    plant_1.x = x0.copy()
    plant_1.reset_rng()
    sim_cc = Simulation(
        t_end=t_end, plant=plant_1, reference=ref_1, sensor=_sensor, estimator=_estimator, controller=cc_mpc
    )
    sim_cc.run()
    return ref_1, sim_cc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Results
    We extract the results from the simulation loggers and plot them.
    """)
    return


@app.cell
def _(Simulation, np, p_max, pd, plot_controls, plt, ref_1, sim_cc, sim_nom):
    def extract_sim_results(sim: Simulation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        results_df = pd.DataFrame(sim.logger.universal_logs)
        time = results_df["t"].to_numpy()
        X = np.array([log["y"] for log in sim.logger.universal_logs])
        U = np.array([log["u"] for log in sim.logger.universal_logs])
        X_open_loop = np.array([log["X_opt"] for log in sim.logger.component_logs["controller"]])
        return (time, X, U, X_open_loop)

    time_nom, X_nom, U_nom, _X_ol_nom = extract_sim_results(sim_nom)
    time_cc, X_cc, U_cc, _X_ol_cc = extract_sim_results(sim_cc)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_nom, X_nom[:, 0], "r-", label="Nominal MPC")
    ax.plot(time_cc, X_cc[:, 0], "b-", label="Chance Constrained MPC")
    # Plotting Position
    ax.axhline(np.atleast_1d(ref_1.step_value)[0], label="ref.", linestyle="--")
    ax.axhline(p_max, color="k", linestyle="--", label="Actual Bound ($p_{max}$)")
    # Nominal Results
    ax.set_xlabel("Time [s]")
    # Chance Constrained Results
    ax.set_ylabel("Position $p$")
    ax.legend()
    ax.set_title("Stochastic MPC: Position vs Time")
    ax.grid(visible=True)
    # Bounds
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_controls(time_nom, U_nom, labels=["Nominal Control"], fig=fig, ax=ax, title="Control Actions")
    plot_controls(time_cc, U_cc, labels=["CC Control"], fig=fig, ax=ax)
    # Plotting Controls
    plt.show()
    return


if __name__ == "__main__":
    app.run()
