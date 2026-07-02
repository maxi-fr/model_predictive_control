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
    # Closed-Loop Linear Model Predictive Control

    This notebook demonstrates a closed-loop Model Predictive Control (MPC) simulation for a simple unstable linear 2D system using the `LinearOCP` class, which uses a QP formulation.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from simulate.estimator import IdentityEstimator
    from simulate.reference import StepReference
    from simulate.sensor import GaussianSensor
    from simulate.simulation import Simulation

    from model_predictive_control.constraints import ConstraintList, LinearConstraint
    from model_predictive_control.dynamics import LinearDynamics
    from model_predictive_control.mpc import LinearMPC
    from model_predictive_control.ocp import LinearOCP
    from model_predictive_control.plots import plot_controls, plot_mpc_trajectories

    return (
        ConstraintList,
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
        plot_mpc_trajectories,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Linear System Dynamics

    We define a generic unstable linear system $x_{k+1} = A x_k + B u_k$.
    """)
    return


@app.cell
def _(np):
    A = np.array([[1.0, 0.1], [0.5, 1.0]])
    B = np.array([[0.0], [0.1]])

    nx = A.shape[1]
    nu = B.shape[1]
    return A, B, nu, nx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Objective Function and Constraints

    We use a standard quadratic objective to penalize state deviations and control effort, and box constraints for safety.
    """)
    return


@app.cell
def _(np, nu, nx):
    # Objective matrices
    Q = np.diag([100.0, 10.0])
    R = np.array([[0.1]])

    np.zeros(nx)
    np.zeros(nu)
    np.zeros((nx, nu))

    # Terminal objective
    np.diag([1000.0, 100.0])

    # Constraints
    # F x + G u <= h
    u_max_val = 50.0
    x_max_val = 2.0

    # Box constraints
    F = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
    G = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [-1.0]])
    h = np.array([x_max_val, x_max_val, x_max_val, x_max_val, u_max_val, u_max_val])

    # Terminal constraints (only on state)
    F_term = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    h_term = np.array([x_max_val, x_max_val, x_max_val, x_max_val])

    # Bounds for plotting
    x_min = np.array([-x_max_val, -x_max_val])
    x_max = np.array([x_max_val, x_max_val])
    u_min = np.array([-u_max_val])
    u_max = np.array([u_max_val])
    return F, F_term, G, Q, R, h, h_term, u_max, u_min, x_max, x_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. OCP Setup and Closed-Loop Simulation
    """)
    return


@app.cell
def _(
    A,
    B,
    ConstraintList,
    F,
    F_term,
    G,
    GaussianSensor,
    IdentityEstimator,
    LinearConstraint,
    LinearDynamics,
    LinearMPC,
    LinearOCP,
    Q,
    R,
    Simulation,
    StepReference,
    h,
    h_term,
    np,
    nu,
    nx,
):
    N_horizon = 20
    N_sim = 40
    dt = 0.1
    t_end = N_sim * dt

    cl = ConstraintList()
    cl.add(LinearConstraint(F=F, G=G, h=h), range(N_horizon))
    cl.add(LinearConstraint(F=F_term, h=h_term, nu=nu), [N_horizon])

    # Use LinearDynamics with dt
    dynamics = LinearDynamics(A, B, dt=dt)

    ocp = LinearOCP(
        N=N_horizon,
        dt=dt,
        dynamics=dynamics,
        Q=Q,
        R=R,
        constraints=cl,
    )

    # Setup using LinearMPC wrapper with multiple shooting (sparse) and qrqp backend
    setup_args = {
        "method": "multiple_shooting",
        "dynamics_type": "discrete",
        "solver": "qrqp",
        "solver_opts": {"print_iter": False, "print_header": False},
    }

    mpc = LinearMPC(linear_ocp=ocp, dt=dt, setup_args=setup_args)

    # Simulation setup
    x0_val = np.array([1.5, 0.0])
    dynamics.x = x0_val

    ref = StepReference(dt=dt, step_value=np.zeros(nx))
    sensor = GaussianSensor(dt=dt, std_dev=0.0)
    estimator = IdentityEstimator(dt=dt)

    sim = Simulation(t_end=t_end, plant=dynamics, reference=ref, sensor=sensor, estimator=estimator, controller=mpc)

    sim.run()
    return (sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Plot Results
    """)
    return


@app.cell
def _(
    np,
    pd,
    plot_controls,
    plot_mpc_trajectories,
    plt,
    sim,
    u_max,
    u_min,
    x_max,
    x_min,
):
    # Extract results from logger
    results_df = pd.DataFrame(sim.logger.universal_logs)
    time_vec = results_df["t"].to_numpy()

    # Extract X_closed_loop and X_open_loop from the logs
    X_closed_loop = np.array([log["y"] for log in sim.logger.universal_logs])
    U_closed_loop = np.array([log["u"] for log in sim.logger.universal_logs])

    # Open loop predictions from controller logs
    X_open_loop = np.array([log["X_opt"] for log in sim.logger.component_logs["controller"]])

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot states with open loop predictions
    plot_mpc_trajectories(
        time_vec,
        X_closed_loop,
        X_open_loop,
        labels=["State 1", "State 2"],
        fig=fig,
        ax=axs[0],
        title="Closed-Loop MPC Trajectories with Open-Loop Predictions",
        bounds=[(x_min[0], x_max[0]), (x_min[1], x_max[1])],
        step_interval=4,
    )

    # Plot controls
    plot_controls(
        time_vec,
        U_closed_loop,
        labels=["Control"],
        fig=fig,
        ax=axs[1],
        title="Closed-Loop Control Action",
        bounds=[(u_min[0], u_max[0])],
    )

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
