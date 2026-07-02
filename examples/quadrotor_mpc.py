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
    # Closed-Loop Model Predictive Control for a 3D Quadrotor Tracking an S-Shaped Trajectory

    This notebook demonstrates how to formulate and solve a closed-loop model predictive control (MPC) problem for a full 3D quadrotor tracking a time-varying reference trajectory using the `MPC` and `OCP` classes from the `model_predictive_control` package, integrated with the `simulate` framework.
    """)
    return


@app.cell
def _():
    from typing import Any

    import casadi as ca
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pydantic import BaseModel, ConfigDict
    from simulate.estimator import IdentityEstimator
    from simulate.integrator import rk4
    from simulate.reference import Reference
    from simulate.sensor import GaussianSensor
    from simulate.simulation import Simulation

    from model_predictive_control.constraints import ConstraintList, ControlBoundConstraint, StateBoundConstraint
    from model_predictive_control.dynamics import Dynamics
    from model_predictive_control.mpc import MPC
    from model_predictive_control.objective import LQRObjective
    from model_predictive_control.ocp import OCP
    from model_predictive_control.plots import plot_controls, plot_mpc_trajectories

    return (
        Any,
        BaseModel,
        ConfigDict,
        ConstraintList,
        ControlBoundConstraint,
        Dynamics,
        GaussianSensor,
        IdentityEstimator,
        LQRObjective,
        MPC,
        OCP,
        Reference,
        Simulation,
        StateBoundConstraint,
        ca,
        np,
        pd,
        plot_controls,
        plot_mpc_trajectories,
        plt,
        rk4,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. System Dynamics

    We define the physical parameters of the quadrotor and derive the non-linear 3D equations of motion.
    """)
    return


@app.cell
def _(ca, np):
    # Physical parameters
    m = 1.0  # Mass (kg)
    g = 9.81  # Gravity (m/s^2)
    arm_length = 0.25  # Arm length (m)
    c_tau = 0.01  # Thrust-to-torque coefficient

    # Moments of inertia
    J_x = 0.02
    J_y = 0.02
    J_z = 0.04
    J = np.diag([J_x, J_y, J_z])
    inv_J = np.linalg.inv(J)

    # Dimensions
    nx = 12
    nu = 4

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)

    # Unpack states
    x[0:3]  # [x, y, z]
    v_vel = x[3:6]  # [v_x, v_y, v_z]
    eta = x[6:9]  # [phi, theta, psi]
    omega = x[9:12]  # [p, q, r]

    # Unpack controls
    T1, T2, T3, T4 = u[0], u[1], u[2], u[3]
    F_thrust = T1 + T2 + T3 + T4

    # Torques
    tau_x = arm_length * (T4 - T2)
    tau_y = arm_length * (T1 - T3)
    tau_z = c_tau * (T1 - T2 + T3 - T4)
    tau = ca.vertcat(tau_x, tau_y, tau_z)

    # Kinematics
    phi, theta, psi = eta[0], eta[1], eta[2]

    R_z = ca.vcat([ca.hcat([ca.cos(psi), -ca.sin(psi), 0]), ca.hcat([ca.sin(psi), ca.cos(psi), 0]), ca.hcat([0, 0, 1])])
    R_y = ca.vcat(
        [ca.hcat([ca.cos(theta), 0, ca.sin(theta)]), ca.hcat([0, 1, 0]), ca.hcat([-ca.sin(theta), 0, ca.cos(theta)])]
    )
    R_x = ca.vcat([ca.hcat([1, 0, 0]), ca.hcat([0, ca.cos(phi), -ca.sin(phi)]), ca.hcat([0, ca.sin(phi), ca.cos(phi)])])
    R_IB = R_z @ R_y @ R_x

    # Translational dynamics
    g_vec = ca.vertcat(0, 0, -g)
    T_B = ca.vertcat(0, 0, F_thrust)
    v_dot = g_vec + (R_IB @ T_B) / m

    # Rotational kinematics matrix
    W = ca.vcat(
        [
            ca.hcat([1, ca.sin(phi) * ca.tan(theta), ca.cos(phi) * ca.tan(theta)]),
            ca.hcat([0, ca.cos(phi), -ca.sin(phi)]),
            ca.hcat([0, ca.sin(phi) / ca.cos(theta), ca.cos(phi) / ca.cos(theta)]),
        ]
    )
    eta_dot = W @ omega

    # Rotational dynamics (Euler's equations)
    omega_dot = inv_J @ (tau - ca.cross(omega, J @ omega))

    # Full state derivative
    x_dot = ca.vertcat(v_vel, v_dot, eta_dot, omega_dot)
    dynamics_func = ca.Function("dynamics", [x, u], [x_dot])
    return dynamics_func, g, m, nu, nx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Objective and Constraints
    """)
    return


@app.cell
def _(
    ConstraintList,
    ControlBoundConstraint,
    LQRObjective,
    StateBoundConstraint,
    np,
    nu,
):
    N = 40  # OCP Horizon length
    dt = 0.1

    # State weights
    Q_diag = [
        100.0,
        100.0,
        200.0,  # Position: [x, y, z]
        10.0,
        10.0,
        10.0,  # Velocity: [v_x, v_y, v_z]
        10.0,
        10.0,
        50.0,  # Angles:   [phi, theta, psi]
        1.0,
        1.0,
        1.0,  # Rates:    [p, q, r]
    ]
    Q = np.diag(Q_diag)
    R = np.diag([1.0, 1.0, 1.0, 1.0])
    Qf = Q * 5.0

    objective = LQRObjective(Q, R, Qf, N)

    u_min_val = 0.0
    u_max_val = 3.0
    u_min = np.array([u_min_val] * nu)
    u_max = np.array([u_max_val] * nu)

    inf = 1e9
    x_min = np.array([-inf, -inf, 0.0, -inf, -inf, -inf, -np.pi / 3, -np.pi / 3, -inf, -inf, -inf, -inf])
    x_max = np.array([inf, inf, inf, inf, inf, inf, np.pi / 3, np.pi / 3, inf, inf, inf, inf])

    state_bounds = StateBoundConstraint(x_min, x_max)
    control_bounds = ControlBoundConstraint(u_min, u_max)
    cl = ConstraintList()
    cl.add(state_bounds, slice(None))
    cl.add(control_bounds, slice(0, N))
    return N, cl, dt, objective, u_max_val, u_min_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Reference Trajectory and MPC Setup
    """)
    return


@app.cell
def _(
    Any,
    BaseModel,
    ConfigDict,
    Dynamics,
    MPC,
    N,
    OCP,
    Reference,
    cl,
    dt,
    dynamics_func,
    g,
    m,
    np,
    nu,
    nx,
    objective,
    rk4,
):
    N_sim = N * 3
    np.arange(N_sim + 1) * dt
    (N_sim + N) * dt
    time_ref = np.arange(0, N_sim + N + 1) * dt

    X_ref_full = np.zeros((N_sim + N + 1, nx))
    U_ref_full = np.zeros((N_sim + N, nu))

    hover_thrust = m * g / 4.0

    for k in range(N_sim + N + 1):
        t = time_ref[k]
        # S-shape trajectory
        X_ref_full[k, 0] = 1.0 - 1.0 * t
        X_ref_full[k, 1] = 2.0 * np.sin(2 * np.pi * t / (N * 3 * dt))
        X_ref_full[k, 2] = 1.0 + 0.5 * t
        X_ref_full[k, 3] = 1.0
        X_ref_full[k, 4] = 2.0 * (2 * np.pi / (N * 3 * dt)) * np.cos(2 * np.pi * t / (N * 3 * dt))
        X_ref_full[k, 5] = 0.0

    for k in range(N_sim + N):
        U_ref_full[k, :] = hover_thrust

    class TrajectoryReferenceLog(BaseModel):
        """Log for TrajectoryReference."""

        model_config = ConfigDict(arbitrary_types_allowed=True)
        k: int

    # Custom reference generator for time-varying trajectory

    class TrajectoryReference(Reference[TrajectoryReferenceLog]):
        """Generates an S-shaped trajectory reference."""

        def __init__(self, dt: float, X_ref: np.ndarray, U_ref: np.ndarray, N_horizon: int) -> None:
            super().__init__(dt)
            self.X_ref = X_ref
            self.U_ref = U_ref
            self.N_horizon = N_horizon

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "TrajectoryReference":
            """Not implemented."""
            raise NotImplementedError

        def step(self, t: float) -> tuple[np.ndarray, TrajectoryReferenceLog]:
            """Execute step."""
            result, log = self._execute_zoh(t, self.update)
            return np.atleast_1d(result), log

        def update(self, t: float) -> tuple[Any, TrajectoryReferenceLog]:
            """Update reference."""
            k = round(t / self.dt)
            # Log 1D version of the current state reference for UniversalLog validation
            self.X_ref[k]

            # Internal reference passed to controller is the full horizon
            xk_ref_horizon = self.X_ref[k : k + self.N_horizon + 1]
            uk_ref_horizon = self.U_ref[k : k + self.N_horizon]

            return (xk_ref_horizon, uk_ref_horizon), TrajectoryReferenceLog(k=k)

    # Initialize Dynamics (Plant)
    dynamics = Dynamics(dynamics_func, dt=dt, integrator=rk4)

    # Initialize OCP and MPC (Controller)
    ocp = OCP(
        N=N,
        dt=dt,
        objective=objective,
        dynamics=dynamics,
        constraints=cl,
    )

    setup_args = {
        "method": "collocation",
        "dynamics_type": "continuous",
        "solver": "ipopt",
        "solver_opts": {"print_level": 0},
    }

    mpc = MPC(ocp, dt=dt, setup_args=setup_args, X_guess=X_ref_full[0 : N + 1, :], U_guess=U_ref_full[0:N, :])
    return (
        N_sim,
        TrajectoryReference,
        TrajectoryReferenceLog,
        U_ref_full,
        X_ref_full,
        dynamics,
        mpc,
        time_ref,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Closed-Loop Simulation
    """)
    return


@app.cell
def _(
    Any,
    GaussianSensor,
    IdentityEstimator,
    MPC,
    N,
    N_sim,
    Simulation,
    TrajectoryReference,
    TrajectoryReferenceLog,
    U_ref_full,
    X_ref_full,
    dt,
    dynamics,
    mpc,
    np,
):
    x0 = X_ref_full[0, :].copy()
    dynamics.x = x0

    def patched_step(self: TrajectoryReference, t: float) -> tuple[np.ndarray, TrajectoryReferenceLog]:
        """Return only 1D reference for UniversalLog validation."""
        k = round(t / self.dt)
        # We call update indirectly via ZOH, but we ignore its return for the 1D signal
        _, log = self._execute_zoh(t, self.update)
        return self.X_ref[k], log

    TrajectoryReference.step = patched_step  # type: ignore[assignment]

    def patched_mpc_step(self: MPC, t: float, ref: np.ndarray, x_hat: np.ndarray) -> tuple[np.ndarray, Any]:  # noqa: ARG001
        """Regenerate horizon reference from 1D signal."""
        k = round(t / self.dt)
        xk_ref_horizon = ref_gen.X_ref[k : k + self.N + 1]
        uk_ref_horizon = ref_gen.U_ref[k : k + self.N]
        return original_mpc_step(t, (xk_ref_horizon, uk_ref_horizon), x_hat)

    original_mpc_step = mpc.step
    mpc.step = patched_mpc_step.__get__(mpc, MPC)  # type: ignore[assignment]

    ref_gen = TrajectoryReference(dt, X_ref_full, U_ref_full, N)
    sensor = GaussianSensor(dt, std_dev=0.0)
    estimator = IdentityEstimator(dt)

    sim = Simulation(
        t_end=N_sim * dt, plant=dynamics, reference=ref_gen, sensor=sensor, estimator=estimator, controller=mpc
    )

    sim.run()
    return (sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Visualize Results
    """)
    return


@app.cell
def _(
    X_ref_full,
    np,
    pd,
    plot_controls,
    plot_mpc_trajectories,
    plt,
    sim,
    time_ref,
    u_max_val,
    u_min_val,
):
    # Extract results
    results_df = pd.DataFrame(sim.logger.universal_logs)
    time_vec = results_df["t"].to_numpy()
    X_closed_loop = np.array([log["y"] for log in sim.logger.universal_logs])
    U_closed_loop = np.array([log["u"] for log in sim.logger.universal_logs])
    X_open_loop = np.array([log["X_opt"] for log in sim.logger.component_logs["controller"]])

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Position X, Y, Z vs Reference
    for idx, label in enumerate(["X [m]", "Y [m]", "Z [m]"]):
        axs[0].plot(time_vec, X_closed_loop[:, idx], label=f"Closed-Loop {label}", linewidth=2)
        axs[0].plot(
            time_ref[: len(time_vec)],
            X_ref_full[: len(time_vec), idx],
            label=f"Reference {label}",
            linestyle="--",
            alpha=0.7,
        )
    axs[0].set_title("Position Tracking")
    axs[0].set_ylabel("Position")
    axs[0].legend(loc="upper right", ncol=3)
    axs[0].grid()

    # Euler Angles with open-loop predictions
    plot_mpc_trajectories(
        time_vec,
        X_closed_loop,
        X_open_loop,
        indices=[6, 7, 8],
        labels=[r"Roll ($\phi$)", r"Pitch ($\theta$)", r"Yaw ($\psi$)"],
        fig=fig,
        ax=axs[1],
        title="Euler Angles [rad] with Open-Loop Predictions",
        ylabel="Angle [rad]",
        bounds=[(-np.pi / 3, np.pi / 3), (-np.pi / 3, np.pi / 3), None],
        step_interval=5,
    )

    # Controls
    plot_controls(
        time_vec,
        U_closed_loop,
        labels=["$T_1$", "$T_2$", "$T_3$", "$T_4$"],
        fig=fig,
        ax=axs[2],
        title="Motor Thrusts [N]",
        bounds=[(u_min_val, u_max_val)] * 4,
        step=True,
    )

    plt.tight_layout()
    plt.show()

    # 3D Path Plot
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.set_aspect("equal")

    ax_3d.plot(
        X_ref_full[: len(time_vec), 0],
        X_ref_full[: len(time_vec), 1],
        X_ref_full[: len(time_vec), 2],
        "--",
        color="gray",
        label="Reference Path",
    )
    ax_3d.plot(
        X_closed_loop[:, 0], X_closed_loop[:, 1], X_closed_loop[:, 2], "-b", linewidth=2, label="Closed-Loop Path"
    )
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.set_title("3D Quadrotor Trajectory (MPC)")
    ax_3d.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
