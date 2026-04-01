import numpy as np
import casadi as ca
from model_predictive_control.ocp import (
    OCP,
    LinearOCP,
    quadratic_objective,
    tracking_objective,
    terminal_quadratic_objective,
    terminal_tracking_objective,
)

def test_ocp_with_tracking_reference():
    nx = 2
    nu = 1
    N = 5
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    x_ref = ca.MX.sym("x_ref", nx)
    u_ref = ca.MX.sym("u_ref", nu)

    dyn = ca.Function("dyn", [x, u], [ca.vertcat(x[0] + dt*x[1], x[1] + dt*u[0])], ["x", "u"], ["f"])

    Q = np.eye(nx) * 10
    R = np.eye(nu) * 0.1

    obj = tracking_objective(Q, R)
    term_obj = terminal_tracking_objective(Q * 10, np.zeros((nx, 1)))

    ocp = OCP(N=N, dt=dt, objective=obj, terminal_objective=term_obj, dynamics=dyn)

    ocp.setup(method="multiple_shooting", dynamics_type="discrete", solver="ipopt", solver_opts={"print_level": 0})

    # Test solving with a constant reference
    x0 = np.array([0.0, 0.0])
    X_ref = np.ones((N + 1, nx)) * 2.0
    U_ref = np.zeros((N, nu))

    X_opt, U_opt, status = ocp.solve(x0, x_ref=X_ref, u_ref=U_ref)

    assert status == "Solve_Succeeded"
    # The state should move towards the reference [2.0, 2.0]
    assert X_opt[-1, 0] > 0.5
    assert X_opt[-1, 1] > 0.5

def test_linear_ocp_with_tracking_reference():
    nx = 2
    nu = 1
    N = 15
    dt = 0.1

    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    Q = np.eye(nx) * 10
    R = np.eye(nu)
    Qf = np.eye(nx) * 50

    lin_ocp = LinearOCP(
        N=N,
        dt=dt,
        A=A,
        B=B,
        Q=Q,
        R=R,
        Qf=Qf
    )

    lin_ocp.setup(method="multiple_shooting", dynamics_type="discrete", solver="qrqp", solver_opts={"print_iter": False, "print_header": False})

    x0 = np.array([0.0, 0.0])
    X_ref = np.ones((N + 1, nx)) * 1.0
    U_ref = np.zeros((N, nu))

    X_opt, U_opt, status = lin_ocp.solve(x0, x_ref=X_ref, u_ref=U_ref)

    assert status == "success"
    # It should track X_ref
    assert np.allclose(X_opt[-1], [1.0, 1.0], atol=0.2)

def test_linearize_with_reference():
    nx = 2
    nu = 1
    N = 5
    dt = 0.1

    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    x_ref = ca.MX.sym("x_ref", nx)
    u_ref = ca.MX.sym("u_ref", nu)

    dyn = ca.Function("dyn", [x, u], [ca.vertcat(x[0] + dt*x[1], x[1] + dt*u[0])], ["x", "u"], ["f"])

    Q = np.eye(nx)
    R = np.eye(nu)

    obj = tracking_objective(Q, R)

    ocp = OCP(N=N, dt=dt, objective=obj, dynamics=dyn)

    x_bar = np.zeros(nx)
    u_bar = np.zeros(nu)
    X_ref = np.ones((N + 1, nx)) * 2.0
    U_ref = np.zeros((N, nu))

    # Linearize the tracking objective
    lin_ocp = ocp.linearize(x_bar, u_bar, dynamics_type="discrete", x_ref=X_ref, u_ref=U_ref)

    # For (x - x_ref)^T Q (x - x_ref)
    # The linear term q around x=0 is -2 * Q * x_ref.
    # But wait, Q here is 1/2 of the hessian! So tracking_obj returns dx^T Q dx.
    # The gradient wrt x at x=0 is 2 Q (x - x_ref) = -2 Q x_ref.
    # So q_func returns -2 Q x_ref. But wait, we define tracking_obj as dx.T @ Q @ dx!
    # Let's check the linear term q in the linearized OCP.
    expected_q = -2 * Q @ X_ref[0]

    assert np.allclose(lin_ocp.q[0], expected_q)
