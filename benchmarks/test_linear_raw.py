"""Benchmark for linear MPC without abstractions (pure OSQP)."""

import numpy as np
import scipy.sparse as sparse
import osqp

def setup_raw_linearized_inverted_pendulum_mpc():
    """Set up the linearized inverted pendulum MPC benchmark using pure OSQP."""
    m_cart = 1.0
    m_pend = 0.1
    length = 0.5
    g = 9.81

    denominator = m_cart
    A_cont = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m_pend * g / denominator, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, (m_cart + m_pend) * g / (length * denominator), 0.0],
    ])
    B_cont = np.array([[0.0], [1.0 / denominator], [0.0], [-1.0 / (length * denominator)]])

    dt = 0.05
    A = np.eye(4) + A_cont * dt
    B = B_cont * dt

    N = 20
    nx = 4
    nu = 1

    Q = np.diag([10.0, 1.0, 10.0, 1.0])
    R = np.array([[0.1]])
    Qf = np.diag([100.0, 10.0, 100.0, 10.0])

    # Variables: y = [x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
    ny = N * (nx + nu) + nx

    # Construct P
    P_blocks = []
    for _ in range(N):
        P_blocks.append(Q)
        P_blocks.append(R)
    P_blocks.append(Qf)
    P = sparse.block_diag(P_blocks, format='csc')
    
    # Construct q
    q = np.zeros(ny)

    # Dynamics matrix A_dyn
    A_dyn = sparse.dok_matrix(((N + 1) * nx, ny))
    for i in range(nx):
        A_dyn[i, i] = 1.0
    for k in range(N):
        idx_x_k = k * (nx + nu)
        idx_u_k = k * (nx + nu) + nx
        idx_x_next = (k + 1) * (nx + nu)
        row_start = (k + 1) * nx
        
        for i in range(nx):
            A_dyn[row_start + i, idx_x_next + i] = 1.0
            for j in range(nx):
                if A[i, j] != 0:
                    A_dyn[row_start + i, idx_x_k + j] = -A[i, j]
            for j in range(nu):
                if B[i, j] != 0:
                    A_dyn[row_start + i, idx_u_k + j] = -B[i, j]

    A_dyn = A_dyn.tocsc()

    # Bounds matrix A_bnd
    A_bnd = sparse.eye(ny, format='csc')

    # Full constraints matrix
    A_osqp = sparse.vstack([A_dyn, A_bnd], format='csc')

    # Bounds
    u_max_val = 20.0
    p_max_val = 2.0

    l_dyn = np.zeros((N + 1) * nx)
    u_dyn = np.zeros((N + 1) * nx)
    
    l_bnd = -np.inf * np.ones(ny)
    u_bnd = np.inf * np.ones(ny)

    for k in range(N):
        idx_x = k * (nx + nu)
        idx_u = k * (nx + nu) + nx
        l_bnd[idx_x + 0] = -p_max_val
        u_bnd[idx_x + 0] = p_max_val
        l_bnd[idx_u + 0] = -u_max_val
        u_bnd[idx_u + 0] = u_max_val

    idx_x_N = N * (nx + nu)
    l_bnd[idx_x_N + 0] = -p_max_val
    u_bnd[idx_x_N + 0] = p_max_val

    return P, q, A_osqp, l_dyn, u_dyn, l_bnd, u_bnd, A, B, N, nx, nu


def test_linear_raw_ocp_solve(benchmark) -> None:
    """Benchmark solving the raw linear OCP with OSQP."""
    P, q, A_osqp, l_dyn, u_dyn, l_bnd, u_bnd, A, B, N, nx, nu = setup_raw_linearized_inverted_pendulum_mpc()
    
    x0_val = np.array([0.0, 0.0, 0.5, 0.0])

    def solve(x0):
        # Update bounds for initial condition
        l = np.concatenate([np.concatenate([x0, l_dyn[nx:]]), l_bnd])
        u = np.concatenate([np.concatenate([x0, u_dyn[nx:]]), u_bnd])
        
        prob = osqp.OSQP()
        prob.setup(P, q, A_osqp, l, u, verbose=False)
        return prob.solve()

    result = benchmark(solve, x0_val)
    
    assert result.info.status_val in [1, 2] # 1 is solved, 2 is solved inaccurate
    benchmark.extra_info["osqp_iterations"] = result.info.iter


def test_linear_raw_mpc_step(benchmark) -> None:
    """Benchmark stepping the raw linear MPC loop with OSQP."""
    P, q, A_osqp, l_dyn, u_dyn, l_bnd, u_bnd, A, B, N, nx, nu = setup_raw_linearized_inverted_pendulum_mpc()
    
    x_current = np.array([0.0, 0.0, 0.1, 0.0])
    num_steps = 50

    def run_mpc_loop() -> list[int]:
        iterations_list = []
        x = x_current.copy()
        
        # Setup prob once
        prob = osqp.OSQP()
        
        # Preallocate arrays
        l = np.concatenate([np.concatenate([x, l_dyn[nx:]]), l_bnd])
        u = np.concatenate([np.concatenate([x, u_dyn[nx:]]), u_bnd])
        
        prob.setup(P, q, A_osqp, l, u, verbose=False)
        
        for _ in range(num_steps):
            # Update bounds for x0 in-place
            l[:nx] = x
            u[:nx] = x
            
            prob.update(l=l, u=u)
            
            result = prob.solve()
            iterations_list.append(result.info.iter)
            
            # Extract u0
            u_opt = result.x[nx:nx+nu]
            
            x = A @ x + B @ u_opt

        return iterations_list

    iterations_list = benchmark(run_mpc_loop)

    valid_iters = [it for it in iterations_list if it >= 0]
    if valid_iters:
        benchmark.extra_info["avg_solver_iterations"] = sum(valid_iters) / len(valid_iters)
        benchmark.extra_info["total_solver_iterations"] = sum(valid_iters)
    else:
        benchmark.extra_info["avg_solver_iterations"] = -1
        benchmark.extra_info["total_solver_iterations"] = -1
