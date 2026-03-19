import casadi as ca

class OCP:
    objective: ca.Function

    dymanics: ca.Function

    eq_constraints: ca.Function
    in_eq_constraints: ca.Function




def Quadratic_Objective(Q, R, q, r, N):
    nx = Q.shape[0]
    nu = R.shape[0]
    x = ca.sym("x", nx)
    u = ca.sym("u", nu)

    return ca.Function("quadr_obj", [x, u], [x.T @ Q @ x + x.T @ q + u.T @ R @ u + u.T @ r + x.T @ N @ u], ["x", "u"], ["f"])