import casadi as ca
import numpy as np


def sqp_solver(x, f, g, lbx=None, ubx=None, lbg=None, ubg=None, max_iter=50, tol=1e-6):
    """
    Standard SQP solver implementation using CasADi.

    Parameters:
        f (CasADi MX or SX): Objective function.
        g (CasADi MX or SX): Constraints.
        x0 (numpy array): Initial guess.
        lbx, ubx (array-like, optional): Bounds for decision variables.
        lbg, ubg (array-like, optional): Bounds for constraints.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        dict: Solution dictionary with keys 'x' and 'f'.
    """
    n = len(x)
    xk = ca.DM(x)
    x_sym = ca.MX.sym('x', n)

    grad_f = ca.gradient(f(x_sym), x_sym)
    jac_g = ca.jacobian(g(x_sym), x_sym)
    hess_lagrangian = ca.hessian(f(x_sym), x_sym)[0]

    grad_fk_fun = ca.Function('grad_fk', [x_sym], [grad_f])
    jac_gk_fun = ca.Function('jac_gk', [x_sym], [jac_g])
    hess_lk_fun = ca.Function('hess_lk', [x_sym], [hess_lagrangian])
    g_fun = ca.Function('g_fun', [x_sym], [g(x_sym)])

    for iter in range(max_iter):
        # Evaluate derivatives at current iterate
        grad_fk = grad_fk_fun(xk)
        jac_gk = jac_gk_fun(xk)
        hess_lk = hess_lk_fun(xk)
        gk = g_fun(xk)

        p = ca.MX.sym('p', n)

        # Form and solve QP subproblem
        qp = {
            'x': p,
            'f': 0.5 * ca.mtimes([p.T, hess_lk, p]) + grad_fk.T @ p,
            'g': jac_gk @ p + gk
        }

        qp_solver = ca.qpsol('qp_solver', 'qpoases', qp)

        alpha = qp_solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)['x']

        xk += alpha

        if ca.norm_2(alpha) < tol:
            break

    return {'x': xk, 'f': f(xk)}


# Example usage
if __name__ == "__main__":
    x = ca.MX.sym('x', 2)

    # Objective function
    f = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

    # Constraints: x[0]^2 + x[1]^2 <= 1 and x[0] = x[1]
    g = lambda x: ca.vertcat(x[0] ** 2 + x[1] ** 2 - 1,
                             x[0] - x[1] / 2)

    # Initial guess
    x0 = np.array([2.0, 2.0])

    solution = sqp_solver(
        x0=x0,
        f=f,
        g=g,
        lbx=[-np.inf, -np.inf],
        ubx=[np.inf, np.inf],
        lbg=[-np.inf, 0.0],  # inequality and equality constraints
        ubg=[0.0, 0.0],
    )

    print("Optimal solution:", solution['x'])
    print("Optimal objective value:", solution['f'])
