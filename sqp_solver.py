# sqp_solver.py
import casadi as ca
import numpy as np
from track_constraints import Pl_expr, Pu_expr


class SQPSolver:
    def __init__(self, name, solver_type, nlp_dict, lbx, ubx, lbg, ubg, solver_opts=None):
        # NLP dictionary
        self.x = nlp_dict['x']
        self.f = nlp_dict['f']
        self.g = nlp_dict['g']
        self.opts = solver_opts or {}
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        # Precompiled functions
        self.f_fun = ca.Function('f', [self.x], [self.f])
        self.g_fun = ca.Function('g', [self.x], [self.g])
        self.grad_f_fun = ca.Function('grad_f', [self.x], [ca.gradient(self.f, self.x)])
        self.jac_g_fun = ca.Function('jac_g', [self.x], [ca.jacobian(self.g, self.x)])
        self.hess_fun = ca.Function('hess_l', [self.x], [ca.hessian(self.f, self.x)[0]])

        # Lagrange multipliers (same length as g)
        self.lam_g = ca.MX.sym('lam_g', self.g.size1())
        self.lam_x = ca.MX.sym('lam_x', self.x.size1())  # Note: size1() returns number of rows
        # Lagrangian function
        self.L = self.f + ca.dot(self.lam_g, self.g)
        # Hessian of the Lagrangian (arguments are x, lam_g)
        self.hess_lagrangian_fun = ca.Function('hess_l', [self.x, self.lam_g], [ca.hessian(self.L, self.x)[0]])

    def __call__(self, **kwargs):
        x0 = np.array(kwargs.get('x0')).flatten()

        max_iter = self.opts.get('ipopt.max_iter', 100)
        tol = self.opts.get('ipopt.tol', 1e-6)

        xk = x0.copy()
        lam_gk = np.zeros(self.g_fun(xk).shape[0])
        for it in range(max_iter):
            grad = self.grad_f_fun(xk).full().flatten()
            gval = self.g_fun(xk).full().flatten()
            jac_g = self.jac_g_fun(xk).full()
            # Compute Hessian of Lagrangian using current multipliers
            Hk = self.hess_lagrangian_fun(xk, lam_gk).full()
            p = ca.MX.sym('p', xk.size)
            Hk = ca.DM(Hk)
            grad = ca.DM(grad)
            jac_g = ca.DM(jac_g)

            qp = {'x': p,
                  'f': 0.5 * ca.mtimes([p.T, Hk, p]) + ca.dot(grad, p),
                  'g': jac_g @ p}
            qp_opts = {'printLevel': 'low', 'nWSR': 10000, 'error_on_fail': False}
            qpsolver = ca.qpsol('qpsol', 'qpoases', qp, qp_opts)
            qp_res = qpsolver(
                lbg=self.lbg, ubg=self.ubg,
                lbx=self.lbx, ubx=self.ubx
            )

            pk = qp_res['x'].full().flatten()
            # Update Lagrange multipliers
            lam_gk = qp_res['lam_g'].full().flatten() if 'lam_g' in qp_res else lam_gk
            xk1 = xk + pk

            if np.linalg.norm(pk) < tol:
                break
            xk = xk1

        return {
            'x': xk,
            'f': self.f_fun(xk).full().item(),
            'g': self.g_fun(xk).full().flatten(),
            'success': True,
            'status': 0,
            'iterations': it + 1,
        }


def sqp_solver(name, solver_type, nlp_dict, lbx, ubx, lbg, ubg, solver_opts=None):
    return SQPSolver(name, solver_type, nlp_dict, lbx, ubx, lbg, ubg, solver_opts)
