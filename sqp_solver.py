# sqp_solver.py
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class SQPSolver:
    def __init__(self, nlp, w0, lbw, ubw, lbg, ubg, max_iter=50, tol=1e-6):
        self.nlp = nlp
        self.w = w0
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        self.max_iter = max_iter
        self.tol = tol

        # CasADi functions
        self.F = ca.Function('F', [nlp['x']], [nlp['f'], nlp['g']])
        self.Jac = self.F.jacobian()
        self.HessLag = ca.Function('HessLag', [nlp['x'], ca.MX.sym('lam', nlp['g'].size(1))],
                                   [ca.hessian(nlp['f'] + ca.dot(nlp['g'], ca.MX.sym('lam', nlp['g'].size(1))), nlp['x'])[0]])

        self.iterations = []
        self.objective_history = []

    def solve(self):
        w = self.w.copy()

        for k in range(self.max_iter):
            # Evaluate objective and constraints
            f_val, g_val = self.F(w)

            # Jacobian
            J_f, J_g = self.Jac(w)

            # Hessian of the Lagrangian (simplified, using f only)
            H = self.HessLag(w, np.zeros_like(g_val))

            # Solve QP subproblem (simplified approach)
            qp = {'h': H, 'a': J_g, 'g': J_f.T}
            S = ca.qpsol('S', 'qpoases', qp)
            sol = S(lbx=self.lbw - w, ubx=self.ubw - w,
                    lba=self.lbg - g_val, uba=self.ubg - g_val)

            dw = sol['x'].full().flatten()
            lam = sol['lam_a'].full().flatten()

            # Update w
            w += dw

            # Logging
            obj_val = float(f_val)
            constraint_violation = np.linalg.norm(g_val, np.inf)
            self.iterations.append(k)
            self.objective_history.append(obj_val)

            print(f"Iter {k}: Obj = {obj_val:.6f}, Constr Viol = {constraint_violation:.3e}")

            # Check convergence
            if np.linalg.norm(dw, np.inf) < self.tol and constraint_violation < self.tol:
                print("SQP converged.")
                break

        return w, lam

    def plot_convergence(self):
        plt.figure()
        plt.plot(self.iterations, self.objective_history, '-o')
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title('SQP Convergence History')
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == '__main__':
    from ocp_formulation import setup_ocp

    gear, dt, N = 2, 0.1, 50
    solver, nlp, _ = setup_ocp(gear, dt, N)

    w0 = np.zeros(nlp['x'].shape[0])
    lbw = -np.inf * np.ones(nlp['x'].shape[0])
    ubw = np.inf * np.ones(nlp['x'].shape[0])
    lbg = np.zeros(nlp['g'].shape[0])
    ubg = np.zeros(nlp['g'].shape[0])

    sqp_solver = SQPSolver(nlp, w0, lbw, ubw, lbg, ubg)
    w_opt, lam_opt = sqp_solver.solve()
    sqp_solver.plot_convergence()
