# ocp_formulation.py
import casadi as ca
from car_model import make_car_integrator
from multiple_shooting import setup_multiple_shooting_ocp

def setup_ocp(gear: int, dt: float, N: int, objective: str = 'control_energy'):
    """
    Set up a complete CasADi NLP for the Optimal Control Problem (OCP).

    Parameters
    ----------
    gear : int
        Gear selection for the entire trajectory.
    dt : float
        Time step for discretization.
    N : int
        Number of shooting intervals.
    objective : str
        Objective function type ('control_energy' or others).

    Returns
    -------
    solver : casadi.Function
        CasADi NLP solver function.
    nlp_dict : dict
        Dictionary containing NLP elements.
    integrator : casadi.Function
        CasADi integrator used for the dynamics.
    """
    nx, nu = 7, 3  # State and control dimensions

    # Create integrator
    integrator = make_car_integrator(gear, dt)

    # Define shooting intervals
    t_shooting = [i * dt for i in range(N + 1)]

    # Setup multiple shooting OCP
    w, X_end, F2, F3, S_vars, U_vars, _, _ = setup_multiple_shooting_ocp(
        integrator, t_shooting, nx, nu
    )

    # Objective function
    J = 0
    if objective == 'control_energy':
        for u in U_vars:
            J += ca.sumsqr(u)

    # Combine constraints
    g = ca.vertcat(F2, F3)

    # NLP formulation
    nlp_dict = {'x': w, 'f': J, 'g': g}

    # Solver configuration
    solver_opts = {'ipopt.print_level': 0, 'print_time': False}
    solver = ca.nlpsol('solver', 'ipopt', nlp_dict, solver_opts)

    return solver, nlp_dict, integrator

# Example usage
if __name__ == '__main__':
    gear, dt, N = 2, 0.1, 50
    solver, nlp, integrator = setup_ocp(gear, dt, N)

    # Initial guess and bounds
    w0 = [0] * nlp['x'].shape[0]
    lbw = [-ca.inf] * nlp['x'].shape[0]
    ubw = [ca.inf] * nlp['x'].shape[0]
    lbg = [0] * nlp['g'].shape[0]
    ubg = [0] * nlp['g'].shape[0]

    # Solve NLP
    solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print("Optimal solution:", solution['x'])
