# ocp_formulation.py
import casadi as ca
from car_model import make_car_integrator, make_scaled_integrator
from multiple_shooting import setup_multiple_shooting_ocp
from track_constraints import track_constraints

def setup_ocp(gear: int, dt: float, N: int, objective: str = 'control_energy', use_soft_track: bool = True):
    """
    Set up a complete CasADi NLP using multiple shooting (M3) for the Optimal Control Problem (OCP).

    Parameters
    ----------
    gear : int
        Gear selection for the entire trajectory.
    dt : float
        Initial time step (used for estimating time horizon).
    N : int
        Number of shooting intervals.
    objective : str
        Objective function type ('control_energy' or others).
    use_soft_track : bool
        If True, apply track constraint as soft penalty; otherwise as hard constraint.

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

    # Create symbolic final time T
    T_var = ca.MX.sym('T')
    t_shooting = [i / N for i in range(N + 1)]  # Normalized shooting points

    # Create integrator (scaled by dt placeholder)
    integrator = make_scaled_integrator(gear)  # unit dt, scaled later by T_var

    # Setup multiple shooting OCP
    w, X_end, F2, F3, S_vars, U_vars, _, _ = setup_multiple_shooting_ocp(
    integrator,
    t_shooting,
    nx,
    nu,
    use_final_time=True,
    enforce_control_bounds=[(-0.5, 0.5), (0, 15000), (0, 1)]
    )

    # Extract T_var from decision variable vector
    T = w[-1]  # final variable is T

    # Objective function
    J = 0
    if objective == 'control_energy':
        for u in U_vars:
            J += ca.sumsqr(u) * (T / N)

    # Track constraints
    track_con = track_constraints()
    g = [F2, F3]
    lbg = [0] * F2.shape[0] + [-ca.inf] * F3.shape[0]
    ubg = [0] * F2.shape[0] + [0] * F3.shape[0]

    for s in S_vars:
        g_track = track_con(s[0:2])  
        if use_soft_track:
            J += 1e4 * ca.sumsqr(ca.fmax(0, g_track))
        else:
            g.append(g_track)
            lbg += [-ca.inf, -ca.inf]
            ubg += [0, 0]

    # Initial state constraint: s_0 = x0
    x0 = ca.DM([-30, 0, 10, 0, 0, 0, 0]) 
    g.insert(0, S_vars[0] - x0)           
    lbg = [0] * nx + lbg                  
    ubg = [0] * nx + ubg                 
    
    # Terminal constraint: reach specific position at final state
    xf_target = ca.DM([130, 0])
    terminal = X_end[-1][0:2] - xf_target
    g.append(terminal)
    lbg += [-1.0, -1.0]
    ubg += [1.0, 1.0]

    # Combine all constraints
    g = ca.vertcat(*g)

    # NLP formulation
    nlp_dict = {'x': w, 'f': J, 'g': g}

    # Solver configuration
    solver_opts = {
    'ipopt.print_level': 5,
    'print_time': True,
    'ipopt.tol': 1e-4,  # main tol
    'ipopt.constr_viol_tol': 1e-4,  # con tol
    'ipopt.acceptable_tol': 1e-2,
    'ipopt.acceptable_constr_viol_tol': 1e-2,
    'ipopt.acceptable_iter': 5,
    'ipopt.max_iter': 50
     }
    solver = ca.nlpsol('solver', 'ipopt', nlp_dict, solver_opts)

    return solver, nlp_dict, integrator, lbg, ubg
