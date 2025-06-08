import casadi as ca
from typing import Callable, List, Tuple, Optional

def setup_multiple_shooting_ocp(
    integrator: ca.Function,
    t_shooting: List[float],
    nx: int,
    nu: int,
    n_params: int = 0,
    use_final_time: bool = False,
    enforce_state_nonneg: bool = False,
    enforce_control_bounds: Optional[List[Tuple[float, float]]] = None,
    enforce_param_nonneg: bool = False
) -> Tuple[ca.MX, List[ca.MX], ca.MX, ca.MX, List[ca.MX], List[ca.MX], Optional[ca.MX], Optional[ca.MX]]:
    """
    Set up a direct multiple shooting discretization for optimal control problems.

    Parameters
    ----------
    integrator : ca.Function
        CasADi integrator with signature integrator(x0, p) -> {'xf': ...}
    t_shooting : List[float]
        Shooting nodes (N+1 nodes for N intervals)
    nx : int
        State dimension
    nu : int
        Control dimension
    n_params : int
        Number of global parameters (default 0)
    use_final_time : bool
        If True, optimize final time T (adds T to decision variables)
    enforce_state_nonneg : bool
        If True, add state >= 0 constraints
    enforce_control_bounds : List[Tuple[float, float]]
        List of (lower, upper) tuples for each control variable (optional)
    enforce_param_nonneg : bool
        If True, enforce p >= 0

    Returns
    -------
    w        : MX                # All decision variables [s_0,...,s_{N-1}, u_0,...,u_{N-1}, (p), (T)]
    X_end    : List[MX]          # End state for each interval
    F2       : MX                # Continuity constraints (shooting connection)
    F3       : MX                # Inequality constraints (states, controls, params)
    S_vars   : List[MX]          # Symbolic shooting state variables
    U_vars   : List[MX]          # Symbolic control variables for each interval
    P_var    : Optional[MX]      # Parameter vector (or None)
    T_var    : Optional[MX]      # Final time variable (or None)
    """
    N = len(t_shooting) - 1
    S_vars = [ca.MX.sym(f's_{i}', nx) for i in range(N)]
    U_vars = [ca.MX.sym(f'u_{i}', nu) for i in range(N)]

    P_var = ca.MX.sym('p', n_params) if n_params > 0 else None
    T_var = ca.MX.sym('T') if use_final_time else None

    # For w:  [s_0, ..., s_{N-1}, u_0, ..., u_{N-1}, (p), (T)]
    w_list = S_vars + U_vars
    if P_var is not None:
        w_list.append(P_var)
    if T_var is not None:
        w_list.append(T_var)
    w = ca.vertcat(*w_list)

    # Collect interval endpoints, continuity constraints
    X_end, F2_terms = [], []
    for i in range(N):
        # Pack integrator parameters: usually [control, (params), (T)]
        # You can change to match your integrator signature
        p_args = [U_vars[i]]
        if P_var is not None:
            p_args.append(P_var)
        if T_var is not None:
            p_args.append(T_var)
        p_i = ca.vertcat(*p_args) if len(p_args) > 1 else p_args[0]

        # Call CasADi integrator
        res = integrator(x0=S_vars[i], p=p_i)
        x_end = res['xf']
        X_end.append(x_end)
        if i < N - 1:
            F2_terms.append(x_end - S_vars[i + 1])

    F2 = ca.vertcat(*F2_terms) if F2_terms else ca.MX.zeros(0)

    # Collect inequality constraints (F3)
    F3_list = []
    if enforce_state_nonneg:
        for s in S_vars:
            F3_list.append(s)
    if enforce_control_bounds is not None:
        # enforce_control_bounds: list of (lb, ub) for each control variable
        for u in U_vars:
            for j, (lb, ub) in enumerate(enforce_control_bounds):
                if lb is not None:
                    F3_list.append(u[j] - lb)
                if ub is not None:
                    F3_list.append(ub - u[j])
    if enforce_param_nonneg and P_var is not None:
        F3_list.append(P_var)

    F3 = ca.vertcat(*F3_list) if F3_list else ca.MX.zeros(0)

    return w, X_end, F2, F3, S_vars, U_vars, P_var, T_var
