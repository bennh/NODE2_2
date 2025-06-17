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
    Set up a direct multiple shooting discretization for optimal control problems,
    with time scaling support via dt = T / N passed to the integrator.
    """
    N = len(t_shooting) - 1
    S_vars = [ca.MX.sym(f's_{i}', nx) for i in range(N)]
    U_vars = [ca.MX.sym(f'u_{i}', nu) for i in range(N)]

    P_var = ca.MX.sym('p', n_params) if n_params > 0 else None
    T_var = ca.MX.sym('T') if use_final_time else None

    w_list = S_vars + U_vars
    if P_var is not None:
        w_list.append(P_var)
    if T_var is not None:
        w_list.append(T_var)
    w = ca.vertcat(*w_list)

    # Time scaling
    T_val = T_var if T_var is not None else ca.MX(1.0)

    # Call integrator with p_i = [u_i, T/N]
    X_end, F2_terms = [], []
    for i in range(N):
        dt_i = T_val / N
        p_i = ca.vertcat(U_vars[i], dt_i)  # now p_i has length 4: [wd, FB, f, dt]
        res = integrator(x0=S_vars[i], p=p_i)
        x_end = res['xf']
        X_end.append(x_end)
        if i < N - 1:
            F2_terms.append(x_end - S_vars[i + 1])
    F2 = ca.vertcat(*F2_terms) if F2_terms else ca.MX()

    # Inequality constraints
    F3_list = []
    if enforce_state_nonneg:
        for s in S_vars:
            F3_list.append(s)
    if enforce_control_bounds is not None:
        for u in U_vars:
            for j, (lb, ub) in enumerate(enforce_control_bounds):
                if lb is not None:
                    F3_list.append(u[j] - lb)
                if ub is not None:
                    F3_list.append(ub - u[j])
    if enforce_param_nonneg and P_var is not None:
        F3_list.append(P_var)
    F3 = ca.vertcat(*F3_list) if F3_list else ca.MX()

    return w, X_end, F2, F3, S_vars, U_vars, P_var, T_var