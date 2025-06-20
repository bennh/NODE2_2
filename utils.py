# utils.py
import numpy as np
import matplotlib.pyplot as plt
from track_constraints import Pl_vis, Pu_vis
from car_model import make_car_integrator
from scipy.interpolate import CubicSpline, BSpline


def unpack_solution(w_opt, N, integrator, nx=7, nu=3):
    """
    Unpack flat NLP solution vector into state and control arrays.

    Parameters
    ----------
    w_opt : array-like
        Optimal decision variable vector (from solver).
    N : int
        Number of shooting intervals.
    nx : int
        Dimension of state vector.
    nu : int
        Dimension of control vector.

    Returns
    -------
    X_opt : ndarray of shape (N, nx)
        State trajectory (shooting node states).
    U_opt : ndarray of shape (N, nu)
        Control trajectory (piecewise constant over each interval).
    T_opt : float
        Optimal final time.
    """
    w = np.array(w_opt).flatten()
    # 1) Node states, controls, and terminal time
    X_flat = w[: N * nx]
    U_flat = w[N * nx: N * (nx + nu)]
    T_opt = float(w[-1])

    X_nodes = X_flat.reshape((N, nx))  # shape (N, nx)
    U_opt = U_flat.reshape((N, nu))  # shape (N, nu)

    # 2) Integrate the final segment
    dt = T_opt / N
    x_last = X_nodes[-1, :]  # shape (nx,)
    u_last = U_opt[-1, :]  # shape (nu,)

    # Call integrator, receive a (nx,1) or (nx,) CasADi DM/SX
    x1_mx = integrator(x_last, u_last, dt)
    # If integrator returns a tuple/list, take the first element
    if isinstance(x1_mx, (tuple, list)):
        x1_mx = x1_mx[0]
    # Convert to NumPy and squeeze to (nx,)
    x1_np = np.array(x1_mx.full()).squeeze()

    assert x1_np.shape == (nx,), f"x1_np shape is {x1_np.shape}, expected {(nx,)}"

    # 3) Concatenate
    X_full = np.vstack([X_nodes, x1_np])  # shape (N+1, nx)
    return X_full, U_opt, T_opt


def warm_start_from_w(w_old, N_old, N_new, integrator, nx=7, nu=3):
    """
    Generate an initial guess w0 for a smaller shooting grid (N_new)
    by spline-interpolating the solution w_old from a larger grid (N_old).

    Parameters
    ----------
    w_old : array-like, shape (N_old*nx + N_old*nu + 1,)
        Optimal decision vector from the big-grid solution.
    N_old : int
        Number of shooting intervals in w_old.
    N_new : int
        Desired smaller number of shooting intervals.
    integrator : casadi.Function
        Integrator to reconstruct the final state in unpack_solution.
    nx : int
        State dimension.
    nu : int
        Control dimension.

    Returns
    -------
    w0_new : ndarray, shape (N_new*nx + N_new*nu + 1,)
        Warm-start guess for the small-grid problem.
    """
    # 1) Unpack old solution into state (N_old+1,nx), control (N_old,nu), time
    X_old, U_old, T_old = unpack_solution(w_old, N_old, integrator, nx=nx, nu=nu)

    # 2) Spline-interpolate states over normalized time [0,1]
    t_old_state = np.linspace(0, 1, N_old + 1)
    t_new_state = np.linspace(0, 1, N_new + 1)
    X_new = np.zeros((N_new + 1, nx))
    for i in range(nx):
        cs = CubicSpline(t_old_state, X_old[:, i])
        X_new[:, i] = cs(t_new_state)

    # 3) Spline-interpolate controls at segment midpoints
    t_old_ctrl = (t_old_state[:-1] + t_old_state[1:]) / 2
    t_new_ctrl = (t_new_state[:-1] + t_new_state[1:]) / 2
    U_new = np.zeros((N_new, nu))
    for j in range(nu):
        csu = CubicSpline(t_old_ctrl, U_old[:, j])
        U_new[:, j] = csu(t_new_ctrl)

    # 4) Pack the new decision vector w0_new
    w0_new = np.zeros(N_new * nx + N_new * nu + 1)
    #   a) states s_0 ... s_{N_new-1}
    for k in range(N_new):
        w0_new[k * nx:(k + 1) * nx] = X_new[k]
    #   b) controls u_0 ... u_{N_new-1}
    for k in range(N_new):
        idx = N_new * nx + k * nu
        w0_new[idx:idx + nu] = U_new[k]
    #   c) final time
    w0_new[-1] = T_old

    return w0_new


def plot_trajectory(X, T, track=True, title='Trajectory with Track Constraints'):
    """
    Plot the vehicle trajectory and optionally the track boundaries.

    Parameters
    ----------
    X : ndarray of shape (N, nx)
        State trajectory with [cx, cy, ...] structure.
    T : float
        Final time (for title display).
    track : bool
        Whether to overlay track boundaries using Pl/Pu.
    title : str
        Plot title.
    """
    cx, cy = X[:, 0], X[:, 1]

    plt.figure(figsize=(10, 4))
    plt.plot(cx, cy, 'k-', linewidth=3, label='Vehicle trajectory')

    if track:
        x_plot = np.linspace(-30, 140, 500)
        B = 1.5
        Pl = np.array([Pl_vis(x) for x in x_plot])
        Pu = np.array([Pu_vis(x) for x in x_plot])
        plt.plot(x_plot, Pl, 'r--', label='Lower boundary $P_l(x)+B/2$')
        plt.plot(x_plot, Pu, 'b--', label='Upper boundary $P_u(x)-B/2$')
        plt.fill_between(x_plot, Pl, Pu, color='gray', alpha=0.2)

    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.title(f"{title}\nFinal time T = {T:.2f} s")
    # plt.axis('equal')
    plt.grid(True)
    plt.legend()
    # plt.tight_layout()
    plt.ylim(-5, 10)
    plt.show()


def show_interactive_simulation(N=50, dt=0.1, gear=2):
    """
    Display an interactive widget to simulate car trajectory
    with constant control input over time.

    Parameters
    ----------
    N : int
        Number of simulation steps.
    dt : float
        Time step.
    gear : int
        Gear setting (1-5).
    """
    import ipywidgets as widgets
    from ipywidgets import interactive_output, VBox
    from IPython.display import display
    from car_model import make_car_integrator

    F = make_car_integrator(gear)

    # Track data
    x_plot = np.linspace(-30, 140, 500)
    B = 1.5
    Pl_num = np.array([Pl_vis(xi) for xi in x_plot])
    Pu_num = np.array([Pu_vis(xi) for xi in x_plot])

    def simulate_and_plot(wd, FB, f):
        u = np.array([wd, FB, f, dt])
        x = np.array([-30.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        xs = [x]
        for _ in range(N):
            x = F(x0=x, p=u)['xf'].full().flatten()
            xs.append(x)
        xs = np.array(xs)
        cx = xs[:, 0]
        cy = xs[:, 1]

        plt.figure(figsize=(10, 4))
        plt.clf()
        plt.plot(x_plot, Pl_num, 'r--', label='Lower Boundary $P_l(x)+B/2$')
        plt.plot(x_plot, Pu_num, 'b--', label='Upper Boundary $P_u(x)-B/2$')
        plt.plot(cx, cy, 'k-', linewidth=2, label='Car Trajectory')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Trajectory with Variable Constant Controls')
        plt.legend()
        # plt.axis('equal')
        plt.grid(True)
        # plt.tight_layout()
        plt.ylim(-5, 10)
        plt.show()

    # Sliders
    wd_slider = widgets.FloatSlider(min=-0.5, max=0.5, step=0.01, value=0.0, description='wd')
    FB_slider = widgets.IntSlider(min=0, max=15000, step=500, value=3000, description='FB')
    f_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.5, description='f')

    ui = VBox([wd_slider, FB_slider, f_slider])
    out = interactive_output(simulate_and_plot, {'wd': wd_slider, 'FB': FB_slider, 'f': f_slider})

    display(ui, out)
