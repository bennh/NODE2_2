import numpy as np
import matplotlib.pyplot as plt
from track_constraints import Pl_vis, Pu_vis

def unpack_solution(w_opt, N, nx=7, nu=3):
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
    w_opt = np.array(w_opt).flatten()
    X_flat = w_opt[:N * nx]
    U_flat = w_opt[N * nx:N * (nx + nu)]
    T_opt = w_opt[-1]

    X_opt = X_flat.reshape((N, nx))
    U_opt = U_flat.reshape((N, nu))
    return X_opt, U_opt, T_opt


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
        Pl = np.array([Pl_vis(x) for x in x_plot]) + B / 2
        Pu = np.array([Pu_vis(x) for x in x_plot]) - B / 2
        plt.plot(x_plot, Pl, 'r--', label='Lower boundary $P_l(x)+B/2$')
        plt.plot(x_plot, Pu, 'b--', label='Upper boundary $P_u(x)-B/2$')
        plt.fill_between(x_plot, Pl, Pu, color='gray', alpha=0.2)

    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.title(f"{title}\nFinal time T = {T:.2f} s")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
