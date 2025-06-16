import casadi as ca
import numpy as np

def make_car_integrator(gear: int) -> ca.Function:
    """
    Create CasADi integrator for the car model with fixed gear,
    where time step dt is passed as a symbolic input.

    Returns
    -------
    integrator : casadi.Function
        CasADi integrator F(x0, [u; dt]) → x_next
    """
    # Symbolic variables
    x = ca.MX.sym('x', 7)
    u = ca.MX.sym('u', 3)
    dt = ca.MX.sym('dt')  # Time step as symbolic input
    p = ca.vertcat(u, dt)

    # State unpacking
    cx, cy, v, delta, beta, psi, wz = x[0:7]
    wd, FB, f = u[0:3]

    # Vehicle parameters
    m = 1239
    g = 9.81
    lf, lr = 1.19016, 1.37484
    eSP = 0.5
    R = 0.302
    Izz = 1752
    cw, rho, A = 0.3, 1.2495, 1.4379
    it = 3.91
    ig = [3.91, 2.002, 1.33, 1.0, 0.805]
    igm = ig[gear - 1]

    # Tire model
    Bf, Br = 10.96, 12.67
    Cf, Cr = 1.3, 1.3
    Df, Dr = 4560.4, 3947.81
    Ef, Er = -0.5, -0.5

    alpha_f = delta - ca.atan2(lf * wz - v * ca.sin(beta), v * ca.cos(beta))
    alpha_r = ca.atan2(lr * wz + v * ca.sin(beta), v * ca.cos(beta))

    Fsf = Df * ca.sin(Cf * ca.atan(Bf * alpha_f - Ef * (Bf * alpha_f - ca.atan(Bf * alpha_f))))
    Fsr = Dr * ca.sin(Cr * ca.atan(Br * alpha_r - Er * (Br * alpha_r - ca.atan(Br * alpha_r))))

    FBf = (2/3) * FB
    FBr = (1/3) * FB
    FRf = (m * lr * g / (lf + lr)) * (0.009 + 0.002 * v / 100 + 0.0003 * (v / 100)**4)
    FRr = (m * lf * g / (lf + lr)) * (0.009 + 0.002 * v / 100 + 0.0003 * (v / 100)**4)
    FAx = 0.5 * cw * rho * A * v**2

    w_mot = igm * it * v / R
    f1 = 1 - ca.exp(-3 * f)
    f2 = -37.8 + 1.54 * w_mot - 0.0019 * w_mot**2
    f3 = -34.9 - 0.04775 * w_mot
    Mmot = f1 * f2 + (1 - f1) * f3
    Mwheel = igm * it * Mmot
    Flf = -FBf - FRf
    Flr = Mwheel / R - FBr - FRr

    # Dynamics
    rhs = ca.vertcat(
        v * ca.cos(psi - beta),
        v * ca.sin(psi - beta),
        (Flr * ca.cos(beta) + Flf * ca.cos(delta + beta) - Fsr * ca.sin(beta) - Fsf * ca.sin(delta + beta) - FAx) / m,
        wd,
        wz - (Flr * ca.sin(beta) + Flf * ca.sin(delta + beta) + Fsr * ca.cos(beta) + Fsf * ca.cos(delta + beta)) / (m * v),
        wz,
        (Fsf * lf * ca.cos(delta) - Fsr * lr - rho * eSP + Flf * lf * ca.sin(delta)) / Izz
    )

    # Create integrator with symbolic dt
    ode = {'x': x, 'p': p, 'ode': rhs}
    opts = {'tf': 1.0}  # Time scaling via dt included in p
    integrator = ca.integrator('car_integrator', 'rk', ode, opts)

    return integrator
