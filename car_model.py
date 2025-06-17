import casadi as ca

def make_car_integrator(gear: int) -> ca.Function:
    """
    Create CasADi integrator for the car model with time-scaled input.
    
    Parameters
    ----------
    gear : int
        Selected gear (1-5)

    Returns
    -------
    integrator : casadi.Function
        Function F(x0, p) → x_next, where p = [wd, FB, f, dt]
    """
    # Symbolic variables
    x = ca.MX.sym('x', 7)  # [cx, cy, v, delta, beta, psi, wz]
    p = ca.MX.sym('p', 4)  # [wd, FB, f, dt]

    wd, FB, f, dt = p[0], p[1], p[2], p[3]
    cx, cy, v, delta, beta, psi, wz = x[0], x[1], x[2], x[3], x[4], x[5], x[6]

    # Vehicle parameters
    m = 1239
    g = 9.81
    lf, lr = 1.19016, 1.37484
    eSP, R = 0.5, 0.302
    Izz = 1752
    cw, rho, A = 0.3, 1.2495, 1.4379
    it = 3.91
    ig = [3.91, 2.002, 1.33, 1.0, 0.805]
    igm = ig[gear - 1]

    # Tire model parameters
    Bf, Br = 10.96, 12.67
    Cf, Cr = 1.3, 1.3
    Df, Dr = 4560.4, 3947.81
    Ef, Er = -0.5, -0.5

    # Slip angles
    alpha_f = delta - ca.atan2(lf * wz - v * ca.sin(beta), v * ca.cos(beta))
    alpha_r = ca.atan2(lr * wz + v * ca.sin(beta), v * ca.cos(beta))

    # Tire forces
    Fsf = Df * ca.sin(Cf * ca.atan(Bf * alpha_f - Ef * (Bf * alpha_f - ca.atan(Bf * alpha_f))))
    Fsr = Dr * ca.sin(Cr * ca.atan(Br * alpha_r - Er * (Br * alpha_r - ca.atan(Br * alpha_r))))

    # Braking and resistance forces
    FBf = (2 / 3) * FB
    FBr = (1 / 3) * FB
    FRf = (m * lr * g / (lf + lr)) * (0.009 + 0.002 * v / 100 + 0.0003 * (v / 100)**4)
    FRr = (m * lf * g / (lf + lr)) * (0.009 + 0.002 * v / 100 + 0.0003 * (v / 100)**4)
    FAx = 0.5 * cw * rho * A * v**2

    # Engine model
    w_mot = igm * it * v / R
    f1 = 1 - ca.exp(-3 * f)
    f2 = -37.8 + 1.54 * w_mot - 0.0019 * w_mot ** 2
    f3 = -34.9 - 0.04775 * w_mot
    Mmot = f1 * f2 + (1 - f1) * f3
    Mwheel = igm * it * Mmot

    # Longitudinal forces
    Flf = -FBf - FRf
    Flr = Mwheel / R - FBr - FRr

    # Dynamics equations
    rhs = ca.vertcat(
        v * ca.cos(psi - beta),
        v * ca.sin(psi - beta),
        (Flr * ca.cos(beta) + Flf * ca.cos(delta + beta)
         - Fsr * ca.sin(beta) - Fsf * ca.sin(delta + beta) - FAx) / m,
        wd,
        wz - (Flr * ca.sin(beta) + Flf * ca.sin(delta + beta)
              + Fsr * ca.cos(beta) + Fsf * ca.cos(delta + beta)) / (m * v + 1e-3),
        wz,
        (Fsf * lf * ca.cos(delta) - Fsr * lr + Flf * lf * ca.sin(delta)) / Izz
    )

    # CasADi integrator
    ode = {'x': x, 'p': p, 'ode': rhs}
    opts = {}  # tf is dynamic
    integrator = ca.integrator('car_integrator', 'rk', ode, opts)
    return integrator


def make_scaled_integrator(gear: int) -> ca.Function:
    base_integrator = make_car_integrator(gear)

    x0 = ca.MX.sym('x0', 7)
    u = ca.MX.sym('u', 3)
    dt = ca.MX.sym('dt')
    p = ca.vertcat(u, 1.0)  # tf=1.0 ⇒ unit time step

    x1_unit = base_integrator(x0=x0, p=p)['xf']
    x1_scaled = x0 + dt * (x1_unit - x0)  # Linear time-scaling

    return ca.Function('scaled_integrator', [x0, u, dt], [x1_scaled])