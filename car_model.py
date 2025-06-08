# car_model.py
import casadi as ca
import numpy as np

def make_car_integrator(gear: int, dt: float) -> ca.Function:
    """
    Create CasADi integrator for the car model with fixed gear.
    
    Arguments:
    ----------
    gear : int
        Gear value, fixed over the whole time.
    dt : float
        Time step for integration.
        
    Returns:
    --------
    integrator : casadi.Function
        CasADi integrator F(x0, u) → x_next.
    """
    # === Declare symbolic variables ===
    x = ca.MX.sym('x', 7)      # [cx, cy, v, delta, beta, psi, wz]
    u = ca.MX.sym('u', 3)      # [wd, FB, f]

    cx, cy, v, delta, beta, psi, wz = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    wd, FB, f = u[0], u[1], u[2]

    # === Parameters from Gerdts/Kirches ===
    # vehicle
    m = 1239       # kg
    g = 9.81       # m/s^2
    lf = 1.19016   # m
    lr = 1.37484   # m
    eSP = 0.5      # m
    R = 0.302      # m
    Izz = 1752     # kg*m^2
    cw = 0.3       # air drag
    rho = 1.2495   # air density
    A = 1.4379     # effective area
    it = 3.91

    ig = [3.91, 2.002, 1.33, 1.0, 0.805]  # gear ratio
    igm = ig[gear - 1]  # select gear ratio

    # Pacejka parameters
    Bf, Br = 10.96, 12.67
    Cf, Cr = 1.3, 1.3
    Df, Dr = 4560.4, 3947.81
    Ef, Er = -0.5, -0.5

    # === Auxiliary: slip angles ===
    alpha_f = delta - ca.atan2(lf * wz - v * ca.sin(beta), v * ca.cos(beta))
    alpha_r = ca.atan2(lr * wz + v * ca.sin(beta), v * ca.cos(beta))

    # === Tire forces ===
    Fsf = Df * ca.sin(Cf * ca.atan(Bf * alpha_f - Ef * (Bf * alpha_f - ca.atan(Bf * alpha_f))))
    Fsr = Dr * ca.sin(Cr * ca.atan(Br * alpha_r - Er * (Br * alpha_r - ca.atan(Br * alpha_r))))

    # === Longitudinal force ===
    FBf = (2/3) * FB
    FBr = (1/3) * FB
    FRf = (m * lr * g / (lf + lr)) * (0.009 + 0.002 * v / 100 + 0.0003 * (v / 100) ** 4)
    FRr = (m * lf * g / (lf + lr)) * (0.009 + 0.002 * v / 100 + 0.0003 * (v / 100) ** 4)
    FAx = 0.5 * cw * rho * A * v**2

    # === Engine torque ===
    w_mot = igm * it * v / R
    f1 = 1 - ca.exp(-3 * f)
    f2 = -37.8 + 1.54 * w_mot - 0.0019 * w_mot ** 2
    f3 = -34.9 - 0.04775 * w_mot
    Mmot = f1 * f2 + (1 - f1) * f3
    Mwheel = igm * it * Mmot
    Flf = -FBf - FRf
    Flr = Mwheel / R - FBr - FRr

    # === Equations of motion ===
    #dx = ca.MX.sym('dx', 7)
    rhs = ca.vertcat(
    v * ca.cos(psi - beta),
    v * ca.sin(psi - beta),
    (Flr * ca.cos(beta) + Flf * ca.cos(delta + beta) - Fsr * ca.sin(beta) - Fsf * ca.sin(delta + beta) - FAx) / m,
    wd,
    wz - (Flr * ca.sin(beta) + Flf * ca.sin(delta + beta) + Fsr * ca.cos(beta) + Fsf * ca.cos(delta + beta)) / (m * v),
    wz,
    (Fsf * lf * ca.cos(delta) - Fsr * lr - rho * eSP + Flf * lf * ca.sin(delta)) / Izz
    )

    # === Integrator ===
    ode = {'x': x, 'p': u, 'ode': rhs}
    opts = {'tf': dt}
    integrator = ca.integrator('car_integrator', 'rk', ode, opts)

    return integrator
