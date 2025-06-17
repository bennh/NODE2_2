# track_constraints.py
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Numerical functions defining the track boundaries (for visualization)
def Pl_vis(x):
    h2 = 3.5
    if x <= 44: return 0
    if x <= 44.5: return 4 * h2 * (x - 44) ** 3
    if x <= 45: return 4 * h2 * (x - 45) ** 3 + h2
    if x <= 70: return h2
    if x <= 70.5: return 4 * h2 * (70 - x) ** 3 + h2
    if x <= 71: return 4 * h2 * (71 - x) ** 3
    return 0

def Pu_vis(x):
    B = 1.5
    h1 = 1.1 * B + 0.25
    h3 = 1.2 * B + 3.75
    h4 = 1.3 * B + 0.25
    if x <= 15: return h1
    if x <= 15.5: return 4 * (h3 - h1) * (x - 15) ** 3 + h1
    if x <= 16: return 4 * (h3 - h1) * (x - 16) ** 3 + h3
    if x <= 94: return h3
    if x <= 94.5: return 4 * (h3 - h4) * (94 - x) ** 3 + h3
    if x <= 95: return 4 * (h3 - h4) * (95 - x) ** 3 + h4
    return h4

# CasADi-compatible expressions for symbolic computation
def Pl_expr(x):
    h2 = 3.5
    return ca.if_else(x <= 44, 0,
           ca.if_else(x <= 44.5, 4 * h2 * (x - 44)**3,
           ca.if_else(x <= 45, 4 * h2 * (x - 45)**3 + h2,
           ca.if_else(x <= 70, h2,
           ca.if_else(x <= 70.5, 4 * h2 * (70 - x)**3 + h2,
           ca.if_else(x <= 71, 4 * h2 * (71 - x)**3,
           0))))))

def Pu_expr(x):
    B = 1.5
    h1 = 1.1 * B + 0.25
    h3 = 1.2 * B + 3.75
    h4 = 1.3 * B + 0.25
    return ca.if_else(x <= 15, h1,
           ca.if_else(x <= 15.5, 4 * (h3 - h1) * (x - 15)**3 + h1,
           ca.if_else(x <= 16, 4 * (h3 - h1) * (x - 16)**3 + h3,
           ca.if_else(x <= 94, h3,
           ca.if_else(x <= 94.5, 4 * (h3 - h4) * (94 - x)**3 + h3,
           ca.if_else(x <= 95, 4 * (h3 - h4) * (95 - x)**3 + h4,
           h4))))))

class TrackConstraints:
    def __init__(self):
        self.B = 1.5

    def track_constraints_function(self):
        X = ca.MX.sym('X', 2)
        x_pos, y_pos = X[0], X[1]

        Pl = Pl_expr(x_pos) + self.B / 2
        Pu = Pu_expr(x_pos) - self.B / 2

        g1 = Pl - y_pos  # Must be above lower boundary
        g2 = y_pos - Pu  # Must be below upper boundary

        return ca.Function('track_constraints', [X], [ca.vertcat(g1, g2)])

    def visualize_track(self):
        x_plot = np.linspace(-30, 140, 500)
        Pl_raw = np.array([Pl_vis(xi) for xi in x_plot]) + self.B / 2
        Pu_raw = np.array([Pu_vis(xi) for xi in x_plot]) - self.B / 2

        plt.figure(figsize=(10, 4))
        plt.plot(x_plot, Pl_raw, 'r--', label='Lower boundary $P_l(x)$')
        plt.plot(x_plot, Pu_raw, 'b--', label='Upper boundary $P_u(x)$')
        plt.fill_between(x_plot, Pl_raw, Pu_raw, color='gray', alpha=0.2)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Track Constraints Visualization')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

# CasADi function to use in OCP
def track_constraints():
    tc = TrackConstraints()
    return tc.track_constraints_function()

# Visualization example
if __name__ == '__main__':
    tc = TrackConstraints()
    tc.visualize_track()
