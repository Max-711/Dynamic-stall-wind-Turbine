"""Single-DOF blade oscillator coupled to a dynamic stall model.

    m*xdd + c*xd + k1*x + k3*x^3 = F_y

    U_x = U*cos(alpha_mean)          U_y = U*sin(alpha_mean)
    alpha = arctan((U_y - xd)/U_x)   V_rel = sqrt(U_x^2 + (U_y - xd)^2)
    F_y   = 0.5*rho*V_rel^2*chord*(C_L*cos(alpha) + C_D*sin(alpha))

"""

import numpy as np
from input import Params, DEG


class Oscillator:
    def __init__(self, params: Params, aero):
        self.p = params
        self.aero = aero
        self.n = 2 + aero.n_states

    def kinematics(self, xd):
        p = self.p
        Ux = p.U * np.cos(p.alpha_mean)
        Uy = p.U * np.sin(p.alpha_mean)
        alpha = np.arctan2(Uy - xd, Ux)
        V = np.hypot(Ux, Uy - xd)
        return alpha, V

    def y0(self, x0=1e-3, xd0=0.0):
        a, V = self.kinematics(xd0)
        return np.concatenate([[x0, xd0], self.aero.y0(a, V)])

    def rhs(self, y):
        p = self.p
        x, xd = y[0], y[1]
        ya = y[2:]
        alpha, V = self.kinematics(xd)
        cl, cd = self.aero.coeffs(ya, alpha, 0.0, V)
        F = 0.5 * p.rho * V ** 2 * p.chord * (cl * np.cos(alpha) + cd * np.sin(alpha))
        xdd = (F - p.c_damp * xd - p.k1 * x - p.k3 * x ** 3) / p.mass
        return np.concatenate([[xd, xdd], self.aero.rhs(ya, alpha, 0.0, V)])

    def run(self, x0=1e-3, xd0=0.0, t_end=None, dt=None):
        p = self.p
        t_end = p.t_end if t_end is None else t_end
        dt = p.dt if dt is None else dt
        n = int(t_end / dt)
        t = np.arange(n) * dt
        X = np.zeros((n, self.n))
        y = self.y0(x0, xd0)
        for i in range(n):
            X[i] = y
            k1 = self.rhs(y)
            k2 = self.rhs(y + dt / 2 * k1)
            k3 = self.rhs(y + dt / 2 * k2)
            k4 = self.rhs(y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return t, X

    @staticmethod
    def amplitude(x, frac=0.5):
        """Half peak-to-peak over the last `frac` of the series."""
        s = x[int(len(x) * (1 - frac)):]
        return 0.5 * (s.max() - s.min())


if __name__ == "__main__":
    from oye import Oye

    p = Params()
    print(f"alpha_mean {p.alpha_mean/DEG:.1f} deg,  zeta {p.zeta},  k3 {p.k3:.0f}\n")
    print(f"{'U [m/s]':>8} {'amplitude [mm]':>16} {'alpha range [deg]':>22}")
    for U in (3, 4, 6, 8):
        p.U = U
        osc = Oscillator(p, Oye(p.polar, p.chord, p.T_f))
        t, X = osc.run(t_end=30.0, dt=1e-3)
        A = osc.amplitude(X[:, 0]) * 1e3
        a, _ = osc.kinematics(X[len(X)//2:, 1])
        print(f"{U:8.1f} {A:16.4f} {a.min()/DEG:14.2f} .. {a.max()/DEG:.2f}")
    print("\n(short 30 s runs -- lengthen t_end for converged limit cycles)")
