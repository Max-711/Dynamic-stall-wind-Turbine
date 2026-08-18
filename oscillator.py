"""Single-DOF blade oscillator coupled to a dynamic stall model.

    m*ydd + c*yd + k1*y + k3*y^3 = F_y

    U_x = U*cos(alpha_mean)            U_y = U*sin(alpha_mean)
    alpha = arctan2(U_y - yd, U_x)    V_rel = hypot(U_x, U_y - yd)
    F_y   = 0.5*rho*V_rel^2*chord*(C_L*cos(alpha) + C_D*sin(alpha))
"""

import numpy as np
from input import Params, DEG
from oye import Oye
from IAG import IAG


class Oscillator:
    def __init__(self, params: Params, aero):
        self.p = params
        self.aero = aero
        self.n = 2 + aero.n_states

    def kinematics(self, yd):
        p = self.p
        Ux = p.U * np.cos(p.alpha_mean)
        Uy = p.U * np.sin(p.alpha_mean)
        w = Uy - yd
        V = np.hypot(Ux, w)
        return np.arctan2(w, Ux), V, -Ux / V ** 2

    def state(self, y=0.0, yd=0.0):
        """Assemble s, with the aerodynamic block steady at (alpha, V_rel)."""
        a, V, _ = self.kinematics(yd)
        return np.concatenate([[y, yd], self.aero.y0(float(a), float(V))])

    def rhs(self, s, ydd_lag=0.0):
        p = self.p
        y, yd = float(s[0]), float(s[1])
        ya = s[2:]
        a, V, da = self.kinematics(yd)
        a, V, ad = float(a), float(V), float(da) * ydd_lag
        cl, cd = self.aero.coeffs(ya, a, ad, V)
        F = 0.5 * p.rho * V ** 2 * p.chord * (cl * np.cos(a) + cd * np.sin(a))
        ydd = (F - p.c_damp * yd - p.k1 * y - p.k3 * y ** 3) / p.mass
        return np.concatenate([[yd, ydd], self.aero.rhs(ya, a, ad, V)])

    def run(self, s_init, t_end=None, dt=None, picard=1):
        p = self.p
        t_end = p.t_end if t_end is None else t_end
        dt = p.dt if dt is None else dt
        n = int(t_end / dt)
        t = np.arange(n) * dt
        S = np.zeros((n, self.n))
        A = np.zeros(n)
        s = np.asarray(s_init, float).copy()
        lag = 0.0
        for i in range(n):
            S[i] = s
            f1 = self.rhs(s, lag)
            for _ in range(picard):
                lag = float(f1[1])
                f1 = self.rhs(s, lag)
            A[i] = lag
            f2 = self.rhs(s + dt / 2 * f1, lag)
            f3 = self.rhs(s + dt / 2 * f2, lag)
            f4 = self.rhs(s + dt * f3, lag)
            s = s + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
            lag = float(f1[1])
        return t, S, A


if __name__ == "__main__":

    p = Params()
    p.U = 4.0
    models = {"Oye": Oye(p.polar, p.chord, p.T_f), "IAG": IAG(p.polar, p.chord)}

    print(f"alpha_mean {p.alpha_mean/DEG:.1f} deg,  U {p.U} m/s,  "
          f"zeta {p.zeta},  k3 {p.k3:.0f},  y(0) 1 mm,  t_end 20 s\n")

    for name, aero in models.items():
        osc = Oscillator(p, aero)
        t, S, A = osc.run(osc.state(y=1e-3), t_end=20.0, dt=1e-3)
        a, V, da_dyd = osc.kinematics(S[:, 1])
        ad = da_dyd * A
        print(f"{name}:  {osc.n} states {aero.names}")
        print(f"  finite            {np.isfinite(S).all()}")
        print(f"  y      [mm]       {S[:, 0].min()*1e3:8.3f} .. {S[:, 0].max()*1e3:8.3f}")
        print(f"  alpha  [deg]      {a.min()/DEG:8.3f} .. {a.max()/DEG:8.3f}")
        print(f"  V_rel  [m/s]      {V.min():8.3f} .. {V.max():8.3f}")
        print(f"  alpha_dot [deg/s] {ad.min()/DEG:8.3f} .. {ad.max()/DEG:8.3f}")
        if osc.n > 3:
            print(f"  vortex x5, x6 max {S[:, 6].max():.4g}, {S[:, 7].max():.4g}"
                  f"   (zero means alpha_dot never reached the gate)")
        print()