"""Single-DOF blade oscillator coupled to a dynamic stall model.

    (m + m_a)*ydd + c*yd + k1*y + k3*y^3 = F_y|_{alpha_dot = 0}

    U_x = U*cos(alpha_mean)           U_y = U*sin(alpha_mean)
    alpha = arctan2(U_y - yd, U_x)    V_rel = hypot(U_x, U_y - yd)
    alpha_dot = -U_x/V_rel^2 * ydd
    Vdot      = -(U_y - yd)*ydd/V_rel
    F_y = 0.5*rho*V_rel^2*chord*C_N
"""

import numpy as np
from input import Params, DEG
from oye import Oye
from IAG import IAG


class Oscillator:
    def __init__(self, params: Params, aero):
        self.p = params
        self.aero = aero
        self.n = 2 + aero.n_states #2(y, yd) + aero.n_states 

    def kinematics(self, yd):
        p = self.p
        Ux = p.U * np.cos(p.alpha_mean)
        Uy = p.U * np.sin(p.alpha_mean)
        w = Uy - yd
        V = np.hypot(Ux, w)
        return np.arctan2(w, Ux), V, -Ux / V ** 2

    def normalforce(self, ya, alpha, alpha_dot, V):
        p = self.p
        cl, cd = self.aero.coeffs(ya, alpha, alpha_dot, V)
        return 0.5 * p.rho * V ** 2 * p.chord * (cl * np.cos(alpha) + cd * np.sin(alpha))

    def added_mass(self, a):
        """
            C_N_I = 4*K_a*(c/V)*alpha_dot                         (38)
            kappa = dF_y/d(alpha_dot) = 0.5*rho*V^2*c * 4*K_a*c/V
                  = 2*rho*V*c^2*K_a
            m_a   = kappa*U_x/V^2 = 2*rho*c^2*K_a*U_x/V

        """
        p = self.p
        K_a = getattr(self.aero, "K_a", 0.0)
        if K_a == 0.0:
            return 0.0
        return 2 * p.rho * p.chord * self.aero.chord * K_a * np.cos(a)  # cos(a) = U_x/V

    def state(self, y=0.0, yd=0.0):
        """Assemble s, with the aerodynamic block steady at (alpha, V_rel)."""
        a, V, _ = self.kinematics(yd)
        return np.concatenate([[y, yd], self.aero.y0(float(a), float(V))])

    def accel(self, s):
        """

        Returns (ydd, alpha, V_rel, alpha_dot, m_a).
        """
        p = self.p
        y, yd = float(s[0]), float(s[1])
        ya = s[2:]
        alpha, V, dad = self.kinematics(yd)        # dad = d(alpha_dot)/d(ydd)
        alpha, V, dad = float(alpha), float(V), float(dad)

        F0 = self.normalforce(ya, alpha, 0.0, V)         
        m_a = self.added_mass(alpha)               # eq:mass
        ydd = (F0 - p.c_damp * yd - p.k1 * y - p.k3 * y ** 3) / (p.mass + m_a)
        return ydd, alpha, V, dad * ydd, m_a


    def rhs(self, s):
        ydd, alpha, V, alpha_dot, _ = self.accel(s)
        return np.concatenate([[float(s[1]), ydd],
                           self.aero.rhs(s[2:], alpha, alpha_dot, V)])

   
    def run(self, s_init, t_end=None, dt=None):
        p = self.p
        t_end = p.t_end if t_end is None else t_end
        dt = p.dt if dt is None else dt
        n = int(t_end / dt)
        t = np.arange(n) * dt
        S = np.zeros((n, self.n))
        D = {k: np.zeros(n) for k in ("ydd", "alpha", "V", "alpha_dot", "m_a")}
        vor = np.zeros(n, bool)
        has_vortex = hasattr(self.aero, "vortex_on")
        s = np.asarray(s_init, float).copy()
        for i in range(n):
            S[i] = s
            ydd, alpha, V, alpha_dot, m_a = self.accel(s)
            D["ydd"][i], D["alpha"][i], D["V"][i] = ydd, alpha, V
            D["alpha_dot"][i], D["m_a"][i] = alpha_dot, m_a
            if has_vortex:
                vor[i] = bool(self.aero.vortex_on(s[2:], alpha, alpha_dot, V))
            f1 = self.rhs(s)
            f2 = self.rhs(s + dt / 2 * f1)
            f3 = self.rhs(s + dt / 2 * f2)
            f4 = self.rhs(s + dt * f3)
            s = s + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        D["vortex"] = vor
        return t, S, D


if __name__ == "__main__":

    p = Params()
    p.U = 4.0
    models = {"Oye": Oye(p.polar, p.chord, p.T_f),
              "IAG": IAG(p.polar, p.chord)}

    print(f"alpha_mean {p.alpha_mean/DEG:.1f} deg,  U {p.U} m/s,  "
          f"zeta {p.zeta},  k3 {p.k3:.0f},  y(0) 1 mm,  t_end 20 s\n")

    for name, aero in models.items():
        osc = Oscillator(p, aero)
        s0 = osc.state(y=1e-3)
        t, S, D = osc.run(s0, t_end=20.0, dt=1e-3)
        half = slice(len(t) // 2, None)          # discard the transient
        print(f"{name}:  {osc.n} states {aero.names}")
        print(f"  finite              {np.isfinite(S).all()}")
        print(f"  y      [mm]         {S[:, 0].min()*1e3:8.3f} .. {S[:, 0].max()*1e3:8.3f}")
        print(f"  alpha  [deg]        {D['alpha'].min()/DEG:8.3f} .. {D['alpha'].max()/DEG:8.3f}")
        print(f"  V_rel  [m/s]        {D['V'].min():8.3f} .. {D['V'].max():8.3f}")
        print(f"  alpha_dot [deg/s]   {D['alpha_dot'].min()/DEG:8.3f} .. {D['alpha_dot'].max()/DEG:8.3f}")
        print(f"  m_a    [kg/m]       {D['m_a'].min():8.3f} .. {D['m_a'].max():8.3f}"
              f"   ({100*D['m_a'].mean()/p.mass:.2f} % of m)")
        if hasattr(aero, "vortex_on"):
            print(f"  vortex duty [%]     {100*D['vortex'][half].mean():8.2f}"
                  f"   (0 or 100 means the trigger is degenerate)")
            print(f"  x5, tau_v max       {S[:, 6].max():.4g}, {S[:, 7].max():.4g}")
        print()
