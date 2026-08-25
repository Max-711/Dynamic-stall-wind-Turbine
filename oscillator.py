"""Single-DOF blade oscillator coupled to a dynamic stall model.

    (m + m_a)*ydd + c*yd + k1*y + k3*y^3 = F_y|_{alpha_dot = 0}

    U_x = U*cos(alpha_mean)           U_y = U*sin(alpha_mean)
    alpha = arctan2(U_y - yd, U_x)    V_rel = hypot(U_x, U_y - yd)
    alpha_dot = -U_x/V_rel^2 * ydd
    F_y = 0.5*rho*V_rel^2*chord*(C_L*cos(alpha) + C_D*sin(alpha))

The aerodynamic force depends on ydd through alpha_dot, so force and
acceleration are coupled algebraically.  The impulsive normal force is
linear in alpha_dot and is the only route by which alpha_dot reaches the
coefficients, so the loop closes in one substitution: the alpha_dot part
of F_y is proportional to ydd and moves to the left hand side as an
apparent mass m_a, given in closed form by added_mass() below.  No
iteration is involved and there is no lagged ydd.
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

    def force(self, ya, a, ad, V):
        p = self.p
        cl, cd = self.aero.coeffs(ya, a, ad, V)
        return 0.5 * p.rho * V ** 2 * p.chord * (cl * np.cos(a) + cd * np.sin(a))

    def added_mass(self, a):
        """m_a of the thesis, Ch5 eq:mass.

            C_N_I = 4*K_a*L/V * alpha_dot                 L = length in C_N_I
            dC_L/d(alpha_dot)  = cos(alpha) * 4*K_a*L/V   C_L = C_N*cos - C_T*sin
            dC_D/d(alpha_dot)  = 0                        eq (53) has no alpha_dot

            kappa = dF_y/d(alpha_dot)
                  = 0.5*rho*V^2*c * cos(alpha) * cos(alpha)*4*K_a*L/V
                  = 2*rho*V*c*L*K_a*cos^2(alpha)

            m_a   = kappa*U_x/V^2 = 2*rho*c*L*K_a*cos^3(alpha)

        The third power of the cosine, rather than the first, follows from
        the force convention of Ch3 eq:force_cn,
        F_y = 0.5*rho*V^2*c*(C_L*cos + C_D*sin).  That equals 0.5*rho*V^2*c*C_N
        only when C_L and C_D are the exact rotation of (C_N, C_T), and the
        drag of Bangga eq (53) is not, so C_N_I reaches F_y through cos^2.

        A model with no impulsive term carries no K_a, and m_a is zero there:
        the structural equation reduces to the plain one.
        """
        p = self.p
        K_a = getattr(self.aero, "K_a", 0.0)
        if K_a == 0.0:
            return 0.0
        L = self.aero.chord            # the length inside C_N_I  --  [I7]
        return 2 * p.rho * p.chord * L * K_a * np.cos(a) ** 3

    def state(self, y=0.0, yd=0.0):
        """Assemble s, with the aerodynamic block steady at (alpha, V_rel)."""
        a, V, _ = self.kinematics(yd)
        return np.concatenate([[y, yd], self.aero.y0(float(a), float(V))])

    def accel(self, s):
        """Evaluation order of the thesis, Ch5 sec 5.3.

        Returns (ydd, alpha, V_rel, alpha_dot, m_a).
        """
        p = self.p
        y, yd = float(s[0]), float(s[1])
        ya = s[2:]
        a, V, dad = self.kinematics(yd)        # dad = d(alpha_dot)/d(ydd)
        a, V, dad = float(a), float(V), float(dad)

        F0 = self.force(ya, a, 0.0, V)         # everything free of alpha_dot
        m_a = self.added_mass(a)               # eq:mass

        ydd = (F0 - p.c_damp * yd - p.k1 * y - p.k3 * y ** 3) / (p.mass + m_a)
        return ydd, a, V, dad * ydd, m_a

    def rhs(self, s):
        ydd, a, V, ad, _ = self.accel(s)
        return np.concatenate([[float(s[1]), ydd], self.aero.rhs(s[2:], a, ad, V)])

    # ---- checks on the two assumptions the closure rests on --------------

    def affine_residual(self, s):
        """Departure of F_y from being affine in alpha_dot, per unit force.

        The closure needs F_y = F_y(0) + kappa*alpha_dot exactly.  Any term
        quadratic in alpha_dot shows up here as a nonzero second difference.
        """
        ya = s[2:]
        a, V, _ = self.kinematics(float(s[1]))
        a, V = float(a), float(V)
        f0, f1, f2 = (self.force(ya, a, d, V) for d in (0.0, 1.0, 2.0))
        return abs(f2 - 2 * f1 + f0) / max(abs(f0), abs(f1 - f0), 1e-30)

    def mass_residual(self, s):
        """Relative disagreement between eq:mass and the model's own force.

        added_mass() is the analytic derivative of whatever force() returns.
        Reading the same slope straight off two evaluations of force() must
        reproduce it.  If the aerodynamic model is edited without eq:mass
        following, this stops being at round-off.
        """
        ya = s[2:]
        a, V, dad = self.kinematics(float(s[1]))
        a, V, dad = float(a), float(V), float(dad)
        kappa = self.force(ya, a, 1.0, V) - self.force(ya, a, 0.0, V)
        measured = -kappa * dad
        formula = self.added_mass(a)
        return abs(measured - formula) / max(abs(measured), abs(formula), 1e-30)

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
            ydd, a, V, ad, m_a = self.accel(s)
            D["ydd"][i], D["alpha"][i], D["V"][i] = ydd, a, V
            D["alpha_dot"][i], D["m_a"][i] = ad, m_a
            if has_vortex:
                vor[i] = bool(self.aero.vortex_on(s[2:], a, ad, V))
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
    models = {"Oye": Oye(p.polar, p.chord, p.T_f), "IAG": IAG(p.polar, p.chord)}

    print(f"alpha_mean {p.alpha_mean/DEG:.1f} deg,  U {p.U} m/s,  "
          f"zeta {p.zeta},  k3 {p.k3:.0f},  y(0) 1 mm,  t_end 20 s\n")

    for name, aero in models.items():
        osc = Oscillator(p, aero)
        s0 = osc.state(y=1e-3)
        t, S, D = osc.run(s0, t_end=20.0, dt=1e-3)
        half = slice(len(t) // 2, None)          # discard the transient
        print(f"{name}:  {osc.n} states {aero.names}")
        print(f"  finite              {np.isfinite(S).all()}")
        print(f"  affine residual     {osc.affine_residual(s0):.2e}   (F_y linear in alpha_dot)")
        print(f"  eq:mass residual    {osc.mass_residual(s0):.2e}   (added_mass vs the model)")
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
