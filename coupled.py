"""Closed-form coupled IAG / oscillator system.

The aerodynamic normal force is *affine* in the angle-of-attack rate,

    F_y(z, alpha_dot) = F_y(z, 0) + Lambda(z) * alpha_dot,
    Lambda(z) = dF_y/dalpha_dot        (non-circulatory / added-mass term)

and the section kinematics give alpha_dot as a linear function of the
structural acceleration,

    alpha = arctan((U_y - xd)/U_x)  =>  d alpha/d xd = -U_x/V^2 = kappa,
    alpha_dot = kappa * xdd.

Substituting into the equation of motion

    m*xdd + c*xd + k1*x + k3*x^3 = F_y(z, alpha_dot)

and collecting xdd on the left gives an explicit ODE,

    (m - Lambda*kappa) * xdd = F_y(z, 0) - c*xd - k1*x - k3*x^3,

i.e. the apparent "algebraic loop" is just an added mass.  No sub-iteration,
no freezing of the aerodynamic force over a time step, no partitioned time
integration -- the model is a closed-form system  zdot = F(z)  that any
ODE solver can advance.

States (n = 8):   z = [x, xd, x1, x2, x3, x4, x5, tau_v]
Parameters:       U (wind speed), alpha_mean (inflow angle)
"""

import numpy as np
from input import Params, DEG


class CoupledOscillator:

    def __init__(self, params: Params, aero, force_ext=None):
        self.p = params
        self.aero = aero
        self.n = 2 + aero.n_states
        self.names = ["x", "xd"] + list(aero.names)
        # optional prescribed force F(t) replacing the aerodynamic model,
        # used for verification of the structural solver against hand solutions
        self.force_ext = force_ext

    # ---------------------------------------------------------------- kinematics
    def kinematics(self, xd):
        """alpha, V and kappa = d(alpha)/d(xd) = d(alpha_dot)/d(xdd)."""
        p = self.p
        Ux = p.U * np.cos(p.alpha_mean)
        Uy = p.U * np.sin(p.alpha_mean)
        alpha = np.arctan2(Uy - xd, Ux)
        V = np.hypot(Ux, Uy - xd)
        kappa = -Ux / V ** 2
        return alpha, V, kappa

    # ---------------------------------------------------------------- force
    def force(self, ya, alpha, alpha_dot, V):
        p = self.p
        cl, cd = self.aero.coeffs(ya, alpha, alpha_dot, V)
        return 0.5 * p.rho * V ** 2 * p.chord * (cl * np.cos(alpha) + cd * np.sin(alpha))

    def added_mass(self, ya, alpha, V):
        """Lambda = dF_y/d(alpha_dot).  Exact, because F_y is affine in alpha_dot."""
        return self.force(ya, alpha, 1.0, V) - self.force(ya, alpha, 0.0, V)

    # ---------------------------------------------------------------- rhs
    def rhs(self, t, z):
        p = self.p
        x, xd, ya = z[0], z[1], z[2:]
        alpha, V, kappa = self.kinematics(xd)

        if self.force_ext is None:
            F0 = self.force(ya, alpha, 0.0, V)      # circulatory part
            Lam = self.added_mass(ya, alpha, V)     # dF/d(alpha_dot)
        else:
            F0, Lam = float(self.force_ext(t)), 0.0

        m_eff = p.mass - Lam * kappa                # generalised (added) mass
        xdd = (F0 - p.c_damp * xd - p.k1 * x - p.k3 * x ** 3) / m_eff

        if self.force_ext is not None:              # structural solver only
            return np.concatenate([[xd, xdd], np.zeros(self.aero.n_states)])

        alpha_dot = kappa * xdd                     # now known, exactly
        return np.concatenate([[xd, xdd], self.aero.rhs(ya, alpha, alpha_dot, V)])

    def rhs_z(self, z):
        """Autonomous form, for fsolve / Jacobian."""
        return self.rhs(0.0, z)

    # ---------------------------------------------------------------- uncoupled reference
    def rhs_uncoupled(self, t, z):
        """The old model: alpha_dot forced to zero (no coupling at all)."""
        p = self.p
        x, xd, ya = z[0], z[1], z[2:]
        alpha, V, _ = self.kinematics(xd)
        F = self.force(ya, alpha, 0.0, V)
        xdd = (F - p.c_damp * xd - p.k1 * x - p.k3 * x ** 3) / p.mass
        return np.concatenate([[xd, xdd], self.aero.rhs(ya, alpha, 0.0, V)])

    # ---------------------------------------------------------------- init
    def z0(self, x0=1e-3, xd0=0.0):
        alpha, V, _ = self.kinematics(xd0)
        return np.concatenate([[x0, xd0], self.aero.y0(alpha, V)])


if __name__ == "__main__":
    from IAG_Continuous import IAG

    p = Params()
    m = CoupledOscillator(p, IAG(p.polar, p.chord))
    z = m.z0()
    alpha, V, kappa = m.kinematics(0.0)

    # 1. is the force really affine in alpha_dot?
    ad = np.array([0.0, 0.5, 1.0, 2.0])
    F = np.array([m.force(z[2:], alpha, a, V) for a in ad])
    lin = F[0] + (F[2] - F[0]) * ad
    print("affine check  max|F - linear fit| =", f"{np.max(np.abs(F - lin)):.3e}")

    # 2. how big is the added mass?
    Lam = m.added_mass(z[2:], alpha, V)
    m_add = -Lam * kappa
    print(f"kappa = dalpha/dxd     {kappa:+.5f} rad/(m/s)")
    print(f"Lambda = dF/dalpha_dot {Lam:+.3f} N/(rad/s)")
    print(f"added mass             {m_add:+.3f} kg/m   "
          f"({100*m_add/p.mass:+.2f} % of m = {p.mass} kg/m)")

    # 3. analytic cross-check:  m_add = 2*rho*c^2*K_a*cos^3(alpha)
    a_th = 2 * p.rho * p.chord ** 2 * m.aero.K_a * np.cos(alpha) ** 3
    print(f"analytic 2*rho*c^2*Ka*cos^3(alpha) = {a_th:.3f} kg/m")
