import numpy as np
from input import Polar, DEG


class Oye:
    """
        tau * df/dt = f_st(alpha) - f,      tau = T_f * chord / V
        C_L = f*C_L_att + (1-f)*C_L_fs

    Static limit: f -> f_st reproduces the measured polar exactly.
    """

    n_states = 1
    names = ["f"]

    def __init__(self, polar: Polar, chord: float, T_f: float = 1.7):
        self.polar = polar
        self.chord = chord
        self.T_f = T_f

    def y0(self, alpha, V):
        return np.array([float(self.polar.f_sep(alpha))])

    def rhs(self, y, alpha, alpha_dot, V):
        tau = max(self.T_f * self.chord / max(V, 1e-9), 1e-9) #防止V=0
        return np.array([(float(self.polar.f_sep(alpha)) - y[0]) / tau])

    def coeffs(self, y, alpha, alpha_dot, V):
        """Return (C_L, C_D)."""
        f = float(np.clip(y[0], 0.0, 1.0))
        cl = f * float(self.polar.cl_inv(alpha)) + (1 - f) * float(self.polar.cl_fs(alpha))
        return cl, float(self.polar.cd(alpha))

if __name__ == "__main__":
    p = Polar()
    m = Oye(p, chord=0.25)
    print("static consistency (f held at f_st):\n")
    print(f"{'alpha':>7} {'model':>9} {'table':>9} {'error':>11}")
    err = 0.0
    for d in (8, 10, 12, 14, 15.3, 18, 20, 22, 25, 28):
        a = d * DEG
        cl, _ = m.coeffs(m.y0(a, 10.0), a, 0.0, 10.0)
        e = cl - float(p.cl(a))
        err = max(err, abs(e))
        print(f"{d:7.1f} {cl:9.5f} {float(p.cl(a)):9.5f} {e:+11.2e}")
    print(f"\nmax error {err:.2e}  {'PASS' if err < 5e-3 else 'FAIL'}")
