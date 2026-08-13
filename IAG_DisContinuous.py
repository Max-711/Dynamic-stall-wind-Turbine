import numpy as np
from input import Polar, DEG


class IAG:
    """
        Ts = c / (2V)                       semi-chord time

        Ts * dx1/dt   = b1*A1*a - b1*x1                         (34)
        Ts * dx2/dt   = b2*A2*a - b2*x2                         (35)
        Ts*Tp*dx3/dt  = -x3 + C_N_P                             (40)
        Ts*Tf*dx4/dt  = -x4 + f_st(a_f)                         (43)
        Ts*Tv*dx5/dt  = -x5 + Ts*Tv*dC_V/dt                     (45)
        tau_v         : discrete accumulator                    (49)

        a_e   = a34*(1-A1-A2) + x1 + x2                         (36)
        C_N_C = dCN*sin(a_e - a0)                               (37)
        C_N_I = 4*Ka*(c/V)*a_dot                                (38)
        C_N_P = C_N_C + C_N_I                                   (39)
        a_f   = a0 + x3/dCN                                     (41)
        C_N_f = dCN*((1+sqrt(x4))/2)^2*sin(a_e-a0) + C_N_I      (44)
        C_L   = (C_N_f + x5)*cos(a) - C_T(a_f)*sin(a)           (50,52)

    """

    n_states = 6
    names = ["x1", "x2", "x3", "x4", "x5", "tau_v"]

    # Table 1, state-space IAG.  
    A1, A2 = 0.3, 0.7
    b1, b2 = 0.7, 0.53
    K_a = 0.75
    T_p, T_f, T_v, T_vl = 1.7, 3.0, 6.0, 6.0

    def __init__(self, polar: Polar, chord: float):
        self.polar = polar
        self.chord = chord
        self.dCN = polar.C_N_alpha
        self.a0 = polar.alpha_0
        self.CN_crit = polar.CN1        # C_N max from the static polar   (47)

    def f_st(self, a):
        """Kirchhoff inversion, sinusoidal form."""
        den = self.dCN * np.sin(a - self.a0)                              # (42)
        if abs(den) < 1e-9:
            return 1.0
        s = 2 * np.sqrt(max(float(self.polar.cn(a)) / den, 0.0)) - 1
        return float(np.clip(s ** 2, 0.0, 1.0)) if s > 0 else 0.0

    def ct(self, a):
        """C_T = C_D cos(a) - C_L sin(a), the sign Eq 52 needs."""
        return float(self.polar.cd(a)) * np.cos(a) - float(self.polar.cl(a)) * np.sin(a)

    def a34(self, alpha, alpha_dot, V):
        return alpha + self.chord / (2 * max(V, 1e-9)) * alpha_dot

    def inner(self, y, alpha, alpha_dot, V):
        """(a_e, C_N_C, C_N_I, a_f) -- shared by rhs and coeffs."""
        a_e = self.a34(alpha, alpha_dot, V) * (1 - self.A1 - self.A2) \
            + y[0] + y[1]                                                
        cn_c = self.dCN * np.sin(a_e - self.a0)                          
        cn_i = 4 * self.K_a * self.chord / max(V, 1e-9) * alpha_dot       
        a_f = self.a0 + y[2] / self.dCN                                 
        return a_e, cn_c, cn_i, a_f
    
    def stalled(self, y, alpha, alpha_dot, V):
        return alpha_dot > 0 and y[2] > self.CN_crit

    def vortex_on(self, y, alpha, alpha_dot, V):
        return alpha_dot > 0 and y[2] > self.CN_crit and y[5] < self.T_vl
    

    def clock(self, y, alpha, alpha_dot, V):
        r = V / self.chord
        return 0.45 * r if self.stalled(y, alpha, alpha_dot, V) else -r * y[5]

    def y0(self, alpha, V):
        x3 = self.dCN * np.sin(alpha - self.a0)
        return np.array([self.A1 * alpha, self.A2 * alpha,
                         x3, self.f_st(self.a0 + x3 / self.dCN), 0.0, 0.0])

    def rhs(self, y, alpha, alpha_dot, V):
        Ts = self.chord / (2 * max(V, 1e-9))
        x1, x2, x3, x4, x5, tv = y
        a_e, cn_c, cn_i, a_f = self.inner(y, alpha, alpha_dot, V)

        a34 = self.a34(alpha, alpha_dot, V)
        x1d = (self.b1 * self.A1 * a34 - self.b1 * x1) / Ts              
        x2d = (self.b2 * self.A2 * a34 - self.b2 * x2) / Ts               
        x3d = (-x3 + cn_c + cn_i) / (Ts * self.T_p)                      
        x4d = (-x4 + self.f_st(a_f)) / (Ts * self.T_f)  

        tvd = self.clock(y, alpha, alpha_dot, V)

        if self.vortex_on(y, alpha, alpha_dot, V):
            a_ed = alpha_dot * (1 - self.A1 - self.A2) + x1d + x2d
            s = np.sqrt(max(x4, 1e-12))
            cvd = (self.dCN * a_ed * (1 - 0.25 * (1 + s) ** 2)           
                   - 0.25 * cn_c * (1 + s) * x4d / s)
        else:
            cvd = 0.0
        x5d = -x5 / (Ts * self.T_v) + cvd                                 

        return np.array([x1d, x2d, x3d, x4d, x5d, tvd])

    def coeffs(self, y, alpha, alpha_dot, V):
        """Return (C_L, C_D)."""
        x4 = float(np.clip(y[3], 0.0, 1.0))
        a_e, cn_c, cn_i, a_f = self.inner(y, alpha, alpha_dot, V)
        s = np.sqrt(x4)

        cn_f = self.dCN * ((1 + s) / 2) ** 2 * np.sin(a_e - self.a0) + cn_i   # (44)
        cl = (cn_f + y[4]) * np.cos(alpha) - self.ct(a_f) * np.sin(alpha)     # (50,52)

        cd_a, cd_0 = float(self.polar.cd(alpha)), float(self.polar.cd(self.a0))
        fv = np.sqrt(self.f_st(alpha))
        cd = (cd_a + (alpha - a_e) * cn_c                                     # (53)
              + (cd_a - cd_0) * (((1 - s) / 2) ** 2 - ((1 - fv) / 2) ** 2)
              + y[4] * np.sin(alpha))
        return float(cl), float(cd)


if __name__ == "__main__":
    p = Polar()
    m = IAG(p, chord=0.25)
    print("static consistency (states held at equilibrium):\n")
    print(f"{'alpha':>7} {'model':>9} {'table':>9} {'error':>11}")
    err = 0.0
    for d in (8, 10, 12, 14, 15.3, 18, 20, 22, 25, 28):
        a = d * DEG
        cl, _ = m.coeffs(m.y0(a, 10.0), a, 0.0, 10.0)
        e = cl - float(p.cl(a))
        err = max(err, abs(e))
        print(f"{d:7.1f} {cl:9.5f} {float(p.cl(a)):9.5f} {e:+11.2e}")
    print(f"\nmax error {err:.2e}")
    print(f"C_N_CRIT = {m.CN_crit:.4f}")
