import os
import numpy as np
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline

DEG = np.pi / 180.0
F_FLOOR = 1e-4         


class Polar:
    alpha_0 = -0.38 * DEG      
    C_N_alpha = 7.12499         
    alpha_stall = 15.3 * DEG
    CN1 = 0.920154          

    def __init__(self, path="S809_polar.csv"):
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        rows = []
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or line[0].isalpha():
                continue
            rows.append([float(v) for v in line.split(",")[:4]])
        d = np.array(rows)

        self.a = d[:, 0] * DEG
        self.cl_t, self.cd_t = d[:, 1], d[:, 2]
        self._cl = CubicSpline(self.a, self.cl_t)
        self._cd = CubicSpline(self.a, self.cd_t)

        # separation point, inverted from the measured normal force
        cn_t = self.cl_t * np.cos(self.a) + self.cd_t * np.sin(self.a)
        lin = self.C_N_alpha * (self.a - self.alpha_0)
        ratio = np.where(np.abs(lin) > 1e-6, cn_t / np.where(np.abs(lin) > 1e-6, lin, 1), 1)
        sq = 2 * np.sqrt(np.clip(ratio, 0, None)) - 1
        f_t = np.clip(np.where(sq > 0, sq ** 2, 0), 0, 1)
        f_t = np.where(np.abs(self.a - self.alpha_0) < 0.5 * DEG, 1.0, f_t)
        self._f = CubicSpline(self.a, f_t)

        # fully separated branch, defined so the Oye blend reproduces the
        # measured polar exactly at f = f_st
        att = self.cl_inv(self.a)
        den = 1 - f_t
        self._clfs = CubicSpline(
            self.a,
            np.where(den > 1e-3, (self.cl_t - f_t * att) / np.where(den > 1e-3, den, 1),
                     0.5 * self.cl_t))

    def cl(self, a):
        return self._cl(a)

    def cd(self, a):
        return self._cd(a)

    def cn(self, a):
        a = np.asarray(a, float)
        return self.cl(a) * np.cos(a) + self.cd(a) * np.sin(a)

    def f_sep(self, a):
        return np.clip(self._f(a), 0.0, 1.0)

    def cl_inv(self, a):
        """Fully attached lift, the f = 1 limit."""
        a = np.asarray(a, float)
        return self.C_N_alpha * (a - self.alpha_0) * np.cos(a)

    def cl_fs(self, a):
        """Fully separated lift, the f = 0 limit."""
        return self._clfs(a)



# Parameters
@dataclass
class Params:
    rho: float = 1.225                  # kg/m^3
    chord: float = 2.7646               # m
    mass: float = 130.019               # kg/m
    f_n: float = 0.687                  # Hz,  first edgewise
    zeta: float = 0.005                 # ElastoDyn BldEdDmp1 = 0.48 %
    k3_over_k1: float = 0.0             # 0 = linear;  1e5 = preliminary report
    alpha_mean: float = 18.5 * DEG
    U: float = 4.0                      # m/s
    T_f: float = 3.0                    # tau = T_f * chord / V

    # numerics
    dt: float = 1e-3
    t_end: float = 120.0

    polar: Polar = field(default_factory=Polar)

    @property
    def omega_n(self):
        return 2 * np.pi * self.f_n

    @property
    def k1(self):
        return self.mass * self.omega_n ** 2

    @property
    def k3(self):
        return self.k3_over_k1 * self.k1

    @property
    def c_damp(self):
        return 2 * self.zeta * self.mass * self.omega_n


if __name__ == "__main__":
    p = Params()
    print(f"chord {p.chord} m   mass {p.mass} kg/m   f_n {p.f_n} Hz   zeta {p.zeta}")
    print(f"k1 {p.k1:.2f}   c {p.c_damp:.4f}   k3 {p.k3:.1f}")
    print(f"\n{'alpha':>7} {'C_L':>8} {'C_D':>8} {'f_st':>8}")
    for d in (0, 8, 15.3, 18.5, 20, 28, 40):
        a = d * DEG
        print(f"{d:7.1f} {float(p.polar.cl(a)):8.4f} {float(p.polar.cd(a)):8.4f} "
              f"{float(p.polar.f_sep(a)):8.4f}")
