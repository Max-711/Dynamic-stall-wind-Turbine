import numpy as np
from scipy.interpolate import CubicSpline

class Motion:
    def __init__(self, t, alpha, V, chord):
        self.t = np.asarray(t, float)
        self.alpha = np.asarray(alpha, float)
        self.V = V
        self.chord = chord
        self._s = CubicSpline(self.t, self.alpha)

    @classmethod
    def sine(cls, mean_deg, amp_deg, k, chord, V, n_cycles=8, n_per_cycle=400):
        """alpha = mean + amp*sin(w t),  w = 2*k*V/chord."""
        w = 2 * k * V / chord
        T = 2 * np.pi / w
        t = np.linspace(0, n_cycles * T, n_cycles * n_per_cycle + 1)
        a = np.radians(mean_deg) + np.radians(amp_deg) * np.sin(w * t)
        return cls(t, a, V, chord)

    def a(self, t):
        return float(self._s(t))

    def ad(self, t):
        return float(self._s(t, 1))