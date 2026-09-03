"""Chapter 7 pipeline applied to the Oye model (3-state coupled system)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import brentq
from input import Params, DEG
from coupled import CoupledOscillator
from oye import Oye
from stability import leading, jacobian, equilibrium

p = Params(); p.alpha_mean = 18.5*DEG
m = CoupledOscillator(p, Oye(p.polar, p.chord, T_f=3.0))
print(f"Oye coupled system: n = {m.n} states {m.names}")

p.U = 4.0
z, res, ier = equilibrium(m)
print(f"\nequilibrium at U=4 (residual {res:.2e}): " +
      "  ".join(f"{n}={v:+.6f}" for n, v in zip(m.names, z)))
Lam = m.added_mass(z[2:], *m.kinematics(0.0)[:2][::1][:1] + (m.kinematics(0.0)[1],))
ev = np.linalg.eigvals(jacobian(m.rhs_z, z))
print("eigenvalues:", "  ".join(f"{e.real:+.5f}{e.imag:+.5f}j" for e in np.sort_complex(ev)))

print(f"\n{'U':>6} {'maxRe':>11} {'f [Hz]':>9}")
prev = None
for U in np.arange(4., 81., 4.):
    p.U = U
    try:
        s, f, _, ze, r = leading(m)
        if r < 1e-8:
            print(f"{U:6.1f} {s:11.5f} {f:9.4f}")
            if prev is not None and prev[1] < 0 < s: pass
            prev = (U, s)
    except Exception as e:
        print(f"{U:6.1f}   failed")
