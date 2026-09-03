import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from input import Params, DEG
from oye import Oye
from motion import Motion
from integrator import rk4
from error import error
from loadreference import load_reference

CHORD, U = 0.25, 10.0          
PERIOD = 2 * np.pi / 5.6       # alpha = 18 + 10*sin(5.6 t) deg, k = 0.07

t, alpha, cl_ref, cl_iag_ref = load_reference()
motion = Motion(t, alpha, U, CHORD) # this is the motion of the airfoil, with alpha(t) and V(t)
p = Params() # get static polar data for the airfoil

print(f"reference: {len(t)} samples, {t[-1]/PERIOD:.2f} cycles, "
      f"alpha {alpha.min()/DEG:.0f}..{alpha.max()/DEG:.0f} deg\n")



T_f = 3.0
r = rk4(Oye(p.polar, CHORD, T_f), motion)
e = error(t, r["C_L"], cl_ref, PERIOD)

print(f"Oye,  T_f = {p.T_f}  (cycle {e['cycle']})")
print(f"  RMS      {e['rms']:.5f}   ({e['pct']:.2f} % of C_L range)")
print(f"  max err  {e['max']:.4f}")
print(f"  Cl_max   {e['cl_max']:.4f}  vs {e['cl_max_ref']:.4f} Bladed")
print(f"  Cl_min   {e['cl_min']:.4f}  vs {e['cl_min_ref']:.4f} Bladed")

m = (t >= e["cycle"] * PERIOD) & (t < (e["cycle"] + 1) * PERIOD)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

ax[0].plot(t, cl_ref, "k-", lw=2, label="Bladed")
ax[0].plot(t, r["C_L"], "r-", lw=1.2, label=f"present ($T_f$={T_f:g})")
ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("$C_L$"); ax[0].set_title("$C_L(t)$")

aa = np.linspace(8, 28, 300) * DEG
ax[1].plot(alpha[m]/DEG, cl_ref[m], "k-", lw=2, label="Bladed")
ax[1].plot(alpha[m]/DEG, r["C_L"][m], "r-", lw=1.5, label="present")
ax[1].plot(aa/DEG, p.polar.cl(aa), "b--", lw=1, label="static")
ax[1].set_xlabel(r"$\alpha$ [deg]"); ax[1].set_ylabel("$C_L$")
ax[1].set_title(f"hysteresis loop (cycle {e['cycle']})")

ax[2].plot(t, r["C_L"] - cl_ref, color="crimson", lw=1.2)
ax[2].axhline(0, color="k", lw=.8)
ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("present $-$ Bladed"); ax[2].set_title("residual")

for a_ in ax:
    a_.grid(alpha=.3)
    if a_ is not ax[2]:
        a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{sys.argv[1] if len(sys.argv) > 1 else '.'}/fig_verify_oye.png", dpi=140)
print("\n  -> fig_verify_oye.png")