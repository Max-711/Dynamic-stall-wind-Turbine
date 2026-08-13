import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from input import Params, DEG
from IAG_Continuous import IAG
# from IAG_DisContinuous import IAG
from motion import Motion
from integrator import rk4
from error import error
from loadreference import load_reference

CHORD, U = 0.25, 10.0
PERIOD = 2 * np.pi / 5.6

t, alpha, _, cl_ref = load_reference()
motion = Motion(t, alpha, U, CHORD)
p = Params()

print(f"reference: {len(t)} samples, {t[-1]/PERIOD:.2f} cycles, "
      f"alpha {alpha.min()/DEG:.0f}..{alpha.max()/DEG:.0f} deg\n")

model = IAG(p.polar, CHORD)
r = rk4(model, motion)
e = error(t, r["C_L"], cl_ref, PERIOD)

print(f"IAG,  T_p {IAG.T_p}  T_f {IAG.T_f}  T_v {IAG.T_v}   (cycle {e['cycle']})")
print(f"  RMS      {e['rms']:.5f}   ({e['pct']:.2f} % of C_L range)")
print(f"  max err  {e['max']:.4f}")
print(f"  Cl_max   {e['cl_max']:.4f}  vs {e['cl_max_ref']:.4f} Bladed")
print(f"  Cl_min   {e['cl_min']:.4f}  vs {e['cl_min_ref']:.4f} Bladed")

# the vortex state is the whole reason IAG differs from Oye -- check it fires
print(f"\n  x5 (vortex C_N) range {r['x5'].min():+.4f} .. {r['x5'].max():+.4f}")

m = (t >= e["cycle"] * PERIOD) & (t < (e["cycle"] + 1) * PERIOD)
fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.3))


ax[0].plot(t, r["tau_v"], "r-", lw=1.2, label="present")
ax[0].axhline(IAG.T_vl, color="k", ls="--", lw=1, label=rf"$T_{{vl}}$ = {IAG.T_vl:g}")
ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"$\tau_v$"); ax[0].set_title(r"$\tau_v(t)$")

ax[1].plot(alpha[m]/DEG, r["tau_v"][m], "r-", lw=1.5, label="present")
ax[1].axhline(IAG.T_vl, color="k", ls="--", lw=1, label=rf"$T_{{vl}}$ = {IAG.T_vl:g}")
ax[1].set_xlabel(r"$\alpha$ [deg]"); ax[1].set_ylabel(r"$\tau_v$")
ax[1].set_title(f"hysteresis loop (cycle {e['cycle']})")

aa = np.linspace(8, 28, 300) * DEG
ax[2].plot(alpha[m]/DEG, cl_ref[m], "k-", lw=2, label="Bladed")
ax[2].plot(alpha[m]/DEG, r["C_L"][m], "r-", lw=1.5, label="present")
ax[2].plot(aa/DEG, p.polar.cl(aa), color="0.7", ls="--", lw=1, label="static")
ax[2].set_xlabel(r"$\alpha$ [deg]"); ax[2].set_ylabel("$C_L$")
ax[2].set_title(f"hysteresis loop (cycle {e['cycle']})")

ax[3].plot(t, r["C_L"] - cl_ref, color="crimson", lw=1.2)
ax[3].axhline(0, color="k", lw=.8)
ax[3].set_xlabel("t [s]"); ax[3].set_ylabel("present $-$ Bladed"); ax[3].set_title("residual")

plt.plot()

for a_ in ax:
    a_.grid(alpha=.3)
    if a_ is not ax[3]:
        a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_verify_iag.png", dpi=140)
print("\n  -> fig_verify_iag.png")
