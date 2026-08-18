import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from input import Params, DEG
from IAG_Continuous import IAG as IAG_C
from IAG_DisContinuous import IAG as IAG_D
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

model_d = IAG_D(p.polar, CHORD)
r_d = rk4(model_d, motion)
e_d = error(t, r_d["C_L"], cl_ref, PERIOD)

model_c = IAG_C(p.polar, CHORD)
r_c = rk4(model_c, motion)
e_c = error(t, r_c["C_L"], cl_ref, PERIOD)

print(f"IAG disc,  T_p {IAG_D.T_p}  T_f {IAG_D.T_f}  T_v {IAG_D.T_v}   (cycle {e_d['cycle']})")
print(f"  RMS      {e_d['rms']:.5f}   ({e_d['pct']:.2f} % of C_L range)")
print(f"  max err  {e_d['max']:.4f}")
print(f"  Cl_max   {e_d['cl_max']:.4f}  vs {e_d['cl_max_ref']:.4f} Bladed")
print(f"  Cl_min   {e_d['cl_min']:.4f}  vs {e_d['cl_min_ref']:.4f} Bladed")

print(f"\nIAG cont,  T_p {IAG_C.T_p}  T_f {IAG_C.T_f}  T_v {IAG_C.T_v}   (cycle {e_c['cycle']})")
print(f"  RMS      {e_c['rms']:.5f}   ({e_c['pct']:.2f} % of C_L range)")
print(f"  max err  {e_c['max']:.4f}")
print(f"  Cl_max   {e_c['cl_max']:.4f}  vs {e_c['cl_max_ref']:.4f} Bladed")
print(f"  Cl_min   {e_c['cl_min']:.4f}  vs {e_c['cl_min_ref']:.4f} Bladed")

# the vortex state is the whole reason IAG differs from Oye -- check it fires
print(f"\n  x5 (vortex C_N) range disc {r_d['x5'].min():+.4f} .. {r_d['x5'].max():+.4f}"
      f"   cont {r_c['x5'].min():+.4f} .. {r_c['x5'].max():+.4f}")

e = e_d
m = (t >= e["cycle"] * PERIOD) & (t < (e["cycle"] + 1) * PERIOD)
fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.3))


ax[0].plot(t, r_d["tau_v"], "r-", lw=1.2, label="disc")
ax[0].plot(t, r_c["tau_v"], "b--", lw=1.2, label="cont")
ax[0].axhline(IAG_D.T_vl, color="k", ls="--", lw=1, label=rf"$T_{{vl}}$ = {IAG_D.T_vl:g}")
ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"$\tau_v$"); ax[0].set_title(r"$\tau_v(t)$")

ax[1].plot(alpha[m]/DEG, r_d["tau_v"][m], "r-", lw=1.5, label="disc")
ax[1].plot(alpha[m]/DEG, r_c["tau_v"][m], "b--", lw=1.5, label="cont")
ax[1].axhline(IAG_D.T_vl, color="k", ls="--", lw=1, label=rf"$T_{{vl}}$ = {IAG_D.T_vl:g}")
ax[1].set_xlabel(r"$\alpha$ [deg]"); ax[1].set_ylabel(r"$\tau_v$")
ax[1].set_title(f"hysteresis loop (cycle {e['cycle']})")

aa = np.linspace(8, 28, 300) * DEG
ax[2].plot(alpha[m]/DEG, cl_ref[m], "k-", lw=2, label="Bladed")
ax[2].plot(alpha[m]/DEG, r_d["C_L"][m], "r-", lw=1.5, label="IAG disc")
ax[2].plot(alpha[m]/DEG, r_c["C_L"][m], "b--", lw=1.5, label="IAG cont")
ax[2].plot(aa/DEG, p.polar.cl(aa), color="0.7", ls="--", lw=1, label="static")
ax[2].set_xlabel(r"$\alpha$ [deg]"); ax[2].set_ylabel("$C_L$")
ax[2].set_title(f"hysteresis loop (cycle {e['cycle']})")

ax[3].plot(t, r_d["C_L"] - cl_ref, color="r", lw=1.2, label="disc")
ax[3].plot(t, r_c["C_L"] - cl_ref, color="b", ls="--", lw=1.2, label="cont")
ax[3].axhline(0, color="k", lw=.8)
ax[3].set_xlabel("t [s]"); ax[3].set_ylabel("present $-$ Bladed"); ax[3].set_title("residual")

plt.plot()

for a_ in ax:
    a_.grid(alpha=.3)
    a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_verify_iag.png", dpi=140)
print("\n  -> fig_verify_iag.png")
