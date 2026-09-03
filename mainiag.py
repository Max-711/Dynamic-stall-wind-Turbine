import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from input import Params, DEG
from IAG_Continuous import IAG as IAG_C
from IAG_DisContinuous import IAG as IAG_D
from motion import Motion
from integrator import rk4
from error import error
from loadreference import load_reference

OUT = sys.argv[1] if len(sys.argv) > 1 else "."   # where the figures are written
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
aa = np.linspace(8, 28, 300) * DEG

# --- Fig 1: lift against the Bladed reference -------------------------
f1, a1 = plt.subplots(1, 2, figsize=(11, 4.5))

a1[0].plot(alpha[m]/DEG, cl_ref[m], "k-", lw=2, label="Bladed")
a1[0].plot(alpha[m]/DEG, r_d["C_L"][m], "r-", lw=1.5, label="IAG disc")
a1[0].plot(alpha[m]/DEG, r_c["C_L"][m], "b--", lw=1.5, label="IAG cont")
a1[0].plot(aa/DEG, p.polar.cl(aa), color="0.7", ls="--", lw=1, label="static")
a1[0].set_xlabel(r"$\alpha$ [deg]"); a1[0].set_ylabel("$C_L$")
a1[0].set_title(f"(a) hysteresis loop (cycle {e['cycle']})")

a1[1].plot(t, r_d["C_L"] - cl_ref, "r-", lw=1.2, label="disc")
a1[1].plot(t, r_c["C_L"] - cl_ref, "b--", lw=1.2, label="cont")
a1[1].axhline(0, color="k", lw=.8)
a1[1].set_xlabel("t [s]"); a1[1].set_ylabel("present $-$ Bladed")
a1[1].set_title("(b) residual")

# --- Fig 2: vortex states --------------------------------------------
f2, a2 = plt.subplots(1, 2, figsize=(11, 4.5))

a2[0].plot(t, r_d["tau_v"], "r-", lw=1.2, label="$x_6$")
a2[0].axhline(IAG_D.T_vl, color="k", ls="--", lw=1, label=rf"$T_{{VL}}$ = {IAG_D.T_vl:g}")
a2[0].set_xlabel("t [s]"); a2[0].set_ylabel("$x_6$")
a2[0].set_title("(a) vortex travel state")

a2[1].plot(t, r_c["C_L"] - r_d["C_L"], "g-", lw=1.2, label="cont $-$ disc")
a2[1].axhline(0, color="k", lw=.8)
a2[1].set_xlabel("t [s]"); a2[1].set_ylabel("$\\Delta C_L$")
a2[1].set_title("(b) cost of smoothing")

for fig, axs, name in ((f1, a1, "fig_iag_cl"), (f2, a2, "fig_iag_vortex")):
    for a_ in axs:
        a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=140)
print("\n  -> fig_iag_cl.pdf, fig_iag_vortex.pdf")
