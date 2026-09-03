"""Chapter 7 bifurcation diagram: edgewise LCO amplitude vs wind speed.

Writes fig_bifurcation.pdf (the version that is in the thesis) -- up sweep,
down sweep, hysteresis window and a detail inset.

Usage:  python3 fig_bif2.py <output dir>       (default: current directory)

Data provenance
---------------
up sweep    lco_branch.npy
            <- plot_bif.py, which parses lco.log and lco2.log
            <- printed by lco.py and lco2.py (nonlinear time marching from a
               1 mm disturbance about the equilibrium at each U)
down sweep  hard-coded in the array `down` below, because it was produced by a
            resumable run that only printed to the console:
            <- sweep_down2.py (U = 50 .. 32.0, each U continued from the cycle
               reached at the previous U)
UH = 32.7706   Hopf point, from bifurcation.py (brentq on max Re(lambda))
UF = 31.9      fold of the limit cycle, from fold_refine.py (last U at which
               the cycle is still self-sustaining after 400 s)
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1] if len(sys.argv)>1 else "."
UH, UF = 32.7706, 31.9
up = np.load("lco_branch.npy")                      # small initial condition
down = np.array(   # from sweep_down2.py, see docstring
    [[50,1314.19],[46,994.71],[43,770.55],[40,562.77],[38,408.20],
                 [36,267.21],[35,200.26],[34,148.10],[33.5,125.33],[33,103.87],
                 [32.5,82.71],[32.2,69.15],[32.0,58.33]])
fig, ax = plt.subplots(figsize=(5.8,4.0))
ax.axvspan(UF, UH, color="0.88", zorder=0)

ax.plot(up[:,0], up[:,1], "o-", ms=4, color="0.45", lw=1.0,
        label="up sweep, 1 mm initial disturbance")
ax.plot(down[:,0], down[:,1], "s-", ms=4, color="C3", lw=1.0,
        label="down sweep, continued from the cycle")
ax.axvline(UH, color="C0", lw=1.0, ls="--")
ax.axvline(UF, color="C2", lw=1.0, ls=":")
ax.annotate(f"$U_H={UH:.2f}$", (UH, 980), fontsize=8, rotation=90,
            textcoords="offset points", xytext=(4,0), color="C0")
ax.annotate(f"$U_F\\approx{UF:.1f}$", (UF, 980), fontsize=8, rotation=90,
            textcoords="offset points", xytext=(-13,0), color="C2")
ax.set_xlabel("$U$ [m/s]"); ax.set_ylabel("edgewise amplitude [mm]")
ax.set_ylim(-60,1450)
ax.set_xlim(28, 51); ax.grid(alpha=0.25)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()

axi = fig.add_axes([0.575, 0.235, 0.345, 0.315])
axi.axvspan(UF, UH, color="0.88")
axi.plot(up[:,0], up[:,1], "o-", ms=3, color="0.45", lw=0.9)
axi.plot(down[:,0], down[:,1], "s-", ms=3, color="C3", lw=0.9)
axi.axvline(UH, color="C0", lw=0.8, ls="--"); axi.axvline(UF, color="C2", lw=0.8, ls=":")
axi.set_xlim(31.4, 34.2); axi.set_ylim(-8, 160); axi.tick_params(labelsize=6)
axi.set_title("detail", fontsize=7)
fig.savefig(f"{out}/fig_bifurcation.pdf"); fig.savefig(f"{out}/fig_bifurcation.png", dpi=200)
print("bifurcation figure rewritten")
