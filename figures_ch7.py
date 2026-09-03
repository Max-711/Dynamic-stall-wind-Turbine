"""Regenerate two of the Chapter 7 figures as PDF: fig_stability and fig_twoparam.

Usage:  python3 figures_ch7.py <output dir>      (default: current directory)
Reads:  bifurcation_linear.npz  (from bifurcation.py)
        field2d.npz            (from field2d.py)
The third Chapter 7 figure, fig_bifurcation.pdf, comes from fig_bif2.py.
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1] if len(sys.argv) > 1 else "."

d = np.load("bifurcation_linear.npz"); U_h = float(d["U_h"])
# NB: never hard-code a thesis equation number in a legend -- the numbering
# shifts whenever the text is edited.  Describe the curve instead.
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].axhline(0, color="0.6", lw=0.8)
ax[0].plot(d["Us"], d["s_c"], "-", color="C0", label=r"coupled ($\dot\alpha$ retained)")
ax[0].plot(d["Us"], d["s_u"], "--", color="C1", label=r"$\dot\alpha$ suppressed")
ax[0].plot([U_h], [0], "ko", ms=5)
ax[0].annotate(f"$U_H={U_h:.2f}$ m/s", (U_h, 0), textcoords="offset points",
               xytext=(-72, 16), fontsize=8)
ax[0].set_xlabel("$U$ [m/s]"); ax[0].set_ylabel(r"$\max\,\mathrm{Re}(\lambda)$ [1/s]")
ax[0].legend(fontsize=8, frameon=False)
ax[1].plot(d["Us"], d["f_c"], "-", color="C0", label="coupled")
ax[1].plot(d["Us"], d["f_u"], "--", color="C1", label=r"$\dot\alpha$ suppressed")
ax[1].axhline(0.687, color="0.6", lw=0.8, ls=":")
ax[1].annotate("$f_n$", (5, 0.687), fontsize=8, textcoords="offset points", xytext=(2, 3))
ax[1].set_xlabel("$U$ [m/s]"); ax[1].set_ylabel("frequency [Hz]")
ax[1].legend(fontsize=8, frameon=False)
for a in ax: a.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{out}/fig_stability.pdf")

# NOTE: fig_bifurcation.pdf is NOT produced here -- it is produced by fig_bif2.py,
# which also draws the down sweep and the hysteresis window.  An earlier version of
# this script wrote a simplified (up sweep only) fig_bifurcation.pdf and silently
# overwrote the correct one; that block has been removed.

e = np.load("field2d.npz"); A, U, S = e["A"], e["U"], e["S"]
fig, ax = plt.subplots(figsize=(6.0, 4.0))
UU, AA = np.meshgrid(U, A); lim = np.nanmax(np.abs(S))
cf = ax.contourf(UU, AA, S, levels=np.linspace(-lim, lim, 25), cmap="RdBu_r")
ax.contour(UU, AA, S, levels=[0.0], colors="k", linewidths=1.6)
ax.plot([U_h], [18.5], "ko", ms=5)
ax.annotate(r"$\tilde\alpha=18.5^\circ$", (U_h, 18.5), textcoords="offset points",
            xytext=(6, -12), fontsize=8)
fig.colorbar(cf, ax=ax, label=r"$\max\,\mathrm{Re}(\lambda)$  [1/s]")
ax.set_xlabel("$U$ [m/s]"); ax.set_ylabel(r"$\tilde\alpha$ [deg]")
fig.tight_layout(); fig.savefig(f"{out}/fig_twoparam.pdf")
print("wrote fig_stability.pdf, fig_twoparam.pdf to", out)
