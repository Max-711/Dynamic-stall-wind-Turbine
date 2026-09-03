"""Chapter 7: linear stability and Hopf bifurcation of the coupled system.

Two parameters:  U (wind speed) and alpha_mean (inflow angle).

(a) sweep U, track max Re(lambda) and the flutter frequency
(b) locate the Hopf point  max Re(lambda) = 0  by bisection
(c) time-march the full nonlinear system either side of it to get the
    limit-cycle amplitude -> bifurcation diagram
"""

import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from input import Params, DEG
from coupled import CoupledOscillator
from IAG_Continuous import IAG
from stability import leading

warnings.filterwarnings("ignore")

ALPHA_DEG = 18.5
U_MIN, U_MAX, N_U = 5.0, 60.0, 111


def build():
    p = Params()
    p.alpha_mean = ALPHA_DEG * DEG
    return p, CoupledOscillator(p, IAG(p.polar, p.chord))


# ------------------------------------------------------------------ (a) sweep
def sweep(model, Us, rhs=None):
    s = np.full(len(Us), np.nan)
    f = np.full(len(Us), np.nan)
    for i, U in enumerate(Us):
        model.p.U = U
        try:
            si, fi, ev, z, res = leading(model, rhs)
            if res < 1e-8:
                s[i], f[i] = si, fi
        except Exception:
            pass
    return s, f


# ------------------------------------------------------------------ (c) LCO
def lco_amplitude(model, U, t_end=80.0, x0=1e-3, keep=0.3):
    model.p.U = U
    z_eq = None
    try:
        z_eq = leading(model)[3]
    except Exception:
        pass
    z0 = model.z0(x0=x0) if z_eq is None else z_eq + np.r_[x0, np.zeros(model.n - 1)]
    t_keep = t_end * (1 - keep)
    sol = solve_ivp(model.rhs, (0.0, t_end), z0, method="RK45",
                    rtol=1e-8, atol=1e-10,
                    t_eval=np.linspace(t_keep, t_end, 3000))
    if not sol.success:
        return np.nan
    x = sol.y[0]
    return 0.5 * (x.max() - x.min())


if __name__ == "__main__":
    p, model = build()
    Us = np.linspace(U_MIN, U_MAX, N_U)

    s_c, f_c = sweep(model, Us)
    s_u, f_u = sweep(model, Us, rhs=model.rhs_uncoupled)

    def g(U):
        model.p.U = U
        return leading(model)[0]

    U_h = brentq(g, 25.0, 45.0, xtol=1e-6)
    model.p.U = U_h
    f_h = leading(model)[1]
    print(f"HOPF (coupled)    U = {U_h:.4f} m/s,  f = {f_h:.4f} Hz "
          f"(structural f_n = {p.f_n} Hz)")

    def gu(U):
        model.p.U = U
        return leading(model, model.rhs_uncoupled)[0]
    try:
        U_hu = brentq(gu, 25.0, 55.0, xtol=1e-6)
        model.p.U = U_hu
        f_hu = leading(model, model.rhs_uncoupled)[1]
        print(f"HOPF (uncoupled)  U = {U_hu:.4f} m/s,  f = {f_hu:.4f} Hz")
    except Exception as e:
        U_hu = np.nan
        print("HOPF (uncoupled)  not bracketed:", e)

    np.savez("bifurcation_linear.npz", Us=Us, s_c=s_c, f_c=f_c,
             s_u=s_u, f_u=f_u, U_h=U_h, U_hu=U_hu, alpha_deg=ALPHA_DEG)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].plot(Us, s_c, label="coupled (closed form)")
    ax[0].plot(Us, s_u, "--", label=r"uncoupled ($\dot\alpha\equiv0$)")
    ax[0].plot([U_h], [0], "ko", ms=5)
    ax[0].annotate(f"Hopf\n$U={U_h:.2f}$ m/s", (U_h, 0),
                   textcoords="offset points", xytext=(-70, 18), fontsize=8)
    ax[0].set_xlabel("$U$ [m/s]")
    ax[0].set_ylabel(r"$\max\,\mathrm{Re}(\lambda)$ [1/s]")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].plot(Us, f_c, label="coupled")
    ax[1].plot(Us, f_u, "--", label="uncoupled")
    ax[1].axhline(p.f_n, color="0.6", lw=0.8, ls=":")
    ax[1].annotate("$f_n$", (U_MIN, p.f_n), fontsize=8,
                   textcoords="offset points", xytext=(2, 3))
    ax[1].set_xlabel("$U$ [m/s]")
    ax[1].set_ylabel("frequency [Hz]")
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(rf"IAG + 1-DOF oscillator, $\tilde\alpha={ALPHA_DEG}^\circ$",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_stability.png", dpi=200)
    print("saved fig_stability.png")
