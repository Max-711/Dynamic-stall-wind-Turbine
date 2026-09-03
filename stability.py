"""Linear stability / Hopf analysis of the closed-form coupled system.

    zdot = F(z; U, alpha_mean),     z in R^8

1. equilibrium      F(z*) = 0                    (fsolve)
2. Jacobian         J_ij = dF_i/dz_j             (central differences)
3. eigenvalues      lambda = eig(J)              (numpy)
4. Hopf point       max Re(lambda) crosses 0     (bisection in U)

Nothing here is derived by hand: the Jacobian is evaluated numerically from
the same right-hand side that the time integrator uses, so the two can never
disagree.
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from input import Params, DEG
from coupled import CoupledOscillator


# ---------------------------------------------------------------- equilibrium
def equilibrium(model, z_guess=None):
    z0 = model.z0(x0=0.0) if z_guess is None else z_guess
    z, info, ier, msg = fsolve(model.rhs_z, z0, full_output=True, xtol=1e-13)
    res = np.max(np.abs(model.rhs_z(z)))
    return z, res, ier


# ---------------------------------------------------------------- Jacobian
def jacobian(f, z, rel=1e-6, absmin=1e-8):
    """Central-difference Jacobian of f at z."""
    z = np.asarray(z, float)
    n = z.size
    J = np.empty((n, n))
    for j in range(n):
        h = max(rel * abs(z[j]), absmin)
        zp, zm = z.copy(), z.copy()
        zp[j] += h
        zm[j] -= h
        J[:, j] = (f(zp) - f(zm)) / (2 * h)
    return J


def leading(model, rhs=None, z_guess=None):
    """(max Re(lambda), frequency [Hz], eigenvalues, z_eq, residual)."""
    f = model.rhs_z if rhs is None else (lambda z: rhs(0.0, z))
    z0 = model.z0(x0=0.0) if z_guess is None else z_guess
    z_eq = fsolve(f, z0, xtol=1e-13)
    res = np.max(np.abs(f(z_eq)))
    ev = np.linalg.eigvals(jacobian(f, z_eq))
    k = int(np.argmax(ev.real))
    return ev[k].real, abs(ev[k].imag) / (2 * np.pi), ev, z_eq, res


# ---------------------------------------------------------------- U sweep
def sweep(model, U_list, rhs=None):
    out = []
    z_guess = None
    for U in U_list:
        model.p.U = U
        try:
            s, fq, ev, z_eq, res = leading(model, rhs, z_guess)
            z_guess = z_eq
            out.append((U, s, fq, res))
        except Exception as e:                                    # pragma: no cover
            out.append((U, np.nan, np.nan, np.nan))
    return np.array(out)


def hopf_point(model, U_lo, U_hi, rhs=None):
    def g(U):
        model.p.U = U
        return leading(model, rhs)[0]
    return brentq(g, U_lo, U_hi, xtol=1e-6)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    from IAG_Continuous import IAG

    p = Params()
    model = CoupledOscillator(p, IAG(p.polar, p.chord))

    print(f"alpha_mean = {p.alpha_mean/DEG:.2f} deg,  zeta = {p.zeta},  "
          f"m = {p.mass} kg/m,  f_n = {p.f_n} Hz\n")

    p.U = 4.0
    z_eq, res, ier = equilibrium(model)
    print("equilibrium at U = 4 m/s   (residual %.2e, ier=%d)" % (res, ier))
    for nm, v in zip(model.names, z_eq):
        print(f"   {nm:>6} = {v:+.6f}")

    J = jacobian(model.rhs_z, z_eq)
    ev = np.sort_complex(np.linalg.eigvals(J))
    print("\neigenvalues of the 8x8 Jacobian:")
    for e in ev:
        print(f"   {e.real:+12.5f} {e.imag:+12.5f}j   "
              f"(f = {abs(e.imag)/(2*np.pi):8.4f} Hz)")

    U_list = np.arange(1.0, 12.01, 0.25)
    print(f"\n{'U':>6} {'maxRe (coupled)':>17} {'f [Hz]':>9} "
          f"{'maxRe (uncoupled)':>19} {'f [Hz]':>9}")
    A = sweep(model, U_list)
    B = sweep(model, U_list, rhs=model.rhs_uncoupled)
    for (U, s, fq, _), (_, s2, fq2, _) in zip(A, B):
        print(f"{U:6.2f} {s:17.6f} {fq:9.4f} {s2:19.6f} {fq2:9.4f}")

    np.save("stability_sweep.npy", np.hstack([A, B[:, 1:]]))
    print("\nsaved stability_sweep.npy")
