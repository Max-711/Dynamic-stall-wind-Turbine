"""Verification of the structural oscillator and of the closed-form coupling.

T1  kinematics          kappa = d(alpha)/d(xd) vs finite difference
T2  affine force        F_y(z, alpha_dot) linear in alpha_dot
T3  added-mass algebra  closed-form xdd vs root of the implicit equation
T4  free decay          aero off, exact underdamped solution
T5  step force          aero off, exact step response
T6  harmonic force      aero off, exact frequency response (amplitude + phase)
T7  energy              aero off, zero damping, energy conservation
T8  integrator          solve_ivp vs fixed-step RK4 on the FULL coupled system
T9  Jacobian <-> time   eigenvalue vs log-decrement of the nonlinear simulation
"""

import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
warnings.filterwarnings("ignore")

from input import Params, DEG
from coupled import CoupledOscillator
from IAG_Continuous import IAG
from stability import leading, jacobian

OK = lambda b: "PASS" if b else "**FAIL**"
res = []


def report(name, err, tol, extra=""):
    res.append(err <= tol)
    print(f"  {name:<34} err = {err:10.3e}   tol = {tol:8.1e}   "
          f"{OK(err <= tol)}  {extra}")


p = Params()
aero = IAG(p.polar, p.chord)
m = CoupledOscillator(p, aero)
p.U, p.alpha_mean = 4.0, 18.5 * DEG

print("T1  kinematics")
for xd in (-2.0, -0.3, 0.0, 0.7, 3.0):
    a0, V0, kap = m.kinematics(xd)
    h = 1e-6
    fd = (m.kinematics(xd + h)[0] - m.kinematics(xd - h)[0]) / (2 * h)
    Ux, Uy = p.U * np.cos(p.alpha_mean), p.U * np.sin(p.alpha_mean)
    report(f"kappa vs d(alpha)/d(xd), xd={xd:+.1f}", abs(kap - fd), 1e-8)
    report(f"V vs sqrt(Ux^2+(Uy-xd)^2), xd={xd:+.1f}",
           abs(V0 - np.hypot(Ux, Uy - xd)), 1e-12)

print("\nT2  aerodynamic force is affine in alpha_dot")
z = m.z0(x0=0.02); a0, V0, kap = m.kinematics(0.3)
ad = np.linspace(-3, 3, 13)
F = np.array([m.force(z[2:], a0, a, V0) for a in ad])
lin = np.polyval(np.polyfit(ad, F, 1), ad)
report("max |F - linear fit|", np.max(np.abs(F - lin)), 1e-10)

print("\nT3  added-mass elimination vs implicit root solve")
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    zz = m.z0(x0=rng.uniform(-.3, .3), xd0=rng.uniform(-3, 3))
    zz[2:] += rng.normal(0, .05, aero.n_states)
    x, xd = zz[0], zz[1]
    al, V, kp = m.kinematics(xd)
    # implicit residual: m*xdd = F_y(z, kappa*xdd) - c*xd - k1*x - k3*x^3
    g = lambda a: (p.mass * a - m.force(zz[2:], al, kp * a, V)
                   + p.c_damp * xd + p.k1 * x + p.k3 * x ** 3)
    root = brentq(g, -1e4, 1e4, xtol=1e-14, rtol=1e-15)
    worst = max(worst, abs(root - m.rhs(0.0, zz)[1]))
report("closed form vs brentq root (200 states)", worst, 1e-8)

# ---------------------------------------------------------------- aero off
wn = p.omega_n; zt = p.zeta; wd = wn * np.sqrt(1 - zt ** 2); k = p.k1


def run(F_of_t, z0, t_eval):
    mv = CoupledOscillator(p, aero, force_ext=F_of_t)
    s = solve_ivp(mv.rhs, (0, t_eval[-1]), z0, t_eval=t_eval,
                  rtol=1e-10, atol=1e-13)
    return s.y[0], s.y[1]


print("\nT4  free decay,  m*xdd + c*xd + k*x = 0")
t = np.linspace(0, 25, 4001); x0 = 0.05
x, _ = run(lambda tt: 0.0, np.r_[x0, 0.0, aero.y0(p.alpha_mean, p.U)], t)
xa = x0 * np.exp(-zt * wn * t) * (np.cos(wd * t) + zt * wn / wd * np.sin(wd * t))
report("max |x_num - x_exact| / x0", np.max(np.abs(x - xa)) / x0, 1e-8,
       f"(zeta={zt}, f_n={p.f_n} Hz)")

print("\nT5  step force,  F = F0 for t > 0")
F0 = 500.0
x, _ = run(lambda tt: F0, np.r_[0.0, 0.0, aero.y0(p.alpha_mean, p.U)], t)
xa = F0 / k * (1 - np.exp(-zt * wn * t) * (np.cos(wd * t)
                                           + zt * wn / wd * np.sin(wd * t)))
report("max |x_num - x_exact| (step)", np.max(np.abs(x - xa)) / (F0 / k), 1e-8)
t5 = np.linspace(0, 400, 4001)
x5, _ = run(lambda tt: F0, np.r_[0.0, 0.0, aero.y0(p.alpha_mean, p.U)], t5)
report("static deflection x(400 s) -> F0/k", abs(x5[-1] - F0 / k) / (F0 / k), 5e-3,
       f"(F0/k = {F0/k*1e3:.4f} mm)")

print("\nT6  harmonic force,  F = F0 cos(Om t)   [amplitude and phase]")
for r in (0.5, 1.0, 2.0):
    Om = r * wn
    T = 2 * np.pi / Om
    A_ex = F0 / np.hypot(k - p.mass * Om ** 2, p.c_damp * Om)
    ph_ex = -np.arctan2(p.c_damp * Om, k - p.mass * Om ** 2)
    t6 = np.linspace(0, 10 * T, 4001)
    z6 = np.r_[A_ex * np.cos(ph_ex), -A_ex * Om * np.sin(ph_ex),
               aero.y0(p.alpha_mean, p.U)]          # exact steady state at t=0
    x, _ = run(lambda tt, O=Om: F0 * np.cos(O * tt), z6, t6)
    x_ex = A_ex * np.cos(Om * t6 + ph_ex)
    report(f"steady state vs exact, Om/wn = {r:.1f}",
           np.max(np.abs(x - x_ex)) / A_ex, 1e-8, "(exact IC, no transient)")
    A_num = 0.5 * (x.max() - x.min())
    M = np.c_[np.cos(Om * t6), np.sin(Om * t6)]
    cc, ss = np.linalg.lstsq(M, x, rcond=None)[0]
    ph_num = np.arctan2(-ss, cc)
    report(f"amplitude, Om/wn = {r:.1f}", abs(A_num - A_ex) / A_ex, 2e-4,
           f"A = {A_ex*1e3:.4f} mm")
    report(f"phase,     Om/wn = {r:.1f}", abs(ph_num - ph_ex), 2e-3,
           f"phi = {np.degrees(ph_ex):+7.2f} deg")

print("\nT7  energy conservation,  zero damping, no force")
p.zeta = 0.0
t7 = np.linspace(0, 40, 4001)
mv7 = CoupledOscillator(p, aero, force_ext=lambda tt: 0.0)
s7 = solve_ivp(mv7.rhs, (0, t7[-1]), np.r_[0.05, 0.0, aero.y0(p.alpha_mean, p.U)],
               t_eval=t7, rtol=1e-13, atol=1e-15)
x, v = s7.y[0], s7.y[1]
E = 0.5 * p.mass * v ** 2 + 0.5 * p.k1 * x ** 2
report("max |E - E0| / E0", np.max(np.abs(E - E[0])) / E[0], 1e-10)
p.zeta = 0.005

print("\nT8  solve_ivp vs fixed-step RK4, FULL coupled system")
z0 = m.z0(x0=1e-3)
T8, dt = 2.0, 1e-4
n = int(T8 / dt)
zz = z0.copy()
for i in range(n):
    k1 = m.rhs(0, zz); k2 = m.rhs(0, zz + dt / 2 * k1)
    k3 = m.rhs(0, zz + dt / 2 * k2); k4 = m.rhs(0, zz + dt * k3)
    zz = zz + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
s = solve_ivp(m.rhs, (0, T8), z0, t_eval=[T8], rtol=1e-10, atol=1e-14)
report("max |z_RK4 - z_solve_ivp| (8 states)",
       np.max(np.abs(zz - s.y[:, -1])), 1e-6)

print("\nT9  Jacobian eigenvalue vs decay of the nonlinear simulation")
for U in (4.0, 20.0, 30.0, 35.0):
    p.U = U
    sig, fq, ev, z_eq, r_ = leading(m)
    t9 = np.linspace(0, 40 if sig < -0.02 else 12, 20001)
    s = solve_ivp(m.rhs, (0, t9[-1]), z_eq + np.r_[1e-5, np.zeros(7)],
                  t_eval=t9, rtol=1e-10, atol=1e-14)
    d = s.y[0] - z_eq[0]
    i0 = np.searchsorted(t9, 5.0)
    pk = i0 + np.where((d[i0 + 1:-1] > d[i0:-2]) & (d[i0 + 1:-1] > d[i0 + 2:]))[0] + 1
    pk = pk[d[pk] > 0]
    sl = np.polyfit(t9[pk], np.log(d[pk]), 1)[0]
    f_num = 1.0 / np.mean(np.diff(t9[pk]))
    report(f"growth rate, U = {U:>4.1f} m/s", abs(sl - sig) / max(abs(sig), 1e-9),
           2e-2, f"eig {sig:+.5f}  sim {sl:+.5f}  1/s")
    report(f"frequency,   U = {U:>4.1f} m/s", abs(f_num - fq) / fq, 2e-2,
           f"eig {fq:.5f}  sim {f_num:.5f}  Hz")

print(f"\n{sum(res)}/{len(res)} checks passed")
