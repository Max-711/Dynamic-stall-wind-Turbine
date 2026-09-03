# Thesis code — IAG / Øye aeroelastic oscillator

---

## 1. Core model (never run directly, imported by everything else)

| File | What it is |
|---|---|
| `input.py` | `Params` (structure + flow), `Polar` (S809 table), `DEG` |
| `IAG_Continuous.py` | IAG dynamic-stall model, **smoothed** switches (tanh, widths `eps1..eps3`). Used for everything in Ch 7 — the Jacobian needs a differentiable right-hand side. |
| `IAG_DisContinuous.py` | Same model with **hard** `0/1` switches. Used only in Ch 6 to price the cost of smoothing. |
| `oye.py` | Øye separation-lag model (3 states) |
| `coupled.py` | **The closed-form coupled system** `zdot = F(z)`, z ∈ R^8. Shows that the apparent algebraic loop is an added mass; `added_mass()` gives Λ = dF/dαdot. Run directly for the affinity check and `m_add`. |

## 2. Solvers / analysis

| File | What it does | Writes |
|---|---|---|
| `stability.py` | equilibrium (`fsolve`) → **numerical Jacobian** (central differences on the *same* rhs the integrator uses) → eigenvalues → `sweep`, `hopf_point` | — |
| `bifurcation.py` | U-sweep of max Re(λ) for coupled and αdot-suppressed rhs; Hopf point by `brentq` | `bifurcation_linear.npz` |
| `field2d.py` | same, over the (U, α̃) plane | `field2d.npz` |
| `oye_stability.py` | the Ch 7 pipeline applied to the 3-state Øye system | — |
| `verify_oscillator.py` | **34 automated checks** behind §6.4/§6.5, incl. T8 (solve_ivp vs fixed-step RK4 on the full 8-state system) and T9 (Jacobian eigenvalue vs the growth rate measured from the nonlinear time series) | — |
| `eps_sens.py` | U_H as a function of the smoothing width ε — the evidence that the result is insensitive to ε | — |

## 3. Nonlinear time marching (the LCO branch)

These print to the console; their output was captured by hand. **This is the one
fragile part of the pipeline** — see the provenance notes in §5.

| File | What it does | Writes |
|---|---|---|
| `lco.py`, `lco2.py` | up sweep: amplitude reached from a 1 mm disturbance at each U | console → `lco.log`, `lco2.log` |
| `plot_bif.py` | parses those two logs into the array the figure uses | `lco_branch.npy` |
| `sweep_down2.py` | down sweep, each U continued from the cycle at the previous U (resumable via `sweepstate.npz`) | console + `sweepstate.npz` |
| `fold_refine.py` | locates the fold U_F ≈ 31.9 m/s by running 400 s at each U | console |
| `decay_test.py` | below U_H, checks the decay rate against exp(σt) | console |
| `oye_crit.py`, `oye_down2.py` | the same up/down test for Øye — the evidence for "no coexistence, no hysteresis" (Ch 7 §…) | console |

## 4. Figures — which script makes which thesis figure

The thesis includes only the PDFs in `~/Desktop/UOB_final__1___1_/figures/`.
Every script below takes the output directory as its first argument, so

```
python3 <script>.py ~/Desktop/UOB_final__1___1_/figures
```

writes straight into the thesis. `.png` copies are previews only.

| Thesis figure | Script | Reads |
|---|---|---|
| `fig_iag_cl.pdf`, `fig_iag_vortex.pdf` | `mainiag.py` | `bladed_reference.csv`, `S809_polar.csv` |
| `fig_verify_oye.png` (Ch 6 table only) | `mainoye.py` | same |
| `fig_stability.pdf` | `figures_ch7.py` | `bifurcation_linear.npz` |
| `fig_twoparam.pdf` | `figures_ch7.py` | `field2d.npz` |
| `fig_bifurcation.pdf` | **`fig_bif2.py`** | `lco_branch.npy` + a hard-coded down sweep |
| `fig_compare.pdf` | `fig_compare.py` | `bifurcation_linear.npz` (+ computes the Øye sweep itself) |
| `fig31/32/33_*.pdf` | `figures/mksvg.py` **in the thesis folder**, not here | — |

> **Do not let `figures_ch7.py` write `fig_bifurcation.pdf`.** An earlier version
> of it did, with a simplified up-sweep-only diagram, and it silently overwrote
> the correct figure from `fig_bif2.py`. That block has been removed — keep it
> removed.

## 5. Provenance of the numbers that are hard-coded in figures

| Number | Where it is hard-coded | Where it came from |
|---|---|---|
| `U_H = 32.7706` m/s | `fig_bif2.py`, `fig_compare.py`, `eps_sens.py` | `bifurcation.py` (brentq on max Re λ) |
| `U_F = 31.9` m/s | `fig_bif2.py` | `fold_refine.py` |
| Øye `U_H = 1.1684` m/s | `fig_compare.py`, `oye_crit.py` | `oye_stability.py` / `fig_compare.py` sweep |
| down-sweep amplitudes (13 points) | `down = [...]` in `fig_bif2.py` | `sweep_down2.py` console output |
| `f_n = 0.687` Hz | `figures_ch7.py`, `fig_compare.py` | `input.py` (`Params.f_n`) |

## 6. Reproduce everything, in order

```
python3 coupled.py                                   # affinity + added mass (m_add = 11.98 kg/m, 9.2 % of m)
python3 verify_oscillator.py                         # 34/34 checks must pass
python3 mainiag.py  ../UOB_final__1___1_/figures     # Ch 6 Table 6.1 + two figures
python3 mainoye.py  ../UOB_final__1___1_/figures
python3 bifurcation.py                               # -> bifurcation_linear.npz   (slow)
python3 field2d.py                                   # -> field2d.npz              (slow)
python3 oye_stability.py                             # Øye spectrum
python3 figures_ch7.py ../UOB_final__1___1_/figures  # fig_stability, fig_twoparam
python3 fig_compare.py ../UOB_final__1___1_/figures  # fig_compare
python3 fig_bif2.py    ../UOB_final__1___1_/figures  # fig_bifurcation
```

The LCO branch (`lco*.py`, `sweep_down2.py`, `fold_refine.py`) takes hours and its
results are already frozen in `lco_branch.npy` and in the `down` array of
`fig_bif2.py`. Only re-run it if the structural parameters change.

## 7. `_archive_0828/`

Superseded and duplicate files moved out on 2026-08-28: the pre-`coupled.py`
model (`oscillator.py`, `IAG.py`, `IAGorigin.py` and their mains), `* copy.py`
duplicates, one-off scan scripts, plot scripts replaced by the ones above, and
the loose `.png`/`.pdf` previews. Nothing in the live pipeline imports anything
from there. Delete the folder once you are happy.

Verified after the clean-up: all six thesis PDFs regenerate **byte-identical**
(bar the embedded creation date) and `verify_oscillator.py` still reports 34/34.
