"""Limit-cycle amplitude either side of the Hopf point -> bifurcation diagram."""
import warnings, sys, time
import numpy as np
warnings.filterwarnings("ignore")
from bifurcation import build, lco_amplitude

p, model = build()
d = np.load("bifurcation_linear.npz")
U_h = float(d["U_h"])
Us = np.round(np.concatenate([np.arange(24., U_h, 2.0),
                              np.arange(U_h, 52.1, 2.0)]), 3)
rows = []
t0 = time.time()
for U in Us:
    A = lco_amplitude(model, U, t_end=80.0, x0=1e-3)
    rows.append((U, A))
    print(f"{U:7.2f}  A = {1e3*A:10.4f} mm   [{time.time()-t0:6.1f}s]", flush=True)
np.save("lco.npy", np.array(rows))
print("done")
