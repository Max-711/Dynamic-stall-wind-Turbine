import warnings,time; warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp
from bifurcation import build
from stability import leading
p,m=build()
t0=time.time()
for U in (38.0,40.0,43.0,46.0,50.0):
    p.U=U
    z0=leading(m)[3]+np.r_[1e-3,np.zeros(7)]
    s=solve_ivp(m.rhs,(0,60.0),z0,method="LSODA",rtol=1e-7,atol=1e-9,
                t_eval=np.linspace(42,60,2000))
    if s.success:
        x=s.y[0]; A=0.5*(x.max()-x.min())
        print(f"{U:7.2f}  A = {1e3*A:10.4f} mm   [{time.time()-t0:5.1f}s]",flush=True)
    else:
        print(f"{U:7.2f}  failed",flush=True)
