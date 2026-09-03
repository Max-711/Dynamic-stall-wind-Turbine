import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp
from bifurcation import build
from stability import leading
p,m=build()
d=np.load("sweepstate.npz"); z=d["z"]        # state on the 83 mm cycle at U = 32.5
p.U=32.5
sig=leading(m)[0]
print(f"U = 32.5 m/s   (below U_H = 32.77),  Re(lambda) = {sig:+.5f} 1/s,"
      f"  decay time {1/abs(sig):.0f} s")
t=np.linspace(0,400,8000)
s=solve_ivp(m.rhs,(0,400),z,method="LSODA",rtol=1e-6,atol=1e-9,t_eval=t)
x=s.y[0]-leading(m)[3][0]
print(f"{'window [s]':>14} {'amplitude [mm]':>16} {'ratio':>8}")
prev=None
for a,b in ((0,40),(80,120),(160,200),(240,280),(320,360),(360,400)):
    k=(t>=a)&(t<b); A=0.5*(x[k].max()-x[k].min())*1e3
    r=f"{A/prev:8.3f}" if prev else "       -"
    print(f"{a:6.0f}-{b:<7.0f} {A:16.4f} {r}")
    prev=A
print(f"\npredicted per-80s decay factor exp(80*sigma) = {np.exp(80*sig):.3f}")
