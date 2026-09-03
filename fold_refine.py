import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp
from bifurcation import build
from stability import leading
p,m=build()
z=np.load("sweepstate.npz")["z"]
res=[]
for U in (31.9,31.8,31.7,31.6):
    p.U=U
    s=solve_ivp(m.rhs,(0,400),z,method="LSODA",rtol=1e-6,atol=1e-9,
                t_eval=np.linspace(0,400,8000))
    t=s.t; x=s.y[0]
    A1=0.5*(x[(t>=240)&(t<280)].max()-x[(t>=240)&(t<280)].min())*1e3
    A2=0.5*(x[(t>=360)&(t<400)].max()-x[(t>=360)&(t<400)].min())*1e3
    conv = abs(A2/A1-1)<2e-3 and A2>1
    print(f"U={U:5.2f}   A(240-280)={A1:9.4f}   A(360-400)={A2:9.4f}   "
          f"{'stable LCO' if conv else 'collapsing'}",flush=True)
    res.append((U,A2,conv))
np.save("fold_refine.npy",np.array([(u,a) for u,a,c in res if c]))
