import warnings,time,os; warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp
from input import Params, DEG
from coupled import CoupledOscillator
from oye import Oye
from stability import leading
p=Params(); p.alpha_mean=18.5*DEG
m=CoupledOscillator(p, Oye(p.polar,p.chord,T_f=3.0))
SEQ=[1.5,1.3,1.2,1.15,1.10,1.05,1.00]; st="oyedown.npz"
if os.path.exists(st):
    d=np.load(st); z=d["z"]; k=int(d["k"])
else:
    p.U=2.0; z=leading(m)[3]+np.r_[1e-3,0,0]
    z=solve_ivp(m.rhs,(0,1200),z,method="LSODA",rtol=1e-8,atol=1e-11).y[:,-1]; k=0
t0=time.time()
while k<len(SEQ) and time.time()-t0<28:
    U=SEQ[k]; p.U=U; sig=leading(m)[0]
    s=solve_ivp(m.rhs,(0,1500),z,method="LSODA",rtol=1e-8,atol=1e-11,
                t_eval=np.linspace(900,1500,6000))
    z=s.y[:,-1]; t=s.t; x=s.y[0]
    A1=0.5*(x[t<1200].max()-x[t<1200].min())*1e3
    A2=0.5*(x[t>=1200].max()-x[t>=1200].min())*1e3
    print(f"U={U:5.2f}  Re={sig:+9.5f}  A={A2:9.4f} mm  ratio={A2/max(A1,1e-12):7.4f}"
          f"  {'LCO' if (A2>0.05 and abs(A2/max(A1,1e-12)-1)<5e-3) else 'decaying'}",flush=True)
    k+=1
np.savez(st,z=z,k=k)
print("finished" if k==len(SEQ) else f"[paused {k}/{len(SEQ)}]")
