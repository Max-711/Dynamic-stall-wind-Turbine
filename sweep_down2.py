import warnings,time,os,sys; warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp
from bifurcation import build
from stability import leading
p,m=build(); UH=32.7706
SEQ=[34.0,33.5,33.0,32.5,32.0,31.0]
st="sweepstate.npz"
if os.path.exists(st):
    d=np.load(st); z=d["z"]; k=int(d["k"]); rows=list(d["rows"])
else:
    p.U=34.77; z=leading(m)[3]+np.r_[5e-3,np.zeros(7)]
    z=solve_ivp(m.rhs,(0,80),z,method="LSODA",rtol=1e-7,atol=1e-10).y[:,-1]
    k=0; rows=[]
t0=time.time()
while k<len(SEQ) and time.time()-t0<26:
    U=SEQ[k]; p.U=U
    s=solve_ivp(m.rhs,(0,100.0),z,method="LSODA",rtol=1e-7,atol=1e-10,
                t_eval=np.linspace(75,100,2000))
    z=s.y[:,-1]; A=0.5*(s.y[0].max()-s.y[0].min())*1e3
    print(f"U={U:6.2f}  A={A:10.4f} mm   {'LCO' if A>1 else 'decays to equilibrium'}",flush=True)
    rows.append((U,A)); k+=1
np.savez(st,z=z,k=k,rows=np.array(rows))
print("done" if k==len(SEQ) else f"[paused at {k}/{len(SEQ)}]")
