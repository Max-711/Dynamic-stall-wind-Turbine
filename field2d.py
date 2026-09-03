import warnings; warnings.filterwarnings("ignore")
import numpy as np
from input import Params, DEG
from coupled import CoupledOscillator
from IAG_Continuous import IAG
from stability import leading
p=Params(); aero=IAG(p.polar,p.chord); m=CoupledOscillator(p,aero)
A=np.arange(14.0,23.01,0.25); U=np.concatenate([np.arange(4,42,2.0),np.arange(42,81,4.0)])
S=np.full((len(A),len(U)),np.nan); F=np.full_like(S,np.nan)
for i,ad in enumerate(A):
    p.alpha_mean=ad*DEG; g=None
    for j,u in enumerate(U):
        p.U=u
        try:
            s,f,ev,z,r=leading(m,None,g)
            if r<1e-8: S[i,j],F[i,j],g=s,f,z
            else:
                s,f,ev,z,r=leading(m)          # retry from the default guess
                if r<1e-8: S[i,j],F[i,j],g=s,f,z
        except Exception: pass
    print(f"{ad:5.2f} {np.nanmin(S[i]):+8.4f} {np.nanmax(S[i]):+8.4f} nan={np.isnan(S[i]).sum()}", flush=True)
np.savez("field2d.npz",A=A,U=U,S=S,F=F)
print("saved")
