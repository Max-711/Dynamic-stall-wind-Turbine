import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp
from input import Params, DEG
from coupled import CoupledOscillator
from oye import Oye
from stability import leading
p=Params(); p.alpha_mean=18.5*DEG
m=CoupledOscillator(p, Oye(p.polar,p.chord,T_f=3.0))
UH=1.1684
print("UP sweep (1 mm initial disturbance)")
print(f"{'U':>6} {'maxRe':>10} {'A [mm]':>11}")
for U in (0.6,0.9,1.1,1.3,1.6,2.0,3.0,4.0):
    p.U=U; sig=leading(m)[0]; z=leading(m)[3]+np.r_[1e-3,0,0]
    s=solve_ivp(m.rhs,(0,400),z,method="LSODA",rtol=1e-8,atol=1e-11,
                t_eval=np.linspace(360,400,3000))
    A=0.5*(s.y[0].max()-s.y[0].min())*1e3
    print(f"{U:6.2f} {sig:10.5f} {A:11.4f}",flush=True)
