import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import brentq
from input import Params, DEG
from coupled import CoupledOscillator
from IAG_Continuous import IAG
from stability import leading
p=Params(); p.alpha_mean=18.5*DEG; aero=IAG(p.polar,p.chord); m=CoupledOscillator(p,aero)
print(f"{'eps':>8} {'U_H [m/s]':>11} {'f [Hz]':>9} {'rel. shift':>11}")
base=None
for e in (3e-3,1e-2,3e-2,1e-1,3e-1):
    aero.eps1=aero.eps2=aero.eps3=e
    def g(U):
        p.U=U; return leading(m)[0]
    try:
        Uh=brentq(g,25.,45.,xtol=1e-6); p.U=Uh; f=leading(m)[1]
        base=Uh if base is None else base
        print(f"{e:8.0e} {Uh:11.4f} {f:9.4f} {(Uh-32.7706)/32.7706:+11.2e}")
    except Exception as ex:
        print(f"{e:8.0e}  failed: {ex}")
