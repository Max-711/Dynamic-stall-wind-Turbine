import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from input import Params, DEG
from coupled import CoupledOscillator
from oye import Oye
from stability import leading
out = sys.argv[1] if len(sys.argv)>1 else "."
p=Params(); p.alpha_mean=18.5*DEG
mo=CoupledOscillator(p, Oye(p.polar,p.chord,T_f=3.0))
U=np.concatenate([np.arange(0.25,4,0.25),np.arange(4,60.1,1.0)])
so=np.full(len(U),np.nan); fo=np.full(len(U),np.nan)
for i,u in enumerate(U):
    p.U=u
    try:
        s,f,_,z,r=leading(mo)
        if r<1e-8: so[i],fo[i]=s,f
    except Exception: pass
np.savez("oye_sweep.npz",U=U,s=so,f=fo)
d=np.load("bifurcation_linear.npz")
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].axhline(0,color="0.6",lw=0.8)
ax[0].plot(U,so,"-",color="C2",label="Oye  (3 states)")
ax[0].plot(d["Us"],d["s_c"],"-",color="C0",label="IAG  (8 states)")
ax[0].plot([1.1684],[0],"o",color="C2",ms=5); ax[0].plot([32.7706],[0],"o",color="C0",ms=5)
ax[0].annotate("$1.17$",(1.1684,0),fontsize=8,textcoords="offset points",xytext=(2,10),color="C2")
ax[0].annotate("$32.77$",(32.7706,0),fontsize=8,textcoords="offset points",xytext=(-30,-16),color="C0")
ax[0].set_xlabel("$U$ [m/s]"); ax[0].set_ylabel(r"$\max\,\mathrm{Re}(\lambda)$ [1/s]")
ax[0].set_xlim(0,60); ax[0].set_ylim(-0.8,2.0); ax[0].legend(fontsize=8,frameon=False)
ax[1].plot(U,fo,"-",color="C2",label="Oye")
ax[1].plot(d["Us"],d["f_c"],"-",color="C0",label="IAG")
ax[1].axhline(0.687,color="0.6",lw=0.8,ls=":")
ax[1].annotate("$f_n$",(1,0.687),fontsize=8,textcoords="offset points",xytext=(2,3))
ax[1].set_xlabel("$U$ [m/s]"); ax[1].set_ylabel("frequency [Hz]")
ax[1].set_xlim(0,60); ax[1].legend(fontsize=8,frameon=False)
for a in ax: a.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{out}/fig_compare.pdf"); fig.savefig(f"{out}/fig_compare.png",dpi=200)
print("saved fig_compare;  Oye maxRe at 60 m/s =", f"{so[-1]:.3f}")
