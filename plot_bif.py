import re, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=[]
for fn in ("lco.log","lco2.log"):
    try:
        for ln in open(fn):
            mm=re.match(r"\s*([\d.]+)\s+A =\s+([\d.eE+-]+) mm",ln)
            if mm: rows.append((float(mm.group(1)),float(mm.group(2))))
    except FileNotFoundError: pass
D=np.array(sorted(set(rows)))
U_h=float(np.load("bifurcation_linear.npz")["U_h"])
np.save("lco_branch.npy",D)
fig,ax=plt.subplots(figsize=(5.4,3.8))
st=D[:,0]<U_h
ax.plot(D[st,0],D[st,1],"o-",ms=4,color="0.4",label="decays to equilibrium")
ax.plot(D[~st,0],D[~st,1],"o-",ms=4,color="C3",label="limit cycle")
ax.axvline(U_h,color="C0",lw=1.0,ls="--")
ax.annotate(f"Hopf,  $U={U_h:.2f}$ m/s",(U_h,0.72*D[:,1].max()),fontsize=8,
            textcoords="offset points",xytext=(-108,0))
ax.set_xlabel("$U$ [m/s]"); ax.set_ylabel("edgewise amplitude [mm]")
ax.set_title(r"Bifurcation diagram,  $\tilde\alpha=18.5^{\circ}$",fontsize=10)
ax.grid(alpha=0.25); ax.legend(fontsize=8,frameon=False,loc="upper left")
fig.tight_layout(); fig.savefig("fig_bifurcation.png",dpi=200)
print(len(D),"points;  U range",D[0,0],"-",D[-1,0])
for u,a in D: print(f"  {u:6.2f}  {a:10.4f} mm")
