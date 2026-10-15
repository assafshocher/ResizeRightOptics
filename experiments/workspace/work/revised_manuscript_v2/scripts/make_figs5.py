#!/usr/bin/env python3
"""Family-as-a-function-of-s figures, current fineness gauge (s<1 coarser).
fig_family.pdf: (A) forced symbol heat over (omega, log s) with the visible
freedom wedge opening above s=1; (B) spatial waterfall of K_{1->s}.
fig_family3d.pdf: 3D surface of the spatial family. fig_gaussfam.pdf:
Gaussian camera vs s. fig_sets2d.pdf: corrected-gauge 2D annulus strip."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
plt.rcParams.update({"font.size":8.5,"axes.linewidth":0.6})
OUT='paper_output/figs'
def nospines(ax):
    for sd in ["top","right"]: ax.spines[sd].set_visible(False)

W=np.pi/3; ETA=0.05
H=lambda o: np.clip(1-np.abs(o)/W,0,None)

# ---------- fig_family: frequency portrait + spatial waterfall ----------
fig,(A,B)=plt.subplots(1,2,figsize=(7.0,2.9),dpi=200,
                       gridspec_kw={"width_ratios":[1.15,1]})
wv=np.linspace(0,2.6*W,420)
sv=2.0**np.linspace(-2,2,320)
WW,SS=np.meshgrid(wv,sv)
FORCED=H(WW/SS)*H(WW)/(H(WW)**2+ETA**2)
A.imshow(np.minimum(np.abs(FORCED),3.0),origin="lower",aspect="auto",
         extent=[0,2.6,np.log2(sv[0]),np.log2(sv[-1])],cmap="Greys")
FREE=((H(WW)<=ETA)&(H(WW/SS)>=ETA)).astype(float)
A.imshow(np.dstack([np.ones_like(FREE),0*FREE,0.2*np.ones_like(FREE),0.30*FREE]),
         origin="lower",aspect="auto",extent=[0,2.6,np.log2(sv[0]),np.log2(sv[-1])])
A.axhline(0,color="k",lw=0.6,ls="--")
A.set_yticks([-2,-1,0,1,2]); A.set_yticklabels([r"$\frac{1}{4}$",r"$\frac{1}{2}$",r"$1$",r"$2$",r"$4$"])
A.set_xticks([0,1,2]); A.set_xticklabels([r"$0$",r"$B$",r"$2B$"])
A.set_xlabel(r"$\omega$"); A.set_ylabel(r"$s$")
A.text(1.55,1.35,"sets",color="crimson",fontsize=8)
A.text(0.35,-1.4,"unique action",color="0.25",fontsize=8)
# spatial waterfall
N=8192; dx=0.25
x=(np.arange(N)-N//2)*dx
w=2*np.pi*np.fft.fftfreq(N,dx)
xm=np.abs(x)<=7
slist=2.0**np.linspace(-3,3,13)
cm=plt.cm.viridis
for i,s in enumerate(slist):
    Ks=np.fft.fftshift(np.fft.ifft(H(w/s)*H(w)/(H(w)**2+ETA**2)).real)
    Ks=Ks/np.max(np.abs(Ks))
    B.plot(x[xm],Ks[xm]+2.0*i,color=cm(i/(len(slist)-1)),lw=0.9)
    B.axhline(2.0*i,color="0.9",lw=0.3,zorder=0)
for i,lab in [(0,r"$s=\frac{1}{8}$"),(6,r"$s=1$"),(12,r"$s=8$")]:
    B.text(7.3,2.0*i,lab,fontsize=7.5,va="center")
B.set_xlabel(r"$x$"); B.set_yticks([])
nospines(B); B.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_family.pdf"); plt.close(fig)

# ---------- fig_family3d ----------
fig=plt.figure(figsize=(4.6,3.0),dpi=200)
ax=fig.add_subplot(111,projection="3d")
s3=2.0**np.linspace(-3,3,60)
xm3=np.abs(x)<=8
X3=x[xm3]
S3,XX3=np.meshgrid(np.log2(s3),X3,indexing="ij")
Z=np.zeros_like(S3)
for i,s in enumerate(s3):
    Ks=np.fft.fftshift(np.fft.ifft(H(w/s)*H(w)/(H(w)**2+ETA**2)).real)
    Z[i]=Ks[xm3]/np.max(np.abs(Ks))
ax.plot_surface(XX3,S3,Z,cmap="viridis",linewidth=0,antialiased=True,
                rcount=60,ccount=120)
ax.set_xlabel(r"$x$",labelpad=-4); ax.set_ylabel(r"$\log_2 s$",labelpad=-4)
ax.set_zticks([]); ax.tick_params(pad=-2,labelsize=7)
ax.view_init(elev=35,azim=-60)
fig.savefig(f"{OUT}/fig_family3d.pdf",bbox_inches="tight"); plt.close(fig)

# ---------- fig_gaussfam ----------
SIG=2.0; ETg=0.03
G=lambda o: np.exp(-np.clip(SIG**2*o**2/2,0,700))
fig,ax=plt.subplots(figsize=(3.6,2.7),dpi=200)
wv=np.linspace(0,3.0,420)
sv=2.0**np.linspace(-2,2,320)
WW,SS=np.meshgrid(wv,sv)
FORCED=G(WW/SS)*G(WW)/(G(WW)**2+ETg**2)
ax.imshow(np.minimum(np.abs(FORCED),3.0),origin="lower",aspect="auto",
          extent=[0,3.0,-2,2],cmap="Greys")
FREE=((G(WW)<=ETg)&(G(WW/SS)>=ETg)).astype(float)
ax.imshow(np.dstack([np.ones_like(FREE),0*FREE,0.2*np.ones_like(FREE),0.30*FREE]),
          origin="lower",aspect="auto",extent=[0,3.0,-2,2])
ax.axhline(0,color="k",lw=0.6,ls="--")
ax.set_yticks([-2,-1,0,1,2]); ax.set_yticklabels([r"$\frac{1}{4}$",r"$\frac{1}{2}$",r"$1$",r"$2$",r"$4$"])
ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$s$")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gaussfam.pdf"); plt.close(fig)

# ---------- fig_sets2d, corrected gauge ----------
N2=192
w2=2*np.pi*np.fft.fftfreq(N2,1.0)
WX,WY=np.meshgrid(w2,w2,indexing="ij")
RHO=np.hypot(WX,WY); W2=np.pi/3; E2=0.04
H2r=lambda r: np.clip(1-r/W2,0,None)
fig,axs=plt.subplots(1,3,figsize=(6.8,2.3),dpi=200)
for ax,ss in zip(axs,[0.6,1.05,2.0]):
    F=H2r(RHO/ss)*H2r(RHO)/(H2r(RHO)**2+E2**2)
    ax.imshow(np.fft.fftshift(np.minimum(np.abs(F),3)),cmap="Greys",vmin=0,vmax=3,
              origin="lower",extent=[w2.min(),-w2.min(),w2.min(),-w2.min()])
    if ss>1:
        ann=np.fft.fftshift(((RHO>=W2)&(RHO<ss*W2)).astype(float))
        ax.imshow(np.dstack([np.ones_like(ann),0*ann,0.2*np.ones_like(ann),0.28*ann]),
                  origin="lower",extent=[w2.min(),-w2.min(),w2.min(),-w2.min()])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(rf"$s={ss}$",fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sets2d.pdf"); plt.close(fig)
print("family figs done")
