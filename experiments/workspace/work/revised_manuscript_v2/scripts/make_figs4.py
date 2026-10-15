#!/usr/bin/env python3
"""v4 figures: fig_prelim (2D kernels: bilinear/bicubic/Lanczos3/Gaussian),
fig_scaling (2D four-phase, sparse off-center grid), fig_cc_process (pipeline)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
plt.rcParams.update({"font.size":8.5,"axes.linewidth":0.6})
OUT='paper_output/figs'
def nospines(ax):
    for sd in ["top","right"]: ax.spines[sd].set_visible(False)
def keys(t,a=-0.5):
    at=np.abs(t)
    return np.where(at<=1,(a+2)*at**3-(a+3)*at**2+1,
           np.where(at<2,a*(at**3-5*at**2+8*at-4),0.0))
def tri(t): return np.clip(1-np.abs(t),0,None)
def lanczos(t,A=3):
    z=np.where(np.abs(t)<1e-9,1e-9,t)
    L=A*np.sin(np.pi*z)*np.sin(np.pi*z/A)/(np.pi**2*z**2)
    return np.where(np.abs(t)<1e-9,1.0,np.where(np.abs(t)<A,L,0.0))
def gauss(t,s=0.7): return np.exp(-t**2/(2*s**2))/(s*np.sqrt(2*np.pi))

# ---------- fig_prelim: 2D kernel cast ----------
xx=np.linspace(-3.2,3.2,321)
X,Y=np.meshgrid(xx,xx)
kers=[("bilinear",tri),("bicubic",keys),("Lanczos-3",lanczos),("Gaussian",gauss)]
fig,axs=plt.subplots(2,4,figsize=(7.0,3.5),dpi=200)
sup={"bilinear":1,"bicubic":2,"Lanczos-3":3,"Gaussian":3}
for j,(name,f) in enumerate(kers):
    K2=f(X)*f(Y)
    vm=np.max(np.abs(K2))
    axs[0][j].imshow(K2,cmap="RdBu_r",vmin=-vm,vmax=vm,extent=[-3.2,3.2,-3.2,3.2])
    axs[0][j].set_title(name,fontsize=8.5)
    axs[0][j].set_xticks([]); axs[0][j].set_yticks([])
    S=sup[name]
    kk=np.arange(-S,S)+0.5           # half-pixel phase taps
    t=f(kk); T=np.outer(t,t)
    vt=np.max(np.abs(T))
    axs[1][j].imshow(T,cmap="RdBu_r",vmin=-vt,vmax=vt)
    axs[1][j].set_xticks(np.arange(-0.5,2*S,1),minor=True)
    axs[1][j].set_yticks(np.arange(-0.5,2*S,1),minor=True)
    axs[1][j].grid(which="minor",color="w",lw=0.8)
    axs[1][j].set_xticks([]); axs[1][j].set_yticks([])
    axs[1][j].set_xlabel(rf"${2*S}\times{2*S}$ taps",fontsize=7.5)
axs[0][0].set_ylabel("continuous")
axs[1][0].set_ylabel("discrete")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_prelim.pdf"); plt.close(fig)

# ---------- fig_scaling: 2D four phases ----------
IN,OUTN=6,4; s=OUTN/IN
gn=(np.arange(OUTN)+0.5)/s-0.5
rng=np.random.default_rng(7)
base=rng.standard_normal((IN,IN))
kk=np.exp(-0.5*(np.arange(-2,3)/1.0)**2); kk/=kk.sum()
img=base
for _ in range(2):
    img=np.apply_along_axis(lambda v: np.convolve(v,kk,mode="same"),0,img)
    img=np.apply_along_axis(lambda v: np.convolve(v,kk,mode="same"),1,img)
GX,GY=np.meshgrid(gn,gn)
fig,axs=plt.subplots(1,4,figsize=(7.2,2.0),dpi=200)
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
a=axs[0]
a.imshow(img,cmap="gray",extent=[-0.5,IN-0.5,IN-0.5,-0.5],alpha=0.55)
px,py=np.meshgrid(np.arange(IN),np.arange(IN))
a.plot(px.ravel(),py.ravel(),".",color="0.25",ms=2.5)
a.plot(GX.ravel(),GY.ravel(),"x",color="crimson",ms=5,mew=1.3)
a.set_title("1: project grid",fontsize=8)
b=axs[1]
b.plot(px.ravel(),py.ravel(),".",color="0.7",ms=2.5)
cx,cy=GX[1,2],GY[1,2]
b.plot([cx],[cy],"x",color="crimson",ms=6,mew=1.4)
nb=[(m1,m2) for m1 in range(IN) for m2 in range(IN)
    if abs(m1-cx)<2 and abs(m2-cy)<2]
b.plot([m[0] for m in nb],[m[1] for m in nb],"o",mfc="none",mec="k",ms=6,mew=1.0)
from matplotlib.patches import Rectangle
b.add_patch(Rectangle((cx-2,cy-2),4,4,fill=True,color="crimson",alpha=0.08))
b.set_xlim(-0.5,IN-0.5); b.set_ylim(IN-0.5,-0.5)
b.set_title("2: neighbors",fontsize=8)
c=axs[2]
gxx=np.linspace(cx-2.2,cx+2.2,300)
gyy=np.linspace(cy-2.2,cy+2.2,300)
GXX,GYY=np.meshgrid(gxx,gyy)
KK=keys(GXX-cx)*keys(GYY-cy)
vk=0.35*np.max(np.abs(KK))
c.plot(px.ravel(),py.ravel(),".",color="0.7",ms=2.5)
c.imshow(KK,cmap="RdBu_r",vmin=-vk,vmax=vk,alpha=0.55,
         extent=[cx-2.2,cx+2.2,cy+2.2,cy-2.2])
for (m1,m2) in nb:
    w=keys(m1-cx)*keys(m2-cy)
    c.plot([m1],[m2],"o",ms=3+9*abs(w),mfc=("#2166ac" if w>0 else "#b2182b"),
           mec="w",mew=0.5,alpha=0.95)
c.plot([cx],[cy],"x",color="crimson",ms=6,mew=1.4)
c.set_xlim(-0.5,IN-0.5); c.set_ylim(IN-0.5,-0.5)
c.set_title(r"3: weights $K(g_n\!-\!m)$",fontsize=8)
d=axs[3]
W=keys(gn[:,None]-np.arange(IN)[None,:])
outimg=W@img@W.T
d.imshow(outimg,cmap="gray",extent=[-0.5,OUTN-0.5,OUTN-0.5,-0.5])
d.set_title("4: apply and sum",fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scaling.pdf"); plt.close(fig)

# ---------- fig_cc_process: pipeline, no nets ----------
fig=plt.figure(figsize=(7.2,2.1),dpi=200)
gs=fig.add_gridspec(1,5,width_ratios=[1,1.15,1,1.15,1],wspace=0.25)
ax1=fig.add_subplot(gs[0])
small=img[::2,::2] if img.shape[0]>=6 else img
ax1.imshow(img[:4,:4],cmap="gray")
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title("samples",fontsize=8)
ax2=fig.add_subplot(gs[1],projection="3d")
sub=img[:4,:4]
Xs,Ys=np.meshgrid(np.arange(4),np.arange(4))
for xi,yi,zi in zip(Xs.ravel(),Ys.ravel(),sub.ravel()):
    ax2.plot([xi,xi],[yi,yi],[0,zi],color="k",lw=0.8)
    ax2.scatter([xi],[yi],[zi],color="k",s=5)
ax2.set_title("bed of nails",fontsize=8)
ax3=fig.add_subplot(gs[2],projection="3d")
gx=np.linspace(-2,2,60); GX2,GY2=np.meshgrid(gx,gx)
ax3.plot_surface(GX2,GY2,keys(GX2)*keys(GY2),cmap="Greys",linewidth=0,antialiased=True)
ax3.set_title(r"kernel $K$",fontsize=8)
ax4=fig.add_subplot(gs[3],projection="3d")
up=np.linspace(0,3,40)
UX,UY=np.meshgrid(up,up)
Wu=keys(up[:,None]-np.arange(4)[None,:])
cont=Wu@sub@Wu.T
ax4.plot_surface(UX,UY,cont,cmap="Greys",linewidth=0,antialiased=True)
ax4.set_title("continuous",fontsize=8)
for ax in (ax2,ax3,ax4):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
ax5=fig.add_subplot(gs[4])
g7=(np.arange(7)+0.5)*4/7-0.5
W7=keys(g7[:,None]-np.arange(4)[None,:])
big=W7@sub@W7.T
g3=(np.arange(3)+0.5)*4/3-0.5
W3=keys(g3[:,None]-np.arange(4)[None,:])
sml=W3@sub@W3.T
ax5.imshow(big,cmap="gray",extent=[0,7,7,0])
ax5.imshow(sml,cmap="gray",extent=[8.2,12.4,4.2,0])
ax5.set_xlim(0,12.6); ax5.set_ylim(7.4,-0.4)
ax5.set_xticks([]); ax5.set_yticks([]); ax5.axis("off")
ax5.set_title("resampled at any scale",fontsize=8)
fig.text(0.405,0.5,"$*$",fontsize=15,ha="center",va="center")
fig.text(0.755,0.5,r"$\rightarrow$",fontsize=11,ha="center",va="center")
fig.savefig(f"{OUT}/fig_cc_process.pdf",bbox_inches="tight"); plt.close(fig)
print("v4 figs done")
