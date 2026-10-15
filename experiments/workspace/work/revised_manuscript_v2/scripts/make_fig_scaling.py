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
fig,axs=plt.subplots(1,4,figsize=(3.5,1.06),dpi=220)
fig.subplots_adjust(wspace=0.05,left=0.005,right=0.995,top=0.85,bottom=0.02)
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
a=axs[0]
a.imshow(img,cmap="gray",extent=[-0.5,IN-0.5,IN-0.5,-0.5],alpha=0.55)
px,py=np.meshgrid(np.arange(IN),np.arange(IN))
a.plot(px.ravel(),py.ravel(),".",color="0.25",ms=1.8)
a.plot(GX.ravel(),GY.ravel(),"x",color="crimson",ms=3.6,mew=1.1)
a.set_title("1: project grid",fontsize=6.6)
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
b.set_title("2: neighbors",fontsize=6.6)
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
c.set_title(r"3: weights",fontsize=6.6)
d=axs[3]
W=keys(gn[:,None]-np.arange(IN)[None,:])
outimg=W@img@W.T
d.imshow(outimg,cmap="gray",extent=[-0.5,OUTN-0.5,OUTN-0.5,-0.5])
d.set_title("4: sum",fontsize=6.6)
fig.savefig('paper_output/figs/fig_scaling.pdf',bbox_inches="tight"); fig.savefig('paper_output/figs/fig_scaling.png',dpi=200,bbox_inches="tight"); print("saved")

