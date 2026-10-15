#!/usr/bin/env python3
"""Figure 1: one image, one question (half the scale), four answers.
Camera: radial band-limited (diffraction-type) OTF. Ground truth = the
camera's own half-zoom capture. Compared: family member K_{1->1/2},
best-fit Gaussian, software bicubic (stretch convention)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.size":8.5})

# ---------- scene ----------
im=Image.open('outputs/paper_revision_2026-09-17/figures/kodim19.png').convert("L")
S=np.asarray(im,float)/255.0
S=S[128:640,:512]                       # 512x512 scene proxy (lighthouse)
NS=512; N1=256; N2=128

def radial_w(N):
    w=2*np.pi*np.fft.fftfreq(N,1.0)
    WX,WY=np.meshgrid(w,w,indexing="ij")
    return np.hypot(WX,WY)

# ---------- camera: cone OTF, cutoff 0.9*pi on the scale-1 grid ----------
Wc=0.9*np.pi                            # in scale-1 pixel units
RS=radial_w(NS)                         # scene grid: scale-1 unit = 2 scene px
H1_scene=np.clip(1-(2*RS)/Wc,0,None)         # camera at scale 1, scene units
Hh_scene=np.clip(1-(4*RS)/Wc,0,None)         # camera at scale 1/2 (2x wider PSF)

F_S=np.fft.fft2(S)
I1=np.real(np.fft.ifft2(F_S*H1_scene))[::2,::2]        # capture at scale 1 (256)
GT=np.real(np.fft.ifft2(F_S*Hh_scene))[::4,::4]        # true half-zoom capture (128)

# ---------- three downscales of I1 to 128 ----------
R1=radial_w(N1)                          # scale-1 grid units
H1=np.clip(1-R1/Wc,0,None)               # camera symbol on its own grid
Hh=np.clip(1-2*R1/Wc,0,None)             # half-zoom camera on scale-1 grid
ETA=1e-3
Kfam=Hh*H1/(H1**2+ETA**2)                # family member K_{1->1/2} (Wiener)

def keys1(t,a=-0.5):
    at=np.abs(t)
    return np.where(at<=1,(a+2)*at**3-(a+3)*at**2+1,
           np.where(at<2,a*(at**3-5*at**2+8*at-4),0.0))
x1=(np.arange(8192)-4096)*0.02
uh=np.fft.fftshift(np.real(np.fft.fft(np.fft.ifftshift(keys1(x1)))*0.02))
wax=np.fft.fftshift(2*np.pi*np.fft.fftfreq(8192,0.02))
w1=2*np.pi*np.fft.fftfreq(N1,1.0)
ub=np.interp(np.abs(2*w1),wax,uh)        # stretched bicubic, separable
Kbic=np.outer(ub,ub)

band=Hh>1e-6                             # fit region: target band
sig=np.linspace(0.3,3.0,400)
errs=[np.sum((np.exp(-s*s*R1[band]**2/2)-Kfam[band])**2) for s in sig]
sg=sig[int(np.argmin(errs))]
Kgau=np.exp(-sg*sg*R1**2/2)

F1=np.fft.fft2(I1)
outs={}
for name,Ksym in [("family",Kfam),("gaussian",Kgau),("bicubic",Kbic)]:
    outs[name]=np.real(np.fft.ifft2(F1*Ksym))[::2,::2]

# ---------- assemble ----------
r0,c0,cw=28,40,72                        # crop window on the 128 grid
def crop(A): return A[r0:r0+cw,c0:c0+cw]
cols=[("camera at half zoom\n(ground truth)",GT),
      (r"family member $K_{1/2}$"+"\n(the derived kernel)",outs["family"]),
      ("best-fit Gaussian\n(the scale-space step)",outs["gaussian"])]
fig,axs=plt.subplots(2,3,figsize=(5.6,3.9),dpi=220)
for j,(t,A) in enumerate(cols):
    axs[0][j].imshow(crop(A),cmap="gray",vmin=0,vmax=1)
    axs[0][j].set_title(t,fontsize=7.6)
    d=np.abs(crop(A)-crop(GT))
    axs[1][j].imshow(d*5,cmap="gray",vmin=0,vmax=1)
    if j==0:
        axs[1][j].text(0.5,0.5,"reference",transform=axs[1][j].transAxes,
                       color="0.6",fontsize=8,ha="center")
    else:
        rmse=np.sqrt(np.mean((A-GT)**2))
        lab=f"RMSE {rmse:.1e}" if rmse<1e-3 else f"RMSE {rmse:.4f}"
        axs[1][j].text(0.04,0.06,lab,transform=axs[1][j].transAxes,
                       color="w",fontsize=7)
    for ax in (axs[0][j],axs[1][j]):
        ax.set_xticks([]); ax.set_yticks([])
axs[1][0].set_ylabel(r"$|\,\cdot-\mathrm{GT}\,|\times5$",fontsize=8)
fig.tight_layout(h_pad=0.5)
fig.savefig('paper_output/figs/fig_four.pdf',bbox_inches="tight")
print("sigma*=%.3f"%sg,
      {k:float(np.sqrt(np.mean((v-GT)**2))) for k,v in outs.items()})
