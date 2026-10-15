#!/usr/bin/env python3
"""Three minimal figures (PDF) for the derivation grand-section.
Fineness gauge: s<1 coarser, s>1 finer, PSF = K_{inf->1}."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":9,"axes.linewidth":0.6,
                     "xtick.direction":"out","ytick.direction":"out"})
import os
os.makedirs('paper_output/figs',exist_ok=True)
N=16384; dx=0.25
x=(np.arange(N)-N//2)*dx
w=2*np.pi*np.fft.fftfreq(N,dx); ws=np.fft.fftshift(w)
W=np.pi/3; ETA=0.05
H=lambda o: np.clip(1-np.abs(o)/W,0,None)
pv=lambda z,e=1e-12: np.where(np.abs(z)>e,1/np.where(np.abs(z)>e,z,1),0)
sp=lambda F: np.fft.fftshift(np.fft.ifft(F).real)/dx
def sinc(u): return np.where(np.abs(u)<1e-12,1.0,np.sin(u)/np.where(np.abs(u)<1e-12,1,u))
h=(W/(2*np.pi))*sinc(W*x/2)**2

# ---- Fig 1: endpoints (two panels) ----
fig,axs=plt.subplots(1,2,figsize=(6.6,2.5),dpi=200)
cm=plt.cm.viridis
a=axs[0]; xm=np.abs(x)<=11
for i,s in enumerate([1/2,1/4,1/8,1/16]):
    Ko=sp(H(w)*pv(H(w/(1/s))))          # coarsening member, output units
    a.plot(x[xm],Ko[xm],color=cm(i/4),lw=1.0)
a.plot(x[xm],h[xm],"k--",lw=1.3)
a.set_xlabel(r"$x$"); a.set_yticks([])
a.text(0.03,0.88,r"$s\to 0$: $K_s^{\mathrm{out}}\to K_0$",transform=a.transAxes)
b=axs[1]
Dec=sp(H(w)/(H(w)**2+ETA**2))
for i,s in enumerate([2,4,8,16]):
    Ku=sp(H(w/s)*H(w)/(H(w)**2+ETA**2))
    b.plot(x[xm],Ku[xm],color=cm(i/4),lw=1.0)
b.plot(x[xm],Dec[xm],"k--",lw=1.3)
b.set_xlabel(r"$x$"); b.set_yticks([])
b.text(0.03,0.88,r"$s\to\infty$: $K_s\to K_\infty$",transform=b.transAxes)
for ax in axs:
    for sd in ["top","right"]: ax.spines[sd].set_visible(False)
fig.tight_layout()
fig.savefig('paper_output/figs/fig_endpoints.pdf')
plt.close(fig)

# ---- Fig 2: the upscaling set (one panel) ----
fig,ax=plt.subplots(figsize=(3.6,2.5),dpi=200)
s=2.0                                    # upscale x2 (fineness gauge)
forced=H(w/s)*H(w)/(H(w)**2+0.03**2)     # hat K_{1->2} Wiener forced part
wm=(ws>=0)&(ws<=2.6*W)
F=np.fft.fftshift(forced)
ax.plot(ws[wm]/W,np.abs(F)[wm],color="k",lw=1.4)
ax.axvspan(1,2,color="crimson",alpha=0.12)
rng=np.random.default_rng(0)
for c,tau,amp in [("#d95f02",0.35,0.5),("#7570b3",0.7,-0.45)]:
    g=amp*np.exp(-0.5*((np.abs(w)-(W+tau*W))/(0.12*W))**2)*(np.abs(w)>W)*(np.abs(w)<2*W)
    ax.plot(ws[wm]/W,np.abs(F+np.fft.fftshift(g))[wm],color=c,lw=0.9)
ax.set_xticks([0,1,2]); ax.set_xticklabels([r"$0$",r"$B$",r"$sB$"])
ax.set_yticks([])
ax.set_xlabel(r"$\omega$")
ax.text(1.5,ax.get_ylim()[1]*0.86,"free",color="crimson",ha="center")
for sd in ["top","right"]: ax.spines[sd].set_visible(False)
fig.tight_layout()
fig.savefig('paper_output/figs/fig_set.pdf')
plt.close(fig)

# ---- Fig 3: scale-space / composition defect (bicubic) ----
Nb=2**15; dxb=1/32.0
xb=(np.arange(Nb)-Nb//2)*dxb
wb=np.fft.fftshift(2*np.pi*np.fft.fftfreq(Nb,dxb))
def keys(x,a=-0.5):
    ax_=np.abs(x)
    return np.where(ax_<=1,(a+2)*ax_**3-(a+3)*ax_**2+1,
           np.where(ax_<2,a*(ax_**3-5*ax_**2+8*ax_-4),0.0))
uh=np.fft.fftshift(np.real(np.fft.fft(np.fft.ifftshift(keys(xb)))*dxb))
u=lambda q: np.interp(q,wb,uh)
f=np.linspace(0,1,400)
one=u(np.pi*f); two=u(np.pi*f)*u(np.pi*f/2)
fig,ax=plt.subplots(figsize=(3.6,2.5),dpi=200)
ax.plot(f,two,color="k",lw=1.4,label=r"consistent family ($\times\frac{1}{2}$ twice)")
ax.plot(f,one,color="crimson",lw=1.4,label=r"stretched kernel ($\times\frac{1}{4}$)")
ax.fill_between(f,two,one,color="crimson",alpha=0.15)
ax.set_xlabel("fraction of Nyquist"); ax.set_ylabel("MTF")
ax.legend(frameon=False,fontsize=8,loc="lower left")
for sd in ["top","right"]: ax.spines[sd].set_visible(False)
fig.tight_layout()
fig.savefig('paper_output/figs/fig_scalespace.pdf')
plt.close(fig)
print("figs done")
