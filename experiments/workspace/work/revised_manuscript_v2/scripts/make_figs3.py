#!/usr/bin/env python3
"""Additional figures v3: CC four-phase resizing, classical scale-space stack,
zone-plate pyramid; regenerate fig_bicubic with corrected labels."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8.5,"axes.linewidth":0.6})
OUT='paper_output/figs'
def nospines(ax):
    for sd in ["top","right"]: ax.spines[sd].set_visible(False)
def keys(t,a=-0.5):
    at=np.abs(t)
    return np.where(at<=1,(a+2)*at**3-(a+3)*at**2+1,
           np.where(at<2,a*(at**3-5*at**2+8*at-4),0.0))

# ---------- fig_scaling: the four phases of resampling ----------
rng=np.random.default_rng(3)
M=9; I=np.cumsum(rng.standard_normal(M)); I=(I-I.min())/(I.max()-I.min())+0.3
s=1.4; Nout=int(np.floor(M*s))
g=(np.arange(Nout)+0.5)/s-0.5          # centered projected grid
fig,axs=plt.subplots(1,4,figsize=(7.2,1.9),dpi=200)
a=axs[0]
a.stem(np.arange(M),I,basefmt=" ",linefmt="k-",markerfmt="ko")
a.plot(g,-0.12*np.ones_like(g),"x",color="crimson",ms=4,clip_on=False)
a.set_title("1: projected grid",fontsize=8)
n0=4; gn=g[n0]; nb=np.arange(int(np.ceil(gn-2)),int(np.floor(gn+2))+1)
b=axs[1]
b.stem(np.arange(M),I,basefmt=" ",linefmt="0.75",markerfmt=" ")
b.stem(nb,I[nb],basefmt=" ",linefmt="k-",markerfmt="ko")
b.axvspan(gn-2,gn+2,color="crimson",alpha=0.10)
b.plot([gn],[-0.12],"x",color="crimson",ms=5,clip_on=False)
b.set_title("2: neighbors",fontsize=8)
c=axs[2]
xx=np.linspace(gn-2.4,gn+2.4,300)
c.plot(xx,keys(xx-gn),color="crimson",lw=1.0)
c.stem(nb,keys(nb-gn),basefmt=" ",linefmt="k-",markerfmt="ko")
c.axhline(0,color="0.85",lw=.4)
c.set_title(r"3: weights $u(g_n\!-\!m)$",fontsize=8)
d=axs[3]
Iout=np.array([np.sum(I[np.clip(nn,0,M-1)]*keys(nn-gg))
               for gg,nn in [(gg,np.arange(int(np.ceil(gg-2)),int(np.floor(gg+2))+1)) for gg in g]])
d.stem(np.arange(Nout),Iout,basefmt=" ",linefmt="k-",markerfmt="ko")
d.set_title("4: apply and sum",fontsize=8)
for ax in axs:
    ax.set_yticks([]); ax.set_xticks([]); nospines(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scaling.pdf"); plt.close(fig)

# ---------- fig_ssclassic: classical Gaussian scale-space stack ----------
Nx=900
xx=np.linspace(0,12,Nx)
sig=np.zeros(Nx)
sig+= (xx>2)*(xx<3.1)*1.0 + (xx>3.6)*(xx<3.9)*1.3
sig+= np.exp(-0.5*((xx-6)/0.15)**2)*1.1 + np.exp(-0.5*((xx-6.8)/0.35)**2)*0.7
sig+= 0.5*np.sin(2*np.pi*xx*2.2)*(xx>8.5)*(xx<11.5)
w=2*np.pi*np.fft.fftfreq(Nx,xx[1]-xx[0])
fig,ax=plt.subplots(figsize=(3.8,2.6),dpi=200)
for i,t in enumerate([0,0.02,0.12,0.6]):
    sm=np.fft.ifft(np.fft.fft(sig)*np.exp(-t*w**2/2)).real
    ax.plot(xx,sm+1.7*i,color="k",lw=0.9)
    ax.text(12.15,1.7*i+0.1,rf"$t={t}$",fontsize=7.5)
ax.set_yticks([]); ax.set_xticks([]); nospines(ax)
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ssclassic.pdf"); plt.close(fig)

# ---------- fig_pyramid: zone plate, consistent pyramid + stretch defect ----------
N=256
yy,xxg=np.mgrid[0:N,0:N]
r2=((xxg-N/2)**2+(yy-N/2)**2)
zp=np.cos(2*np.pi*r2/(2*N))*np.hanning(N)[None,:]*np.hanning(N)[:,None]
def uhat2d(N,dil):
    w1=2*np.pi*np.fft.fftfreq(N,1.0)
    x1=(np.arange(4096)-2048)*0.05
    u1=keys(x1)
    W1=np.fft.fftshift(2*np.pi*np.fft.fftfreq(4096,0.05))
    U1=np.fft.fftshift(np.real(np.fft.fft(np.fft.ifftshift(u1))*0.05))
    q=np.interp(np.abs(dil*w1),W1,U1)
    return np.outer(q,q)
def filt(im,dil):
    return np.fft.ifft2(np.fft.fft2(im)*uhat2d(im.shape[0],dil)).real
L1=filt(zp,2)[::2,::2]
L2=filt(L1,2)[::2,::2]
stretch=filt(zp,4)[::4,::4]
diff=np.abs(L2-stretch)
fig,axs=plt.subplots(1,4,figsize=(7.2,2.0),dpi=200)
for ax,im,t in zip(axs,[zp,L1,L2,diff*10],
                   [r"$s=1$",r"$s=\frac{1}{2}$",r"$s=\frac{1}{4}$ (two octaves)",
                    r"stretch defect ($\times10$)"]):
    ax.imshow(im,cmap="gray",vmin=-1,vmax=1)
    ax.set_title(t,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pyramid.pdf"); plt.close(fig)

# ---------- regenerate fig_bicubic with corrected labels ----------
Nb=2**15; dxb=1/32.0
xb=(np.arange(Nb)-Nb//2)*dxb
wb=np.fft.fftshift(2*np.pi*np.fft.fftfreq(Nb,dxb))
u_sp=keys(xb)
uh=np.fft.fftshift(np.real(np.fft.fft(np.fft.ifftshift(u_sp))*dxb))
u=lambda q: np.interp(q,wb,uh)
Kinf=np.ones(Nb)
for j in range(19): Kinf*=u(wb/2**j)      # = implied camera at scale 1/2
Ki=lambda q: np.interp(q,wb,Kinf)
Kinf_sp=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Kinf)).real)/dxb
fig,axs=plt.subplots(1,3,figsize=(7.0,2.3),dpi=200)
a=axs[0]; xbm=np.abs(xb)<=4
a.plot(xb[xbm],u_sp[xbm],color="0.45",lw=1.0,ls="--",label=r"software cubic $=K_{1/2}$")
a.plot(xb[xbm],Kinf_sp[xbm],color="k",lw=1.2,label=r"implied camera $K_0$")
a.axhline(0,color="0.85",lw=.4); a.set_yticks([]); a.set_xlabel(r"$x$")
a.legend(frameon=False,fontsize=6.5,loc="upper right"); nospines(a)
b=axs[1]
sv=np.linspace(1.0,4.0,601)
pole=np.abs(Ki(2*np.pi*sv))
b.semilogy(sv,np.maximum(pole,1e-13),color="k",lw=1.0)
for k in [1,2,3,4]: b.axvline(k,color="0.85",lw=.4)
b.set_xlabel(r"downscale factor")
b.set_title(r"implied camera at replica frequencies",fontsize=8)
nospines(b)
c=axs[2]
f=np.linspace(0,1,400)
c.plot(f,u(np.pi*f)*u(np.pi*f/2),color="k",lw=1.2,label="consistent (two octaves)")
c.plot(f,u(np.pi*f),color="crimson",lw=1.2,label="stretched kernel")
c.fill_between(f,u(np.pi*f)*u(np.pi*f/2),u(np.pi*f),color="crimson",alpha=0.15)
c.set_xlabel("fraction of Nyquist"); c.set_ylabel("MTF")
c.legend(frameon=False,fontsize=6.5,loc="lower left"); nospines(c)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bicubic.pdf"); plt.close(fig)
print("v3 figs done")
