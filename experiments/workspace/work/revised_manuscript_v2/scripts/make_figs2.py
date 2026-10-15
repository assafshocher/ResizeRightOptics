#!/usr/bin/env python3
"""Figures v2 for the grand-section. Minimal style, PDF output to figs/.
Main: fig_prelim, fig_set, fig_endpoints (3 panels incl. variance arc),
fig_bicubic (3 panels). Appendix: fig_sets2d (strip), fig_gauss, fig_cascade."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
plt.rcParams.update({"font.size":8.5,"axes.linewidth":0.6})
os.makedirs('paper_output/figs',exist_ok=True)
OUT='paper_output/figs'

def nospines(ax):
    for sd in ["top","right"]: ax.spines[sd].set_visible(False)

# ---------- shared machinery ----------
N=16384; dx=0.25
x=(np.arange(N)-N//2)*dx
w=2*np.pi*np.fft.fftfreq(N,dx); ws=np.fft.fftshift(w)
W=np.pi/3; ETA=0.05
H=lambda o: np.clip(1-np.abs(o)/W,0,None)
G=lambda o: np.exp(-np.clip(2.0**2*o**2/2,0,700))
pv=lambda z,e=1e-12: np.where(np.abs(z)>e,1/np.where(np.abs(z)>e,z,1),0)
sp=lambda F: np.fft.fftshift(np.fft.ifft(F).real)/dx
def sinc(u): return np.where(np.abs(u)<1e-12,1.0,np.sin(u)/np.where(np.abs(u)<1e-12,1,u))
h_slit=(W/(2*np.pi))*sinc(W*x/2)**2

# bicubic (Keys) on its own fine grid
Nb=2**15; dxb=1/32.0
xb=(np.arange(Nb)-Nb//2)*dxb
wb=np.fft.fftshift(2*np.pi*np.fft.fftfreq(Nb,dxb))
def keys(t,a=-0.5):
    at=np.abs(t)
    return np.where(at<=1,(a+2)*at**3-(a+3)*at**2+1,
           np.where(at<2,a*(at**3-5*at**2+8*at-4),0.0))
u_sp=keys(xb)
uh=np.fft.fftshift(np.real(np.fft.fft(np.fft.ifftshift(u_sp))*dxb))
u=lambda q: np.interp(q,wb,uh)
Kinf=np.ones(Nb)
for j in range(19): Kinf*=u(wb/2**j)
Ki=lambda q: np.interp(q,wb,Kinf)
Kinf_sp=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Kinf)).real)/dxb

# ---------- fig_prelim: the cast (PSFs and OTFs) ----------
fig,axs=plt.subplots(2,3,figsize=(6.8,3.1),dpi=200)
xm=np.abs(x)<=6; xbm=np.abs(xb)<=6
gsp=np.exp(-x**2/(2*1.2**2))/(1.2*np.sqrt(2*np.pi))
cases=[(r"band-limited",h_slit[xm],x[xm]),(r"Gaussian",gsp[xm],x[xm]),(r"cubic kernel",u_sp[xbm],xb[xbm])]
for ax,(t,f,xx) in zip(axs[0],cases):
    ax.plot(xx,f,color="k",lw=1.1); ax.set_title(t,fontsize=8.5)
    ax.set_yticks([]); ax.axhline(0,color="0.8",lw=.4); nospines(ax)
wm=(ws>=0)&(ws<=1.5*np.pi); wbm=(wb>=0)&(wb<=1.5*np.pi)
specs=[np.fft.fftshift(H(w))[wm],np.exp(-1.2**2*ws**2/2)[wm],np.abs(uh)[wbm]]
xsp=[ws[wm],ws[wm],wb[wbm]]
for ax,S,xx in zip(axs[1],specs,xsp):
    ax.plot(xx/np.pi,S,color="k",lw=1.1); ax.set_yticks([])
    ax.set_xlabel(r"$\omega/\pi$"); ax.axhline(0,color="0.8",lw=.4); nospines(ax)
axs[0][0].set_ylabel("kernel"); axs[1][0].set_ylabel("transform")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_prelim.pdf"); plt.close(fig)

# ---------- fig_set (unchanged design) ----------
fig,ax=plt.subplots(figsize=(3.6,2.5),dpi=200)
s=2.0
forced=H(w/s)*H(w)/(H(w)**2+0.03**2)
wm2=(ws>=0)&(ws<=2.6*W)
F=np.fft.fftshift(forced)
ax.plot(ws[wm2]/W,np.abs(F)[wm2],color="k",lw=1.4)
ax.axvspan(1,2,color="crimson",alpha=0.12)
for c,tau,amp in [("#d95f02",0.35,0.5),("#7570b3",0.7,-0.45)]:
    g=amp*np.exp(-0.5*((np.abs(w)-(W+tau*W))/(0.12*W))**2)*(np.abs(w)>W)*(np.abs(w)<2*W)
    ax.plot(ws[wm2]/W,np.abs(F+np.fft.fftshift(g))[wm2],color=c,lw=0.9)
ax.set_xticks([0,1,2]); ax.set_xticklabels([r"$0$",r"$B$",r"$sB$"]); ax.set_yticks([])
ax.set_xlabel(r"$\omega$"); ax.text(1.5,ax.get_ylim()[1]*0.86,"free",color="crimson",ha="center")
nospines(ax); fig.tight_layout(); fig.savefig(f"{OUT}/fig_set.pdf"); plt.close(fig)

# ---------- fig_endpoints: 3 panels incl. variance arc ----------
fig,axs=plt.subplots(1,3,figsize=(7.0,2.3),dpi=200)
cm=plt.cm.viridis
a=axs[0]; xm=np.abs(x)<=11
for i,ss in enumerate([1/2,1/4,1/8,1/16]):
    a.plot(x[xm],sp(H(w)*pv(H(w/ss)))[xm],color=cm(i/4),lw=0.9)
a.plot(x[xm],h_slit[xm],"k--",lw=1.2)
a.set_yticks([]); a.set_xlabel(r"$x$"); nospines(a)
a.set_title(r"$s\to0$: $\to K_0$",fontsize=8.5)
b=axs[1]
Dec=sp(H(w)/(H(w)**2+ETA**2))
for i,ss in enumerate([2,4,8,16]):
    b.plot(x[xm],sp(H(w/ss)*H(w)/(H(w)**2+ETA**2))[xm],color=cm(i/4),lw=0.9)
b.plot(x[xm],Dec[xm],"k--",lw=1.2)
b.set_yticks([]); b.set_xlabel(r"$x$"); nospines(b)
b.set_title(r"$s\to\infty$: $\to K_\infty$",fontsize=8.5)
c=axs[2]; SG=2.0
ssv=np.exp(np.linspace(np.log(1/16),np.log(16),41)); vv=[]
for ss in ssv:
    Kk=sp(G(w)*pv(G(w/ss))) if ss<=1 else sp(G(w/ss)*G(w)/(G(w)**2+ETA**2))
    m0=np.sum(Kk)*dx; m1=np.sum(x*Kk)*dx/m0
    vv.append(np.sum((x-m1)**2*Kk)*dx/m0)
c.semilogx(ssv,vv,color="k",lw=1.2)
c.axhline(SG**2,color="0.6",ls="--",lw=.7); c.axhline(-SG**2,color="0.6",ls="--",lw=.7)
c.axhline(0,color="0.85",lw=.4); c.axvline(1,color="0.85",lw=.4)
c.set_xlabel(r"$s$"); c.set_title("variance of $K_s$",fontsize=8.5)
c.text(0.07,SG**2*0.7,r"$+\kappa_2$",fontsize=8); c.text(0.07,-SG**2*0.93,r"$-\kappa_2$",fontsize=8)
nospines(c)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_endpoints.pdf"); plt.close(fig)

# ---------- fig_bicubic: 3 panels ----------
fig,axs=plt.subplots(1,3,figsize=(7.0,2.3),dpi=200)
a=axs[0]; xbm=np.abs(xb)<=4
a.plot(xb[xbm],u_sp[xbm],color="0.45",lw=1.0,ls="--",label=r"software cubic $=K_{1/2}$")
a.plot(xb[xbm],Kinf_sp[xbm],color="k",lw=1.2,label=r"implied $K_0$")
a.axhline(0,color="0.85",lw=.4); a.set_yticks([]); a.set_xlabel(r"$x$")
a.legend(frameon=False,fontsize=6.5,loc="upper right"); nospines(a)
b=axs[1]
sv=np.linspace(1.0,4.0,601)
pole=np.abs(Ki(2*np.pi*sv))
b.semilogy(sv,np.maximum(pole,1e-13),color="k",lw=1.0)
for k in [1,2,3,4]: b.axvline(k,color="0.85",lw=.4)
b.set_xlabel(r"$s^{-1}$ (downscale factor)"); b.set_title(r"$|\hat K_0(2\pi s^{-1})|$",fontsize=8.5)
nospines(b)
c=axs[2]
f=np.linspace(0,1,400)
c.plot(f,u(np.pi*f)*u(np.pi*f/2),color="k",lw=1.2,label="consistent (two octaves)")
c.plot(f,u(np.pi*f),color="crimson",lw=1.2,label="stretched kernel")
c.fill_between(f,u(np.pi*f)*u(np.pi*f/2),u(np.pi*f),color="crimson",alpha=0.15)
c.set_xlabel("fraction of Nyquist"); c.set_ylabel("MTF")
c.legend(frameon=False,fontsize=6.5,loc="lower left"); nospines(c)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bicubic.pdf"); plt.close(fig)

# ---------- appendix: fig_sets2d strip ----------
N2=192
w2=2*np.pi*np.fft.fftfreq(N2,1.0)
WX,WY=np.meshgrid(w2,w2,indexing="ij")
RHO=np.hypot(WX,WY); W2=np.pi/3; E2=0.04
H2r=lambda r: np.clip(1-r/W2,0,None)
fig,axs=plt.subplots(1,3,figsize=(6.8,2.3),dpi=200)
for ax,ss in zip(axs,[1.05,0.8,0.45]):
    den=H2r(RHO/ss); F=H2r(RHO)*den/(den**2+E2**2)
    ax.imshow(np.fft.fftshift(np.minimum(np.abs(F),3)),cmap="Greys",vmin=0,vmax=3,
              origin="lower",extent=[w2.min(),-w2.min(),w2.min(),-w2.min()])
    if ss<1:
        ann=np.fft.fftshift(((RHO>ss*W2)&(RHO<W2)).astype(float))
        ax.imshow(np.dstack([np.ones_like(ann),0*ann,0.2*np.ones_like(ann),0.28*ann]),
                  origin="lower",extent=[w2.min(),-w2.min(),w2.min(),-w2.min()])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(rf"$s={ss}$",fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sets2d.pdf"); plt.close(fig)

# ---------- appendix: fig_gauss funnel ----------
fig,ax=plt.subplots(figsize=(3.6,2.4),dpi=200)
sg=0.65; SIG=2.0; ETAg=0.03; ACAP=1.8
Hg=lambda o: np.exp(-np.clip(SIG**2*o**2/2,0,700))
Kpg=Hg(w)*Hg(w/sg)/(Hg(w/sg)**2+ETAg**2)
A=np.minimum(ETAg*np.exp(np.clip(SIG**2*w**2/(2*sg*sg),0,50)),ACAP)
M=((Hg(w)>=ETAg)&(Hg(w/sg)<=ETAg)).astype(float)
K0s=np.fft.fftshift(Kpg); As=np.fft.fftshift(A*M)
wmg=(ws>=0)&(ws<=3.0)
ax.fill_between(ws[wmg],(K0s-As)[wmg],(K0s+As)[wmg],color="crimson",alpha=0.15)
ax.plot(ws[wmg],K0s[wmg],color="k",lw=1.2)
for cc,tau,amp in [("#d95f02",0.8,0.6),("#7570b3",1.0,-0.5)]:
    weff=(sg/SIG)*np.sqrt(2*np.log(1/ETAg))
    g=amp*np.minimum(A,ACAP)*np.exp(-0.5*((np.abs(w)-tau*weff)/(0.14*weff))**2)*M
    ax.plot(ws[wmg],np.fft.fftshift(Kpg+g)[wmg],color=cc,lw=0.8)
ax.set_yticks([]); ax.set_xlabel(r"$\omega$"); nospines(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gauss.pdf"); plt.close(fig)

# ---------- appendix: fig_cascade ----------
rng=np.random.default_rng(5)
hh=lambda o: np.exp(-1.4**2*o**2/2)*sinc(1.0*o)
r=np.fft.ifft(np.fft.fft(rng.standard_normal(N))*np.exp(-w**2/(2*0.5**2))).real
eps=0.03*r/np.max(np.abs(r))*np.clip(np.abs(w)/0.5,0,1)
u_est=hh(w)/np.maximum(hh(0.5*w),1e-9)*(1+eps)
Hc=np.ones(N)
for j in range(9):
    Hc*=np.interp(np.fft.fftshift(w)*0.5**j,ws,np.fft.fftshift(u_est))
Hc=np.fft.ifftshift(Hc)
fig,axs=plt.subplots(1,2,figsize=(6.2,2.3),dpi=200)
a=axs[0]; xm=np.abs(x)<=8
a.plot(x[xm],sp(u_est)[xm],color="crimson",lw=1.1,label=r"estimated $K_{1/2}$")
a.plot(x[xm],sp(hh(w))[xm],"k--",lw=1.1,label=r"true $K_0$")
a.axhline(0,color="0.85",lw=.4); a.set_yticks([]); a.set_xlabel(r"$x$")
a.legend(frameon=False,fontsize=6.5); nospines(a)
b=axs[1]
b.plot(x[xm],sp(Hc)[xm],color="k",lw=1.1,label="cascade estimate")
b.plot(x[xm],sp(hh(w))[xm],"k--",lw=1.1,label=r"true $K_0$")
b.axhline(0,color="0.85",lw=.4); b.set_yticks([]); b.set_xlabel(r"$x$")
b.legend(frameon=False,fontsize=6.5); nospines(b)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_cascade.pdf"); plt.close(fig)
print("all figs done")
