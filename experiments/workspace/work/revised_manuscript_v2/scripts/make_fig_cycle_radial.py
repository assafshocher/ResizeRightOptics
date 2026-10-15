"""Cycle experiment for radial cameras (Airy, disk). The family side is non-separable and is applied
exactly on the periodic 2D grid; the classical side is what practice does regardless of camera:
separable bicubic interpolation up, stretched bicubic down. Same 3x3 layout as make_fig_cycle.py.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.special import j1
from PIL import Image

def keys(x, a=-0.5):
    x = np.abs(x); y = np.zeros_like(x); m1 = x < 1; m2 = (x >= 1) & (x < 2)
    y[m1] = (a+2)*x[m1]**3 - (a+3)*x[m1]**2 + 1
    y[m2] = a*x[m2]**3 - 5*a*x[m2]**2 + 8*a*x[m2] - 4*a
    return y
def airy_otf(rho, rc=0.9*np.pi):
    x = np.clip(rho/rc, 0, 1)
    return np.where(rho < rc, (2/np.pi)*(np.arccos(x) - x*np.sqrt(1-x**2)), 0.0)
def sinc(x): return np.sinc(x/np.pi)
FLOOR = 0.03   # the camera's MTF is floored at 3%: zero below, so every kept frequency is safely invertible
def floor_(H): return H*(H > FLOOR)
def disk_otf(rho, R=1.6):
    z = R*rho; out = np.ones_like(z); nz = z > 1e-9; out[nz] = 2*j1(z[nz])/z[nz]; return out
CAMERAS = {"Airy": lambda X, Y: floor_(airy_otf(np.hypot(X, Y))),
           "disk": lambda X, Y: floor_(disk_otf(np.hypot(X, Y)))}

n = 256; s = 2
img = np.asarray(Image.open('outputs/paper_revision_2026-09-17/figures/kodim19.png').convert("L"), dtype=float)/255.0
crop = img[180:436, 200:456]
def freqs(N): return 2*np.pi*np.fft.fftfreq(N)*N/n           # absolute frequencies for an N-grid over the same 256-unit period
w1 = freqs(n); WX, WY = np.meshgrid(w1, w1, indexing="ij"); RHO = np.hypot(WX, WY)
band = (np.abs(WX) < np.pi) & (np.abs(WY) < np.pi)          # Nyquist row/column dropped
def pinv(a): return np.where(np.abs(a) > 1e-12, 1/np.where(np.abs(a) > 1e-12, a, 1), 0.0)

def fam_up(cap, H):
    """K_{1->2} applied exactly: fine grid positions p = m/2 - 1/4 on a 512-grid."""
    Fh = np.fft.fft2(cap)*band*H(WX/s, WY/s)*pinv(H(WX, WY))
    Fh = np.fft.fftshift(Fh)*np.exp(-1j*(np.fft.fftshift(WX)+np.fft.fftshift(WY))/4)
    pad = np.zeros((s*n, s*n), complex); o = (s*n-n)//2; pad[o:o+n, o:o+n] = Fh
    return np.real(np.fft.ifft2(np.fft.ifftshift(pad)))*s*s
def fam_dn(F, H):
    """K_{2->1} applied exactly from the fine samples: train spectrum, restrict to the band, multiply, read at integers."""
    G = np.fft.fftshift(np.fft.fft2(F))
    WXf, WYf = np.meshgrid(np.fft.fftshift(freqs(s*n)), np.fft.fftshift(freqs(s*n)), indexing="ij")
    G = G*np.exp(1j*(WXf+WYf)/4)                                # train spectrum at the fine-grid frequencies
    o = (s*n-n)//2; Gb = G[o:o+n, o:o+n]                        # |w| <= pi block (centered)
    Gb = np.fft.ifftshift(Gb)*band*H(WX, WY)*pinv(H(WX/s, WY/s))
    return np.real(np.fft.ifft2(Gb))/(s*s)

ints = np.arange(n).astype(float); p_up = np.arange(n*s)/s - (s-1)/(2*s)
def synth2(out_pos, in_pos, symbol, I):
    """apply a band-limited symbol on the periodic grid: inputs at in_pos (per axis), outputs at out_pos."""
    Eo = np.exp(1j*np.outer(out_pos, w1)); Ei = np.exp(1j*np.outer(in_pos, w1))
    T = Ei.conj().T @ I @ Ei.conj()                     # train spectrum on the 256-grid frequencies
    return np.real(Eo @ (symbol*band*T) @ Eo.T)/(n*n)
def matrix(out_pos, in_pos, kern, support):
    M = np.zeros((len(out_pos), len(in_pos)))
    for i, o in enumerate(out_pos):
        d = o - in_pos; sel = np.abs(d) <= support; M[i, sel] = kern(d[sel]); ssum = M[i].sum()
        if ssum != 0: M[i] /= ssum
    return M
Mup_c = matrix(p_up, ints, keys, 2.0); Mdn_c = matrix(ints, p_up, keys, 2.0)
c = slice(48, 208)
def rmse(A, B): return np.sqrt(np.mean((A[c, c]-B[c, c])**2))

def dense2d(symbol, W=8, step=0.25):
    """kernel with the given band-limited symbol on a dense grid, window +-W, via zero-padded ifft2."""
    N = int(n/step); pad = np.zeros((N, N), complex); o = (N-n)//2
    pad[o:o+n, o:o+n] = np.fft.fftshift(symbol*band)
    k = np.real(np.fft.ifft2(np.fft.ifftshift(pad)))*(N*N)/(n*n)
    k = np.fft.fftshift(k); cidx = N//2; half = int(W/step)
    return k[cidx-half:cidx+half+1, cidx-half:cidx+half+1]
def sep2d(f1d): return np.outer(f1d, f1d)
NORM = SymLogNorm(linthresh=0.001, vmin=-1, vmax=1, base=10)
def psnr(e): return 20*np.log10(1/e)
def nz(M): return M/np.abs(M).max()

def experiment(name):
    H = CAMERAS[name]
    cap = np.real(np.fft.ifft2(np.fft.fft2(crop)*band*H(WX, WY)))
    up_c_img = synth2(p_up, ints, H(WX, WY), cap)                        # PSF as the interpolant
    rt_c = synth2(ints, p_up, H(s*WX, s*WY), up_c_img)/(s*s)              # PSF stretched to the output scale
    rt_f = fam_dn(fam_up(cap, H), H)
    e_c, e_f = rmse(rt_c, cap), rmse(rt_f, cap)
    W = 8; xd = np.arange(-W, W+0.001, 0.25)
    psf = dense2d(H(WX, WY), W); upf = dense2d(H(WX/s, WY/s)*pinv(H(WX, WY)), W); dnf = dense2d(H(WX, WY)*pinv(H(WX/s, WY/s)), W)
    upc = psf; dnc = dense2d(H(s*WX, s*WY), W)
    mid = len(xd)//2
    prof_c = dnc[mid]/dnc[mid].max(); prof_f = dnf[mid]/dnf[mid].max()
    if name == "disk":                      # draw the disk camera's own kernels as the disk itself, on a fine grid
        xf_ = np.arange(-W, W+0.001, 0.02); XX, YY = np.meshgrid(xf_, xf_, indexing="ij"); RR = np.hypot(XX, YY)
        psf = (RR <= 1.6).astype(float); upc = psf; dnc = (RR <= 3.2).astype(float); prof_c = (np.abs(xd) <= 3.2).astype(float)
    print("%-8s x2 cycle RMSE classic %.1e family %.1e | DC gain %.4f | K21 profile zero crossings %d (classic %d)"
          % (name, e_c, e_f, rt_f.mean()/cap.mean(), (np.diff(np.sign(prof_f)) != 0).sum(), (np.diff(np.sign(prof_c)) != 0).sum()))
    return dict(cap=cap, dc=rt_c-cap, df=rt_f-cap, e_c=e_c, e_f=e_f, psf=nz(psf), upc=nz(upc), upf=nz(upf), dnc=nz(dnc), dnf=nz(dnf),
                xd=xd, pc=prof_c, pf=prof_f)

def draw_grid(axs, name, r):
    vm = np.abs(r["dc"][c, c]).max()
    axs[0, 0].imshow(r["cap"][c, c], cmap="gray", vmin=0, vmax=1); axs[0, 0].set_title("input image")
    axs[0, 1].imshow(r["dc"][c, c], cmap="RdBu_r", vmin=-vm, vmax=vm); axs[0, 1].set_title("classical (PSF)\nRMSE %.1e" % r["e_c"])
    axs[0, 2].imshow(r["df"][c, c], cmap="RdBu_r", vmin=-vm, vmax=vm); axs[0, 2].set_title("$K_{2}$, $K_{1/2}$\nRMSE %.0e" % r["e_f"])
    axs[1, 0].imshow(r["psf"], cmap="RdBu_r", norm=NORM); axs[1, 0].set_title("PSF $K_0$")
    axs[1, 1].imshow(r["upc"], cmap="RdBu_r", norm=NORM); axs[1, 1].set_title("up: classical (= PSF)")
    axs[1, 2].imshow(r["upf"], cmap="RdBu_r", norm=NORM); axs[1, 2].set_title("up: $K_{2}$")
    axs[2, 1].imshow(r["dnc"], cmap="RdBu_r", norm=NORM); axs[2, 1].set_title("down: PSF + AA")
    axs[2, 2].imshow(r["dnf"], cmap="RdBu_r", norm=NORM); axs[2, 2].set_title("down: $K_{1/2}$")
    ax = axs[2, 0]; ax.axhline(0, color="k", lw=0.3)
    ax.plot(r["xd"], r["pf"], color="#1f4e79", lw=1.3, label="$K_{1/2}$", zorder=2)
    ax.plot(r["xd"], r["pc"], color="#c0392b", lw=0.9, ls="--", label="PSF + AA", zorder=3)
    ax.set_yscale("symlog", linthresh=0.001); ax.set_ylim(-0.3, 1.3); ax.set_title("profiles: $K_{1/2}$ blue,\nclassical red", fontsize=6.6)
    for ax in axs.ravel():
        if ax is not axs[2, 0]: ax.set_xticks([]); ax.set_yticks([])
    axs[2, 0].set_xticks([-8, 0, 8]); axs[2, 0].set_yticks([-0.1, 0, 0.1, 1]); axs[2, 0].tick_params(labelsize=5.2, length=2, pad=1)

plt.rcParams.update({"font.size": 7.2, "axes.titlesize": 7.2})
for name in CAMERAS:
    fig, axs = plt.subplots(3, 3, figsize=(3.4, 3.75)); fig.subplots_adjust(wspace=0.08, hspace=0.36, left=0.08, right=0.99, top=0.93, bottom=0.04)
    draw_grid(axs, name, experiment(name))
    tag = name.lower().replace(" + ", "_"); fig.savefig('paper_output/figs/fig_cycle_%s.pdf' % tag, bbox_inches="tight")
    fig.savefig('paper_output/figs/fig_cycle_%s.png' % tag, dpi=200, bbox_inches="tight")
print("saved")
