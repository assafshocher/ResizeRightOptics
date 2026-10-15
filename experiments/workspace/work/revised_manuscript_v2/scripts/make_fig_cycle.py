"""Two misconceptions, for a camera whose PSF is the interpolation kernel itself:
 (1) the classic anti-aliasing stretch is not the consistent x1/16 downscaling kernel;
 (2) up x16 then down x16 with the same interpolant does not close the cycle; the family does.
One row per kernel: capture, two difference maps, two discrete 2D kernels. Bicubic -> main figure,
bilinear and Lanczos-3 -> appendix figure.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def keys(x, a=-0.5):
    x = np.abs(x); y = np.zeros_like(x); m1 = x < 1; m2 = (x >= 1) & (x < 2)
    y[m1] = (a+2)*x[m1]**3 - (a+3)*x[m1]**2 + 1
    y[m2] = a*x[m2]**3 - 5*a*x[m2]**2 + 8*a*x[m2] - 4*a
    return y
def bilinear(x): return np.clip(1 - np.abs(x), 0, None)
def lanczos3(x):
    x = np.asarray(x, float); y = np.sinc(x)*np.sinc(x/3); y[np.abs(x) >= 3] = 0; return y
def gauss(x): return np.exp(-0.5*np.asarray(x, float)**2)/np.sqrt(2*np.pi)
KERNELS = {"bicubic": (keys, 2.0), "bilinear": (bilinear, 1.0), "Lanczos-3": (lanczos3, 3.0), "Gaussian": (gauss, 4.0)}

# fine grid for continuous spectra (absolute units, scale-1 pixel = 1)
dx = 1/32; L = 128; N = int(2*L/dx)
x = (np.arange(N) - N//2)*dx; w = 2*np.pi*np.fft.fftfreq(N, d=dx)
ws = np.fft.fftshift(w)
def spec(kern):
    F = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(kern(x)))*dx))
    return lambda arg: np.interp(np.abs(arg), ws[ws >= 0], F[ws >= 0])

img = np.asarray(Image.open('outputs/paper_revision_2026-09-17/figures/kodim19.png').convert("L"), dtype=float)/255.0
crop = img[180:436, 200:456]; n_in = 256
kk = np.arange(-n_in//2+1, n_in//2); wk = 2*np.pi*kk/n_in   # symmetric: Nyquist bin dropped
ints = np.arange(n_in).astype(float)
def synth(out_pos, in_pos, symbol):
    Eo = np.exp(1j*np.outer(out_pos, wk)); Ei = np.exp(1j*np.outer(in_pos, wk))
    return np.real(Eo @ np.diag(symbol) @ Ei.conj().T)/n_in
def matrix(out_pos, in_pos, kern, support):
    M = np.zeros((len(out_pos), len(in_pos)))
    for i, o in enumerate(out_pos):
        d = o - in_pos; sel = np.abs(d) <= support; M[i, sel] = kern(d[sel])
        ssum = M[i].sum()
        if ssum != 0: M[i] /= ssum
    return M
def apply2(M, I): return M @ I @ M.T
c = slice(48, 208)
def rmse(A, R): return np.sqrt(np.mean((A[c, c]-R[c, c])**2))

from matplotlib.colors import SymLogNorm
def zc(v): return int((np.diff(np.sign(v[np.abs(v) > 1e-9])) != 0).sum())
def experiment(name, s):
    kern, sup = KERNELS[name]; uh = spec(kern)
    p_up = np.arange(n_in*s)/s - (s-1)/(2*s)
    cap = apply2(synth(ints, ints, uh(wk)), crop)                       # capture of the camera with PSF = kern
    Mup_c = matrix(p_up, ints, kern, sup); Mdn_c = matrix(ints, p_up, kern, sup)   # interpolate up, stretch down
    Mup_f = synth(p_up, ints, uh(wk/s)/uh(wk)); Mdn_f = synth(ints, p_up, uh(wk)/uh(wk/s))/s
    rt_c = Mdn_c @ (Mup_c @ cap @ Mup_c.T) @ Mdn_c.T
    rt_f = Mdn_f @ (Mup_f @ cap @ Mup_f.T) @ Mdn_f.T
    e_c, e_f = rmse(rt_c, cap), rmse(rt_f, cap)
    # continuous kernels around a source pixel (periodic, band <= pi; even factors vanish at the band edge)
    W = 4*s; xu = np.arange(-W, W+0.001, 0.05)
    up_c = kern(xu); up_f = synth(128 + xu, np.array([128.0]), uh(wk/s)/uh(wk))[:, 0]
    xdn = xu
    dn_c = kern(xdn/s)/s; dn_f = synth(128 + xdn, np.array([128.0]), uh(s*wk)/uh(wk))[:, 0]
    psf = kern(xu)
    def norm2(v): M = np.outer(v, v); return M/np.abs(M).max()
    print("%-9s x%d cycle RMSE classic %.1e ours %.1e | down: max|ours-classic| %.3f of peak, zero crossings %d vs %d | up: min %.2f vs %.2f, zero crossings %d vs %d"
          % (name, s, e_c, e_f, np.abs(dn_f-dn_c).max()/dn_c.max(), zc(dn_c), zc(dn_f), up_c.min()/up_c.max(), up_f.min()/up_f.max(), zc(up_c), zc(up_f)))
    return dict(cap=cap, dc=rt_c-cap, df=rt_f-cap, e_c=e_c, e_f=e_f, psf=norm2(psf), upc=norm2(up_c), upf=norm2(up_f),
                dnc=norm2(dn_c), dnf=norm2(dn_f), dc1=dn_c/dn_c.max(), df1=dn_f/dn_c.max(), xdn=xdn, s=s)

NORM = SymLogNorm(linthresh=0.001, vmin=-1, vmax=1, base=10)
def psnr(e): return 20*np.log10(1/e)
def draw_grid(axs, name, s, r):
    vm = np.abs(r["dc"][c, c]).max()
    axs[0, 0].imshow(r["cap"][c, c], cmap="gray", vmin=0, vmax=1); axs[0, 0].set_title("input image")
    axs[0, 1].imshow(r["dc"][c, c], cmap="RdBu_r", vmin=-vm, vmax=vm); axs[0, 1].set_title("classical\nRMSE %.1e" % r["e_c"])
    axs[0, 2].imshow(r["df"][c, c], cmap="RdBu_r", vmin=-vm, vmax=vm); axs[0, 2].set_title("$K_{%d}$, $K_{1/%d}$\nRMSE %.0e" % (s, s, r["e_f"]))
    axs[1, 0].imshow(r["psf"], cmap="RdBu_r", norm=NORM); axs[1, 0].set_title("PSF $K_0$")
    axs[1, 1].imshow(r["upc"], cmap="RdBu_r", norm=NORM); axs[1, 1].set_title("up: classical (= PSF)")
    axs[1, 2].imshow(r["upf"], cmap="RdBu_r", norm=NORM); axs[1, 2].set_title("up: $K_{%d}$" % s)
    axs[2, 1].imshow(r["dnc"], cmap="RdBu_r", norm=NORM); axs[2, 1].set_title("down: classical + AA")
    axs[2, 2].imshow(r["dnf"], cmap="RdBu_r", norm=NORM); axs[2, 2].set_title("down: $K_{1/%d}$" % s)
    ax = axs[2, 0]; ax.axhline(0, color="k", lw=0.3)
    ax.plot(r["xdn"], r["df1"], color="#1f4e79", lw=1.3, label="$K_{1/%d}$" % s, zorder=2)
    ax.plot(r["xdn"], r["dc1"], color="#c0392b", lw=0.9, ls="--", label="classical + AA", zorder=3)
    ax.set_yscale("symlog", linthresh=0.001); ax.set_ylim(-0.3, 1.3); ax.set_title("profiles: $K_{1/%d}$ blue,\nclassical red" % s, fontsize=6.6)
    for ax in axs.ravel():
        if ax is not axs[2, 0]: ax.set_xticks([]); ax.set_yticks([])
    axs[2, 0].set_xticks([-4*s, 0, 4*s]); axs[2, 0].set_yticks([-0.1, 0, 0.1, 1])
    axs[2, 0].tick_params(labelsize=5.2, length=2, pad=1)

plt.rcParams.update({"font.size": 7.2, "axes.titlesize": 7.2})
for f in (4, 8, 16): experiment("bicubic", f)   # higher factors: the down kernels only get closer
res = experiment("bicubic", 2)
fig, axs = plt.subplots(3, 3, figsize=(3.4, 3.75)); fig.subplots_adjust(wspace=0.08, hspace=0.36, left=0.08, right=0.99, top=0.93, bottom=0.04)
draw_grid(axs, "bicubic", 2, res)
fig.savefig('paper_output/figs/fig_cycle.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_cycle.png', dpi=170, bbox_inches="tight")

for name in ["bilinear", "Lanczos-3", "Gaussian"]:
    fig, axs = plt.subplots(3, 3, figsize=(3.4, 3.75)); fig.subplots_adjust(wspace=0.08, hspace=0.36, left=0.08, right=0.99, top=0.93, bottom=0.04)
    draw_grid(axs, name, 2, experiment(name, 2))
    tag = name.lower().replace("-", "")
    fig.savefig('paper_output/figs/fig_cycle_%s.pdf' % tag, bbox_inches="tight"); fig.savefig('paper_output/figs/fig_cycle_%s.png' % tag, dpi=170, bbox_inches="tight")
print("saved")
