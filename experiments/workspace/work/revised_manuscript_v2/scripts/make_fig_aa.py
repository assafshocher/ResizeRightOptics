"""Composition through unequal, non-integer ratios (x1/1.6 then x1/2.5 = x1/4) for a bicubic camera,
band-limited at 0.99 pi (the grid's Nyquist bin removed, restoring the hypothesis of Proposition opcomp).
Ours: family members from the PSF; classical: the bicubic kernel stretched to each output scale, real taps."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import ConnectionPatch
from PIL import Image

def keys(x, a=-0.5):
    x = np.abs(x); y = np.zeros_like(x); m1 = x < 1; m2 = (x >= 1) & (x < 2)
    y[m1] = (a+2)*x[m1]**3 - (a+3)*x[m1]**2 + 1
    y[m2] = a*x[m2]**3 - 5*a*x[m2]**2 + 8*a*x[m2] - 4*a
    return y
# continuous spectrum of the cubic
dx = 1/32; L = 128; N = int(2*L/dx); xf = (np.arange(N)-N//2)*dx; wf = 2*np.pi*np.fft.fftfreq(N, d=dx)
U = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(keys(xf)))*dx)); wsf = np.fft.fftshift(wf)
def uh(arg): return np.interp(np.abs(arg), wsf[wsf >= 0], U[wsf >= 0])
BL = 0.99*np.pi
def h(w):                                                     # the camera implied by the x1/2 kernel: telescope (27), band-limited at 0.99 pi
    v = np.ones_like(np.asarray(w, float))
    for j in range(12): v = v*uh(w/2**j)
    return v*(np.abs(w) < BL)
def pinv(v): return np.where(np.abs(v) > 1e-12, 1/np.where(np.abs(v) > 1e-12, v, 1), 0.0)

n = 256
img = np.asarray(Image.open('outputs/paper_revision_2026-09-17/figures/kodim19.png').convert("L"), dtype=float)/255.0
crop = img[180:436, 200:456]
kk = np.arange(-n//2+1, n//2); wk = 2*np.pi*kk/n
ints = np.arange(n).astype(float)
def synth(out_pos, in_pos, symbol, dens=1.0):
    Eo = np.exp(1j*np.outer(out_pos, wk)); Ei = np.exp(1j*np.outer(in_pos, wk))
    return np.real(Eo @ np.diag(symbol) @ Ei.conj().T)/n*dens
def matrix(out_pos, in_pos, kern, support):
    M = np.zeros((len(out_pos), len(in_pos)))
    for i, o in enumerate(out_pos):
        d = o - in_pos; sel = np.abs(d) <= support; M[i, sel] = kern(d[sel]); ssum = M[i].sum()
        if ssum != 0: M[i] /= ssum
    return M
def ap(M, I): return M @ I @ M.T

Mcap = synth(ints, ints, h(wk)); cap = ap(Mcap, crop)
a, b = 1.6, 2.5
g1 = a*np.arange(int(n/a)) + (a-1)/2; g2 = 4*np.arange(n//4) + 1.5
# ours: members of the family, exact on the grid (intermediate grid spacing a, density factor a per axis)
S1 = h(a*wk)*pinv(h(wk)); S12 = h(4*wk)*pinv(h(a*wk)); S = h(4*wk)*pinv(h(wk))
M1 = synth(g1, ints, S1); M12 = synth(g2, g1, S12, dens=a); M = synth(g2, ints, S)
o12 = ap(M12, ap(M1, cap)); o_one = ap(M, cap)
# classical: the cubic stretched to each output scale, real taps, rows normalized
C1 = matrix(g1, ints, lambda d: keys(d/a)/a, 2*a); C2 = matrix(g2, g1, lambda d: keys(d/4)/4, 8.0); C = matrix(g2, ints, lambda d: keys(d/4)/4, 8.0)
c12 = ap(C2, ap(C1, cap)); c_one = ap(C, cap)
cc = slice(8, 56)
def rmse(A, B): return np.sqrt(np.mean((A[cc, cc]-B[cc, cc])**2))
e_ours, e_cls = rmse(o12, o_one), rmse(c12, c_one)
print("x1/1.6 then x1/2.5 vs x1/4: ours %.1e  classical %.1e | DC %.4f %.4f" % (e_ours, e_cls, o12.mean()/cap.mean(), c12.mean()/cap.mean()))

def psnr(e): return 20*np.log10(1/e)
print("PSNR: ours %.1f dB, classical %.1f dB" % (psnr(e_ours), psnr(e_cls)))
W = 16; xd = np.arange(-W, W+0.001, 0.25)
def dense(symbol, x): return synth(128 + x, np.array([128.0]), symbol)[:, 0]
def sym2(v): M = np.outer(v, v); return M/np.abs(M).max()
# every kernel drawn in its own input pixels, window +-16: second-stage kernels act on the intermediate grid (spacing a)
K = {"psf": sym2(dense(h(wk), xd)), "S1": sym2(dense(S1, xd)), "S12": sym2(dense(S12, a*xd)), "S": sym2(dense(S, xd)),
     "C1": sym2(keys(xd/a)/a), "C2": sym2(keys(a*xd/4)/4), "C": sym2(keys(xd/4)/4), "half": sym2(keys(xd/2)/2)}
NORM = SymLogNorm(linthresh=0.001, vmin=-1, vmax=1, base=10)
plt.rcParams.update({"font.size": 6.4, "axes.titlesize": 6.4})
FW, FH = 3.5, 3.6
fig = plt.figure(figsize=(FW, FH))
def sq(x, y, w, title, im, cmap="RdBu_r", norm=NORM, vmin=None, vmax=None):
    ax = fig.add_axes([x, y, w, w*FW/FH]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, pad=2)
    ax.imshow(im, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax); return ax
kw = 0.15; dw = 0.2
ax_src = sq(0.02, 0.74, 0.14, "shared $\\times\\frac{1}{2}$ kernel (bicubic)", K["half"])
ax_in = sq(0.02, 0.40, 0.14, "input image", cap[32:224, 32:224], cmap="gray", norm=None, vmin=0, vmax=1)
d = c12 - c_one; vm = np.abs(d[cc, cc]).max(); d2 = o12 - o_one
ax_c1 = sq(0.26, 0.80, kw, "dilated to $\\times\\frac{1}{1.6}$", K["C1"])
ax_c2 = sq(0.46, 0.80, kw, "dilated to $\\times\\frac{1}{2.5}$", K["C2"])
ax_c = sq(0.36, 0.60, kw, "dilated to $\\times\\frac{1}{4}$", K["C"])
ax_cd = sq(0.68, 0.66, dw, "two steps minus one step\nRMSE %.1e" % e_cls, d[cc, cc], norm=None, vmin=-vm, vmax=vm)
ax_o1 = sq(0.26, 0.29, kw, "$K_{1/1.6}$", K["S1"])
ax_o2 = sq(0.46, 0.29, kw, "$K_{1/2.5}$", K["S12"])
ax_o = sq(0.36, 0.09, kw, "$K_{1/4}$", K["S"])
ax_od = sq(0.68, 0.15, dw, "two steps minus one step\nRMSE %.0e" % e_ours, d2[cc, cc], norm=None, vmin=-vm, vmax=vm)
ax_psf = sq(0.86, 0.40, 0.12, "implied PSF", K["psf"])
def arrow(A, B, xa=(1.0, 0.5), xb=(0.0, 0.5), rad=0.0):
    fig.add_artist(ConnectionPatch(xyA=xa, coordsA=A.transAxes, xyB=xb, coordsB=B.transAxes, arrowstyle="-|>", lw=0.6, color="k",
                                   mutation_scale=6, connectionstyle="arc3,rad=%.2f" % rad))
for A1, A2, A, D, ya1, ya2 in [(ax_c1, ax_c2, ax_c, ax_cd, 0.9, 0.62), (ax_o1, ax_o2, ax_o, ax_od, 0.38, 0.1)]:
    arrow(ax_in, A1, xa=(1.0, ya1)); arrow(ax_in, A, xa=(1.0, ya2)); arrow(A1, A2)
    arrow(A2, D, xb=(0.0, 0.85)); arrow(A, D, xb=(0.0, 0.15))
fig.savefig('paper_output/figs/fig_aa.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_aa.png', dpi=200, bbox_inches="tight")
print("saved")
