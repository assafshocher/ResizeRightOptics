"""Long chains of the stretch convention versus the family (bicubic camera, band-limited at 0.99 pi).
(a) composite transfer functions on the output band for x1/8: one-step stretch, a 9-step stretch chain, the family.
(b) MTF at the output Nyquist against the number of steps, totals x1/8 and x1/32: the stretch chain loses more
    the finer the steps; the family is exact at every count.
(c) RMSE of chain against one step, natural image and white-noise texture, against the number of steps."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def keys(x, a=-0.5):
    x = np.abs(x); y = np.zeros_like(x); m1 = x < 1; m2 = (x >= 1) & (x < 2)
    y[m1] = (a+2)*x[m1]**3 - (a+3)*x[m1]**2 + 1
    y[m2] = a*x[m2]**3 - 5*a*x[m2]**2 + 8*a*x[m2] - 4*a
    return y
dx = 1/32; L = 128; N = int(2*L/dx); xf = (np.arange(N)-N//2)*dx; wf = 2*np.pi*np.fft.fftfreq(N, d=dx)
U = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(keys(xf)))*dx)); wsf = np.fft.fftshift(wf)
def uh(arg): return np.interp(np.abs(arg), wsf[wsf >= 0], U[wsf >= 0])
BL = 0.99*np.pi
def h(w): return uh(w)*(np.abs(w) < BL)
def pinv(v): return np.where(np.abs(v) > 1e-12, 1/np.where(np.abs(v) > 1e-12, v, 1), 0.0)

# (a) transfer functions on the output band, x1/8, steps of equal ratio
F = 8; th = np.linspace(0, np.pi, 300); om = th/F
def chain_mtf(F, m, om):
    D = F**(np.arange(1, m+1)/m); out = np.ones_like(om)
    for Di in D: out = out*uh(Di*om)
    return out
mtf_one, mtf_chain, mtf_fam = uh(F*om), chain_mtf(F, 9, om), uh(F*om)/uh(om)
# (b) MTF at the output Nyquist against the number of steps
ms = np.arange(1, 41)
nyq = {Ft: np.array([chain_mtf(Ft, m, np.array([np.pi/Ft]))[0] for m in ms]) for Ft in (8, 32)}
print("(a) at the output Nyquist, x1/8: one step %.3f, 9-step chain %.3f, family %.3f" % (mtf_one[-1], mtf_chain[-1], mtf_fam[-1]))
print("(b) chain MTF at Nyquist, x1/8: steps 1,2,3,9,20,40 -> %s" % np.array2string(nyq[8][[0, 1, 2, 8, 19, 39]], precision=3))
print("    x1/32: steps 1,2,3,9,20,40 -> %s" % np.array2string(nyq[32][[0, 1, 2, 8, 19, 39]], precision=3))

# (c) lattice pipelines: chain against one step, natural image and white noise, against the number of steps
n = 256
img = np.asarray(Image.open('outputs/paper_revision_2026-09-17/figures/kodim19.png').convert("L"), dtype=float)/255.0
crop = img[180:436, 200:456]
rng = np.random.default_rng(0); noise = 0.5 + 0.25*rng.standard_normal((n, n))
kk = np.arange(-n//2+1, n//2); wk = 2*np.pi*kk/n
def synth(out_pos, in_pos, symbol, dens=1.0):
    Eo = np.exp(1j*np.outer(out_pos, wk)); Ei = np.exp(1j*np.outer(in_pos, wk))
    return np.real(Eo @ np.diag(symbol) @ Ei.conj().T)/n*dens
def matrix(out_pos, in_pos, kern, support):
    M = np.zeros((len(out_pos), len(in_pos)))
    for i, o in enumerate(out_pos):
        d = o - in_pos; sel = np.abs(d) <= support; M[i, sel] = kern(d[sel]); ssum = M[i].sum()
        if ssum != 0: M[i] /= ssum
    return M
def grid(Nn): D = n/Nn; return D*(np.arange(Nn) - (Nn-1)/2) + 127.5, D
def ap(M, I): return M @ I @ M.T
Mcam = synth(*grid(n)[:1]*2, h(wk))
def run(scene, m, Ftot=8):
    cap = ap(Mcam, scene)
    sizes = [int(round(n*Ftot**(-i/m))) for i in range(m+1)]; sizes[-1] = n//Ftot
    Ic, Io = cap, cap; g_in, D_in = grid(sizes[0])
    for Nout in sizes[1:]:
        g_out, D_out = grid(Nout)
        Ic = ap(matrix(g_out, g_in, lambda d: keys(d/D_out)/D_out, 2*D_out), Ic)
        Io = ap(synth(g_out, g_in, h(wk*D_out)*pinv(h(wk*D_in)), dens=D_in), Io)
        g_in, D_in = g_out, D_out
    g1, D1 = grid(n); gN, DN = grid(sizes[-1])
    Ic1 = ap(matrix(gN, g1, lambda d: keys(d/DN)/DN, 2*DN), cap); Io1 = ap(synth(gN, g1, h(wk*DN)*pinv(h(wk*D1))), cap)
    mm = int(0.2*sizes[-1]); sl = slice(mm, sizes[-1]-mm)
    run.last = (Ic - Ic1, Io - Io1, sl)
    return np.sqrt(np.mean((Ic[sl, sl]-Ic1[sl, sl])**2)), np.sqrt(np.mean((Io[sl, sl]-Io1[sl, sl])**2))
steps_c = [1, 2, 3, 4, 6, 9, 12, 18]
res = {name: np.array([run(sc, m) for m in steps_c]) for name, sc in [("natural image", crop), ("white-noise texture", noise)]}
for name in res: print("(c) %s: classical chain vs one step RMSE by steps %s -> %s | ours max %.0e" % (name, steps_c, np.array2string(res[name][:, 0], precision=3), res[name][:, 1].max()))

# zone plate: rings from flat at the center to the output's finest detail at the edge
yy, xx = np.mgrid[0:n, 0:n]; rr = np.hypot(xx-127.5, yy-127.5); Rz = 127.5
zone = 0.5 + 0.5*np.cos(np.pi/(16*Rz)*rr**2)          # instantaneous frequency reaches pi/8 at r = Rz
def outputs(scene, m, Ftot=8):
    cap = ap(Mcam, scene)
    sizes = [int(round(n*Ftot**(-i/m))) for i in range(m+1)]; sizes[-1] = n//Ftot
    Ic = cap; g_in, D_in = grid(sizes[0])
    for Nout in sizes[1:]:
        g_out, D_out = grid(Nout); Ic = ap(matrix(g_out, g_in, lambda d: keys(d/D_out)/D_out, 2*D_out), Ic); g_in, D_in = g_out, D_out
    g1, D1 = grid(n); gN, DN = grid(sizes[-1])
    Ic1 = ap(matrix(gN, g1, lambda d: keys(d/DN)/DN, 2*DN), cap); Io1 = ap(synth(gN, g1, h(wk*DN)*pinv(h(wk*D1))), cap)
    Io = cap; g_in, D_in = grid(sizes[0])
    for Nout in sizes[1:]:
        g_out, D_out = grid(Nout); Io = ap(synth(g_out, g_in, h(wk*D_out)*pinv(h(wk*D_in)), dens=D_in), Io); g_in, D_in = g_out, D_out
    return cap, Ic1, Ic, Io1, Io
M = 40
capZ, oneZ, chainZ, famZ, famchainZ = outputs(zone, M); capL, oneL, chainL, famL, famchainL = outputs(crop, M)
def rm(A, B): return np.sqrt(np.mean((A-B)**2))
print("zone plate, %d steps to x1/8: stretch chain vs one step RMSE %.3f, vs family %.3f, one step vs family %.3f; family chain vs member %.0e"
      % (M, rm(chainZ, oneZ), rm(chainZ, famZ), rm(oneZ, famZ), rm(famchainZ, famZ)))
print("lighthouse, %d steps: stretch chain vs one step RMSE %.3f; family chain vs member %.0e" % (M, rm(chainL, oneL), rm(famchainL, famL)))

from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 6.2, "axes.titlesize": 6.4})
fig = plt.figure(figsize=(7.3, 2.45))
widths = [1, 0.35, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 1.35]
gs = fig.add_gridspec(2, 11, width_ratios=widths, wspace=0.08, hspace=0.32, left=0.01, right=0.99, top=0.82, bottom=0.05)
def cell(ax, im, title, gray=True, vm=None):
    if gray: ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else: ax.imshow(im, cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest")
    ax.set_title(title, pad=2); ax.set_xticks([]); ax.set_yticks([])
axes = {}
for r, (cap_, one_c, chain_c, one_o, chain_o) in enumerate([(capZ, oneZ, chainZ, famZ, famchainZ), (capL, oneL, chainL, famL, famchainL)]):
    vm = np.abs(chain_c - one_c).max()
    cell(fig.add_subplot(gs[r, 0]), cap_, "input" if r == 0 else "")
    axes[(r, "c0")] = a = fig.add_subplot(gs[r, 2]); cell(a, chain_c, "%d steps" % M)
    cell(fig.add_subplot(gs[r, 3]), one_c, "one step")
    axes[(r, "c2")] = a = fig.add_subplot(gs[r, 4]); cell(a, chain_c - one_c, "difference", gray=False, vm=vm)
    a.text(0.5, 0.03, "RMSE %.2g" % rm(chain_c, one_c), transform=a.transAxes, ha="center", va="bottom", fontsize=5.8, color="0.15", bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    axes[(r, "o0")] = a = fig.add_subplot(gs[r, 6]); cell(a, chain_o, "%d steps" % M)
    cell(fig.add_subplot(gs[r, 7]), one_o, "one step")
    axes[(r, "o2")] = a = fig.add_subplot(gs[r, 8]); cell(a, chain_o - one_o, "difference", gray=False, vm=vm)
    a.text(0.5, 0.03, "RMSE %.0e" % rm(chain_o, one_o), transform=a.transAxes, ha="center", va="bottom", fontsize=5.8, color="0.15", bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
fig.canvas.draw()
def bbox(a, b):
    pa, pb = a.get_position(), b.get_position(); return pa.x0, pb.x1
for key, label in [("c", "classical stretch"), ("o", "consistent")]:
    x0, x1 = bbox(axes[(0, key + "0")], axes[(0, key + "2")])
    y1 = axes[(0, key + "0")].get_position().y1; y0 = axes[(1, key + "0")].get_position().y0
    pad, tpad = 0.008, 0.10
    fig.add_artist(FancyBboxPatch((x0 - pad, y0 - pad), (x1 - x0) + 2*pad, (y1 - y0) + tpad + pad, boxstyle="round,pad=0.004",
                                  transform=fig.transFigure, facecolor="#f4f4f4", edgecolor="#999999", lw=0.6, zorder=0))
    fig.text((x0 + x1)/2, y1 + tpad - 0.008, label, ha="center", va="top", fontsize=6.5)
p0, p1 = fig.add_subplot(gs[0, 10]).get_position(), fig.add_subplot(gs[1, 10]).get_position()
for a in fig.axes[-2:]: a.remove()
h = p0.height; yc = (p0.y1 + p1.y0)/2
ax = fig.add_axes([p0.x0, yc - h/2, p0.width, h])
for Ft, col in [(8, "#c0392b"), (32, "#7b241c")]: ax.plot(ms, nyq[Ft], color=col, lw=1.3, label="stretch, $\\times1/%d$" % Ft)
ax.axhline(mtf_fam[-1], color="#1f4e79", lw=1.3, label="consistent")
ax.set_xlim(1, 40); ax.set_ylim(0, 1.05); ax.set_xlabel("number of steps", labelpad=1); ax.set_title("Response at the output Nyquist", pad=3)
ax.legend(frameon=False, fontsize=5.8, loc="upper right"); ax.tick_params(labelsize=5.8)
fig.savefig('paper_output/figs/fig_chain.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_chain.png', dpi=200, bbox_inches="tight")
print("saved")
