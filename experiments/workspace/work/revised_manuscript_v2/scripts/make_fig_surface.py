"""Families as surfaces: K_{1->s}(x) over position and scale (signed-log height), each member restricted to
the source's band |w| <= pi. Anchors: a kernel assumed as the x1/r slice (camera by the telescope) or a PSF
assumed at infinity. Main figure: bicubic at x1/2. Appendix: kernels at slices, PSFs at infinity."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
def keys(x, a=-0.5):
    x = np.abs(x); y = np.zeros_like(x); m1 = x < 1; m2 = (x >= 1) & (x < 2)
    y[m1] = (a+2)*x[m1]**3 - (a+3)*x[m1]**2 + 1; y[m2] = a*x[m2]**3 - 5*a*x[m2]**2 + 8*a*x[m2] - 4*a; return y
def tri(x): return np.clip(1-np.abs(x), 0, None)
def lanczos3(x):
    x = np.asarray(x, float); y = np.sinc(x)*np.sinc(x/3); y[np.abs(x) >= 3] = 0; return y
def boxk(x): x = np.asarray(x, float); return (np.abs(x) < 0.5).astype(float)      # the pixel box: area averaging when stretched
def gauss(x): return np.exp(-0.5*np.asarray(x, float)**2)/np.sqrt(2*np.pi)
dx = 1/32; L = 128; N = int(2*L/dx); xf = (np.arange(N)-N//2)*dx; wf = 2*np.pi*np.fft.fftfreq(N, d=dx); wsf = np.fft.fftshift(wf)
def spec(kern):
    F = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(kern(xf)))*dx))
    return lambda arg: np.interp(np.abs(arg), wsf[wsf >= 0], F[wsf >= 0])
def anchored(kern, r):                                   # camera implied by assuming the x r slice is the stretched kernel
    uh = spec(kern)
    def h(w):
        v = np.ones_like(np.asarray(w, float))
        for j in range(14): v = v*uh(r**j*w)
        return v
    return h
FLOOR = 0.03
def pinv(v, floor=FLOOR): return np.where(np.abs(v) > floor, 1/np.where(np.abs(v) > floor, v, 1), 0.0)
def member(h, s, target_fixed=False):
    """K_{1->s} (source fixed), or K_{s->1} (target fixed). Coarsening members in full; refining members
    restricted to the source's band, all a sampled source carries."""
    if not target_fixed:      # coarsening: nothing is inverted in effect, a tiny floor keeps shapes sharp
        sym = h(wf/s)*pinv(h(wf), 0.03 if s <= 1 else FLOOR)*((s <= 1) | (np.abs(wf) <= np.pi))
    else:
        sym = h(wf)*pinv(h(wf/s), 0.03 if s >= 1 else FLOOR)*((s >= 1) | (np.abs(wf) <= s*np.pi))
    k = np.fft.fftshift(np.real(np.fft.ifft(sym)))/dx
    return k/np.abs(k).max()
EPS = 0.01
def slog(z): return np.sign(z)*np.log10(1 + np.abs(z)/EPS)/np.log10(1 + 1/EPS)
W = 24; sel = np.abs(xf) <= W
def draw(ax, h, band_log, title, target_fixed=False, exact=None, tfs=7.2, Wp=None):
    W = Wp if Wp else 24; sel = np.abs(xf) <= W
    """target_fixed=True draws K_s in OUTPUT pixels (old target-fixed member at source scale 1/s), ratio axis from 1/32 to 4,
    the PSF at the s->0 end; otherwise K_s in input pixels, ratio axis from 1/8 to 4, the deconvolver at the s->inf end."""
    coarser = (target_fixed == "coarser"); hybrid = (target_fixed == "psf")
    lo, hi = (-3, 2) if not target_fixed else ((-2, 2) if coarser else (-5, 2))
    if hybrid:                               # output-pixel rows on [1/32,4], a gap, then a plateau carrying the PSF at "s=0"
        logs = np.concatenate([np.linspace(-6.35, -6.15, 5), np.linspace(-6.0, -5.15, 6), np.linspace(lo, hi, 76)])
        band = np.zeros(len(logs), bool); band[:5] = True; gap = np.zeros(len(logs), bool); gap[5:11] = True
    elif band_log is None:
        logs = np.linspace(lo, hi, 81); band = np.zeros(81, bool)
    else:                                    # a plateau of the axis at the assumed scale, clamped inside the axis
        d = 0.1; start = min(max(band_log - d, lo), hi - 2*d)
        nl = int(round(76*(start - lo)/max(hi - lo - 2*d, 1e-9)))
        left = np.linspace(lo, start, nl, endpoint=False) if nl > 0 else np.array([])
        plateau = np.linspace(start, start + 2*d, 5)
        right = np.linspace(start + 2*d, hi, 76 - nl + 1)[1:] if 76 - nl > 0 else np.array([])
        logs = np.concatenate([left, plateau, right]); band = np.zeros(len(logs), bool); band[nl:nl+5] = True
    if coarser: Z = np.array([(member(h, 2.0**ls, False) if ls > 0 else member(h, 2.0**(-ls), True))[sel] for ls in logs])
    elif hybrid: Z = np.array([member(h, 2.0**(-ls), True)[sel] for ls in logs])
    else: Z = np.array([member(h, 2.0**(-ls) if target_fixed else 2.0**ls, target_fixed)[sel] for ls in logs])
    if band_log is not None and exact is not None: Z[band] = exact[sel]/np.abs(exact).max()
    if hybrid: Z[gap] = np.nan
    X, S = np.meshgrid(xf[sel], logs); Zs = slog(Z)
    colors = cm.RdBu_r((Zs + 1)/2)
    Zf = Zs.copy()
    if np.isnan(Zf).any():                    # shade on a gap-filled copy so the gap's edge does not darken
        bad = np.isnan(Zf).any(1); good = np.where(~bad)[0]
        for i in np.where(bad)[0]: Zf[i] = Zf[good[np.argmin(np.abs(good - i))]]
    shade = 0.55 + 0.45*LightSource(azdeg=315, altdeg=45).hillshade(Zf, vert_exag=2.0)
    colors[:, :, :3] *= shade[:, :, None]                     # manual shading, so the band below can stay flat
    band_quads = band & np.roll(band, -1); band_quads[-1] = band[-1]      # a quad is red only if both its rows are in the plateau
    colors[band_quads, :, :3] = np.array([0.75, 0.18, 0.17]); colors[band_quads, :, 3] = 1.0
    ax.plot_surface(X, S, Zs, facecolors=colors, rstride=1, cstride=4, linewidth=0, edgecolor="none", antialiased=True, shade=False)
    ax.set_xlabel("$x$ (%s pixels)" % ("coarser image's" if coarser else ("output" if target_fixed else "input")), labelpad=-4, fontsize=7); ax.set_xticks([-W, 0, W])
    if hybrid:
        ax.set_ylabel("$s$ ($K_s$, output pixels)", labelpad=-2, fontsize=7)
        ax.set_yticks([-6.25, -5, -3, -1, 0, 2]); ax.set_yticklabels(["$0$", "$1/32$", "$1/8$", "$1/2$", "$1$", "$4$"]); ax.set_ylim(-6.35, 2)
        ax.text(W*1.30, -5.75, -1.25, "$\\cdots$", fontsize=9, ha="center", va="center")
    elif coarser:
        ax.set_ylabel("$s$ ($K_s$)", labelpad=-2, fontsize=7)
        ax.set_yticks([-2, -1, 0, 1, 2]); ax.set_yticklabels(["$1/4$", "$1/2$", "$1$", "$2$", "$4$"]); ax.set_ylim(-2, 2)
    elif not target_fixed:
        ax.set_ylabel("$s$ ($K_s$, input pixels)", labelpad=-2, fontsize=7)
        ax.set_yticks([-3, -2, -1, 0, 2]); ax.set_yticklabels(["$1/8$", "$1/4$", "$1/2$", "$1$", "$4$"]); ax.set_ylim(-3, 2)
    else:
        ax.set_ylabel("$s$ ($K_s$, output pixels)", labelpad=-2, fontsize=7)
        ax.set_yticks([-5, -3, -1, 0, 2]); ax.set_yticklabels(["$1/32$", "$1/8$", "$1/2$", "$1$", "$4$"]); ax.set_ylim(-5, 2)
    ax.set_zticks([slog(np.array(-1.0)), slog(np.array(-0.1)), 0, slog(np.array(0.1)), 1]); ax.set_zticklabels(["$-1$", "$-0.1$", "$0$", "$0.1$", "$1$"])
    ax.tick_params(labelsize=6, pad=-2); ax.view_init(elev=34, azim=-52); ax.set_box_aspect((1.6, 1.4, 0.6))
    ax.set_xlim(-W, W); ax.set_zlim(-1, 1.2); ax.set_zlabel("")
    if band_log is not None:                     # gentle dotted posts from the stripe's ends to the floor; tick label in red
        zb = float(Zs[band][0, 0]), float(Zs[band][0, -1])
        ax.plot([-W, -W], [band_log, band_log], [-1, zb[0]], color="#c0392b", lw=0.8, ls=":", alpha=0.7, zorder=0)
        ax.plot([W, W], [band_log, band_log], [-1, zb[1]], color="#c0392b", lw=0.8, ls=":", alpha=0.7, zorder=0)
        ticks = list(ax.get_yticks()); labels = ax.yaxis.get_ticklabels()
        for tk, lb in zip(ticks, labels):
            if abs(tk - band_log) < 1e-6 or (hybrid and abs(tk + 6.25) < 1e-6): lb.set_color("#c0392b"); lb.set_fontweight("bold")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): pane.set_alpha(0.0)
    ax.text(0.95*W, hi, 1.35, title, zdir="x", fontsize=tfs, ha="right", va="bottom")

fig = plt.figure(figsize=(3.6, 3.0)); ax = fig.add_subplot(111, projection="3d")
draw(ax, anchored(keys, 0.5), -1.0, "assumed: bicubic at $\\times1/2$", exact=0.5*keys(0.5*xf))
fig.subplots_adjust(left=0, right=1, top=1.02, bottom=0.0)
fig.savefig('paper_output/figs/fig_surface.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_surface.png', dpi=400, bbox_inches="tight")

def airy1d(w, rc=0.9*np.pi): return np.clip(1 - np.abs(w)/rc, 0, None)        # slit aperture: OTF a triangle, PSF sinc^2
def box(w): return (np.abs(w) < np.pi).astype(float)                            # ideal band-limited camera: PSF a sinc
def psf_spatial(hh): k = np.fft.fftshift(np.real(np.fft.ifft(hh(wf))))/dx; return k
gallery = [("kernels assumed at a slice", [(anchored(tri, 0.5), -1.0, "assumed: bilinear at $\\times1/2$", 0.5*tri(0.5*xf)),
                                            (anchored(lanczos3, 0.5), -1.0, "assumed: Lanczos-3 at $\\times1/2$", 0.5*lanczos3(0.5*xf)),
                                            (anchored(boxk, 0.5), -1.0, "assumed: box (area average) at $\\times1/2$", 0.5*boxk(0.5*xf)),
                                            (anchored(keys, 0.25), -2.0, "assumed: bicubic at $\\times1/4$", 0.25*keys(0.25*xf))]),
           ("PSFs assumed, output pixels", [(spec(keys), -5.0, "assumed: PSF = bicubic", psf_spatial(spec(keys))),
                                        (spec(gauss), -5.0, "assumed: PSF = Gaussian, $\\sigma=1$", psf_spatial(spec(gauss))),
                                        (box, -5.0, "assumed: PSF = sinc (ideal band limit)", psf_spatial(box)),
                                        (airy1d, -5.0, "assumed: PSF = sinc$^2$ (1D Airy)", psf_spatial(airy1d))])]
for tag, items in zip(["slices", "psfs"], [g[1] for g in gallery]):
    fig = plt.figure(figsize=(7.2, 6.0))
    for k, (h, bl, title, ex) in enumerate(items):
        ax = fig.add_subplot(2, 2, k+1, projection="3d"); draw(ax, h, bl, title, target_fixed=(tag == "psfs"), exact=ex)
    fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0.0, wspace=0.0, hspace=0.02)
    fig.savefig('paper_output/figs/fig_surfaces_%s.pdf' % tag, bbox_inches="tight"); fig.savefig('paper_output/figs/fig_surfaces_%s.png' % tag, dpi=300, bbox_inches="tight")
print("saved")

# main-text figure*: three anchors, three views
fig = plt.figure(figsize=(7.4, 2.7))
for k, (h, bl, title, ex, tf) in enumerate([(anchored(keys, 0.5), -1.0, "assumed: bicubic at $\\times1/2$", 0.5*keys(0.5*xf), False),
                                            (anchored(lanczos3, 0.25), -2.0, "assumed: Lanczos-3 at $\\times1/4$", 0.25*lanczos3(0.25*xf), False),
                                            (box, -5.0, "assumed: PSF = sinc", psf_spatial(box), True)]):
    ax = fig.add_subplot(1, 3, k+1, projection="3d"); draw(ax, h, bl, title, target_fixed=tf, exact=ex, tfs=8.0)
fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0.0, wspace=0.12)
fig.savefig('paper_output/figs/fig_surface_main.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_surface_main.png', dpi=300, bbox_inches="tight")

# shape evolution: each member in the pixels of the coarser image, both endpoints finite
from scipy.special import j1
def disk(w, r=2.0):                                      # defocus disk of radius r px, band-limited at pi: zero inside the band, contrast reversal beyond
    z = np.abs(w)*r; return np.where(np.abs(w) <= np.pi, np.where(z < 1e-9, 1.0, 2*j1(z)/np.where(z < 1e-9, 1, z)), 0.0)
fig = plt.figure(figsize=(7.4, 2.9))
for k, (h, bl, title, ex) in enumerate([(spec(gauss), None, "PSF = Gaussian, $\\sigma=1$", None),
                                        (disk, None, "PSF = defocus disk, $r=2$", None),
                                        (anchored(keys, 0.5), -1.0, "assumed: bicubic at $\\times1/2$", keys(xf))]):
    ax = fig.add_subplot(1, 3, k+1, projection="3d"); draw(ax, h, bl, title, target_fixed="coarser", exact=ex, tfs=8.0)
fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0.0, wspace=0.12)
fig.savefig('paper_output/figs/fig_surface_coarser.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_surface_coarser.png', dpi=300, bbox_inches="tight")

# the classical corner: source fixed, Gaussian PSF; the coarsening half is Gaussian scale-space exactly
fig = plt.figure(figsize=(3.6, 3.0)); ax = fig.add_subplot(111, projection="3d")
draw(ax, spec(gauss), None, "Gaussian camera: $s<1$ classical, $s>1$ new")
fig.subplots_adjust(left=0, right=1, top=1.02, bottom=0.0)
fig.savefig('paper_output/figs/fig_surface_gauss.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_surface_gauss.png', dpi=400, bbox_inches="tight")
import numpy as _np
h = spec(gauss); k = member(h, 0.5); g3 = _np.exp(-xf**2/(2*3.0)); g3 = g3/g3.max(); g4 = _np.exp(-xf**2/(2*4.0)); g4 = g4/g4.max()
print("Gaussian PSF, K_{1->1/2}: vs Gaussian of variance 3 (semigroup step) max diff %.4f; vs variance 4 (stretch) %.3f" % (_np.abs(k-g3).max(), _np.abs(k-g4).max()))
