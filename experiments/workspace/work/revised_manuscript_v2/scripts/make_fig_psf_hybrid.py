src = open(__import__('pathlib').Path(__file__).with_name('make_fig_surface.py')).read().split("# main-text figure*")[0]
src = src.replace('fig.savefig("/home/claude/paper_cvpr/figs/fig_surface.pdf"', 'pass #').replace('fig.savefig("/home/claude/paper_cvpr/figs/fig_surface.png"', 'pass #')
exec(src)
import numpy as np, os
from scipy.special import j1
from PIL import Image
def diskc(w, r=2.0): z = np.abs(w)*r; return np.where(np.abs(w) <= np.pi, np.where(z < 1e-9, 1.0, 2*j1(z)/np.where(z < 1e-9, 1, z)), 0.0)
def boxpsf(w, L=3.0): return np.sinc(L*w/(2*np.pi))*(np.abs(w) <= np.pi)
def coma_sym(w, a=1.2): return (1/(1 + 1j*w*a)**2)*(np.abs(w) <= np.pi)             # skewed (comatic) PSF x e^{-x/a} on x>=0: complex symbol, band-limited
def coma_psf(): return psf_spatial(coma_sym)
def tent2(x): return np.clip(1 - np.abs(x)/2.0, 0, None)                            # 2-px triangle PSF
OUT = 'paper_output/figs/surf'; os.makedirs(OUT, exist_ok=True)
panels = [("h_gauss", spec(gauss), "assumed: PSF = Gaussian, $\\sigma=1$", psf_spatial(spec(gauss))),
          ("h_disk",  diskc,       "assumed: PSF = defocus disk, $r=2$",   psf_spatial(diskc)),
          ("h_box",   boxpsf,      "assumed: PSF = box, 3 px (motion blur)", psf_spatial(boxpsf)),
          ("h_coma",  coma_sym,    "assumed: PSF = comatic (skewed)",       coma_psf()),
          ("h_tent",  spec(tent2), "assumed: PSF = triangle, 2 px",         psf_spatial(spec(tent2))),
          ("h_airy",  airy1d,      "assumed: PSF = sinc$^2$ (1D Airy)",     psf_spatial(airy1d))]
for stem, h, title, ex in panels:
    fig = plt.figure(figsize=(3.6, 3.0)); ax = fig.add_subplot(111, projection="3d")
    draw(ax, h, -4.25, title, target_fixed="psf", exact=ex, tfs=8.0)
    fig.subplots_adjust(left=0, right=1, top=1.02, bottom=0.0)
    fig.savefig("%s/%s.png" % (OUT, stem), dpi=300, bbox_inches="tight"); plt.close(fig); print("panel", stem)
ims = [Image.open("%s/%s.png" % (OUT, p[0])) for p in panels]; w = max(i.width for i in ims); hh = max(i.height for i in ims)
canvas = Image.new("RGB", (3*w, 2*hh), "white")
for k, im in enumerate(ims): canvas.paste(im, ((k % 3)*w, (k//3)*hh))
canvas.save('paper_output/figs/fig_psf_hybrid_candidates.png'); print("done")
