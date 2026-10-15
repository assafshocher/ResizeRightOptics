# Ten surfaces as separate panels (input pixels for slice anchors, output pixels for PSF anchors), plus composites.
src = open(__import__('pathlib').Path(__file__).with_name('make_fig_surface.py')).read().split("# main-text figure*")[0]
src = src.replace('fig.savefig("/home/claude/paper_cvpr/figs/fig_surface.pdf"', 'pass #').replace('fig.savefig("/home/claude/paper_cvpr/figs/fig_surface.png"', 'pass #')
exec(src)
import numpy as np, os
from scipy.special import j1
from PIL import Image
def diskc(w, r=2.0): z = np.abs(w)*r; return np.where(np.abs(w) <= np.pi, np.where(z < 1e-9, 1.0, 2*j1(z)/np.where(z < 1e-9, 1, z)), 0.0)
def boxpsf(w, L=3.0): return np.sinc(L*w/(2*np.pi))*(np.abs(w) <= np.pi)   # box of width L px (motion blur): MTF sinc(L w/2), zero at 2pi/L inside the band, cut at pi
def coma_sym(w, a=1.2): return (1/(1 + 1j*w*a)**2)*(np.abs(w) <= np.pi)      # comatic (skewed) PSF x e^{-x/a}, x>=0: complex symbol, band-limited
OUT = 'paper_output/figs/surf'; os.makedirs(OUT, exist_ok=True)
panels = [  # (file stem, camera, band_log, title, exact stripe, target_fixed)
 ("s_bicubic_half",  anchored(keys, 0.5),     -1.0, "assumed: bicubic at $\\times1/2$",           0.5*keys(0.5*xf),        False),
 ("s_bicubic_quarter", anchored(keys, 0.25),  -2.0, "assumed: bicubic at $\\times1/4$",           0.25*keys(0.25*xf),      False),
 ("s_bilinear_half", anchored(tri, 0.5),      -1.0, "assumed: bilinear at $\\times1/2$",          0.5*tri(0.5*xf),         False),
 ("s_lanczos_half",  anchored(lanczos3, 0.5), -1.0, "assumed: Lanczos-3 at $\\times1/2$",         0.5*lanczos3(0.5*xf),    False),
 ("s_box_half",      anchored(boxk, 0.5),     -1.0, "assumed: box (area average) at $\\times1/2$", 0.5*boxk(0.5*xf),       False),
 ("s_box_quarter",   anchored(boxk, 0.25),    -2.0, "assumed: box (area average) at $\\times1/4$", 0.25*boxk(0.25*xf),     False),
 ("p_gauss",         spec(gauss),             -6.25, "assumed: PSF = Gaussian, $\\sigma=1$",       psf_spatial(spec(gauss)), "psf"),
 ("p_box",           boxpsf,                  -6.25, "assumed: PSF = box, 3 px (motion blur)",     psf_spatial(boxpsf),      "psf"),
 ("p_disk",          diskc,                   -6.25, "assumed: PSF = defocus disk, $r=2$",         psf_spatial(diskc),       "psf"),
 ("p_coma",          coma_sym,                -6.25, "assumed: PSF = comatic (skewed)",            psf_spatial(coma_sym),    "psf"),
 ("p_airy",          airy1d,                  -6.25, "assumed: PSF = sinc$^2$ (1D Airy)",          psf_spatial(airy1d),      "psf")]
for stem, h, bl, title, ex, tf in panels:
    fig = plt.figure(figsize=(3.6, 3.0)); ax = fig.add_subplot(111, projection="3d")
    draw(ax, h, bl, title, target_fixed=tf, exact=ex, tfs=8.0, Wp=(12 if tf == "psf" else None))
    fig.subplots_adjust(left=0, right=1, top=1.02, bottom=0.0)
    fig.savefig("%s/%s.png" % (OUT, stem), dpi=300, bbox_inches="tight"); plt.close(fig); print("panel", stem)
def compose(stems, ncol, name):
    ims = [Image.open("%s/%s.png" % (OUT, s)) for s in stems]
    w = max(i.width for i in ims); hgt = max(i.height for i in ims)
    nrow = int(np.ceil(len(ims)/ncol)); canvas = Image.new("RGB", (ncol*w, nrow*hgt), "white")
    for k, im in enumerate(ims): canvas.paste(im, ((k % ncol)*w + (w-im.width)//2, (k//ncol)*hgt + (hgt-im.height)//2))
    canvas.save('paper_output/figs/%s.png' % name); print("composite", name, canvas.size)
compose(["s_bicubic_half", "s_box_quarter", "p_gauss"], 3, "fig_surfaces_main")
compose(["s_bicubic_quarter", "s_bilinear_half", "s_lanczos_half", "s_box_half"], 2, "fig_surfaces_slices")
compose(["p_gauss", "p_disk", "p_box", "p_airy"], 2, "fig_surfaces_psfs")
