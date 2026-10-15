"""Reproduce the supplied 15-degree lighthouse test with matched x/y shifts.

The finite Fourier-shear realization is reversible; it is not an exact
Euclidean rotation of an arbitrary periodic finite image. No numerical
matrix inverse is used. RMSE is absolute on the central 160x160 crop.
Source image: https://r0k.us/graphics/kodak/kodak/kodim19.png
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.special import erf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NORM = SymLogNorm(linthresh=0.001, vmin=-1, vmax=1, base=10)
SRC, CLS, CON = "#eef1f4", "#faf0ee", "#eef4ef"

def keys(x, a=-0.5):
    x = np.abs(x)
    return np.where(x < 1, (a+2)*x**3-(a+3)*x**2+1,
                    np.where(x < 2, a*x**3-5*a*x**2+8*a*x-4*a, 0))

dx, L = 1/32, 128
Nf = int(2*L/dx)
xf = (np.arange(Nf)-Nf//2)*dx
wf = 2*np.pi*np.fft.fftfreq(Nf, d=dx)
U = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(keys(xf)))*dx))
wsf = np.fft.fftshift(wf)
def uh(arg):
    return np.interp(np.abs(arg), wsf[wsf >= 0], U[wsf >= 0])
BL = 0.65*np.pi
def h(w):
    v = np.ones_like(np.asarray(w, float))
    for j in range(12):
        v *= uh(w/2**j)
    return v*(np.abs(w) < BL)

n = 384
img = np.asarray(Image.open(ROOT / "figures/kodim19.png").convert("L"), dtype=float)/255
crop = img[116:500, 64:448]
assert crop.shape == (n, n)
wk = 2*np.pi*np.fft.fftfreq(n)
WY, WX = np.meshgrid(wk, wk, indexing="ij")
R = np.hypot(WX, WY)
H2 = h(R)
cap0 = np.fft.ifft2(np.fft.fft2(crop)*H2).real
psf = np.fft.fftshift(np.fft.ifft2(H2)).real
xs0 = np.arange(n)-(n-1)/2
w1 = 0.5*(1-erf((np.abs(xs0)-108)/12))
cap = cap0*np.outer(w1, w1)
theta, t = np.deg2rad(15.0), (0.37, 0.21)
max_discarded_imag = []
def real_part(z):
    max_discarded_imag.append(float(np.max(np.abs(z.imag))))
    return z.real
def shift_fft(I, tx, ty):
    # x is the column coordinate and y the row coordinate, in both arms.
    return real_part(np.fft.ifft2(np.fft.fft2(I)*np.exp(-1j*(WX*tx+WY*ty))))
def shear_rows(I, a):
    phase = np.exp(-1j*np.outer(a*xs0, wk))
    return real_part(np.fft.ifft(np.fft.fft(I, axis=1)*phase, axis=1))
def shear_cols(I, b):
    phase = np.exp(-1j*np.outer(wk, b*xs0))
    return real_part(np.fft.ifft(np.fft.fft(I, axis=0)*phase, axis=0))
def rotate_consistent(I, th):
    a, b = -np.tan(th/2), np.sin(th)
    return shear_rows(shear_cols(shear_rows(I, a), b), a)
def T_consistent(I):
    return shift_fft(rotate_consistent(I, theta), *t)
def Tinv_consistent(I):
    return rotate_consistent(shift_fft(I, -t[0], -t[1]), -theta)

c = (n-1)/2
YY, XX = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
def fwd_map():
    y, x = YY-c-t[1], XX-c-t[0]
    return np.cos(theta)*y-np.sin(theta)*x+c, np.sin(theta)*y+np.cos(theta)*x+c
def inv_map():
    y, x = YY-c, XX-c
    return np.cos(theta)*y+np.sin(theta)*x+c+t[1], -np.sin(theta)*y+np.cos(theta)*x+c+t[0]
def resample_classical(I, mapping):
    ys, xs = mapping()
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    out = np.zeros_like(I)
    for dy in range(-1, 3):
        for dx_ in range(-1, 3):
            out += keys(ys-(y0+dy))*keys(xs-(x0+dx_))*I[(y0+dy)%n, (x0+dx_)%n]
    return out

CC = slice(n//2-80, n//2+80)
def rmse(A, B):
    return float(np.sqrt(np.mean((A[CC, CC]-B[CC, CC])**2)))
Jc = resample_classical(cap, fwd_map)
back_c = resample_classical(Jc, inv_map)
Jo = T_consistent(cap)
back_o = Tinv_consistent(Jo)
e_cls, e_con = rmse(back_c, cap), rmse(back_o, cap)
contrast = float(cap[CC, CC].std())
results = {"classical_rmse": e_cls, "consistent_rmse": e_con,
           "central_crop_std": contrast,
           "classical_relative_rmse": e_cls/contrast,
           "consistent_relative_rmse": e_con/contrast,
           "max_discarded_imaginary_part": max(max_discarded_imag),
           "windowed_out_of_band_energy_fraction": float(np.sum(np.abs(np.fft.fft2(cap)[R >= BL])**2)/np.sum(np.abs(np.fft.fft2(cap))**2)),
           "discrete_model_psf_min": float(psf.min()),
           "translation_xy": list(t), "angle_degrees": 15,
           "crop": [116, 500, 64, 448], "evaluation_crop": [112,272,112,272]}
print(json.dumps(results, indent=2))
assert e_con < 1e-12
(ROOT / "geometry_results.json").write_text(json.dumps(results, indent=2)+"\n")

plt.rcParams.update({"font.size": 15.2, "axes.titlesize": 15.2,
                    "pdf.fonttype": 42, "ps.fonttype": 42})
fig = plt.figure(figsize=(7.0, 4.3))
gs = fig.add_gridspec(2, 4, width_ratios=[1,1,1,1], wspace=0.24, hspace=0.70,
                      left=0.03, right=0.945, top=0.86, bottom=0.13)
vm = np.abs((back_c-cap)[CC, CC]).max()
axin = fig.add_subplot(gs[0,0])
axin.imshow(cap[CC, CC], cmap="gray", vmin=0, vmax=1)
axin.set_title("input", pad=2)
axpsf = fig.add_subplot(gs[1,0])
axpsf.imshow((psf/np.abs(psf).max())[n//2-16:n//2+16,n//2-16:n//2+16], cmap="RdBu_r", norm=NORM)
axpsf.set_title("model PSF", pad=2)
rows = [("classical", Jc, back_c, back_c-cap, e_cls, "%.1e", CLS),
        ("Fourier shears", Jo, back_o, back_o-cap, e_con, "%.0e", CON)]
axes = []
for i, (nm,J,bk,err,e,fmt,col) in enumerate(rows):
    a1,a2,a3 = [fig.add_subplot(gs[i,j]) for j in (1,2,3)]
    a1.imshow(J[CC,CC], cmap="gray", vmin=0, vmax=1); a1.set_title("transformed", pad=2)
    a2.imshow(bk[CC,CC], cmap="gray", vmin=0, vmax=1); a2.set_title("back", pad=2)
    a3.imshow(err[CC,CC], cmap="RdBu_r", vmin=-vm, vmax=vm); a3.set_title("round-trip\nerror", pad=2, linespacing=0.95)
    for ax in (a1,a2,a3): ax.set_xticks([]); ax.set_yticks([])
    axes.append((a1,a2,a3,nm,col,e,fmt))
for ax in (axin,axpsf): ax.set_xticks([]); ax.set_yticks([])
fig.text(0.5,1.005,"rotate $15^\\circ$ and shift $(%.2f,\\,%.2f)$ px, then back" % t,
         ha="center",va="top",fontsize=16.0)
fig.canvas.draw()
def bb(ax): return ax.get_position()
def box(x0,y0,x1,y1,color,pad=0.012):
    fig.add_artist(FancyBboxPatch((x0-pad,y0-pad), x1-x0+2*pad,y1-y0+2*pad,
        boxstyle="round,pad=0.004,rounding_size=0.02",fc=color,ec="none",zorder=-5,transform=fig.transFigure))
b_in,b_ps = bb(axin),bb(axpsf)
box(b_in.x0,b_ps.y0-0.09,b_in.x1,b_in.y1+0.10,SRC)
ar = dict(arrowstyle="-|>",color="0.4",lw=0.8,mutation_scale=7,shrinkA=0,shrinkB=0)
xsplit,ymid = b_in.x1+0.035,(b_ps.y0+b_in.y1)/2
fig.add_artist(plt.Line2D([b_in.x1+0.016,xsplit],[ymid,ymid],color="0.4",lw=0.8))
for a1,a2,a3,nm,col,e_,fmt_ in axes:
    y = (bb(a1).y0+bb(a1).y1)/2
    box(bb(a1).x0,bb(a1).y0-0.09,bb(a3).x1+0.040,bb(a1).y1+0.10,col)
    fig.add_artist(plt.Line2D([xsplit,xsplit],[ymid,y],color="0.4",lw=0.8))
    fig.add_artist(FancyArrowPatch((xsplit,y),(bb(a1).x0-0.008,y),**ar))
    for s_,d_ in ((a1,a2),(a2,a3)):
        fig.add_artist(FancyArrowPatch((bb(s_).x1+0.008,y),(bb(d_).x0-0.008,y),**ar))
    fig.text((bb(a1).x0+bb(a2).x1)/2,bb(a1).y0-0.073,nm,ha="center",va="bottom",fontsize=16.8,color="0.25")
    fig.text((bb(a3).x0+bb(a3).x1)/2,bb(a3).y0-0.028,("RMSE "+fmt_)%e_,ha="center",va="top",fontsize=14.8,color="0.25")
fig.savefig(ROOT / "figs/fig_rigid.pdf",bbox_inches="tight",facecolor="white")
fig.savefig(ROOT / "figs/fig_rigid.png",dpi=300,bbox_inches="tight",facecolor="white")
