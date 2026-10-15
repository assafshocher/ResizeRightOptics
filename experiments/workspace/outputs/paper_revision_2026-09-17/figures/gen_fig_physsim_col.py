"""A camera computed from optics, not chosen as a kernel: 1D pupil with defocus + spherical aberration, diffraction cutoff at
0.9 pi rad/px (Nyquist-sampled), 100% fill-factor pixel, read noise. A planar random texture captured at distances d, 2d, 3d
by exact convolution + sampling on a fine grid. Tests: consistency, composition, spectral ratio, telescope."""
from pathlib import Path
import json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
rng = np.random.default_rng(11)
dx = 1/24; N = 2**15; x = (np.arange(N) - N//2)*dx; w = 2*np.pi*np.fft.fftfreq(N, d=dx)
# ---- the camera from a pupil: OTF = normalized autocorrelation of P(u) = exp(i phi(u)), |u|<=1, at shift w/wc
wc = 0.9*np.pi; W20, W40 = 0.18, 0.10                                   # defocus and spherical aberration, in waves
uu = np.linspace(-1, 1, 4001); du = uu[1]-uu[0]
P = np.exp(1j*2*np.pi*(W20*uu**2 + W40*uu**4))
def otf_lens(om):
    sh = np.abs(om)/wc*2.0                                              # pupil autocorrelation: shift in pupil units (full width 2)
    out = np.zeros_like(om, dtype=complex)
    for i, s_ in enumerate(sh):
        if s_ >= 2: continue
        m = uu >= -1 + s_; out[i] = np.sum(P[m]*np.conj(np.interp(uu[m]-s_, uu, P.real) + 1j*np.interp(uu[m]-s_, uu, P.imag)))*du
    return out/out[np.argmin(np.abs(om))]
OM = np.linspace(0, 1.2*np.pi, 600); L = otf_lens(OM)
def H(om): return np.interp(np.abs(om), OM, L.real, right=0.0)*np.sinc(om/(2*np.pi))     # lens OTF (real part; even pupil) x pixel box
Hw = H(w); psf = np.fft.fftshift(np.real(np.fft.ifft(Hw)))/dx
# ---- scene and captures (read noise 0.5% of full scale)
sig = 0.005
scene = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal(N))*np.minimum(1.0, 1.5/np.maximum(np.abs(w), 0.05))))   # ~1/f texture
scene = 0.5 + 0.12*scene/scene.std()
def capture(s):
    I = np.real(np.fft.ifft(np.fft.fft(scene)*H(w/s))); st = int(round(1/(s*dx))); v = I[N//2 % st::st]; return v + sig*rng.standard_normal(v.shape)
def pinv(v, floor=1e-3): return np.where(np.abs(v) > floor, 1/np.where(np.abs(v) > floor, v, 1), 0.0)
def member(s, band=True): K = H(w/s)*pinv(H(w)); return K*((np.abs(w) <= np.pi) if band else 1.0)
def matched_stretch(s, anchor=0.5):
    """Unit-mass dilation of the same anchor kernel, not of an arbitrary cubic.

    A_s(x)=(s/anchor) K_anchor((s/anchor)x), hence
    Ahat_s(omega)=Khat_anchor(anchor*omega/s). Evaluate the continuous
    lookup-defined anchor directly so equality at the anchor is exact.
    No spatial truncation or extra phase normalization is introduced.
    """
    om = anchor*w/s
    return H(om/anchor)*pinv(H(om))*(np.abs(om) <= np.pi)
def keys(t, a=-0.5):
    t = np.abs(t); return np.where(t < 1, (a+2)*t**3 - (a+3)*t**2 + 1, np.where(t < 2, a*t**3 - 5*a*t**2 + 8*a*t - 4*a, 0.0))
def stretch(s): return np.fft.fft(np.fft.ifftshift(keys(x*s)*s))*dx                                            # dilated bicubic, mass 1
def apply(I, K, s_in, s):                                                # input samples at spacing 1/s_in; read at n/(s_in s)
    st_in = int(round(1/(s_in*dx))); nails = np.zeros(N); nails[N//2 % st_in::st_in] = I
    Ks = np.interp(np.abs(w)/s_in, np.abs(w)[np.argsort(np.abs(w))], K[np.argsort(np.abs(w))]) if s_in != 1 else K   # kernel in input pixels -> scene units
    out = np.real(np.fft.ifft(np.fft.fft(nails)*Ks))/dx*(1/s_in) ; st = int(round(1/(s_in*s*dx))); return out[N//2 % st::st]
I1, I2, I3 = capture(1.0), capture(0.5), capture(1/3)
m = 200; err = lambda a, b: np.sqrt(np.mean((a[m:-m]-b[m:-m])**2))/np.sqrt(np.mean((b[m:-m]-b[m:-m].mean())**2))
res = {}
# (1) consistency: predict the x1/2 and x1/3 captures from the x1 capture
for s, Is, nm in [(0.5, I2, "x1/2"), (1/3, I3, "x1/3")]:
    res["cons_fam_"+nm] = err(apply(I1, member(s), 1.0, s), Is); res["cons_str_"+nm] = err(apply(I1, stretch(s), 1.0, s), Is)
    res["cons_matched_"+nm] = err(apply(I1, matched_stretch(s), 1.0, s), Is)
res["anchor_multiplier_max_abs_diff"] = float(np.max(np.abs(member(0.5)-matched_stretch(0.5))))
assert res["anchor_multiplier_max_abs_diff"] == 0.0
res["anchor_output_max_abs_diff"] = float(np.max(np.abs(apply(I1, member(0.5), 1.0, 0.5)-apply(I1, matched_stretch(0.5), 1.0, 0.5))))
assert res["anchor_output_max_abs_diff"] == 0.0
res["noise_reference"] = sig/np.sqrt(np.mean((I2[m:-m]-I2[m:-m].mean())**2))*np.sqrt(2)
# (2) composition: x1/2 then x2/3 vs x1/3 in one step
chain_fam = apply(apply(I1, member(0.5), 1.0, 0.5), member(2/3), 0.5, 2/3); one_fam = apply(I1, member(1/3), 1.0, 1/3)
chain_str = apply(apply(I1, stretch(0.5), 1.0, 0.5), stretch(2/3), 0.5, 2/3); one_str = apply(I1, stretch(1/3), 1.0, 1/3)
res["comp_fam"] = err(chain_fam, one_fam); res["comp_str"] = err(chain_str, one_str)
chain_matched = apply(apply(I1, matched_stretch(0.5), 1.0, 0.5), matched_stretch(2/3), 0.5, 2/3)
one_matched = apply(I1, matched_stretch(1/3), 1.0, 1/3)
res["comp_matched"] = err(chain_matched, one_matched)
# (3) spectral ratio: |I_{1/2}|^2 / |I_1|^2 over the shared band vs |K_{1/2}|^2 (registration-free)
def pspec(v, nseg=2048):
    segs = [v[i:i+nseg]*np.hanning(nseg) for i in range(0, len(v)-nseg, nseg//2)]
    return np.mean([np.abs(np.fft.rfft(s_-s_.mean()))**2 for s_ in segs], 0)/np.sum(np.hanning(nseg))**2
P1, P2 = pspec(I1, 512), pspec(I2, 256)                             # both cover 0..pi rad/px of their own pixels
f1 = np.linspace(0, np.pi, len(P1)); f2 = np.linspace(0, np.pi, len(P2))
ratio = P2/np.interp(f2/2, f1, P1)                                    # I_2 at its frequency nu (per coarse px) vs I_1 at nu/2 (per fine px)
pred_fam = np.abs(H(f2)*pinv(H(f2/2)))**2                              # |K_{1/2}(omega)|^2 at omega = nu/2 (scene units)
pred_str = np.abs(np.interp(f2/2, np.abs(w[:N//2]), np.abs(stretch(0.5))[:N//2]))**2
# (4) telescope: PSF from the x1/2 member alone
Kh = member(0.5); Kpsf = np.ones_like(w, dtype=complex)
for j in range(1, 14): Kpsf = Kpsf*np.interp(np.abs(w)/2**j, np.abs(w[:N//2]), Kh[:N//2].real)
res["telescope_err"] = np.max(np.abs(Kpsf.real - Hw)[np.abs(w) <= 0.85*np.pi])
for k, v in res.items(): print("%-18s %.2e" % (k, v))
print("min |OTF| inside 0.6 pi: %.3f" % np.min(np.abs(H(np.linspace(0, 0.6*np.pi, 300)))))
# ---- single-column figure
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                    "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.0))
keep = (f2 > 0.05) & (f2 < 0.95*np.pi)
ax = axs[0]
ax.semilogy(f2[keep]/np.pi, ratio[keep], color="0.5", lw=0.85,
            label="measured")
ax.semilogy(f2[keep]/np.pi, pred_fam[keep], "k", lw=1.15,
            label="ours = matched AA")
ax.semilogy(f2[keep]/np.pi, pred_str[keep], color="#D55E00", lw=1.15, ls="--",
            label="cubic AA")
ax.set_ylim(1e-3, 60)
ax.set_title("Spectrum ratio", fontsize=8)
ax.set_xlabel("coarse Nyquist fraction", fontsize=7)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=1, labelspacing=0.2)
ins = ax.inset_axes([0.12, 0.06, 0.35, 0.32])
sel = np.abs(x) <= 6
ins.plot(x[sel], psf[sel]/psf.max(), "k", lw=1.0)
ins.set_title("PSF", fontsize=7, pad=1)
ins.tick_params(labelsize=7, length=2)
ins.set_yticks([])

ax = axs[1]
labels = ["$1/2$", "$1/3$", "comp."]
fam = [res["cons_fam_x1/2"], res["cons_fam_x1/3"], res["comp_fam"]]
st = [res["cons_str_x1/2"], res["cons_str_x1/3"], res["comp_str"]]
xx = np.arange(3)
matched = [res["cons_matched_x1/2"], res["cons_matched_x1/3"], res["comp_matched"]]
ax.bar(xx-0.24, fam, 0.24, color="k", label="ours")
ax.bar(xx, matched, 0.24, color="#0072B2", hatch="///", linewidth=0.3, label="matched AA")
ax.bar(xx+0.24, st, 0.24, color="#D55E00", label="cubic AA")
ax.axhline(res["noise_reference"], color="0.5", lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_ylim(2e-5, 50)
ax.set_xticks(xx)
ax.set_xticklabels(labels, fontsize=7)
ax.tick_params(labelsize=7)
ax.set_xlabel("prediction scale / chain", fontsize=7)
ax.set_ylabel("relative RMSE", fontsize=7, labelpad=1)
ax.set_title("Prediction / composition", fontsize=8)
ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=1, labelspacing=0.2)

fig.tight_layout(pad=0.28, w_pad=0.35)
out = Path(__file__).resolve().parents[1] / "figs" / "fig_physsim_col.png"
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix(".pdf"))
(out.parents[1] / "optical_results.json").write_text(json.dumps(res, indent=2)+"\n")
print("saved", out)
