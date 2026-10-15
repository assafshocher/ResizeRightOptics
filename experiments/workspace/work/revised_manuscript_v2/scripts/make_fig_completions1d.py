import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("make_fig_sets1d.py").read().split("red = ")[0])          # camera, captures, sym(), apply(), apply_fine()
red = "#c0392b"
H2 = h(w/2); D = h(w)*np.where(H2 > 1e-6, 1/np.where(H2 > 1e-6, H2, 1), 0.0)     # downscaler from the fine capture
U0 = apply(I1, sym(2.0), 2.0)                                                       # canonical upscale (one action)
# a replica-reading member of the continuum set: free part on the source's first replica band (2pi-B, 2pi+B)
Zrep = 0.9*np.exp(-((np.abs(w) - 2*np.pi)/(0.6*B))**2)*(np.abs(np.abs(w) - 2*np.pi) < B)
Urep = apply(I1, sym(2.0, Zrep) if False else sym(2.0) + Zrep, 2.0)
# a non-convolution completion: random annulus content
nullz = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal(N))*((np.abs(w) > B) & (np.abs(w) <= 2*B))))[N//2 % 8::8]
nullz *= 0.6*np.abs(U0).max()/np.abs(nullz).max()
# the classical bicubic upscale: Keys kernel read at the x2 grid, no camera
def keys(t, a=-0.5):
    t = np.abs(t); return np.where(t < 1, (a+2)*t**3 - (a+3)*t**2 + 1, np.where(t < 2, a*t**3 - 5*a*t**2 + 8*a*t - 4*a, 0.0))
Kbic = np.fft.fft(np.fft.ifftshift(keys(x)))*dx
Ubic = apply(I1, Kbic, 2.0)
cands = [("canonical $\\mathbf{D}^{\\dagger}y$ (every member)", U0, "k"), ("true fine capture $I_2$", I2, "#2c7fb8"),
         ("replica-reading member", Urep, red), ("$\\mathbf{D}^{\\dagger}y$ + random annulus content", U0 + nullz, "#7f8c8d"),
         ("classical bicubic upscale", Ubic, "#e67e22")]
n0f = 20; nf = 40
fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.0), gridspec_kw=dict(height_ratios=[1.5, 1.0], hspace=0.45))
ax = axs[0]
for k, (nm, u, c) in enumerate(cands): ax.plot(np.arange(nf), u[n0f:n0f+nf] + 0.0, color=c, lw=0.9 if k else 1.3, label=nm, alpha=0.95)
ax.set_title("fine images: completions of $y$, and one convention", fontsize=7.5, pad=3); ax.set_xlabel("fine pixel", fontsize=6.5, labelpad=1)
ax.tick_params(labelsize=6); ax.set_xlim(0, nf-1); ax.legend(fontsize=5.0, frameon=False, loc="upper right", handlelength=1.2)
ax = axs[1]
for nm, u, c in cands:
    d = apply_fine(u, D, 0.5); e = np.abs(d[n0f//2:n0f//2+nf//2] - I1[n0f//2:n0f//2+nf//2]).max()/np.abs(I1).max()
    ax.plot(np.arange(nf//2), d[n0f//2:n0f//2+nf//2], color=c, lw=0.9, alpha=0.95); print("%-45s  |D x - y| / |y| = %.1e" % (nm, e))
ax.plot(np.arange(nf//2), I1[n0f//2:n0f//2+nf//2], "k--", lw=0.6)
ax.set_title("their downscales $\\mathbf{D}x$: all equal $y$ except the convention", fontsize=7.5, pad=3); ax.set_xlabel("coarse pixel", fontsize=6.5, labelpad=1)
ax.tick_params(labelsize=6); ax.set_xlim(0, nf//2-1)
fig.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.12)
fig.savefig('paper_output/figs/fig_completions1d.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_completions1d.png', dpi=300, bbox_inches="tight")
