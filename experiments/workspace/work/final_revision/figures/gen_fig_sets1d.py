import os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
rng = np.random.default_rng(5)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
dx = 1/16; N = 2**13; x = (np.arange(N) - N//2)*dx; w = 2*np.pi*np.fft.fftfreq(N, d=dx)
B = 0.6*np.pi                                                   # camera band, below Nyquist: smooth bump MTF
def h(v): return np.where(np.abs(v) <= B, 1 - 0.6*(v/B)**2, 0.0)          # MTF 0.4 at the band edge: bounded deconvolvers
def free_region(v):                                             # where a sampled capture has nothing: off every replica band B+2 pi k
    d = np.abs((v + np.pi) % (2*np.pi) - np.pi); return d > B
def sym(s, Z=None):
    H1 = h(w); K = h(w/s)*np.where(H1 > 1e-6, 1/np.where(H1 > 1e-6, H1, 1), 0.0)
    if Z is not None: K = K + Z*free_region(w)
    return K
def rand_Z(amp, odd=False):                                     # smooth random free part on the first free sliver; odd about Nyquist => taps vanish at integers
    n = 7; c = rng.standard_normal(n)*amp; d = rng.standard_normal(n)*amp; t = (np.abs(w) - np.pi)/(np.pi - B)
    Z = sum(d[k]*np.sin(np.pi*(k+1)*t) for k in range(n)) if odd else sum(c[k]*np.cos(0.5*np.pi*k*t) + d[k]*np.sin(0.5*np.pi*(k+1)*t) for k in range(n))
    return Z*np.exp(-0.7*t**2)*(np.abs(w) <= 3*np.pi)
def kernel(K): return np.fft.fftshift(np.real(np.fft.ifft(K)))/dx
def apply(I1, K, s):                                            # bed of nails, convolve, read at n/s
    nails = np.zeros(N); st1 = int(round(1/dx)); nails[N//2 % st1::st1] = I1
    out = np.real(np.fft.ifft(np.fft.fft(nails)*K))/dx; st = int(round(1/(s*dx))); return out[N//2 % st::st]
def apply_fine(I2, K, s):                                       # fine samples (spacing 1/2) as input: bed of nails on the fine lattice
    nails = np.zeros(N); st2 = int(round(0.5/dx)); nails[N//2 % st2::st2] = I2
    out = np.real(np.fft.ifft(np.fft.fft(nails)*K))/(dx/0.5); st = int(round(0.5/(s*dx))); return out[N//2 % st::st]
scene = rng.standard_normal(N); S = np.fft.fft(scene)
I1 = np.real(np.fft.ifft(S*h(w)))[N//2 % 16::16]                # capture at scale 1 (spacing 1)
I2 = np.real(np.fft.ifft(S*h(w/2)))[N//2 % 8::8]                # capture at scale 2 (spacing 1/2, scene units)
red = "#c0392b"; orange = "#e67e22"; W = 4; sel = np.abs(x) <= W
fig, axs = plt.subplots(2, 2, figsize=(3.45, 3.0), gridspec_kw=dict(height_ratios=[1.5, 1.0], hspace=0.5, wspace=0.32))
for j, s in enumerate([1/3, 3.0]):
    K0 = sym(s); k0 = kernel(K0); pk = np.abs(k0).max()
    members = [sym(s, rand_Z(0.35, odd=(s < 1))) for _ in range(120)]; ks = np.array([kernel(K) for K in members])
    reps = []
    if s > 1:                                                 # beyond the digital band: free part on the replica band (2pi-B, 2pi+B)
        for _ in range(7):
            a = rng.uniform(0.3, 0.9)*rng.choice([-1, 1]); c0 = 2*np.pi + rng.uniform(-0.3, 0.3)*B
            Zr = a*np.exp(-((np.abs(w) - c0)/(0.5*B))**2)*(np.abs(np.abs(w) - 2*np.pi) < B); reps.append(sym(s) + Zr)
    kr = np.array([kernel(K) for K in reps]) if reps else None
    ax = axs[0, j]
    nshow = 5 if s < 1 else 3; ntop = 5 if s < 1 else 2
    for i, K in enumerate(ks[:ntop]): ax.plot(x[sel], K[sel]/pk, color=red, lw=0.5, alpha=0.8, label="in-band members" if (i == 0 and j == 1) else None)
    if kr is not None:
        for i, K in enumerate(kr[:2]): ax.plot(x[sel], K[sel]/pk, color=orange, lw=0.6, alpha=0.9, label="replica-band nonmembers" if i == 0 else None)
    ax.plot(x[sel], k0[sel]/pk, color="k", lw=1.1, label="canonical member" if j == 0 else None)
    phases = [0.0] if s < 1 else [-1/3, 0.0, 1/3]
    for ph in phases:
        tx = np.arange(-W, W) + ph; ti = np.round((tx - x[0])/dx).astype(int)
        if s < 1:
            for K in ks[:nshow]: ax.plot(tx, K[ti]/pk, "o", color=red, ms=2.0, alpha=0.8, mew=0)
        ax.vlines(tx, 0, k0[ti]/pk, color="k", lw=0.5); ax.plot(tx, k0[ti]/pk, "o", color="k", ms=2.0, mew=0, label="taps at the output phase" if (ph == phases[0] and j == 0) else None)
    tdiff = max(np.abs(K[ti] - k0[ti]).max() for K in ks)/np.abs(k0[ti]).max()
    print("s=%g: taps of members differ from the canonical taps by up to %.2f of the peak tap" % (s, tdiff))
    ax.set_title("kernels of $\\mathcal{K}_{1/3}$" if s < 1 else "kernels of $\\mathcal{K}_{3}$", fontsize=7.2, pad=3)
    ax.set_xlim(-W, W); ax.tick_params(labelsize=6); ax.set_xlabel("$x$ (input pixels)", fontsize=6.5, labelpad=1)
    if j == 0: ax.set_ylabel("$K_s$, peak-normalized", fontsize=6.5)
    ax = axs[1, j]; L0 = 6; L1 = 22                          # scene window, in input pixels
    ph = 1.0 if s < 1 else -1/3                                 # centered-grid phase of the output samples: 3n+1 for x1/3, n/3-1/3 for x3
    def pos(k): return k/s + ph
    idx = np.arange(int(np.ceil((L0 - ph)*s)), int(np.floor((L1 - ph)*s)) + 1)
    ax.plot(np.arange(L0, L1+1), I1[L0:L1+1], "o", color="0.55", ms=2.2, mew=0, label="input $I_1$" if j == 0 else None)
    ax.plot(np.arange(L0, L1+1), I1[L0:L1+1], "-", color="0.75", lw=0.5)
    o0 = apply(I1, K0, s); yl = np.abs(I1[L0:L1+1]).max()*1.35
    for K in members[:nshow]: ax.plot(pos(idx), apply(I1, K, s)[idx], "-", color=red, lw=0.5, alpha=0.75)
    if kr is not None:
        for K in reps: ax.plot(pos(idx), apply(I1, K, s)[idx], "-", color=orange, lw=0.5, alpha=0.7)
    ax.plot(pos(idx), o0[idx], "o-", color="k", ms=2.2, mew=0, lw=0.8, label="canonical output" if j == 0 else None)
    spread = max(np.abs(apply(I1, K, s)[idx] - o0[idx]).max() for K in members)/np.abs(o0[idx]).max()
    ax.set_xlabel("position (input pixels)", fontsize=6.5, labelpad=1); ax.tick_params(labelsize=6); ax.set_xlim(L0, L1); ax.set_ylim(-yl, yl)
    ax.set_title("outputs $\\mathbf{K}_{1/3}(I_1)$" if s < 1 else "outputs $\\mathbf{K}_{3}(I_1)$", fontsize=7.2, pad=3)
entries = {}
for ax in axs.ravel():
    for hh, ll in zip(*ax.get_legend_handles_labels()): entries.setdefault(ll, hh)
fig.legend(list(entries.values()), list(entries.keys()), fontsize=5.6, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.53, -0.01), handlelength=1.5, columnspacing=0.9, handletextpad=0.5)
fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.2)
fig.savefig(os.path.join(OUT, "fig_sets1d.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_sets1d.png"), dpi=300, bbox_inches="tight")
print("saved")
