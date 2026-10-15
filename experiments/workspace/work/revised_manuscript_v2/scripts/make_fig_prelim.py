import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def tri(t): return np.clip(1-np.abs(t), 0, None)
def keys(x, a=-0.5):
    x = np.abs(x); y = np.zeros_like(x); m1 = x < 1; m2 = (x >= 1) & (x < 2)
    y[m1] = (a+2)*x[m1]**3 - (a+3)*x[m1]**2 + 1; y[m2] = a*x[m2]**3 - 5*a*x[m2]**2 + 8*a*x[m2] - 4*a; return y
def lanczos(t, A=3):
    t = np.asarray(t, float); y = np.sinc(t)*np.sinc(t/A); y[np.abs(t) >= A] = 0; return y
def gauss(t, s=0.7): return np.exp(-t**2/(2*s**2))/(s*np.sqrt(2*np.pi))
xx = np.linspace(-3.2, 3.2, 321); X, Y = np.meshgrid(xx, xx)
kers = [("bilinear", tri, 1), ("bicubic", keys, 2), ("Lanczos-3", lanczos, 3), ("Gaussian", gauss, 3)]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5})
fig, axs = plt.subplots(2, 4, figsize=(4.6, 2.5)); fig.subplots_adjust(wspace=0.08, hspace=0.25, left=0.07, right=0.99, top=0.9, bottom=0.08)
for j, (name, f, S) in enumerate(kers):
    K2 = f(X)*f(Y); vm = np.abs(K2).max()
    axs[0, j].imshow(K2, cmap="RdBu_r", vmin=-vm, vmax=vm); axs[0, j].set_title(name)
    kk = np.arange(-S, S) + 0.5; t = f(kk); T = np.outer(t, t); vt = np.abs(T).max()
    axs[1, j].imshow(T, cmap="RdBu_r", vmin=-vt, vmax=vt, interpolation="nearest")
    axs[1, j].set_xticks(np.arange(-0.5, 2*S, 1), minor=True); axs[1, j].set_yticks(np.arange(-0.5, 2*S, 1), minor=True); axs[1, j].grid(which="minor", color="w", lw=0.6)
    axs[1, j].set_xlabel("$%d\\times%d$ taps" % (2*S, 2*S), labelpad=1)
    for ax in axs[:, j]: ax.set_xticks([]); ax.set_yticks([])
axs[0, 0].set_ylabel("continuous"); axs[1, 0].set_ylabel("discrete")
fig.savefig('paper_output/figs/fig_prelim.pdf', bbox_inches="tight"); print("saved")
