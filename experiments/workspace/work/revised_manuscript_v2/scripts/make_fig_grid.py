import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.size": 6.5})
fig, axs = plt.subplots(2, 1, figsize=(3.4, 2.3)); fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04, hspace=0.9)
for ax, (n_in, s, n_out, title) in zip(axs, [(8, 0.5, 4, "in = 8, s = 1/2, out = 4 (exact)"), (8, 0.7, 6, "in = 8, s = 0.7, out = 6 (rounded)")]):
    g_c = np.arange(n_out)/s + (n_in-1)/2 - (n_out-1)/(2*s)          # centered, eq. (5)
    g_n = np.arange(n_out)/s + 0.5*(1/s - 1)                          # corner-anchored, eq. (4)
    for m in range(n_in): ax.add_patch(Rectangle((m-0.5, 0.55), 1, 0.4, fc="#dfe7f0", ec="#1f4e79", lw=0.6)); ax.plot(m, 0.75, ".", color="#1f4e79", ms=3)
    for gi in g_c: ax.add_patch(Rectangle((gi-0.5/s, 0.0), 1/s, 0.4, fc="none", ec="#c0392b", lw=0.8)); ax.plot(gi, 0.2, "s", color="#c0392b", ms=3.2)
    if abs(n_out - s*n_in) > 1e-9:
        for gi in g_n: ax.plot(gi, 0.2, "s", mfc="none", mec="#c0392b", ms=3.2, mew=0.7)
    ax.axvline((n_in-1)/2, color="#777777", lw=0.6, ls="--", ymin=0.0, ymax=1.0)
    ax.set_title(title, fontsize=6.8, pad=2); ax.set_xlim(-1.2, n_in+0.2); ax.set_ylim(-0.05, 1.0); ax.axis("off")
    ax.text(-1.15, 0.75, "input", fontsize=6, va="center"); ax.text(-1.15, 0.2, "output", fontsize=6, va="center")
fig.savefig('paper_output/figs/fig_grid.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_grid.png', dpi=220, bbox_inches="tight"); print("saved")
