# The projected grid, built as on the slide: g_n = n/s + q, and q fixed by centering.
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
N0, s = 6, 0.47; N = int(np.ceil(N0*s))               # 6 input pixels -> 3 output pixels
q = (N0-1)/2 - (N-1)/(2*s)                             # the centering offset
g = np.arange(N)/s + q
fig, ax = plt.subplots(figsize=(3.45, 2.35)); ax.set_xlim(-0.9, N0 - 0.1); ax.set_ylim(-2.75, 1.35); ax.axis("off")
# input row: boxes of width 1 centered at integers, centers as dots
for m in range(N0): ax.add_patch(plt.Rectangle((m-0.5, 0.0), 1, 0.8, fc="#eaf3ea", ec="0.3", lw=0.8))
ax.plot(np.arange(N0), np.full(N0, 0.4), "o", color="0.15", ms=3.0, mew=0)
ax.text(-0.75, 0.4, "input\n$N_0=%d$" % N0, ha="right", va="center", fontsize=6.5)
# output row: boxes of width 1/s at g_n, centers as green dots
for n in range(N): ax.add_patch(plt.Rectangle((g[n]-0.5/s, -1.55), 1/s, 0.8, fc="white", ec="0.3", lw=0.8))
ax.plot(g, np.full(N, -1.15), "o", color="#2e7d32", ms=4.0, mew=0)
ax.text(-0.75, -1.15, "output\n$N=%d$, $s=%.2f$" % (N, s), ha="right", va="center", fontsize=6.5)
# the projected centers drawn back on the input row (crosses) and the offset q from the first input center
for n in range(N): ax.plot([g[n], g[n]], [-1.15, 0.4], color="#2e7d32", lw=0.5, ls=":")
ax.plot(g, np.full(N, 0.4), "x", color="#2e7d32", ms=4.5, mew=1.0)
ax.annotate("", xy=(g[0], 0.95), xytext=(0, 0.95), arrowprops=dict(arrowstyle="->", lw=0.8, color="#c0392b"))
ax.text((g[0])/2, 1.02, "$q=g_0$", ha="center", va="bottom", fontsize=6.5, color="#c0392b")
ax.annotate("", xy=(g[1], 0.95), xytext=(g[0], 0.95), arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
ax.text((g[0]+g[1])/2, 1.02, "$1/s$", ha="center", va="bottom", fontsize=6.5, color="0.3")
# the two centers (image midpoints) and the centering condition
cin = (N0-1)/2; cout = g[0] + (N-1)/(2*s)
ax.plot([cin, cin], [-1.85, 0.9], color="0.5", lw=0.7, ls="--"); ax.text(cin, -1.95, "input center $\\frac{N_0-1}{2}$", ha="center", va="top", fontsize=6.0, color="0.35")
ax.plot([cout, cout], [-1.6, -1.15], color="#2e7d32", lw=1.2)
ax.annotate("", xy=(cout, -0.5), xytext=(g[0], -0.5), arrowprops=dict(arrowstyle="->", lw=0.8, color="#2e7d32"))
ax.text((g[0]+cout)/2, -0.42, "$\\frac{N-1}{2s}$", ha="center", va="bottom", fontsize=6.5, color="#2e7d32")
ax.text(cin, -2.4, "centering: $q+\\frac{N-1}{2s}=\\frac{N_0-1}{2}$  $\\Rightarrow$  $g_n=\\frac{n}{s}+\\frac{N_0-1}{2}-\\frac{N-1}{2s}$", ha="center", va="top", fontsize=6.6)
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
fig.savefig('paper_output/figs/fig_grid.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_grid.png', dpi=300, bbox_inches="tight"); print("saved q=%.3f g=%s" % (q, np.round(g, 3)))
