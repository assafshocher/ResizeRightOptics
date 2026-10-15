# Teaser: a photograph upscaled x2 and downscaled x2, the classical way and the consistent way; the differences from the input.
src = open("make_fig_cycle.py").read().split("NORM = SymLogNorm")[0]
exec(src)
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7.0, "axes.titlesize": 7.0})
r = experiment("bicubic", 2)
vm = np.abs(r["dc"][c, c]).max()
fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.35))
axs[0].imshow(r["cap"][c, c], cmap="gray", vmin=0, vmax=1); axs[0].set_title("input")
axs[1].imshow(r["dc"][c, c], cmap="RdBu_r", vmin=-vm, vmax=vm); axs[1].set_title("bicubic up, AA down: error\nRMSE %.1e" % r["e_c"], fontsize=6.4)
axs[2].imshow(r["df"][c, c], cmap="RdBu_r", vmin=-vm, vmax=vm); axs[2].set_title("$\\mathbf{K}_2$ up, $\\mathbf{K}_{1/2}$ down: error\nRMSE %.0e" % r["e_f"], fontsize=6.4)
for ax in axs: ax.set_xticks([]); ax.set_yticks([])
fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.02, wspace=0.06)
fig.savefig('paper_output/figs/fig_teaser.pdf', bbox_inches="tight"); fig.savefig('paper_output/figs/fig_teaser.png', dpi=300, bbox_inches="tight"); print("saved")
