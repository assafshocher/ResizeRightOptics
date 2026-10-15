"""Ordinary-bicubic Set5 appendix: latest untapered realization, all scales.

Read-only experiment summaries; do not use the superseded ridge/tapered curves.
The approximate anchor, hard retained-band boundary, and grid qualification
must be described in the caption. Share the main figure's publication style.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from gen_fig_zssr_six_panel_paper import RC, STYLES, IMAGES, STEPS, SCALES, digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_panels():
    panels, hashes = {}, {}
    for scale in SCALES:
        path = ROOT / f"untapered_set5_x{scale}_comparison.json"
        data = json.loads(path.read_text())
        assert data["passed"] and data["complete"] and data["scale"] == scale
        assert data["images"] == IMAGES and data["m"] == STEPS.tolist()
        assert data["seed"] == 0
        reports = data["per_image"]
        assert [r["image"] for r in reports] == IMAGES
        means = {}
        for label, key in dict(ours="untapered", aa="aa", grid="grid").items():
            values = np.array([r["psnr_y"][key] for r in reports], dtype=float)
            assert values.shape == (5, 10) and np.isfinite(values).all()
            means[label] = values.mean(0)
            assert np.allclose(means[label], data["mean_psnr_y"][key], rtol=0, atol=1e-10)
        baseline = float(np.mean([r["baseline_y"] for r in reports]))
        assert abs(baseline-data["baseline_mean_y"]) < 1e-10
        panels[scale] = dict(means=means, baseline=baseline)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    return panels, hashes


def draw():
    panels, hashes = load_panels()
    limits = {}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(7., 1.7), sharex=True)
        fig.subplots_adjust(left=.088, right=.797, bottom=.15, top=.97, wspace=.25)
        texts = []
        for ax, scale in zip(axes, SCALES):
            panel = panels[scale]
            for method in ("grid", "aa", "ours"):
                style = STYLES[method]
                ax.plot(STEPS, panel["means"][method], **style, markersize=1.9,
                        markeredgewidth=.5,
                        markerfacecolor=style["color"] if method == "ours" else "white",
                        zorder=4 if method == "ours" else 3)
            ax.axhline(panel["baseline"], color=".52", linewidth=.85,
                       linestyle=(0, (4, 2.3)), zorder=2)
            values = np.concatenate([*panel["means"].values(), [panel["baseline"]]])
            lo, hi = float(values.min()), float(values.max())
            span = max(hi-lo, 1.)
            limits[f"x{scale}"] = [lo-.04*span, hi+.06*span]
            ax.set_ylim(limits[f"x{scale}"])
            ax.set_xlim(.65, 10.35)
            ax.set_xticks([1, 4, 7, 10])
            ax.set_xticks(STEPS, minor=True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))
            ax.tick_params(which="major", length=2.5, width=.6, pad=1.8)
            ax.tick_params(which="minor", length=1.3, width=.4)
            ax.grid(axis="y", color=".88", linewidth=.45, zorder=0)
            ax.set_axisbelow(True)
            texts.append(ax.text(.045, .065, rf"$\times{scale}$", transform=ax.transAxes,
                                 fontsize=10.5, va="bottom", ha="left"))
        texts.append(fig.text(.025, .56, "Ordinary\nbicubic-AA", fontsize=10,
                              rotation=90, ha="center", va="center", linespacing=1.05))
        texts.append(fig.text(.823, .975, "Set5", fontsize=10.5, fontweight="bold", va="top"))
        texts.append(fig.text(.823, .835, "Y-PSNR (dB)", fontsize=9.2, va="top"))
        handles = [Line2D([], [], **STYLES[m], markersize=2.3, markeredgewidth=.5,
                          markerfacecolor=STYLES[m]["color"] if m == "ours" else "white")
                   for m in ("ours", "aa", "grid")]
        handles.append(Line2D([], [], color=".52", linewidth=.85, linestyle=(0, (4, 2.3))))
        texts.append(fig.legend(handles, ["Ours", "AA", "AA (PyTorch grid)", "Bicubic interpolation"],
                                loc="upper left", bbox_to_anchor=(.814, .660), fontsize=8.1,
                                handlelength=1.65, handletextpad=.5, labelspacing=.65,
                                borderaxespad=0, borderpad=0))
        texts.append(fig.text(.823, .115, "Learned steps $m$", fontsize=9, va="top"))
        fig.canvas.draw()
        renderer, bounds = fig.canvas.get_renderer(), fig.bbox
        for ax in axes:
            texts.extend(ax.get_xticklabels()+ax.get_yticklabels())
        for artist in texts:
            if artist.get_visible():
                box = artist.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= bounds.width and box.y1 <= bounds.height, (artist, box)
        name = "fig_zssr_regular_bicubic_appendix"
        for extension in ("pdf", "png"):
            fig.savefig(HERE / f"{name}.{extension}", dpi=300, facecolor="white")
        plt.close(fig)
    report = dict(passed=True, source_sha256=hashes, figure_inches=[7., 1.7],
                  images=IMAGES, steps=STEPS.tolist(), seed=0, panel_ylim=limits,
                  independent_y_axes=True, data_modified=False,
                  qualification="Latest untapered realization; hard retained-band integration; approximate ordinary-bicubic anchor. Grid arm modifies downsampling only.")
    (HERE / f"{name}_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    draw()
