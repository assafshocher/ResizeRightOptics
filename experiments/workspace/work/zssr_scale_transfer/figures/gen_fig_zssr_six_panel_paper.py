"""Compact figure* layout: band-limited bicubic and Gaussian, all three scales.

Read-only with respect to experimental data and frozen plotting code. Validate
the complete five-image means before plotting; retain every step and baseline.
Each panel has its own labeled y-axis and data-driven limits. Side labels and legend use the vertical space already
occupied by the two rows, so no title/legend bands are added above or below.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]
STEPS = np.arange(1, 11)
SCALES = (2, 3, 4)
GAUSSIAN_KEYS = dict(ours="gaussian_consistent", aa="gaussian_aa", grid="aa_grid_gaussian")
STYLES = dict(ours=dict(color="#D55E00", marker="o", linestyle="-", linewidth=1.35),
              aa=dict(color="#0072B2", marker="^", linestyle=(0, (4, 1.5)), linewidth=1.1),
              grid=dict(color="#CC79A7", marker="s", linestyle=(0, (1.5, 1.5)), linewidth=1.05))
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
      "mathtext.fontset": "stix", "font.size": 9,
      "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
      "axes.spines.top": False, "axes.spines.right": False,
      "axes.linewidth": .65, "legend.frameon": False,
      "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curve(values):
    a = np.asarray(values, dtype=float)
    assert a.shape == (10,) and np.isfinite(a).all()
    return a


def load_panels():
    panels, hashes = {}, {}
    for scale in SCALES:
        path = HERE / f"matched_bandlimited_x{scale}_summary.json"
        d = json.loads(path.read_text())
        assert d["complete"] and d["missing_records"] == 0
        assert d["images"] == IMAGES and d["m"] == STEPS.tolist()
        assert d["mode"] == "bandlimited" and d["scale"] == scale
        means = {}
        for method in STYLES:
            per_image = np.stack([curve(d["per_image_psnr_y"][method][image]) for image in IMAGES])
            means[method] = per_image.mean(0)
            assert np.allclose(means[method], curve(d["mean_psnr_y"][method]), rtol=0, atol=1e-10)
        # Trace all archived source results; the baseline is not separately
        # serialized per-image in these early summary files.
        baselines = {image: [] for image in IMAGES}
        for relative, expected_hash in d["source_results_sha256"].items():
            source = ROOT / relative
            assert digest(source) == expected_hash, ("Changed original data", relative)
            result = json.loads(source.read_text())
            baselines[result["config"]["image"]].append(result["bicubic"]["psnr_y"])
        for image in IMAGES:
            assert baselines[image] and max(baselines[image])-min(baselines[image]) < 1e-7
        baseline = float(np.mean([baselines[image][0] for image in IMAGES]))
        assert abs(baseline-d["baseline_mean_y"]) < 1e-7
        assert means["ours"][0] == means["aa"][0]
        panels["bandlimited", scale] = dict(means=means, baseline=baseline)
        hashes[str(path.relative_to(ROOT))] = digest(path)

        path = ROOT / f"runs/gaussian_sigma2_suite_v1/summary_x{scale}.json"
        d = json.loads(path.read_text())
        assert d["passed"] and d["complete"] and d["scale"] == scale
        assert d["images"] == IMAGES and d["m"] == STEPS.tolist() and d["seed"] == 0
        assert d["anchor_sigma"] == 2.
        assert [r["image"] for r in d["per_image"]] == IMAGES
        means = {}
        for method, original_key in GAUSSIAN_KEYS.items():
            values = np.stack([curve(r["psnr_y"][original_key]) for r in d["per_image"]])
            means[method] = values.mean(0)
            assert np.allclose(means[method], curve(d["mean_psnr_y"][original_key]), rtol=0, atol=1e-10)
        baseline = float(np.mean([r["baseline_y"] for r in d["per_image"]]))
        assert abs(baseline-d["baseline_mean_y"]) < 1e-10
        assert means["ours"][0] == means["aa"][0]
        panels["gaussian", scale] = dict(means=means, baseline=baseline)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    return panels, hashes


def draw(height=2.6):
    panels, hashes = load_panels()
    panel_ylim = {}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 3, figsize=(7.0, height), sharex=True)
        fig.subplots_adjust(left=.088, right=.797, bottom=.083, top=.980,
                            wspace=.25, hspace=.17)
        artists = []
        for row, family in enumerate(("bandlimited", "gaussian")):
            for col, scale in enumerate(SCALES):
                ax = axes[row, col]
                panel = panels[family, scale]
                for method in ("grid", "aa", "ours"):
                    style = STYLES[method]
                    ax.plot(STEPS, panel["means"][method], **style, markersize=1.9,
                            markeredgewidth=.5,
                            markerfacecolor=style["color"] if method == "ours" else "white",
                            zorder=4 if method == "ours" else 3)
                ax.axhline(panel["baseline"], color=".52", linewidth=.85,
                           linestyle=(0, (4, 2.3)), zorder=2)
                ax.set_xlim(.65, 10.35)
                values = np.concatenate([*panel["means"].values(), [panel["baseline"]]])
                lo, hi = float(values.min()), float(values.max())
                span = max(hi-lo, 1.)
                limits = [lo-.04*span, hi+.06*span]
                ax.set_ylim(limits)
                panel_ylim[f"{family}_x{scale}"] = limits
                ax.set_xticks([1, 4, 7, 10])
                ax.set_xticks(STEPS, minor=True)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))
                ax.tick_params(which="major", length=2.5, width=.6, pad=1.8)
                ax.tick_params(which="minor", length=1.3, width=.4)
                ax.grid(axis="y", color=".88", linewidth=.45, zorder=0)
                ax.set_axisbelow(True)
                if row == 0:
                    ax.tick_params(axis="x", labelbottom=False)
                artists.append(ax.text(.045, .065, rf"$\times{scale}$", transform=ax.transAxes,
                                       fontsize=10.5, va="bottom", ha="left"))
            position = axes[row, 0].get_position()
            label = "Band-limited\nbicubic" if row == 0 else "Gaussian\n$\\sigma=2$"
            artists.append(fig.text(.025, (position.y0+position.y1)/2, label,
                                    fontsize=10, rotation=90, ha="center", va="center",
                                    multialignment="center", linespacing=1.05))

        # Sidebar replaces separate title/legend/x-label rows.
        artists.append(fig.text(.823, .965, "Set5", fontsize=10.5, fontweight="bold", va="top"))
        artists.append(fig.text(.823, .873, "Y-PSNR (dB)", fontsize=9.2, va="top"))
        handles = []
        for method in ("ours", "aa", "grid"):
            style = STYLES[method]
            handles.append(Line2D([], [], **style, markersize=2.3, markeredgewidth=.5,
                                  markerfacecolor=style["color"] if method == "ours" else "white"))
        handles.append(Line2D([], [], color=".52", linewidth=.85, linestyle=(0, (4, 2.3))))
        legend = fig.legend(handles, ["Ours", "AA", "AA (PyTorch grid)", "Bicubic interpolation"],
                            loc="upper left", bbox_to_anchor=(.814, .720),
                            fontsize=8.1, handlelength=1.65, handletextpad=.5,
                            labelspacing=1.2, borderaxespad=0, borderpad=0)
        artists.append(legend)
        artists.append(fig.text(.823, .205, "Learned steps $m$", fontsize=9, va="top"))
        artists.append(fig.text(.823, .127, "$1\\;\\longrightarrow\\;10$", fontsize=9, va="top", color=".35"))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        for artist in artists:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= bounds.width and box.y1 <= bounds.height, (artist, box)
        for ax in axes.flat:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                if label.get_visible():
                    box = label.get_window_extent(renderer)
                    assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= bounds.width and box.y1 <= bounds.height
        name = "fig_zssr_six_panel_paper"
        for extension in ("pdf", "png"):
            fig.savefig(HERE / f"{name}.{extension}", dpi=300, facecolor="white")
        plt.close(fig)
    report = dict(passed=True, source_sha256=hashes, figure_inches=[7.,height],
                  rows=["bandlimited", "gaussian"], columns=list(SCALES),
                  independent_y_axes=True, panel_ylim=panel_ylim,
                  steps=STEPS.tolist(), images=IMAGES,
                  total_panels=6, method_curves=18, baseline_lines=6,
                  data_modified=False, all_labels_within_canvas=True,
                  qualification="Five-image means, one seed; final RGB clipping; grid arm changes only downsampling coordinates.")
    (HERE / f"{name}_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=2.6)
    args = parser.parse_args()
    draw(args.height)
