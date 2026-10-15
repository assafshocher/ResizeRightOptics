"""Set14 counterpart of the BSD100 six-panel ZSSR figure.

The curves are recomputed from every completed per-image result file and
checked against the server summary. The plotted observation families match
the existing Set5 and BSD100 six-panel figures.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from gen_fig_zssr_bsd100 import FAMILIES, METHODS, RC, SCALES, STEPS, STYLES


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "set14_inputs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panels():
    summary_path = INPUTS / "summary.json"
    summary = {}
    for row in json.loads(summary_path.read_text()):
        if row["experiment"] != "zssr" or row["dataset"] != "Set14":
            continue
        key = (row["family"], int(row["scale"]), row["mode"],
               int(row["m"]), row["method"])
        assert key not in summary, ("duplicate summary row", key)
        summary[key] = row

    hashes = {str(summary_path.relative_to(HERE)): digest(summary_path)}
    panels = {}
    image_sets = []
    compared_rows = set()
    for family in FAMILIES:
        for scale in SCALES:
            cell_dir = INPUTS / "zssr" / "Set14" / family / f"x{scale}"
            files = sorted(cell_dir.glob("*/results.json"))
            assert len(files) == 14, (family, scale, "expected 14 images", len(files))
            image_ids = [path.parent.name for path in files]
            assert len(set(image_ids)) == 14, (family, scale, "duplicate image")
            image_sets.append(set(image_ids))
            methods = METHODS[family]
            per_image = {name: [] for name in methods}
            baselines = []

            for path in files:
                record = json.loads(path.read_text())
                assert record["complete"] is True, path
                assert record["config"]["image"] == path.parent.name, path
                assert int(record["config"]["scale"]) == scale, path
                assert int(record["config"]["seed"]) == 0, path
                assert len(record["results"]) == 30, path
                keyed = {}
                for row in record["results"]:
                    assert row["dataset"] == "Set14" and row["image"] == path.parent.name, path
                    assert row["family"] == family and int(row["scale"]) == scale, path
                    assert int(row["seed"]) == 0, path
                    key = (int(row["m"]), row["method"])
                    assert key not in keyed, (path, key)
                    value = float(row["psnr_y"])
                    assert math.isfinite(value), (path, key)
                    keyed[key] = value
                assert set(keyed) == {(int(m), method) for m in STEPS
                                      for method in methods.values()}, path
                for name, method in methods.items():
                    per_image[name].append([keyed[int(m), method] for m in STEPS])
                baseline = float(record["bicubic"]["psnr_y"])
                assert math.isfinite(baseline), path
                baselines.append(baseline)
                hashes[str(path.relative_to(HERE))] = digest(path)

            means = {}
            for name, method in methods.items():
                values = np.asarray(per_image[name], dtype=float)
                assert values.shape == (14, 10)
                means[name] = values.mean(axis=0)
                for step, value in zip(STEPS, means[name]):
                    key = (family, scale, "float", int(step), method)
                    reference = summary[key]
                    assert reference["complete_dataset"] is True, key
                    assert reference["n_images"] == 14, key
                    assert abs(float(reference["mean_psnr_y"]) - float(value)) < 1e-9, (
                        "summary mismatch", key, value, reference["mean_psnr_y"])
                    compared_rows.add(key)
            assert np.array_equal(means["ours"][:1], means["aa"][:1]), (
                family, scale, "m=1 anchor")
            panels[family, scale] = {"means": means, "baseline": float(np.mean(baselines))}

    assert all(ids == image_sets[0] for ids in image_sets[1:]), "image sets differ"
    target_rows = {key for key in summary if key[0] in FAMILIES and
                   key[1] in SCALES and key[2] == "float"}
    assert compared_rows == target_rows and len(compared_rows) == 180, (
        "missing/extra summary rows", len(compared_rows), len(target_rows))

    outside = []
    for scale in SCALES:
        for step in (1, 10):
            for method in ("consistent_untapered", "aa", "aa_grid_literal"):
                key = ("bicubic", scale, "float", step, method)
                row = summary[key]
                assert row["complete_dataset"] is True and row["n_images"] == 14, key
                outside.append({"family": "bicubic", "scale": scale, "m": step,
                                "method": method, "mean_psnr_y": float(row["mean_psnr_y"])})
    return panels, hashes, sorted(image_sets[0]), outside


def draw(height=2.6):
    assert height >= 2.5
    panels, hashes, images, outside = load_panels()
    panel_ylim = {}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 3, figsize=(7.0, height), sharex=True)
        fig.subplots_adjust(left=.088, right=.797, bottom=.083, top=.980,
                            wspace=.25, hspace=.17)
        margin_artists, sidebar_artists, row_artists = [], [], []
        for row, family in enumerate(FAMILIES):
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
                span = max(hi - lo, 1.)
                limits = [lo - .04 * span, hi + .06 * span]
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
                margin_artists.append(ax.text(.045, .065, rf"$\times{scale}$",
                                              transform=ax.transAxes, fontsize=10.5,
                                              va="bottom", ha="left"))
            pos = axes[row, 0].get_position()
            label = "Band-limited\nbicubic" if row == 0 else "Gaussian\n$\\sigma=2$"
            row_artists.append(fig.text(.025, (pos.y0 + pos.y1) / 2, label,
                                        fontsize=10, rotation=90, ha="center", va="center",
                                        multialignment="center", linespacing=1.05))

        sidebar_artists.append(fig.text(.823, .965, "Set14", fontsize=10.5,
                                         fontweight="bold", va="top"))
        sidebar_artists.append(fig.text(.823, .873, "Y-PSNR (dB)", fontsize=9.2, va="top"))
        handles = []
        for method in ("ours", "aa", "grid"):
            style = STYLES[method]
            handles.append(Line2D([], [], **style, markersize=2.3, markeredgewidth=.5,
                                  markerfacecolor=style["color"] if method == "ours" else "white"))
        handles.append(Line2D([], [], color=".52", linewidth=.85, linestyle=(0, (4, 2.3))))
        sidebar_artists.append(fig.legend(handles,
                              ["Camera law", "AA", "AA (PyTorch grid)", "Bicubic interpolation"],
                              loc="upper left", bbox_to_anchor=(.814, .720),
                              fontsize=8.1, handlelength=1.65, handletextpad=.5,
                              labelspacing=1.2, borderaxespad=0, borderpad=0))
        sidebar_artists.append(fig.text(.823, .205, "Learned steps $m$", fontsize=9, va="top"))
        sidebar_artists.append(fig.text(.823, .127, "$1\\;\\longrightarrow\\;10$",
                                        fontsize=9, va="top", color=".35"))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        for artist in margin_artists + sidebar_artists + row_artists:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= bounds.width and box.y1 <= bounds.height, (
                "artist outside canvas", artist, box)
        data_right = max(ax.bbox.x1 for ax in axes.flat)
        data_left = min(ax.bbox.x0 for ax in axes.flat)
        assert all(a.get_window_extent(renderer).x0 > data_right for a in sidebar_artists)
        assert all(a.get_window_extent(renderer).x1 < data_left for a in row_artists)
        for ax in axes.flat:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                if label.get_visible():
                    box = label.get_window_extent(renderer)
                    assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= bounds.width and box.y1 <= bounds.height

        name = "fig_zssr_set14_six_panel"
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"{name}.{ext}", dpi=300, facecolor="white")
        plt.close(fig)

    report = {
        "passed": True,
        "source_sha256": hashes,
        "figure_inches": [7., height],
        "rows": list(FAMILIES),
        "columns": list(SCALES),
        "independent_y_axes": True,
        "panel_ylim": panel_ylim,
        "steps": STEPS.tolist(),
        "images": images,
        "images_per_panel": 14,
        "validated_summary_rows": 180,
        "total_panels": 6,
        "method_curves": 18,
        "baseline_lines": 6,
        "data_modified": False,
        "all_labels_within_canvas": True,
        "bicubic_family_not_plotted_m1_m10": outside,
        "qualification": "14-image means, one seed; final RGB clipping. Bicubic-observation results are not part of this Set5-matched six-panel comparison but remain in the source snapshot.",
    }
    (HERE / f"{name}_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=2.6)
    args = parser.parse_args()
    draw(args.height)
