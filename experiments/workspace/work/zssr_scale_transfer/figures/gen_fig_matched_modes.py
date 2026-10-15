"""Six matched-anchor panels; missing images never enter partial means."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]
MODES = ["bandlimited", "literal"]
METHODS = ["ours", "aa", "grid"]
STEPS = list(range(1, 11))
STYLE = {"ours": ("Ours", "#D55E00", "D", "-"),
         "aa": ("AA", "#0072B2", "o", "--"),
         "grid": ("AA (PyTorch grid)", "#CC79A7", "s", "-.")}


def source(mode, scale, image, method, m):
    if mode == "bandlimited" and method != "grid":
        key = "consistent" if method == "ours" else "aa_bandlimited"
        run = (f"set5_x{scale}_m1_10" if scale != 2 else
               "first_image_residual_v1" if m <= 5 else "set5_x2_m6_10")
    else:
        run = f"matched_{mode}_x{scale}_v1"
        key = ("aa_grid_bandlimited" if mode == "bandlimited" else
               {"ours": "consistent_literal", "aa": "aa", "grid": "aa_grid_literal"}[method])
    return ROOT / "runs" / run / f"{image}_seed0" / "results.json", key


def aggregate(mode, scale, allow_partial=False):
    values = np.full((5, 3, 10), np.nan)
    baselines = np.full(5, np.nan)
    cache, hashes, records, missing = {}, {}, [], []
    expected = {"steps": 10000, "lr": .0001, "batch": 1, "crop": 64,
                "width": 64, "depth": 8, "seed": 0, "residual": True,
                "input_kind": mode, "radius": 32*scale}
    for i, image in enumerate(IMAGES):
        coordinates = set()
        for j, method in enumerate(METHODS):
            for k, m in enumerate(STEPS):
                path, key = source(mode, scale, image, method, m)
                if not path.exists():
                    missing.append([image, method, m])
                    continue
                if path not in cache:
                    raw = path.read_bytes()
                    result = json.loads(raw)
                    config = result["config"]
                    assert config["image"] == image and config.get("scale", 2) == scale
                    for name, value in expected.items():
                        assert config[name] == value, (path, name, config[name], value)
                    rows = {(r["m"], r["method"]): r for r in result["results"]}
                    assert len(rows) == len(result["results"])
                    cache[path] = (result, rows)
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
                result, rows = cache[path]
                base = result["bicubic"]["psnr_y"]
                if np.isfinite(baselines[i]):
                    assert abs(base-baselines[i]) < 1e-7
                baselines[i] = base
                if (m, key) not in rows:
                    missing.append([image, method, m])
                    continue
                row = rows[m, key]
                assert np.isfinite(row["psnr_y"])
                values[i, j, k] = row["psnr_y"]
                coordinates.add(row["coordinate_sha256"])
                records.append(dict(mode=mode, scale=scale, image=image, m=m,
                    method=method, psnr_y=row["psnr_y"], bicubic_y=base,
                    source=str(path.relative_to(ROOT)), source_method=key,
                    shared_anchor=bool(row.get("reused_identical_anchor_run"))))
        assert len(coordinates) <= 1, (mode, scale, image, "crop schedule changed")
        if np.isfinite(values[i, :2, 0]).all():
            assert values[i, 0, 0] == values[i, 1, 0], (mode, scale, image, "anchor")
    if missing and not allow_partial:
        raise ValueError((mode, scale, "incomplete", missing[:5], len(missing)))
    # numpy mean propagates NaN; it is never a mean over fewer than five images.
    means = values.mean(0)
    def clean(a):
        return [float(v) if np.isfinite(v) else None for v in a]
    summary = {"mode": mode, "scale": scale, "m": STEPS,
        "complete": not missing, "missing_records": len(missing),
        "images": IMAGES, "mean_psnr_y": {key: clean(means[j]) for j, key in enumerate(METHODS)},
        "baseline_mean_y": float(baselines.mean()) if np.isfinite(baselines).all() else None,
        "per_image_psnr_y": {key: {image: clean(values[i, j]) for i, image in enumerate(IMAGES)}
                             for j, key in enumerate(METHODS)},
        "aggregation": "Arithmetic mean of all five per-image Y-PSNR values; one seed",
        "literal_caveat": "Literal-mode ours is an anchor-preserving ridge extension, eta=0.001; not exact sampled consistency under aliasing",
        "grid_caveat": "RR filters/normalization/reflection, PyTorch floor and half-pixel grid; NOT unmodified torch.interpolate",
        "source_results_sha256": hashes}
    (HERE / f"matched_{mode}_x{scale}_summary.json").write_text(json.dumps(summary, indent=2))
    if records:
        with (HERE / f"matched_{mode}_x{scale}_per_image.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (HERE / f"matched_{mode}_x{scale}_means.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["m", *METHODS, "bicubic"])
        for k, m in enumerate(STEPS):
            writer.writerow([m, *[summary["mean_psnr_y"][key][k] for key in METHODS],
                             summary["baseline_mean_y"]])
    return summary


def panel(ax, summary):
    for method in METHODS:
        label, color, marker, style = STYLE[method]
        values = [np.nan if v is None else v for v in summary["mean_psnr_y"][method]]
        ax.plot(STEPS, values, label=label, color=color, marker=marker,
                linestyle=style, markersize=2.2, markeredgewidth=.6,
                markerfacecolor=color if method == "ours" else "white",
                linewidth=1.5 if method == "ours" else 1.15, zorder=5 if method == "ours" else 3)
    if summary["baseline_mean_y"] is not None:
        ax.axhline(summary["baseline_mean_y"], color=".5", linestyle=":",
                   linewidth=1, label="Bicubic only")
    else:
        ax.plot([], [], color=".5", linestyle=":", label="Bicubic only")
    ax.set_xlim(.7, 10.3)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xticks(STEPS, minor=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color=".86", linewidth=.5)
    ax.tick_params(pad=2)
    ax.set_xlabel(r"Learned steps $m$", labelpad=3)
    ax.set_title(rf"$\times {summary['scale']}$", pad=5)
    if not summary["complete"]:
        ax.text(.5, .04, f"IN PROGRESS · {summary['missing_records']} missing",
                ha="center", va="bottom", transform=ax.transAxes, fontsize=7, color=".4")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    summaries = {(mode, scale): aggregate(mode, scale, args.allow_partial)
                 for mode in MODES for scale in (2, 3, 4)}
    titles = {"bandlimited": "Band-limited bicubic", "literal": "Literal bicubic (ours: regularized)"}
    for (mode, scale), summary in summaries.items():
        if not summary["complete"]:
            continue
        fig, ax = plt.subplots(figsize=(3.45, 2.8))
        panel(ax, summary)
        ax.set_ylabel("Mean Y-PSNR (dB)")
        ax.set_title(titles[mode] + rf" — $\times {scale}$", fontsize=8.5)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
                   handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(.53, .01))
        fig.subplots_adjust(left=.18, right=.98, bottom=.35, top=.89)
        save(fig, f"fig_matched_{mode}_x{scale}")
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9))
    for i, mode in enumerate(MODES):
        for j, scale in enumerate((2, 3, 4)):
            panel(axes[i, j], summaries[mode, scale])
        axes[i, 0].set_ylabel("Mean Y-PSNR (dB)")
    fig.text(.54, .98, titles["bandlimited"], ha="center", va="top", fontsize=10)
    fig.text(.54, .535, titles["literal"], ha="center", va="top", fontsize=10)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4,
               handlelength=1.6, columnspacing=1.15, bbox_to_anchor=(.53, .005))
    fig.subplots_adjust(left=.09, right=.99, bottom=.13, top=.91, wspace=.32, hspace=.78)
    complete = all(s["complete"] for s in summaries.values())
    name = "fig_set5_zssr_matched_six_panel" + ("" if complete else "_progress")
    save(fig, name)
    print(json.dumps({"complete": complete, "figure": str(HERE / f"{name}.png"),
                      "panels": [{"mode": s["mode"], "scale": s["scale"],
                                  "missing": s["missing_records"]} for s in summaries.values()]}), flush=True)


if __name__ == "__main__":
    main()
