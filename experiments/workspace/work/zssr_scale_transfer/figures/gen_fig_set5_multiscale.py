"""Validate and plot complete fixed multiscale measurements; no missing-case means."""
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
METHODS = ["consistent", "aa_bandlimited", "aa", "torch_aa"]
STEPS = list(range(1, 11))
STYLE = {
    "consistent": ("Ours", "#D55E00", "D", "-"),
    "aa_bandlimited": ("AA (same band)", "#0072B2", "o", "--"),
    "aa": ("RR bicubic-AA", "#009E73", "^", "-."),
    "torch_aa": ("PyTorch bicubic-AA", "#CC79A7", "s", (0, (3, 1, 1, 1))),
}


def source_run(scale, method, m):
    if scale != 2:
        return f"set5_x{scale}_m1_10"
    if method == "torch_aa":
        return "set5_x2_torch_m1_10"
    return "first_image_residual_v1" if m <= 5 else "set5_x2_m6_10"


def aggregate(scale):
    values = np.empty((5, 4, 10))
    baseline = np.empty(5)
    cache, hashes, records = {}, {}, []
    expected = {"steps": 10000, "lr": .0001, "batch": 1, "crop": 64,
                "width": 64, "depth": 8, "seed": 0, "residual": True,
                "input_kind": "bandlimited", "radius": 32*scale}
    cores = {2: [64]*5, 3: [64, 64, 64, 64, 60], 4: [64, 56, 48, 54, 41]}[scale]
    for i, image in enumerate(IMAGES):
        coords = set()
        for j, method in enumerate(METHODS):
            for k, m in enumerate(STEPS):
                run = source_run(scale, method, m)
                path = ROOT / "runs" / run / f"{image}_seed0" / "results.json"
                if path not in cache:
                    raw = path.read_bytes()
                    result = json.loads(raw)
                    config = result["config"]
                    assert config["image"] == image
                    assert config.get("scale", 2) == scale
                    assert config.get("rr_weight_device", "cpu") == ("cpu" if scale == 2 else "cuda")
                    assert config.get("crop_margin", 0) == (0 if run == "first_image_residual_v1" else 16)
                    for key, value in expected.items():
                        assert config[key] == value, (path, key, config[key], value)
                    rows = {(r["m"], r["method"]): r for r in result["results"]}
                    assert len(rows) == len(result["results"]), (path, "duplicates")
                    cache[path] = (result, rows)
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
                result, rows = cache[path]
                row = rows[m, method]  # Missing measurements are errors, not zeros.
                assert row["core_size"] == cores[i]
                coords.add(row["coordinate_sha256"])
                values[i, j, k] = row["psnr_y"]
                if j == 0 and k == 0:
                    baseline[i] = result["bicubic"]["psnr_y"]
                assert abs(baseline[i]-result["bicubic"]["psnr_y"]) < 1e-7
                records.append({"scale": scale, "image": image, "m": m,
                    "method": method, "psnr_y": row["psnr_y"], "psnr_rgb": row["psnr_rgb"],
                    "bicubic_y": baseline[i], "gain_over_bicubic_y": row["psnr_y"]-baseline[i],
                    "source": str(path.relative_to(ROOT))})
        # RNG is reset identically for every m/method. Dataset-size adaptation
        # may change this hash between images/scales, never between methods.
        assert len(coords) == 1, (scale, image, "crop mismatch", coords)
        assert values[i, 0, 0] == values[i, 1, 0], "Shared one-step anchor"
    assert np.isfinite(values).all()
    means = values.mean(0)
    delta = values[:, :1, :] - values[:, 1:, :]
    summary = {"scale": scale, "m": STEPS, "images": IMAGES, "methods": METHODS,
        "aggregation": "Arithmetic mean of five per-image Y-PSNR values in dB",
        "one_seed": True, "no_back_projection": True, "border_shave": scale,
        "baseline_mean_y": float(baseline.mean()),
        "baseline_per_image_y": dict(zip(IMAGES, baseline.tolist())),
        "mean_psnr_y": dict(zip(METHODS, means.tolist())),
        "per_image_psnr_y": {method: dict(zip(IMAGES, values[:, j].tolist()))
                             for j, method in enumerate(METHODS)},
        "ours_minus": {method: {"mean": delta[:, j].mean(0).tolist(),
                               "wins": (delta[:, j] > 1e-9).sum(0).tolist(),
                               "min": delta[:, j].min(0).tolist(),
                               "max": delta[:, j].max(0).tolist()}
                       for j, method in enumerate(METHODS[1:])},
        "source_results_sha256": hashes,
        "software_arm_caveat": "Native PyTorch training grid and filter; RR upsampling shared. Not a pure AA-law comparison.",
        "input_caveat": "Common filtered bicubic-AA, not standard literal-bicubic benchmark scores."}
    (HERE / f"set5_x{scale}_m1_10_summary.json").write_text(json.dumps(summary, indent=2))
    with (HERE / f"set5_x{scale}_m1_10_per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (HERE / f"set5_x{scale}_m1_10_means.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", *METHODS, "bicubic", *[f"ours_minus_{v}" for v in METHODS[1:]]])
        for k, m in enumerate(STEPS):
            writer.writerow([m, *means[:, k], baseline.mean(), *delta[:, :, k].mean(0)])
    return summary


def panel(ax, summary, methods=METHODS):
    for method in methods:
        label, color, marker, style = STYLE[method]
        ax.plot(STEPS, summary["mean_psnr_y"][method], label=label,
                color=color, marker=marker, linestyle=style, markersize=2.5,
                markeredgewidth=.65, markerfacecolor=color if method == "consistent" else "white",
                linewidth=1.6 if method == "consistent" else 1.2,
                zorder=5 if method == "consistent" else 3)
    ax.axhline(summary["baseline_mean_y"], color=".5", linestyle=":",
               linewidth=1.0, label="Bicubic only")
    ax.set_xlim(.7, 10.3)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xticks(STEPS, minor=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color=".86", linewidth=.5)
    ax.tick_params(pad=2)
    ax.set_xlabel(r"Learned steps $m$", labelpad=4)
    ax.set_title(rf"$\times {summary['scale']}$", pad=6)


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", pad_inches=.04)
    fig.savefig(HERE / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4], choices=[2, 3, 4])
    parser.add_argument("--without-pytorch", action="store_true",
                        help="Make an additional display-only variant; retain full data and original plots")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 8.5, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    summaries = [aggregate(scale) for scale in args.scales]
    shown_methods = [m for m in METHODS if not args.without_pytorch or m != "torch_aa"]
    suffix = "_no_pytorch" if args.without_pytorch else ""
    for summary in summaries:
        fig, ax = plt.subplots(figsize=(3.45, 2.75))
        panel(ax, summary, shown_methods)
        ax.set_ylabel("Mean Y-PSNR (dB)")
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
                   handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(.54, .0))
        fig.subplots_adjust(left=.18, right=.98, bottom=.37, top=.89)
        save(fig, f"fig_set5_zssr_x{summary['scale']}_m1_10{suffix}")
    if len(summaries) > 1:
        fig, axes = plt.subplots(1, len(summaries), figsize=(7.0, 2.65))
        for ax, summary in zip(axes, summaries):
            panel(ax, summary, shown_methods)
        axes[0].set_ylabel("Mean Y-PSNR (dB)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                   handlelength=1.6, columnspacing=1.4, bbox_to_anchor=(.53, .0))
        fig.subplots_adjust(left=.085, right=.99, bottom=.34, top=.89, wspace=.30)
        save(fig, f"fig_set5_zssr_multiscale_m1_10{suffix}")
    print(json.dumps({s["scale"]: {"baseline": s["baseline_mean_y"],
        "means": s["mean_psnr_y"]} for s in summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
