"""Set5 x2: equal-image mean PSNR; optional archived two-panel comparison.

Read saved measurements only; require all five images and every planned case.
The original baby no-AA diagnostics are retained in their source files but
excluded from this three-arm comparison at the user's explicit request.
No confidence intervals are inferred from a single training seed.
"""
from __future__ import annotations

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
IMAGES = ("baby", "bird", "butterfly", "head", "woman")
METHODS = ("consistent", "aa_bandlimited", "aa")
STEPS = np.arange(1, 6)
STYLE = {
    "consistent": ("Ours", "#D55E00", "D", "-"),
    "aa_bandlimited": ("AA (same band)", "#0072B2", "o", "--"),
    "aa": ("RR bicubic-AA", "#009E73", "^", "-."),
}


def aggregate(run):
    run = run.resolve()
    values = np.zeros((len(IMAGES), len(METHODS), len(STEPS)))
    rgb = np.zeros_like(values)
    baseline = np.zeros(len(IMAGES))
    source_hashes, per_image_rows = {}, []
    expected = {"steps": 10000, "lr": 0.0001, "batch": 1, "crop": 64,
                "radius": 64, "width": 64, "depth": 8, "seed": 0,
                "residual": True, "input_kind": "bandlimited"}
    for i, name in enumerate(IMAGES):
        path = run / f"{name}_seed0" / "results.json"
        raw = path.read_bytes()
        result = json.loads(raw)
        config = result["config"]
        assert config["image"] == name
        assert config.get("scale", 2) == 2
        assert config.get("rr_weight_device", "cpu") == "cpu"
        for key, value in expected.items():
            assert config[key] == value, (name, key, config[key], value)
        rows = {(r["m"], r["method"]): r for r in result["results"]}
        assert len(rows) == len(result["results"]), "Duplicate measurements"
        for m in STEPS:
            assert len({rows[int(m), method]["coordinate_sha256"] for method in METHODS}) == 1
        baseline[i] = result["bicubic"]["psnr_y"]
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        for j, method in enumerate(METHODS):
            for k, m in enumerate(STEPS):
                row = rows[int(m), method]
                values[i, j, k] = row["psnr_y"]
                rgb[i, j, k] = row["psnr_rgb"]
                per_image_rows.append({"image": name, "m": int(m), "method": method,
                                       "psnr_y": row["psnr_y"], "psnr_rgb": row["psnr_rgb"],
                                       "bicubic_y": baseline[i],
                                       "gain_over_bicubic_y": row["psnr_y"]-baseline[i]})
        assert values[i, 0, 0] == values[i, 1, 0], "One-step anchor must agree"
    assert np.isfinite(values).all()
    means = values.mean(axis=0)
    deltas = values[:, :1, :] - values[:, 1:, :]
    summary = {
        "protocol": {**expected, "scale": 2, "images": list(IMAGES), "n_images": 5,
                     "m": STEPS.tolist(), "methods": list(METHODS),
                     "aggregation": "Arithmetic mean of per-image PSNR in dB; equal image weights",
                     "differences": "Per-image paired PSNR differences, then arithmetic mean",
                     "border_shave": 2, "no_back_projection": True,
                     "evaluation": "Clipped float RGB, BT.601 Y, peak 1; common filtered bicubic-AA input",
                     "uncertainty": "One seed per image. No training-seed error bars or significance claim.",
                     "excluded_diagnostics": "Baby no-AA retained in original results; not in this plot"},
        "baseline_mean_y": float(baseline.mean()),
        "baseline_per_image_y": dict(zip(IMAGES, baseline.tolist())),
        "methods": {
            method: {"mean_psnr_y": means[j].tolist(),
                     "mean_psnr_rgb": rgb[:, j, :].mean(axis=0).tolist(),
                     "per_image_psnr_y": dict(zip(IMAGES, values[:, j, :].tolist()))}
            for j, method in enumerate(METHODS)
        },
        "advantages": {
            method: {"mean": deltas[:, j, :].mean(axis=0).tolist(),
                     "min_across_images": deltas[:, j, :].min(axis=0).tolist(),
                     "max_across_images": deltas[:, j, :].max(axis=0).tolist(),
                     "wins": (deltas[:, j, :] > 1e-9).sum(axis=0).tolist(),
                     "ties": (np.abs(deltas[:, j, :]) <= 1e-9).sum(axis=0).tolist(),
                     "per_image": dict(zip(IMAGES, deltas[:, j, :].tolist()))}
            for j, method in enumerate(METHODS[1:])
        },
        "source_results_sha256": source_hashes,
    }
    (HERE / "set5_x2_summary.json").write_text(json.dumps(summary, indent=2))
    with (HERE / "set5_x2_per_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    with (HERE / "set5_x2_means.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["m", *METHODS, "bicubic", "ours_minus_matched_AA", "ours_minus_RR_AA"])
        for k, m in enumerate(STEPS):
            writer.writerow([int(m), *means[:, k], baseline.mean(), *deltas[:, :, k].mean(axis=0)])
    return summary


def plot(summary, wide=False):
    fig, (absolute, delta) = plt.subplots(1, 2, figsize=(6.75, 2.8) if wide else (3.45, 2.55))
    for method in METHODS:
        label, color, marker, linestyle = STYLE[method]
        absolute.plot(STEPS, summary["methods"][method]["mean_psnr_y"],
                      color=color, marker=marker, linestyle=linestyle,
                      markersize=2.3 if not wide else 3.0,
                      markerfacecolor=color if method == "consistent" else "white",
                      markeredgewidth=0.65, linewidth=1.5 if method == "consistent" else 1.15,
                      label=label, zorder=5 if method == "consistent" else 3)
        if method != "consistent":
            delta.plot(STEPS, summary["advantages"][method]["mean"],
                       color=color, marker=marker, linestyle=linestyle,
                       markerfacecolor="white", markeredgewidth=0.65,
                       markersize=2.3 if not wide else 3.0, linewidth=1.25)
    absolute.axhline(summary["baseline_mean_y"], color="0.55", linestyle=":",
                     linewidth=1.0, label="Bicubic only")
    delta.axhline(0, color="0.45", linewidth=0.7, zorder=1)
    absolute.set_title("(a) PSNR (dB)", loc="left", pad=5)
    delta.set_title("(b) Gain (dB)", loc="left", pad=5)
    for ax in (absolute, delta):
        ax.set_xlim(.82, 5.18)
        ax.set_xticks(STEPS)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="0.86", linewidth=.5)
        ax.tick_params(pad=2)
    handles, labels = absolute.get_legend_handles_labels()
    if wide:
        absolute.set_ylabel("Mean Y-PSNR")
        delta.set_ylabel("Ours − AA")
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .00),
                   ncol=4, handlelength=1.5, columnspacing=1.1)
        fig.subplots_adjust(left=.08, right=.99, bottom=.29, top=.79, wspace=.30)
        label_y = .17
        name = "fig_set5_zssr_x2_wide"
    else:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015),
                   ncol=2, handlelength=1.3, columnspacing=1.0, labelspacing=.45,
                   handletextpad=.45)
        fig.subplots_adjust(left=.115, right=.985, bottom=.40, top=.79, wspace=.35)
        label_y = .265
        name = "fig_set5_zssr_x2_two_panel"
    fig.text(.5, .97, r"Set5 mean · ZSSR $\times 2$", ha="center", va="top", fontsize=10)
    fig.text(.5, label_y, r"Learned steps $m$", ha="center", va="center", fontsize=10)
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_absolute(summary):
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for method in METHODS:
        label, color, marker, linestyle = STYLE[method]
        ax.plot(STEPS, summary["methods"][method]["mean_psnr_y"],
                color=color, marker=marker, linestyle=linestyle,
                markersize=2.8, markeredgewidth=.65,
                markerfacecolor=color if method == "consistent" else "white",
                linewidth=1.6 if method == "consistent" else 1.2,
                label=label, zorder=5 if method == "consistent" else 3)
    ax.axhline(summary["baseline_mean_y"], color="0.55", linestyle=":",
               linewidth=1.0, label="Bicubic only")
    ax.set_xlim(.82, 5.18)
    ax.set_xticks(STEPS)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_ylabel("Y-PSNR (dB)")
    ax.set_xlabel(r"Learned steps $m$", labelpad=4)
    ax.grid(axis="y", color="0.86", linewidth=.5)
    ax.tick_params(pad=2)
    fig.text(.57, .975, r"Set5 mean · ZSSR $\times 2$", ha="center", va="top", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.54, .01),
               ncol=2, handlelength=1.5, columnspacing=1.1, labelspacing=.45,
               handletextpad=.45)
    fig.subplots_adjust(left=.18, right=.985, bottom=.36, top=.86)
    fig.savefig(HERE / "fig_set5_zssr_x2.pdf")
    fig.savefig(HERE / "fig_set5_zssr_x2.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "first_image_residual_v1")
    parser.add_argument("--two-panel", action="store_true", help="Also recreate archived comparison layouts")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5, "axes.titlesize": 10.2, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    summary = aggregate(args.run)
    plot_absolute(summary)
    if args.two_panel:
        plot(summary)
        plot(summary, wide=True)
    print(json.dumps({"baseline_mean_y": summary["baseline_mean_y"],
                      "advantages": summary["advantages"]}, indent=2))


if __name__ == "__main__":
    main()
