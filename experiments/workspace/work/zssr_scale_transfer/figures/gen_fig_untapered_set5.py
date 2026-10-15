"""Plot verified Set5 x2 means from the complete untapered comparison.

Run after run_untapered_set5.py has produced its verified comparison JSON.
Every curve is an arithmetic mean of the same five image-level Y-PSNR
measurements, with one seed. No uncertainty intervals are inferred.
"""
from __future__ import annotations

import argparse
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
STEPS = list(range(1, 11))
METHODS = ["untapered", "tapered", "aa", "grid"]
# Okabe-Ito colors, with line styles and small markers for print legibility.
STYLE = {
    "untapered": ("Ours (untapered)", "#D55E00", "-", "o"),
    "tapered": ("Previous ours (tapered)", "#009E73", "--", "s"),
    "aa": ("Bicubic AA", "#0072B2", "-.", "^"),
    "grid": ("AA (PyTorch grid)", "#E69F00", ":", "D"),
}


def finite_scores(value, shape, label):
    """Reject missing, nonnumeric, nonfinite, or incorrectly shaped scores."""
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{label}: expected numeric scores with shape {shape}")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: every score must be finite")
    return array


def load_verified(path):
    summary = json.loads(path.read_text())
    if summary.get("complete") is not True or summary.get("passed") is not True:
        raise ValueError("The comparison must be complete and have passed verification")
    if summary.get("scale") != 2:
        raise ValueError("Expected a scale-2 comparison")
    if summary.get("images") != IMAGES or summary.get("m") != STEPS:
        raise ValueError("Expected all five Set5 images and steps 1 through 10")
    reports = summary.get("per_image")
    if not isinstance(reports, list) or len(reports) != len(IMAGES):
        raise ValueError("Expected exactly five per-image reports")
    if [report.get("image") for report in reports] != IMAGES:
        raise ValueError("Per-image reports must contain each Set5 image exactly once")

    means = {}
    for method in METHODS:
        scores = np.stack([
            finite_scores(report["psnr_y"][method], (len(STEPS),),
                          f"{report['image']}/{method}")
            for report in reports
        ])
        saved_mean = finite_scores(summary["mean_psnr_y"][method], (len(STEPS),),
                                   f"mean/{method}")
        recomputed_mean = scores.mean(axis=0)
        if not np.allclose(saved_mean, recomputed_mean, rtol=0, atol=1e-10):
            raise ValueError(f"{method}: saved means disagree with the five image scores")
        means[method] = recomputed_mean

    baselines = finite_scores([report["baseline_y"] for report in reports],
                              (len(IMAGES),), "per-image bicubic baselines")
    saved_baseline = finite_scores(summary["baseline_mean_y"], (), "baseline mean")
    baseline_mean = float(baselines.mean())
    if not np.isclose(saved_baseline, baseline_mean, rtol=0, atol=1e-10):
        raise ValueError("Saved baseline mean disagrees with the five image baselines")
    return means, baseline_mean


def plot(means, baseline_mean, output_dir, omit_tapered=False, scale=2, title=None):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(6.75, 4.3))
    for method in METHODS:
        if omit_tapered and method == "tapered":
            continue
        label, color, linestyle, marker = STYLE[method]
        ours = method == "untapered"
        ax.plot(STEPS, means[method], label=label, color=color,
                linestyle=linestyle, marker=marker, markersize=2.4 if ours else 2.2,
                markerfacecolor=color if ours else "white", markeredgewidth=0.65,
                linewidth=1.8 if ours else 1.45, zorder=5 if ours else 3)
    ax.axhline(baseline_mean, label="Bicubic interpolation", color="0.55",
               linestyle=(0, (5, 3)), linewidth=1.15, zorder=2)
    ax.set_title(title or f"Set5 ×{scale}", pad=9)
    ax.set_xlabel("Steps", labelpad=6)
    ax.set_ylabel("Mean Y-PSNR (dB)", labelpad=7)
    ax.set_xlim(0.8, 10.2)
    ax.set_xticks(STEPS)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.margins(y=0.1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.tick_params(pad=3)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.54, 0.015), ncol=2, handlelength=2.3,
               columnspacing=1.8, handletextpad=0.6, labelspacing=0.55)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.33)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "png"):
        suffix = "_without_old" if omit_tapered else ""
        path = output_dir / f"fig_untapered_set5_x{scale}{suffix}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "untapered_set5_x2_comparison.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--omit-tapered", action="store_true")
    args = parser.parse_args()
    means, baseline_mean = load_verified(args.input)
    for path in plot(means, baseline_mean, args.output_dir, args.omit_tapered):
        print(path.resolve())


if __name__ == "__main__":
    main()
