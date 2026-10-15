"""Two-protocol scale sweep for the HPatches resize-path audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = [
    "Camera-consistent Gaussian",
    "Pillow Lanczos-3 AA",
    "OpenCV area",
    "OpenCV bicubic",
    "OpenCV Lanczos-4",
]
COLORS = {
    "Camera-consistent Gaussian": "#D55E00",
    "Pillow Lanczos-3 AA": "#0072B2",
    "OpenCV area": "#009E73",
    "OpenCV bicubic": "#CC79A7",
    "OpenCV Lanczos-4": "#56B4E9",
}
MARKERS = ["D", "o", "s", "^", "v"]


def read_summary(factor: int, normalized: bool) -> dict:
    suffix = "_norm" if normalized else ""
    path = ROOT / f"hpatches_geometry_x{factor}{suffix}" / "hpatches_geometry_summary.json"
    return json.loads(path.read_text())


def add_panel(axis: plt.Axes, factors: list[int], normalized: bool) -> None:
    summaries = [read_summary(factor, normalized) for factor in factors]
    metric = "success_2_5_output_px" if normalized else "success_10px"
    for index, method in enumerate(METHODS):
        entries = [summary["groups"]["all"][method]["chain"][metric] for summary in summaries]
        mean = 100 * np.array([entry["mean"] for entry in entries])
        lower = 100 * np.array([entry["low"] for entry in entries])
        upper = 100 * np.array([entry["high"] for entry in entries])
        axis.errorbar(
            factors,
            mean,
            yerr=[mean - lower, upper - mean],
            color=COLORS[method],
            marker=MARKERS[index],
            linestyle="-" if index <= 2 else "--",
            linewidth=1.6 if index == 0 else 1.1,
            markersize=4.3,
            elinewidth=0.65,
            capsize=1.6,
            label=method,
            zorder=10 - index,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(factors, [str(factor) for factor in factors])
    axis.set_xlim(min(factors) / 1.15, max(factors) * 1.1)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Downscale factor")
    axis.grid(alpha=0.13)
    axis.set_axisbelow(True)
    axis.set_title(
        "10 original px; RANSAC 5 original px"
        if not normalized else "2.5 output px; RANSAC 1.25 output px",
        fontsize=8,
        pad=8,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.75, 3.25), sharey=True)
    add_panel(left, [2, 4, 8, 12, 16], False)
    add_panel(right, [4, 8, 12, 16], True)
    left.set_ylabel("Five-step homography success (%)")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.27, top=0.91, wspace=0.17)
    fig.savefig(ROOT / "hpatches_geometry_scale_sweep.pdf")
    fig.savefig(ROOT / "hpatches_geometry_scale_sweep.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
