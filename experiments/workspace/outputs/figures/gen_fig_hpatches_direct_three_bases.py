"""Direct-only matched-anchor figure for linear, cubic, and fixed Gaussian.

Displays all tested factors rather than selecting only favorable endpoints.
Absolute success uses a full 0-100% axis; the inferential panel plots paired
sequence-bootstrap differences with a zero line. The x32 band denotes the
feature-count floor of this HPatches/SIFT protocol, not a physical limit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = {
    "common": ROOT / "hpatches_matched_anchor_full" / "hpatches_matched_anchor_summary.json",
    "x20": ROOT / "hpatches_matched_anchor_direct_x20_full" / "direct_summary.json",
    "x24": ROOT / "hpatches_matched_anchor_x24_full" / "hpatches_matched_anchor_summary.json",
    "x32": ROOT / "hpatches_matched_anchor_x32_full" / "hpatches_matched_anchor_summary.json",
    "gaussian16": ROOT / "hpatches_matched_anchor_direct_gaussian_x16_full" / "direct_summary.json",
    "gaussian_sweep": ROOT / "hpatches_matched_anchor_direct_gaussian_sweep_full" / "direct_summary.json",
}
OUT = HERE / "fig_hpatches_direct_three_bases"
FACTORS = (8, 12, 16, 20, 24, 32)
METRIC = "success_2_5_output_px"
BASES = (
    ("linear tent", "Linear", "#0072B2", "o", -0.55),
    ("Keys cubic a=-0.5", "Cubic", "#D55E00", "s", 0.0),
    ("Gaussian (x2 Nyquist MTF matched)", "Gaussian", "#009E73", "^", +0.55),
)


def result(sources: dict, base: str, factor: int) -> dict:
    if base.startswith("Gaussian"):
        key = "x20" if factor == 20 else "gaussian16" if factor == 16 else "gaussian_sweep"
    else:
        key = "x20" if factor == 20 else "x24" if factor == 24 else "x32" if factor == 32 else "common"
    return sources[key]["bases"][base]["factors"][str(factor)]


def law_metric(entry: dict, law: str) -> dict:
    value = entry["laws"][law]
    return value["full direct"][METRIC] if "full direct" in value else value[METRIC]


def difference_metric(entry: dict) -> dict:
    differences = entry["differences"]
    key = "consistent_minus_AA:full direct" if "consistent_minus_AA:full direct" in differences else "consistent_minus_AA"
    return differences[key][METRIC]


def main() -> None:
    sources = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.3,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (absolute, delta) = plt.subplots(1, 2, figsize=(6.75, 2.95))
    for axis in (absolute, delta):
        axis.axvspan(28, 34, color="#F1F1F1", zorder=0)
    absolute.text(
        30.8, 68, "≈3 matches/pair\n(8 needed)",
        rotation=90, ha="center", va="center", fontsize=7.2, color="#626262"
    )
    all_lows: list[float] = []
    all_highs: list[float] = []
    for base, label, color, marker, offset in BASES:
        anchor_source = "x20" if base.startswith("Gaussian") else "common"
        anchor = sources[anchor_source]["bases"][base]["anchor"][METRIC]["mean"] * 100
        entries = [result(sources, base, factor) for factor in FACTORS]
        x = np.array((2, *FACTORS), dtype=float)
        for law, linestyle, filled, suffix, width in (
            ("consistent telescope", "-", True, "consistent", 1.55),
            ("AA stretch", "--", False, "AA", 1.05),
        ):
            y = [anchor] + [law_metric(entry, law)["mean"] * 100 for entry in entries]
            absolute.plot(
                x, y, color=color, linestyle=linestyle, marker=marker,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color, markeredgewidth=1.0,
                markersize=4.2, linewidth=width, label=f"{label}: {suffix}",
                zorder=4 if filled else 3,
            )
        diffs = [difference_metric(entry) for entry in entries]
        means = np.array([item["mean"] for item in diffs]) * 100
        lows = np.array([item["low"] for item in diffs]) * 100
        highs = np.array([item["high"] for item in diffs]) * 100
        all_lows.extend(lows)
        all_highs.extend(highs)
        delta.errorbar(
            np.array(FACTORS, dtype=float) + offset, means,
            yerr=[means - lows, highs - means], fmt=marker, linestyle="none",
            color=color, markerfacecolor=color, markeredgewidth=1.0,
            markersize=4.2, capsize=1.6, elinewidth=0.85, zorder=4,
        )

    absolute.set_title("(a) Direct homography success", loc="left", pad=5)
    absolute.set_xlabel("Downscale factor")
    absolute.set_ylabel("Success (%)")
    absolute.set_xlim(0.8, 34)
    absolute.set_ylim(0, 100)
    absolute.set_xticks((2, *FACTORS), [f"{factor}×" for factor in (2, *FACTORS)])
    absolute.set_yticks((0, 25, 50, 75, 100))
    absolute.grid(axis="y", color="#DADADA", linewidth=0.5)

    delta.axhline(0, color="#555555", linewidth=0.8, zorder=1)
    delta.set_title("(b) Matched-kernel difference", loc="left", pad=5)
    delta.set_xlabel("Downscale factor")
    delta.set_ylabel("Consistent − AA (points)")
    delta.set_xlim(6.8, 34)
    delta.set_xticks(FACTORS, [f"{factor}×" for factor in FACTORS])
    delta.set_ylim(
        min(-4.5, np.floor(min(all_lows) - 0.5)),
        max(6.5, np.ceil(max(all_highs) + 0.5)),
    )
    delta.grid(axis="y", color="#DADADA", linewidth=0.5)

    handles, labels = absolute.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.91, wspace=0.31)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
