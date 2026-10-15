"""Direct-only HPatches figure: absolute success and paired advantage.

Combines the existing 8/12/16 matched-anchor summary with follow-ups.
The optional 32x version marks the feature-starved task-floor regime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "hpatches_matched_anchor_full" / "hpatches_matched_anchor_summary.json"
FOLLOWUP = HERE.parent / "hpatches_matched_anchor_x24_full" / "hpatches_matched_anchor_summary.json"
FLOORUP = HERE.parent / "hpatches_matched_anchor_x32_full" / "hpatches_matched_anchor_summary.json"
METRIC = "success_2_5_output_px"
BASES = (
    ("linear tent", "Linear", "#0072B2", "o", -0.35),
    ("Keys cubic a=-0.5", "Cubic", "#D55E00", "s", +0.35),
)


def factor_result(base: str, factor: int, original: dict, followup: dict, floorup: dict) -> dict:
    source = floorup if factor == 32 else followup if factor == 24 else original
    return source["bases"][base]["factors"][str(factor)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-32", action="store_true")
    args = parser.parse_args()
    original = json.loads(ORIGINAL.read_text())
    followup = json.loads(FOLLOWUP.read_text())
    floorup = json.loads(FLOORUP.read_text()) if args.include_32 else {}
    factors = (8, 12, 16, 24, 32) if args.include_32 else (8, 12, 16, 24)
    out = HERE / (
        "fig_hpatches_direct_combined_x32" if args.include_32
        else "fig_hpatches_direct_combined"
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (absolute, delta) = plt.subplots(1, 2, figsize=(6.75, 2.7))
    if args.include_32:
        for axis in (absolute, delta):
            axis.axvspan(28, 34, color="#F1F1F1", zorder=0)
        absolute.text(
            30.9, 69, "≈3 matches/pair\n(8 needed)", rotation=90,
            ha="center", va="center", fontsize=7.2, color="#626262"
        )
    delta_low: list[float] = []
    delta_high: list[float] = []

    for base, label, color, marker, offset in BASES:
        anchor = original["bases"][base]["anchor"][METRIC]["mean"] * 100
        x = np.array((2, *factors), dtype=float)
        results = [factor_result(base, factor, original, followup, floorup) for factor in factors]
        for law, linestyle, filled, suffix, width in (
            ("consistent telescope", "-", True, "consistent", 1.6),
            ("AA stretch", "--", False, "AA", 1.1),
        ):
            y = [anchor] + [
                result["laws"][law]["full direct"][METRIC]["mean"] * 100
                for result in results
            ]
            absolute.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=4.5,
                linewidth=width,
                label=f"{label}: {suffix}",
                zorder=4 if filled else 3,
            )

        differences = [
            result["differences"]["consistent_minus_AA:full direct"][METRIC]
            for result in results
        ]
        means = np.array([entry["mean"] for entry in differences]) * 100
        lows = np.array([entry["low"] for entry in differences]) * 100
        highs = np.array([entry["high"] for entry in differences]) * 100
        delta_low.extend(lows)
        delta_high.extend(highs)
        delta.errorbar(
            np.array(factors, dtype=float) + offset,
            means,
            yerr=[means - lows, highs - means],
            fmt=marker,
            linestyle="none",
            color=color,
            markerfacecolor=color,
            markeredgewidth=1.1,
            markersize=4.5,
            capsize=1.7,
            elinewidth=0.9,
            zorder=4,
        )

    absolute.set_title("(a) Direct homography success", loc="left", pad=5)
    absolute.set_ylabel("Success (%)")
    absolute.set_xlabel("Downscale factor")
    absolute.set_xlim(0.8, 34 if args.include_32 else 25.2)
    absolute.set_ylim(0, 100)
    absolute.set_xticks((2, *factors), [f"{factor}×" for factor in (2, *factors)])
    absolute.set_yticks((0, 25, 50, 75, 100))
    absolute.grid(axis="y", color="#DADADA", linewidth=0.5)

    delta.axhline(0, color="#555555", linewidth=0.8, zorder=1)
    delta.set_title("(b) Matched-kernel difference", loc="left", pad=5)
    delta.set_ylabel("Consistent − AA (points)")
    delta.set_xlabel("Downscale factor")
    delta.set_xlim(6.8, 34 if args.include_32 else 25.2)
    delta.set_xticks(factors, [f"{factor}×" for factor in factors])
    bottom = min(-2.5, np.floor(min(delta_low) - 0.5))
    top = max(6.5, np.ceil(max(delta_high) + 0.5))
    delta.set_ylim(bottom, top)
    delta.grid(axis="y", color="#DADADA", linewidth=0.5)

    handles, labels = absolute.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.89, wspace=0.31)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
