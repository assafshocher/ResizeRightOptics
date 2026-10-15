"""Compact, direct-only HPatches effect plot for the matched 1/2-kernel audit.

Reads the already computed sequence-bootstrap summary; it does not rerun or
alter the experiment. The plotted quantity is a paired *difference*, so its
axis may be narrow without truncating an absolute success-rate axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "hpatches_matched_anchor_full" / "hpatches_matched_anchor_summary.json"
OUT = HERE / "fig_hpatches_direct_advantage"
FACTOR_ORDER = (8, 12, 16)
BASES = (
    ("linear tent", "Linear", "#0072B2", "o", -0.12),
    ("Keys cubic a=-0.5", "Cubic", "#D55E00", "s", +0.12),
)
METRIC = "success_2_5_output_px"


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    ax.axhline(0, color="#444444", linewidth=0.85, zorder=1)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)

    for base, label, color, marker, offset in BASES:
        xs = np.arange(len(FACTOR_ORDER), dtype=float) + offset
        entries = [
            summary["bases"][base]["factors"][str(factor)]["differences"]
            ["consistent_minus_AA:full direct"][METRIC]
            for factor in FACTOR_ORDER
        ]
        mean = np.array([entry["mean"] for entry in entries]) * 100
        low = np.array([entry["low"] for entry in entries]) * 100
        high = np.array([entry["high"] for entry in entries]) * 100
        ax.errorbar(
            xs,
            mean,
            yerr=[mean - low, high - mean],
            fmt=marker,
            linestyle="none",
            color=color,
            ecolor=color,
            markerfacecolor="white" if marker == "s" else color,
            markeredgewidth=1.2,
            markersize=5.0,
            capsize=2.1,
            elinewidth=1.0,
            label=label,
            zorder=3,
        )

    ax.set_xticks(range(len(FACTOR_ORDER)), [f"{factor}×" for factor in FACTOR_ORDER])
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylim(-2.5, 7.0)
    ax.set_yticks((-2, 0, 2, 4, 6))
    ax.set_ylabel("Consistent − AA (percentage points)")
    ax.set_xlabel("Direct downscale factor")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.1, handletextpad=0.3)
    ax.tick_params(length=3, pad=2)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.23, top=0.96)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
