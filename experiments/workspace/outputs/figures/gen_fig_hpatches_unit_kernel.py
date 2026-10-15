"""Two publication figures for the predeclared unit-kernel HPatches protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BASES = (
    ("linear tent", "Linear", "#0072B2", "o", -0.55),
    ("Keys cubic a=-0.5", "Cubic", "#D55E00", "s", 0.0),
    ("Lanczos-3", "Lanczos-3", "#009E73", "^", 0.55),
)
TASKS = (
    ("direct low", "success_2_5_output_px", "Direct low-res", "Success (%, 2.5 low-res px)", "fig_hpatches_unit_direct"),
    ("down then up", "success_5px", "Down then up", "Success (%, 5 original px)", "fig_hpatches_unit_down_up"),
)


def make_figure(
    summary: dict, task: tuple[str, str, str, str, str], tag: str,
    single_column: bool = False,
) -> None:
    path, metric, task_name, ylabel, name = task
    factors = np.array(summary["protocol"]["factors"], dtype=float)
    fig, (absolute, delta) = plt.subplots(
        1, 2, figsize=(3.25, 2.35) if single_column else (6.75, 2.95)
    )
    marker_size = 2.6 if single_column else 4.1
    line_width = 1.9 if single_column else 1.55
    all_lows: list[float] = []
    all_highs: list[float] = []
    if path == "down then up":
        per_base = np.array(
            [
                [
                    summary["bases"][base]["factors"][str(int(factor))]
                    ["paths"][path]["laws"]["unit-kernel consistent"][metric]["mean"] * 100
                    for factor in factors
                ]
                for base, *_ in BASES
            ]
        )
        shared = np.mean(per_base, axis=0)
        absolute.fill_between(
            factors, np.min(per_base, axis=0), np.max(per_base, axis=0),
            color="#333333", alpha=0.15, linewidth=0, zorder=2,
        )
        absolute.plot(
            factors, shared, color="#333333", linestyle="-", marker="D",
            markerfacecolor="#333333", markeredgecolor="#333333",
            markersize=marker_size, linewidth=line_width,
            label="Consistent" if single_column else "Consistent (base mean)",
            zorder=5,
        )
    for base, label, color, marker, offset in BASES:
        entries = [
            summary["bases"][base]["factors"][str(int(factor))]["paths"][path]
            for factor in factors
        ]
        for law, suffix, linestyle, filled in (
            ("unit-kernel consistent", "consistent", "-", True),
            ("AA stretch", "AA", "--", False),
        ):
            if path == "down then up" and law == "unit-kernel consistent":
                continue
            values = [entry["laws"][law][metric]["mean"] * 100 for entry in entries]
            absolute.plot(
                factors, values,
                color=color, linestyle=linestyle, marker=marker,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=0.7 if single_column else 1,
                markersize=marker_size,
                linewidth=(1.8 if filled else 1.35) if single_column
                else (1.45 if filled else 1.0),
                label=f"{label} {suffix}" if single_column
                else f"{label}: {suffix}",
                zorder=4 if filled else 3,
            )
        differences = [entry["difference"][metric] for entry in entries]
        means = 100 * np.array([item["mean"] for item in differences])
        lows = 100 * np.array([item["low"] for item in differences])
        highs = 100 * np.array([item["high"] for item in differences])
        all_lows.extend(lows)
        all_highs.extend(highs)
        delta.errorbar(
            factors + offset, means,
            yerr=[means - lows, highs - means],
            fmt=marker, linestyle="none", color=color,
            markerfacecolor=color, markersize=marker_size,
            markeredgewidth=0.7 if single_column else 1,
            capsize=1.1 if single_column else 1.5,
            elinewidth=1.1 if single_column else 0.8,
            zorder=4,
        )

    absolute.set_title(
        "(a) Success" if single_column
        else f"(a) {task_name}: homography success", loc="left", pad=5
    )
    if not single_column:
        absolute.set_xlabel("Downscale factor")
    absolute.set_ylabel("Success (%)" if single_column else ylabel)
    absolute.set_xlim(2.3, 33.7)
    absolute.set_ylim(0, 100)
    tick_factors = factors[[0, 2, 4, 7]] if single_column else factors
    absolute.set_xticks(
        tick_factors,
        [f"×{int(factor)}" if single_column else f"{int(factor)}×"
         for factor in tick_factors],
    )
    absolute.set_yticks((0, 25, 50, 75, 100))
    absolute.grid(axis="y", color="#DADADA", linewidth=0.5)

    delta.axhline(0, color="#555555", linewidth=0.8, zorder=1)
    delta.set_title(
        "(b) Gain (pp)" if single_column
        else "(b) Paired difference by kernel", loc="left", pad=5
    )
    if not single_column:
        delta.set_xlabel("Downscale factor")
        delta.set_ylabel("Consistent − AA (points)")
    delta.set_xlim(2.3, 33.7)
    delta.set_xticks(
        tick_factors,
        [f"×{int(factor)}" if single_column else f"{int(factor)}×"
         for factor in tick_factors],
    )
    lower = min(-3, np.floor(min(all_lows) - 0.5))
    upper = max(3, np.ceil(max(all_highs) + 0.5))
    delta.set_ylim(lower, upper)
    delta.grid(axis="y", color="#DADADA", linewidth=0.5)

    handles, labels = absolute.get_legend_handles_labels()
    if single_column:
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
            ncol=2, handlelength=1.5, columnspacing=1.2, labelspacing=0.55,
        )
        fig.subplots_adjust(
            left=0.16, right=0.985, bottom=0.35, top=0.87, wspace=0.18
        )
        output_name = f"{name}_single_col{tag}"
    else:
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=4 if path == "down then up" else 3,
        )
        fig.subplots_adjust(
            left=0.09, right=0.985, bottom=0.30, top=0.91, wspace=0.32
        )
        output_name = f"{name}{tag}"
    fig.savefig(HERE / f"{output_name}.pdf")
    fig.savefig(HERE / f"{output_name}.png", dpi=300)
    plt.close(fig)


def make_gain_only_figure(summary: dict) -> None:
    path, metric, *_ = TASKS[1]
    factors = np.array(summary["protocol"]["factors"], dtype=float)
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    all_lows: list[float] = []
    all_highs: list[float] = []

    for base, label, color, marker, offset in BASES:
        differences = [
            summary["bases"][base]["factors"][str(int(factor))]
            ["paths"][path]["difference"][metric]
            for factor in factors
        ]
        means = 100 * np.array([item["mean"] for item in differences])
        lows = 100 * np.array([item["low"] for item in differences])
        highs = 100 * np.array([item["high"] for item in differences])
        all_lows.extend(lows)
        all_highs.extend(highs)
        ax.errorbar(
            factors + offset, means, yerr=[means - lows, highs - means],
            fmt=marker, linestyle="none", color=color, label=label,
            markerfacecolor=color, markersize=5.5, markeredgewidth=1,
            capsize=2.0, elinewidth=1.1, zorder=4,
        )

    ax.axhline(0, color="#555555", linewidth=0.8, zorder=1)
    ax.set_xlim(2.3, 33.7)
    ax.set_xticks((4, 12, 20, 32))
    ax.set_ylim(
        min(-3, np.floor(min(all_lows) - 0.5)),
        max(3, np.ceil(max(all_highs) + 0.5)),
    )
    ax.set_xlabel("Downscale factor (×)")
    ax.set_ylabel("Consistent − AA (pp)")
    ax.grid(axis="y", color="#DADADA", linewidth=0.5)
    fig.legend(
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=3, handletextpad=0.3, columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.36, top=0.95)
    output = HERE / "fig_hpatches_unit_gain_single_col"
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--single-column", action="store_true")
    parser.add_argument("--gain-only", action="store_true")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10 if args.single_column or args.gain_only else 8.5,
            "axes.labelsize": 10.5 if args.single_column or args.gain_only else 8.5,
            "axes.titlesize": 10.8 if args.single_column or args.gain_only else 8.7,
            "xtick.labelsize": 9.2 if args.single_column or args.gain_only else 7.3,
            "ytick.labelsize": 9.2 if args.single_column or args.gain_only else 8,
            "legend.fontsize": 9.2 if args.single_column or args.gain_only else 7.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if args.gain_only:
        make_gain_only_figure(summary)
        return
    tasks = TASKS[1:] if args.single_column else TASKS
    for task in tasks:
        make_figure(summary, task, args.tag, args.single_column)


if __name__ == "__main__":
    main()
