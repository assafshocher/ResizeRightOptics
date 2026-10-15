"""Sequence-bootstrap summary and vector figure for the shared-kernel audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hpatches_geometry_make_report import estimate
from hpatches_matched_anchor_experiment import BASES, LAWS, PATHS


METRICS = ("success_2_output_px", "success_2_5_output_px", "matches")
COLORS = {
    "consistent": "#D55E00",
    "AA direct": "#0072B2",
    "AA from anchor": "#CC79A7",
    "AA chain": "#009E73",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def by_sequence(
    rows: list[dict[str, str]],
    conditions: dict[str, str | int],
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if any(row.get(key) != str(value) for key, value in conditions.items()):
            continue
        value = float(row[metric])
        if np.isfinite(value):
            grouped[row["sequence"]].append(value)
    return {
        sequence: float(np.mean(values))
        for sequence, values in grouped.items()
        if values
    }


def paired(
    rows: list[dict[str, str]],
    first: dict[str, str | int],
    second: dict[str, str | int],
    metric: str,
) -> dict[str, float]:
    a = by_sequence(rows, first, metric)
    b = by_sequence(rows, second, metric)
    difference = {sequence: a[sequence] - b[sequence] for sequence in a.keys() & b.keys()}
    return estimate(difference)


def summarize(task_rows: list[dict[str, str]], image_rows: list[dict[str, str]], factors: list[int]) -> dict:
    summary = {"bases": {}}
    for base in BASES:
        base_result = {"anchor": {}, "factors": {}}
        for metric in METRICS:
            base_result["anchor"][metric] = estimate(
                by_sequence(task_rows, {"base": base, "factor": 2}, metric)
            )
        base_result["anchor"]["image_nrmse_percent"] = estimate(
            by_sequence(
                image_rows,
                {"base": base, "factor": 2, "comparison": "anchor consistency vs AA"},
                "nrmse_percent",
            )
        )
        for factor in factors:
            result = {"laws": {}, "differences": {}, "images": {}}
            for law in LAWS:
                result["laws"][law] = {}
                for path in PATHS:
                    conditions = {"base": base, "factor": factor, "law": law, "path": path}
                    result["laws"][law][path] = {
                        metric: estimate(by_sequence(task_rows, conditions, metric))
                        for metric in METRICS
                    }
                for comparison in (
                    f"{law}: full vs anchor direct",
                    f"{law}: anchor direct vs chain",
                ):
                    result["images"][comparison] = estimate(
                        by_sequence(
                            image_rows,
                            {"base": base, "factor": factor, "comparison": comparison},
                            "nrmse_percent",
                        )
                    )
            comparison = "consistent vs AA: anchored direct"
            result["images"][comparison] = estimate(
                by_sequence(
                    image_rows,
                    {"base": base, "factor": factor, "comparison": comparison},
                    "nrmse_percent",
                )
            )
            for path in PATHS:
                result["differences"][f"consistent_minus_AA:{path}"] = {
                    metric: paired(
                        task_rows,
                        {"base": base, "factor": factor, "law": "consistent telescope", "path": path},
                        {"base": base, "factor": factor, "law": "AA stretch", "path": path},
                        metric,
                    )
                    for metric in METRICS
                }
            result["differences"]["AA_chain_minus_anchor_direct"] = {
                metric: paired(
                    task_rows,
                    {"base": base, "factor": factor, "law": "AA stretch", "path": "anchor chain"},
                    {"base": base, "factor": factor, "law": "AA stretch", "path": "anchor direct"},
                    metric,
                )
                for metric in METRICS
            }
            base_result["factors"][str(factor)] = result
        summary["bases"][base] = base_result
    return summary


def make_figure(summary: dict, output: Path) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.2), sharey=True)
    styles = (
        ("consistent", "consistent telescope", "full direct", "D", "-"),
        ("AA direct", "AA stretch", "full direct", "o", "-"),
        ("AA from anchor", "AA stretch", "anchor direct", "^", "--"),
        ("AA chain", "AA stretch", "anchor chain", "s", "--"),
    )
    factors = summary["protocol"]["hard_factors"]
    for axis, base in zip(axes, BASES):
        base_result = summary["bases"][base]
        for label, law, path, marker, linestyle in styles:
            entries = [base_result["anchor"]["success_2_5_output_px"]] + [
                base_result["factors"][str(factor)]["laws"][law][path]["success_2_5_output_px"]
                for factor in factors
            ]
            x = [2, *factors]
            mean = 100 * np.array([entry["mean"] for entry in entries])
            low = 100 * np.array([entry["low"] for entry in entries])
            high = 100 * np.array([entry["high"] for entry in entries])
            axis.errorbar(
                x,
                mean,
                yerr=[mean - low, high - mean],
                label=label,
                color=COLORS[label],
                marker=marker,
                linestyle=linestyle,
                markersize=4.2,
                linewidth=1.5 if label == "consistent" else 1.0,
                elinewidth=0.6,
                capsize=1.4,
                zorder=5 if label == "consistent" else 3,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([2, *factors], [str(factor) for factor in [2, *factors]])
        axis.set_xlim(1.75, max(factors) * 1.08)
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.13)
        axis.set_axisbelow(True)
        axis.set_xlabel("Downscale factor")
        axis.set_title(base, fontsize=9, pad=7)
    axes[0].set_ylabel("GT homography success (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.24, top=0.91, wspace=0.17)
    fig.savefig(output / "hpatches_matched_anchor.pdf")
    fig.savefig(output / "hpatches_matched_anchor.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    task_rows = read_csv(args.results / "task_results.csv")
    image_rows = read_csv(args.results / "image_results.csv")
    protocol = json.loads((args.results / "protocol.json").read_text())
    summary = summarize(task_rows, image_rows, protocol["hard_factors"])
    summary["protocol"] = protocol
    (args.output / "hpatches_matched_anchor_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    make_figure(summary, args.output)
    for base in BASES:
        for factor in protocol["hard_factors"]:
            result = summary["bases"][base]["factors"][str(factor)]
            consistent = result["laws"]["consistent telescope"]["full direct"]["success_2_5_output_px"]["mean"]
            aa_direct = result["laws"]["AA stretch"]["full direct"]["success_2_5_output_px"]["mean"]
            aa_chain = result["laws"]["AA stretch"]["anchor chain"]["success_2_5_output_px"]["mean"]
            print(
                f"{base:20} x{factor:2}: consistent={100*consistent:.1f}% "
                f"AA direct={100*aa_direct:.1f}% AA chain={100*aa_chain:.1f}%"
            )


if __name__ == "__main__":
    main()
