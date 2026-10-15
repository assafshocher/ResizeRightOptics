"""Cluster-bootstrap report and publication plot for HPatches geometry audit."""

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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cluster_values(
    rows: list[dict[str, str]],
    method: str,
    path: str | None,
    metric: str,
    scene_type: str | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        if path is not None and row.get("path") != path:
            continue
        if scene_type is not None and row["type"] != scene_type:
            continue
        value = float(row[metric])
        if np.isfinite(value):
            grouped[row["sequence"]].append(value)
    return {
        sequence: float(np.mean(values))
        for sequence, values in grouped.items()
        if values
    }


def estimate(values: dict[str, float], seed: int = 1337, repeats: int = 4000) -> dict[str, float]:
    array = np.array(list(values.values()), dtype=np.float64)
    if len(array) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    draws = np.mean(array[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(np.mean(array)),
        "low": float(low),
        "high": float(high),
        "n": len(array),
    }


def paired_difference(
    rows: list[dict[str, str]],
    method: str,
    metric: str,
    scene_type: str | None = None,
) -> dict[str, float]:
    direct = cluster_values(rows, method, "direct", metric, scene_type)
    chain = cluster_values(rows, method, "chain", metric, scene_type)
    paired = {
        sequence: chain[sequence] - direct[sequence]
        for sequence in direct.keys() & chain.keys()
    }
    return estimate(paired)


def between_method_difference(
    rows: list[dict[str, str]],
    first_method: str,
    second_method: str,
    path: str,
    metric: str,
) -> dict[str, float]:
    first = cluster_values(rows, first_method, path, metric)
    second = cluster_values(rows, second_method, path, metric)
    paired = {
        sequence: first[sequence] - second[sequence]
        for sequence in first.keys() & second.keys()
    }
    return estimate(paired)


def aggregate(task_rows: list[dict[str, str]], image_rows: list[dict[str, str]]) -> dict:
    task_metrics = ["success_5px", "success_10px", "gt_inliers_5px", "matches"]
    difference_metrics = ["success_5px", "success_10px", "gt_inliers_5px"]
    if task_rows and "success_2_output_px" in task_rows[0]:
        task_metrics.extend(["success_2_output_px", "success_2_5_output_px"])
        difference_metrics.extend(["success_2_output_px", "success_2_5_output_px"])
    groups = {}
    for scene_type in (None, "i", "v"):
        type_name = "all" if scene_type is None else scene_type
        groups[type_name] = {}
        for method in METHODS:
            task_summary = {}
            for path in ("direct", "chain"):
                task_summary[path] = {
                    metric: estimate(
                        cluster_values(task_rows, method, path, metric, scene_type)
                    )
                    for metric in task_metrics
                }
            task_summary["chain_minus_direct"] = {
                metric: paired_difference(task_rows, method, metric, scene_type)
                for metric in difference_metrics
            }
            task_summary["path_nrmse_percent"] = estimate(
                cluster_values(image_rows, method, None, "path_nrmse_percent", scene_type)
            )
            groups[type_name][method] = task_summary
    comparisons = {
        f"Gaussian_chain_minus_{method.replace(' ', '_')}_chain": {
            metric: between_method_difference(
                task_rows,
                "Camera-consistent Gaussian",
                method,
                "chain",
                metric,
            )
            for metric in difference_metrics
        }
        for method in METHODS
        if method != "Camera-consistent Gaussian"
    }
    return {"groups": groups, "comparisons": comparisons}


def make_figure(summary: dict, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.75, 2.85), gridspec_kw={"width_ratios": [1, 1.35]})
    y = np.arange(len(METHODS), dtype=np.float64)
    entries = summary["groups"]["all"]
    normalized_ransac = summary["protocol"]["ransac"].get("threshold_output_pixels") is not None
    task_metric = "success_2_5_output_px" if normalized_ransac else "success_10px"
    nrmse = [entries[method]["path_nrmse_percent"]["mean"] for method in METHODS]
    left.barh(
        y,
        nrmse,
        color=[COLORS[method] for method in METHODS],
        height=0.54,
        edgecolor="white",
        linewidth=0.5,
    )
    left.set_yticks(y, METHODS)
    left.invert_yaxis()
    left.set_xlabel("Direct--chain image NRMSE (%)")
    left.set_xlim(0, max(nrmse) * 1.14)
    left.grid(axis="x", alpha=0.13)
    left.set_axisbelow(True)

    for path, offset, marker, fill in (
        ("direct", -0.13, "o", "white"),
        ("chain", 0.13, "s", None),
    ):
        x = np.array(
            [100 * entries[method][path][task_metric]["mean"] for method in METHODS]
        )
        low = np.array(
            [100 * entries[method][path][task_metric]["low"] for method in METHODS]
        )
        high = np.array(
            [100 * entries[method][path][task_metric]["high"] for method in METHODS]
        )
        for index, method in enumerate(METHODS):
            right.errorbar(
                x[index],
                y[index] + offset,
                xerr=[[x[index] - low[index]], [high[index] - x[index]]],
                fmt=marker,
                markersize=4,
                markerfacecolor=fill or COLORS[method],
                markeredgecolor=COLORS[method],
                ecolor=COLORS[method],
                elinewidth=0.8,
                capsize=2,
                color=COLORS[method],
                zorder=3,
            )
    right.set_yticks(y, [])
    right.invert_yaxis()
    right.set_xlabel(
        "GT homography success within 2.5 output px (%)"
        if normalized_ransac else "GT homography success within 10 original px (%)"
    )
    right.set_xlim(0, 100)
    right.grid(axis="x", alpha=0.13)
    right.set_axisbelow(True)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black", label="direct"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor="black", markeredgecolor="black", label="five-step"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.69, -0.02))
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.26, top=0.96, wspace=0.23)
    denominator = round(1 / summary["protocol"]["fraction"])
    name = f"hpatches_geometry_x{denominator}"
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    task_rows = load_csv(args.results / "task_results.csv")
    image_rows = load_csv(args.results / "image_results.csv")
    summary = aggregate(task_rows, image_rows)
    summary["protocol"] = json.loads((args.results / "protocol.json").read_text())
    (args.output / "hpatches_geometry_summary.json").write_text(json.dumps(summary, indent=2))
    make_figure(summary, args.output)
    for method in METHODS:
        entry = summary["groups"]["all"][method]
        normalized_ransac = summary["protocol"]["ransac"].get("threshold_output_pixels") is not None
        task_metric = "success_2_5_output_px" if normalized_ransac else "success_10px"
        print(
            f"{method:30} "
            f"NRMSE={entry['path_nrmse_percent']['mean']:.3f}% "
            f"{task_metric} direct={100*entry['direct'][task_metric]['mean']:.1f}% "
            f"chain={100*entry['chain'][task_metric]['mean']:.1f}%"
        )


if __name__ == "__main__":
    main()
