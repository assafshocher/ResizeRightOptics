"""Plot verified sigma=2 Gaussian runs without smoothing or seed error bars.

``plot_scale(summary, output_dir, first_image=False)`` saves a single scale.
``plot_all(summaries, output_dir)`` saves the complete Set5 x2/x3/x4 triptych.
Both return the PNG (300 dpi) and vector PDF paths, in that order.

Required summary fields are ``passed``, ``complete``, ``scale``, ``seed``,
``images``, ``m``, ``mean_psnr_y``, ``baseline_mean_y``, and ``per_image``.
Each per-image report contains ``image``, ``psnr_y``, and ``baseline_y``.
The method dictionaries contain all three keys in ``METHODS``. Means are
checked against and plotted from the individual reports, never trusted alone.
Use a separate output directory for the explicitly labeled baby-only preview.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SET5_IMAGES = ("baby", "bird", "butterfly", "head", "woman")
METHODS = ("gaussian_consistent", "gaussian_aa", "aa_grid_gaussian")
STYLES = {
    "gaussian_consistent": ("Ours (Gaussian law)", "#D55E00", "-", "o"),
    "gaussian_aa": ("Gaussian AA", "#0072B2", "-.", "^"),
    "aa_grid_gaussian": ("AA (PyTorch grid)", "#E69F00", ":", "D"),
}
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.formatter.useoffset": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}


@dataclass(frozen=True)
class _VerifiedScale:
    scale: int
    m: np.ndarray
    means: dict[str, np.ndarray]
    baseline: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a dictionary")
    return value


def _required(mapping: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{name} is missing {key!r}")
    return mapping[key]


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _curve(value: Any, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain 10 finite numbers")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain 10 finite numbers") from exc
    if len(items) != 10:
        raise ValueError(f"{name} must contain exactly 10 entries, got {len(items)}")
    return np.asarray([_scalar(v, f"{name}[{i}]") for i, v in enumerate(items)])


def _check_metadata(report: Mapping[str, Any], scale: int, name: str) -> None:
    """Check optional report metadata without requiring runner-specific fields."""
    for key in ("passed", "complete"):
        if key in report and report[key] is not True:
            raise ValueError(f"{name}.{key} must be true when supplied")
    if "scale" in report and _scalar(report["scale"], f"{name}.scale") != scale:
        raise ValueError(f"{name}.scale does not match the summary")
    if "seed" in report and _scalar(report["seed"], f"{name}.seed") != 0:
        raise ValueError(f"{name}.seed must be 0")
    if "anchor_sigma" in report:
        if _scalar(report["anchor_sigma"], f"{name}.anchor_sigma") != 2:
            raise ValueError(f"{name}.anchor_sigma must be 2")
    if "m" in report and not np.array_equal(_curve(report["m"], f"{name}.m"), np.arange(1, 11)):
        raise ValueError(f"{name}.m must be the ordered steps 1 through 10")


def _validate(summary: Mapping[str, Any], first_image: bool) -> _VerifiedScale:
    summary = _mapping(summary, "summary")
    for key in ("passed", "complete"):
        if _required(summary, key, "summary") is not True:
            raise ValueError(f"summary.{key} must be true before plotting")
    raw_scale = _scalar(_required(summary, "scale", "summary"), "summary.scale")
    if raw_scale not in (2, 3, 4):
        raise ValueError("summary.scale must be 2, 3, or 4")
    scale = int(raw_scale)
    if _scalar(_required(summary, "seed", "summary"), "summary.seed") != 0:
        raise ValueError("summary.seed must be 0; these figures describe a single seed")
    m = _curve(_required(summary, "m", "summary"), "summary.m")
    if not np.array_equal(m, np.arange(1, 11)):
        raise ValueError("summary.m must be the ordered steps 1 through 10")
    _check_metadata(summary, scale, "summary")

    expected_images = ("baby",) if first_image else SET5_IMAGES
    images = _required(summary, "images", "summary")
    if not isinstance(images, (list, tuple)) or tuple(images) != expected_images:
        raise ValueError(f"summary.images must be {list(expected_images)!r}")
    reports = _required(summary, "per_image", "summary")
    if not isinstance(reports, (list, tuple)) or len(reports) != len(expected_images):
        raise ValueError("summary.per_image must contain exactly one report per expected image")

    curves = {method: [] for method in METHODS}
    baselines = []
    seen = set()
    for index, raw_report in enumerate(reports):
        name = f"summary.per_image[{index}]"
        report = _mapping(raw_report, name)
        image = _required(report, "image", name)
        if not isinstance(image, str) or image not in expected_images or image in seen:
            raise ValueError(f"{name}.image is unexpected or duplicated: {image!r}")
        seen.add(image)
        _check_metadata(report, scale, name)
        psnr = _mapping(_required(report, "psnr_y", name), f"{name}.psnr_y")
        for method in METHODS:
            curves[method].append(_curve(_required(psnr, method, f"{name}.psnr_y"),
                                         f"{name}.psnr_y.{method}"))
        baselines.append(_scalar(_required(report, "baseline_y", name), f"{name}.baseline_y"))

    supplied = _mapping(_required(summary, "mean_psnr_y", "summary"), "summary.mean_psnr_y")
    means = {}
    for method in METHODS:
        claimed = _curve(_required(supplied, method, "summary.mean_psnr_y"),
                         f"summary.mean_psnr_y.{method}")
        recomputed = np.mean(np.stack(curves[method]), axis=0)
        if not np.allclose(claimed, recomputed, rtol=0, atol=1e-8):
            raise ValueError(f"summary.mean_psnr_y.{method} does not match the per-image mean")
        means[method] = recomputed
    baseline = float(np.mean(baselines))
    claimed_baseline = _scalar(_required(summary, "baseline_mean_y", "summary"),
                               "summary.baseline_mean_y")
    if not np.isclose(claimed_baseline, baseline, rtol=0, atol=1e-8):
        raise ValueError("summary.baseline_mean_y does not match the per-image mean")
    return _VerifiedScale(scale, m, means, baseline)


def _draw(ax: Any, data: _VerifiedScale, title: str, ylabel: bool = True) -> None:
    for method in METHODS:
        label, color, linestyle, marker = STYLES[method]
        ax.plot(data.m, data.means[method], label=label, color=color,
                linestyle=linestyle, marker=marker, markersize=2.4,
                markeredgewidth=0.55, linewidth=1.55, zorder=3)
    ax.axhline(data.baseline, color="0.55", linestyle="--", linewidth=1.05,
               label="Bicubic interpolation", zorder=2)
    ax.set_title(title, pad=9)
    ax.set_xlabel("Steps (m)")
    if ylabel:
        ax.set_ylabel("Y-PSNR (dB)")
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlim(0.8, 10.2)
    ax.margins(y=0.12)
    ax.grid(axis="y", alpha=0.2, linewidth=0.65)
    ax.set_axisbelow(True)


def _save(fig: Any, output_dir: str | Path, basename: str) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{basename}.{suffix}" for suffix in ("png", "pdf")]
    for path in paths:
        fig.savefig(path, dpi=300, facecolor="white")
    return paths


def plot_scale(summary: Mapping[str, Any], output_dir: str | Path,
               first_image: bool = False) -> list[Path]:
    """Save one verified scale; baby-only data require ``first_image=True``."""
    if not isinstance(first_image, bool):
        raise ValueError("first_image must be a boolean")
    data = _validate(summary, first_image)
    title = (f"Baby ×{data.scale} — first image only" if first_image
             else f"Set5 ×{data.scale} — Gaussian σ=2")
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.75, 3.5))
        try:
            _draw(ax, data, title)
            fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
                       bbox_to_anchor=(0.53, 0.012), columnspacing=1.8, handlelength=2.9)
            fig.subplots_adjust(left=0.115, right=0.975, bottom=0.31, top=0.88)
            return _save(fig, output_dir, f"fig_gaussian_sigma2_x{data.scale}")
        finally:
            plt.close(fig)


def plot_all(summaries: Sequence[Mapping[str, Any]], output_dir: str | Path) -> list[Path]:
    """Save a 1×3 figure after all three complete, five-image means validate."""
    summaries = list(summaries)
    if len(summaries) != 3:
        raise ValueError("plot_all requires exactly three complete Set5 summaries")
    verified = sorted((_validate(summary, False) for summary in summaries), key=lambda x: x.scale)
    if [data.scale for data in verified] != [2, 3, 4]:
        raise ValueError("plot_all requires one complete summary for each scale: 2, 3, 4")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.5))
        try:
            for index, (ax, data) in enumerate(zip(axes, verified)):
                _draw(ax, data, f"Set5 ×{data.scale}", ylabel=index == 0)
            fig.suptitle("Gaussian σ=2", fontsize=11, y=0.98)
            fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
                       bbox_to_anchor=(0.53, 0.012), columnspacing=1.8, handlelength=2.9)
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.31, top=0.79, wspace=0.40)
            return _save(fig, output_dir, "fig_gaussian_sigma2_all_scales")
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+", help="One or three verified summary JSON files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-image", action="store_true", help="Label and validate a baby-only preview")
    args = parser.parse_args()
    if len(args.summaries) not in (1, 3):
        parser.error("provide one summary for a single plot or three for the combined figure")
    if args.first_image and len(args.summaries) != 1:
        parser.error("--first-image accepts exactly one summary")
    summaries = [json.loads(path.read_text()) for path in args.summaries]
    if len(summaries) == 1:
        paths = plot_scale(summaries[0], args.output_dir, first_image=args.first_image)
    else:
        paths = plot_all(summaries, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
