"""Appendix candidate mirroring the path audit for independent physical s and N."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from library_grid_zoom_audit import WRONG_GRID


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "work/library_audit/grid_zoom_full"
OUTPUT = ROOT / "outputs"

METHODS = [
    ("OpenCV Lanczos-4", "OpenCV Lanczos-4", "#A21D21", "o"),
    ("OpenCV bicubic", "OpenCV bicubic", "#D55E00", "s"),
    ("scikit cubic AA", "scikit-image cubic AA", "#CC79A7", "^"),
    ("OpenCV area", "OpenCV area", "#E69F00", "v"),
    ("Pillow linear", "Pillow bilinear", "#7A7A26", "P"),
    ("scikit linear AA", "scikit-image linear AA", "#009E73", "X"),
    ("Pillow cubic AA", "Pillow bicubic", "#56B4E9", "D"),
    ("Pillow Lanczos-3 AA", "Pillow Lanczos", "#0072B2", "<"),
    ("same optics, $N/N_0$ grid", WRONG_GRID, "#111111", "*")
]


def bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    averages = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(averages, .025)), float(np.quantile(averages, .975))


def plot_figure(images: pd.DataFrame, models: pd.DataFrame, protocol: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.1,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.3,
            "legend.fontsize": 5.9,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "savefig.bbox": "tight",
        }
    )
    figure, (axis_error, axis_flip) = plt.subplots(
        1, 2, figsize=(7.05, 3.6),
        gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.45},
    )
    for row, (label, method, color, marker) in enumerate(reversed(METHODS)):
        if method == WRONG_GRID:
            values = 100 * images[images.method == method].nrmse.to_numpy()
            mean, low, high = bootstrap(values, 7000)
            axis_error.axhline(mean, color=color, linewidth=2.0, linestyle="--", label=label, zorder=8)
            axis_error.axhspan(low, high, color=color, alpha=0.08, linewidth=0)
            continue
        means, lows, highs = [], [], []
        for steps in range(1, 6):
            values = 100 * images[(images.method == method) & (images.steps == steps)].nrmse.to_numpy()
            mean, low, high = bootstrap(values, 1000 + 19 * row + steps)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        axis_error.plot(
            range(1, 6), means, color=color, marker=marker, markersize=3.5,
            linewidth=1.25, label=label, zorder=3,
        )
        axis_error.fill_between(range(1, 6), lows, highs, color=color, alpha=.08, linewidth=0)
    axis_error.set_xticks(range(1, 6))
    axis_error.set_xlim(1, 5)
    axis_error.set_yscale("log")
    axis_error.set_ylim(.15, 20)
    axis_error.set_xlabel("number of resize steps")
    axis_error.set_ylabel("mismatch to optical target (NRMSE, %)")
    axis_error.set_title("(a) Same $N$, wrong zoom and/or filtering", loc="left", fontweight="bold")
    axis_error.legend(
        loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2, frameon=False,
        handlelength=1.5, labelspacing=.20, columnspacing=.7,
    )

    positions = np.arange(len(METHODS))
    for model_index, (name, color, marker) in enumerate(
        [("ResNet-18", "#0072B2", "o"), ("MobileNetV3-Small", "#D55E00", "s")]
    ):
        for row, (label, method, _, _) in enumerate(METHODS):
            steps = 0 if method == WRONG_GRID else 5
            values = 100 * models[
                (models.method == method) & (models.steps == steps) & (models.model == name)
            ].top1_flip.astype(float).to_numpy()
            mean, low, high = bootstrap(values, 4000 + 31 * model_index + row)
            axis_flip.errorbar(
                mean, row + (-.13 if model_index == 0 else .13),
                xerr=[[mean-low], [high-mean]], marker=marker, color=color,
                markersize=3.8, capsize=1.4, elinewidth=.8, linewidth=0,
                label=name if row == 0 else None,
            )
    axis_flip.set_yticks(positions)
    axis_flip.set_yticklabels([label for label, _, _, _ in METHODS])
    axis_flip.set_xlim(-2, 102)
    axis_flip.set_xlabel("top-1 change relative to optical target (%)")
    axis_flip.set_title("(b) Predictions differ from target", loc="left", fontweight="bold")
    axis_flip.grid(axis="y", visible=False)
    axis_flip.legend(loc="upper right", frameon=False, fontsize=6)
    figure.text(
        .5, .985,
        f"50 natural textures, simulated clear-pupil camera; $N_0={protocol['source_size']}$, "
        f"$N={protocol['target_size']}$, $s={protocol['physical_scale']:.6f} \\ne N/N_0$",
        ha="center", va="top", fontsize=7.2,
    )
    figure.subplots_adjust(top=.84, bottom=.34, left=.09, right=.99)
    figure.savefig(OUTPUT / "library_grid_zoom_appendix.pdf")
    figure.savefig(OUTPUT / "library_grid_zoom_appendix.png", dpi=300)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(RESULTS / "image_metrics.csv")
    models = pd.read_csv(RESULTS / "model_metrics.csv")
    protocol = json.loads((RESULTS / "results.json").read_text())["protocol"]
    plot_figure(images, models, protocol)
    rows = []
    for method in protocol["methods"] + [WRONG_GRID]:
        steps = 0 if method == WRONG_GRID else 5
        image_values = images[(images.method == method) & (images.steps == steps)]
        model_values = models[(models.method == method) & (models.steps == steps)]
        rows.append(
            {
                "method": method,
                "image_nrmse_percent": 100 * image_values.nrmse.mean(),
                "resnet_flip_percent": 100 * model_values[model_values.model == "ResNet-18"].top1_flip.mean(),
                "mobilenet_flip_percent": 100 * model_values[model_values.model == "MobileNetV3-Small"].top1_flip.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "library_grid_zoom_summary.csv", index=False)
    control = images[images.method == WRONG_GRID]
    grid_mean, grid_low, grid_high = bootstrap(100 * control.nrmse.to_numpy(), 8611)
    natural_methods = summary[summary.method != WRONG_GRID]
    report = f"""# Independent zoom-and-grid appendix candidate

## Result

On 50 natural textures treated as periodic continuous scenes, a computed clear-pupil camera supplied an input capture and a target capture at independently prescribed zoom $s={protocol['physical_scale']:.8f}$ and $N={protocol['target_size']}$ output pixels. A size-only call substitutes $N/N_0={protocol['size_inferred_scale']:.8f}$; the centered projected grids then differ by up to {protocol['max_grid_drift_source_pixels']:.3f} source pixels at the edge. Holding the camera transfer function fixed and changing only those coordinates yields {grid_mean:.3f}% image NRMSE (95% scene-bootstrap interval {grid_low:.3f}--{grid_high:.3f}%) on the central {protocol['evaluation_size']}x{protocol['evaluation_size']} output region.

The 12 common native calls differ from the simulated optical target by {natural_methods.image_nrmse_percent.min():.2f}--{natural_methods.image_nrmse_percent.max():.2f}% after five steps. These **are not grid-only errors**: each method also uses its own interpolation and antialiasing rules. The curves are largely flat or nonmonotonic with step count, so this test does not show accumulating grid phase. The classifier comparisons are relative to the simulated target, not recognition accuracy; target median top-1 confidence is only about 0.27 (ResNet-18) and 0.31 (MobileNetV3-Small), making raw flip rates a secondary, confidence-sensitive observation.

## Five-step table

{summary.to_markdown(index=False, floatfmt='.3f')}

## Audit protocol and limits

- The natural JPGs are texture sources, **not measured camera captures**. Both compared optical images are computed from the same periodic Shannon scene with a clear circular pupil, square-pixel integration, and cutoff $0.9\\pi$.
- $N_0={protocol['source_size']}$, $N={protocol['target_size']}$, $N_0s={protocol['N0_times_s']:.2f}$: the output count is the nearest integer to the prescribed scale's nominal count but is not silently used as the scale.
- The target is synthesized directly on $g_n=(n-(N-1)/2)/s+(N_0-1)/2$. The grid-only arm samples the **same continuous optical target** on the grid obtained by replacing $s$ with $N/N_0$.
- The native arms are unmodified size-only APIs with explicit 255-pixel outputs. Their direct paths and two-to-five-step paths are compared with the common simulated optical target at the same output indices; no crop or padding is used to force image sizes to agree. Metrics use only the central 224x224 region.
- Unlike the preceding path audit, the left panel measures target mismatch, **not direct-versus-chain inconsistency**. The right panel measures top-1 prediction differences from the simulated target, **not correctness**.
- ResizeRight already documents the projected-grid rule; this appendix's contribution would be empirical illustration of its interaction with popular size-only APIs, not a new grid derivation.

## Recommendation

Use this only as a cautious appendix/supplemental audit. The cleanest claim is that a size-only interface has already changed the intended physical zoom before any filtering is considered, and that the complete popular-library calls disagree substantially with a declared optical target. It is **not** evidence that the grid accounts for the full native error or that the proposed family improves classifier accuracy. The grid-only result is modest on these natural textures and more compelling on a controlled high-band example; the two should be reported separately.
"""
    (OUTPUT / "library_grid_zoom_appendix_report.md").write_text(report)
    print(summary.to_string(index=False))
    print(f"max grid displacement {protocol['max_grid_drift_source_pixels']:.4f} source pixels")


if __name__ == "__main__":
    main()
