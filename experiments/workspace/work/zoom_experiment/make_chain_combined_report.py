from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


SMALL = Path("work/zoom_experiment/results/chain_diagnostic/results.json")
LARGE = Path("work/zoom_experiment/results/chain_sensitivity_192/results.json")
OUTPUT = Path("outputs")


def relative_capture_advantage(result: dict, step: int) -> tuple[float, np.ndarray]:
    rows = [row for row in result["rows"] if row["chain_steps"] == step]
    aa = np.array([row["methods"]["stretched_aa"]["chain_to_truth"]["nrmse"] for row in rows])
    consistent = np.array(
        [row["methods"]["gaussian_scale_space"]["chain_to_truth"]["nrmse"] for row in rows]
    )
    values = 100 * (aa - consistent) / aa
    return float(values.mean()), values


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(30000, len(values)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    small = json.loads(SMALL.read_text())
    large = json.loads(LARGE.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SMALL, OUTPUT / "zoom_chain_results_96.json")
    shutil.copy2(LARGE, OUTPUT / "zoom_chain_results_192.json")
    steps = np.arange(1, 6)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"bicubic": "#7A7A7A", "stretched_aa": "#E67E22", "gaussian_scale_space": "#2B7A78"}
    labels = {
        "bicubic": "bicubic",
        "stretched_aa": "stretched Gaussian AA",
        "gaussian_scale_space": "consistent variance increment",
    }
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.05))

    for method in colors:
        values = np.array(
            [large["aggregate"][str(step)][method]["mean_direct_chain_disagreement_nrmse"] for step in steps]
        )
        axes[0].plot(steps, 100 * values, marker="o", linewidth=2, markersize=4, color=colors[method], label=labels[method])
    axes[0].set_title("a  Path disagreement", loc="left", fontweight="bold")
    axes[0].set_xlabel("number of resize steps")
    axes[0].set_ylabel("direct vs. chain NRMSE (%)")
    axes[0].set_xticks(steps)
    axes[0].grid(color="#DDDDDD", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")

    for result, crop, color, marker in [
        (small, "96×96 crop", "#2B7A78", "o"),
        (large, "192×192 crop", "#8E44AD", "s"),
    ]:
        means = []
        low = []
        high = []
        for step in steps:
            mean, per_scene = relative_capture_advantage(result, int(step))
            interval = bootstrap_interval(per_scene, 900 + step + (0 if crop.startswith("96") else 20))
            means.append(mean)
            low.append(mean - interval[0])
            high.append(interval[1] - mean)
        axes[1].errorbar(
            steps,
            means,
            yerr=np.array([low, high]),
            marker=marker,
            markersize=4,
            linewidth=2,
            capsize=2,
            color=color,
            label=crop,
        )
    axes[1].axhline(0, color="#888888", linewidth=0.9)
    axes[1].set_title("b  Advantage on real endpoint", loc="left", fontweight="bold")
    axes[1].set_xlabel("number of resize steps")
    axes[1].set_ylabel("consistent vs. stretched AA (%)")
    axes[1].set_xticks(steps)
    axes[1].grid(color="#DDDDDD", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="best")

    for result, crop, color, marker in [
        (small, "96×96 crop", "#2B7A78", "o"),
        (large, "192×192 crop", "#8E44AD", "s"),
    ]:
        values = np.array(
            [result["aggregate"][str(step)]["oracle_convolution"]["median_approx_reduced_chi_squared"] for step in steps]
        )
        axes[2].plot(steps, values, marker=marker, markersize=4, linewidth=2, color=color, label=crop)
    axes[2].axhline(1, color="#888888", linestyle="--", linewidth=1, label="noise-limited")
    axes[2].set_title("c  Oracle-convolution adequacy", loc="left", fontweight="bold")
    axes[2].set_xlabel("number of resize steps")
    axes[2].set_ylabel(r"median approximate $\chi^2_\nu$")
    axes[2].set_xticks(steps)
    axes[2].grid(color="#DDDDDD", linewidth=0.7)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, loc="upper left")

    figure.suptitle(
        "SR-RAW chain test: the consistency gap is robust; the real-capture advantage is not",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.01,
        -0.02,
        "12 held-out scenes; exact composed adjacent affine grids. Error bars: paired-scene bootstrap 95% intervals. "
        "Oracle convolution: per-pair signed 11×11 kernel, trained/tested on alternating tiles. Source: SR-RAW (Zhang et al., 2020).",
        fontsize=7.2,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / "zoom_chain_combined.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    def final_ratio(result: dict) -> float:
        aa = result["aggregate"]["5"]["stretched_aa"]["mean_direct_chain_disagreement_nrmse"]
        consistent = result["aggregate"]["5"]["gaussian_scale_space"]["mean_direct_chain_disagreement_nrmse"]
        return aa / consistent

    small_advantage, _ = relative_capture_advantage(small, 5)
    large_advantage, _ = relative_capture_advantage(large, 5)
    report = f"""# Combined SR-RAW chain assessment

## What is robust

Using twelve held-out scenes, five adjacent zoom steps, and the exact same composed affine grids for direct and chained paths, stretched Gaussian AA accumulates substantial path dependence. At five steps its direct-versus-chain disagreement is {100*large['aggregate']['5']['stretched_aa']['mean_direct_chain_disagreement_nrmse']:.2f}% NRMSE, compared with {100*large['aggregate']['5']['gaussian_scale_space']['mean_direct_chain_disagreement_nrmse']:.2f}% for the consistent variance-increment family. The ratio is {final_ratio(large):.1f}× on 192×192 crops and {final_ratio(small):.1f}× on 96×96 crops.

This is a strong real-image demonstration that the stretched convention is path-dependent. Because the endpoint affine is defined as the exact composition of the adjacent maps, registration differences cannot produce this direct-versus-chain gap.

## What is not robust

Accuracy against the actual endpoint capture does not yield a stable winner. On 96×96 crops the consistent chain improves over stretched AA by {small_advantage:.1f}% at five steps; on 192×192 crops the value is {large_advantage:.1f}%. Across intermediate chain lengths the advantage is generally small and changes with crop size. This should not be presented as evidence that the Gaussian consistent family better models the real optics.

## Is a convolution adequate?

For each pair, an oracle signed 11×11 shift-invariant kernel with unit sum was fitted on alternating 16-pixel tiles and evaluated on the held-out tiles. On 96×96 crops its median approximate reduced chi-squared rises from {small['aggregate']['1']['oracle_convolution']['median_approx_reduced_chi_squared']:.1f} to {small['aggregate']['5']['oracle_convolution']['median_approx_reduced_chi_squared']:.1f}; on 192×192 crops it rises from {large['aggregate']['1']['oracle_convolution']['median_approx_reduced_chi_squared']:.1f} to {large['aggregate']['5']['oracle_convolution']['median_approx_reduced_chi_squared']:.1f}. Residual lag-one correlations remain large. Thus a global affine warp followed by one spatially invariant convolution does not explain these pairs down to their estimated noise floor.

This diagnostic cannot identify the culprit: parallax, lens distortion, spatial PSF variation, demosaicing, radiometry, and registration error can all contribute. The chi-squared scale is approximate because SR-RAW has no repeated exposures and noise is estimated from demosaiced linear luminance.

## Paper-level conclusion

The chain experiment is useful supporting evidence for the paper's structural composition claim: the inconsistency of stretched AA becomes several-percent and grows with path length, whereas the consistent Gaussian special case stays near the interpolation floor. It is not a validation of the full camera-derived non-Gaussian family, and it does not presently establish superior real-capture accuracy.

The directly estimated non-Gaussian PSF cannot be used safely here without another choice: its transfer has near-zeros, so the canonical ratio becomes unstable unless a visible-band threshold is supplied. Choosing that threshold on these same test images would make the comparison circular.
"""
    (OUTPUT / "zoom_chain_combined_report.md").write_text(report)


if __name__ == "__main__":
    main()
