from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("outputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("work/zoom_experiment/results/multiscene_physical_ls"),
    )
    parser.add_argument("--prefix", default="zoom_multiscene")
    args = parser.parse_args()
    result = json.loads((args.root / "results.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.root / "results.json", OUTPUT / f"{args.prefix}_results.json")

    sections = [
        ("seen", result["test_seen_scale"]),
        ("unseen", result["test_unseen_scale"]),
    ]
    fieldnames = [
        "split",
        "scene",
        "target_index",
        "low_band_ecc_correlation",
        "target_to_source_linear_scale",
        "direct_bicubic_nrmse",
        "stretched_gaussian_aa_nrmse",
        "gaussian_camera_nrmse",
        "estimated_camera_nrmse",
    ]
    with (OUTPUT / f"{args.prefix}_scene_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, section in sections:
            for row in section["rows"]:
                writer.writerow(
                    {
                        "split": split,
                        "scene": row["scene"],
                        "target_index": row["target_index"],
                        "low_band_ecc_correlation": row["low_band_ecc_correlation"],
                        "target_to_source_linear_scale": row["target_to_source_linear_scale"],
                        "direct_bicubic_nrmse": row["direct_bicubic"]["nrmse"],
                        "stretched_gaussian_aa_nrmse": row["stretched_gaussian_aa"]["nrmse"],
                        "gaussian_camera_nrmse": row["gaussian_camera"]["nrmse"],
                        "estimated_camera_nrmse": row["estimated_camera"]["nrmse"],
                    }
                )

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
    colors = {
        "direct_bicubic": "#7A7A7A",
        "stretched_gaussian_aa": "#E67E22",
        "gaussian_camera": "#2B7A78",
        "estimated_camera": "#111111",
    }
    labels = {
        "direct_bicubic": "direct bicubic",
        "stretched_gaussian_aa": "stretched Gaussian AA",
        "gaussian_camera": "consistent Gaussian camera",
        "estimated_camera": "LS-estimated camera",
    }

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.05), gridspec_kw={"width_ratios": [1.55, 1.25, 0.9]})

    methods = list(colors)
    x = np.arange(2)
    width = 0.19
    for index, method in enumerate(methods):
        means = [100 * section["aggregate"][method]["mean_nrmse"] for _, section in sections]
        axes[0].bar(
            x + (index - 1.5) * width,
            means,
            width=width,
            color=colors[method],
            label=labels[method],
        )
    axes[0].set_xticks(x, ["seen zoom\n38 scenes", "unseen zoom\n32 scenes"])
    axes[0].set_ylabel("NRMSE (% robust range)")
    axes[0].set_title("a  Frozen-family capture prediction", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    for index, (split, section) in enumerate(sections):
        rows = section["rows"]
        aa = np.array([row["stretched_gaussian_aa"]["nrmse"] for row in rows])
        camera = np.array([row["gaussian_camera"]["nrmse"] for row in rows])
        improvement = 100 * (aa - camera) / aa
        rng = np.random.default_rng(100 + index)
        jitter = rng.uniform(-0.10, 0.10, size=len(improvement))
        axes[1].scatter(
            np.full(len(improvement), index) + jitter,
            improvement,
            s=19,
            color=colors["gaussian_camera"],
            alpha=0.72,
            linewidth=0,
        )
        axes[1].plot([index - 0.18, index + 0.18], [np.mean(improvement)] * 2, color="black", linewidth=2)
    axes[1].axhline(0, color="#888888", linewidth=0.8)
    axes[1].set_xticks([0, 1], ["seen zoom", "unseen zoom"])
    axes[1].set_ylabel("camera improvement over\nstretched AA (%)")
    axes[1].set_title("b  Per-scene comparison", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axes[1].set_axisbelow(True)

    psf = np.array(result["physical_psf"]["weights"])
    image = axes[2].imshow(psf, cmap="magma", interpolation="nearest")
    axes[2].set_title("c  Direct-LS PSF", loc="left", fontweight="bold")
    axes[2].set_xlabel("pixel offset")
    axes[2].set_ylabel("pixel offset")
    ticks = [0, psf.shape[0] // 2, psf.shape[0] - 1]
    labels_tick = [-(psf.shape[0] // 2), 0, psf.shape[0] // 2]
    axes[2].set_xticks(ticks, labels_tick)
    axes[2].set_yticks(ticks, labels_tick)
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("mass", rotation=270, labelpad=10)

    figure.suptitle(
        "SR-RAW: camera-aware transfer helps, but direct non-Gaussian PSF estimation is not yet competitive",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.01,
        -0.02,
        "One PSF: 8 training scenes; hyperparameters: 4 validation scenes; exact shared affine projected grids. "
        "Source: SR-RAW test set (Zhang et al., 2020).",
        fontsize=7.5,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{args.prefix}_experiment.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    seen = result["test_seen_scale"]["aggregate"]
    unseen = result["test_unseen_scale"]["aggregate"]
    report = f"""# Multi-scene SR-RAW pilot

## Design

- Estimate one nonnegative, normalized, centrosymmetric continuous PSF from 8 training scenes using the affine cross-convolution identity.
- Select radius regularization and pseudoinverse damping on 4 separate validation scenes.
- Freeze the PSF and test on 38 unseen scenes at the estimation zoom and 32 successfully registered unseen scenes at a new zoom.
- Estimate geometry once per pair by SIFT/RANSAC followed by strongly low-pass affine ECC. Every method uses exactly the same projected grid.
- Fit one low-band radiometric gain and bias per pair before all methods.

## Results

| Held-out condition | Direct bicubic | Stretched Gaussian AA | Consistent Gaussian camera | Direct-LS camera |
|---|---:|---:|---:|---:|
| unseen scenes, seen zoom | {100*seen['direct_bicubic']['mean_nrmse']:.3f}% | {100*seen['stretched_gaussian_aa']['mean_nrmse']:.3f}% | {100*seen['gaussian_camera']['mean_nrmse']:.3f}% | {100*seen['estimated_camera']['mean_nrmse']:.3f}% |
| unseen scenes, unseen zoom | {100*unseen['direct_bicubic']['mean_nrmse']:.3f}% | {100*unseen['stretched_gaussian_aa']['mean_nrmse']:.3f}% | {100*unseen['gaussian_camera']['mean_nrmse']:.3f}% | {100*unseen['estimated_camera']['mean_nrmse']:.3f}% |

The Direct-LS camera lowers NRMSE relative to direct bicubic by {100*seen['estimated_vs_direct']['mean_relative_rmse_reduction']:.1f}% at the seen zoom and {100*unseen['estimated_vs_direct']['mean_relative_rmse_reduction']:.1f}% at the unseen zoom. The consistent Gaussian camera lowers it by {100*seen['gaussian_camera_vs_direct']['mean_relative_rmse_reduction']:.1f}% and {100*unseen['gaussian_camera_vs_direct']['mean_relative_rmse_reduction']:.1f}%, respectively.

Against stretched Gaussian AA, the consistent Gaussian camera improves mean NRMSE by only {100*seen['gaussian_camera_vs_stretched_gaussian_aa']['mean_relative_rmse_reduction']:.2f}% at the seen zoom and {100*unseen['gaussian_camera_vs_stretched_gaussian_aa']['mean_relative_rmse_reduction']:.2f}% at the unseen zoom. The latter mean bootstrap interval crosses zero, despite a one-sided Wilcoxon p-value of {unseen['gaussian_camera_vs_stretched_gaussian_aa']['paired_wilcoxon_one_sided_p']:.4g}; the effect is therefore practically tiny and not robust enough to headline.

The directly estimated non-Gaussian PSF is worse than stretched AA by {-100*seen['estimated_vs_stretched_gaussian_aa']['mean_relative_rmse_reduction']:.2f}% at the seen zoom and {-100*unseen['estimated_vs_stretched_gaussian_aa']['mean_relative_rmse_reduction']:.2f}% at the unseen zoom. This shows that the linear kernel estimator runs without joint registration/kernel alternation, but does not yet establish accurate physical-PSF recovery.

## Interpretation

This is encouraging evidence for the camera-sandwich family versus no AA, including transfer to a new zoom. It is not yet an oral-level experiment: a tuned stretched-AA baseline nearly matches the consistent Gaussian camera, and the direct non-Gaussian estimate does not win. The clean next experiment is controlled 2D optics with known PSF and geometry; on real data, a CFA-native forward model and either measured PSF or stronger calibration are needed before making a physical-PSF-estimation claim.

## Limitations

- Central luminance crops from linearly demosaiced RAW, not a CFA-native likelihood.
- Affine geometry is exact only for planar scenes or pure rotation; real scenes have parallax and lens distortion.
- Six of 38 unseen-zoom pairs failed the fixed low-band ECC threshold and are reported as failures.
- Zooming can change the real lens PSF, making frozen-PSF transfer deliberately stringent.
"""
    (OUTPUT / f"{args.prefix}_report.md").write_text(report)


if __name__ == "__main__":
    main()
