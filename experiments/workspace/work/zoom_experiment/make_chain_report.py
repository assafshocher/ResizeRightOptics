from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path("work/zoom_experiment/results/chain_diagnostic")
OUTPUT = Path("outputs")


def main() -> None:
    result = json.loads((ROOT / "results.json").read_text())
    aggregate = result["aggregate"]
    steps = np.array(sorted(int(key) for key in aggregate))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "results.json", OUTPUT / "zoom_chain_results.json")

    with (OUTPUT / "zoom_chain_summary.csv").open("w", newline="") as handle:
        fields = [
            "steps",
            "method",
            "mean_direct_nrmse",
            "mean_chain_nrmse",
            "mean_direct_chain_disagreement_nrmse",
            "median_approx_reduced_chi_squared",
            "median_residual_lag1_correlation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            for method in [
                "bicubic",
                "stretched_aa",
                "gaussian_scale_space",
                "gaussian_kernel",
                "oracle_convolution",
            ]:
                writer.writerow({"steps": int(step), "method": method, **aggregate[str(step)][method]})

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
        "bicubic": "#7A7A7A",
        "stretched_aa": "#E67E22",
        "gaussian_scale_space": "#2B7A78",
    }
    labels = {
        "bicubic": "bicubic",
        "stretched_aa": "stretched Gaussian AA",
        "gaussian_scale_space": "consistent variance increment",
    }
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))

    for method in colors:
        disagreement = np.array(
            [aggregate[str(step)][method]["mean_direct_chain_disagreement_nrmse"] for step in steps]
        )
        axes[0].plot(
            steps,
            100 * disagreement,
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[method],
            label=labels[method],
        )
    axes[0].set_title("a  Path disagreement", loc="left", fontweight="bold")
    axes[0].set_xlabel("number of resize steps")
    axes[0].set_ylabel("direct vs. chain NRMSE (%)")
    axes[0].set_xticks(steps)
    axes[0].grid(color="#DDDDDD", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")

    for method in colors:
        chain_error = np.array([aggregate[str(step)][method]["mean_chain_nrmse"] for step in steps])
        axes[1].plot(
            steps,
            100 * chain_error,
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[method],
            label=labels[method],
        )
    axes[1].set_title("b  Error against real capture", loc="left", fontweight="bold")
    axes[1].set_xlabel("number of resize steps")
    axes[1].set_ylabel("chained prediction NRMSE (%)")
    axes[1].set_xticks(steps)
    axes[1].grid(color="#DDDDDD", linewidth=0.7)
    axes[1].set_axisbelow(True)

    chi_squared = np.array(
        [aggregate[str(step)]["oracle_convolution"]["median_approx_reduced_chi_squared"] for step in steps]
    )
    correlation = np.array(
        [aggregate[str(step)]["oracle_convolution"]["median_residual_lag1_correlation"] for step in steps]
    )
    axes[2].plot(steps, chi_squared, marker="o", linewidth=2, color="#8E44AD", label=r"approx. $\chi^2_\nu$")
    axes[2].axhline(1.0, color="#888888", linestyle="--", linewidth=1, label="noise-limited")
    axes[2].set_title("c  Model adequacy", loc="left", fontweight="bold")
    axes[2].set_xlabel("number of resize steps")
    axes[2].set_ylabel(r"median approximate $\chi^2_\nu$")
    axes[2].set_xticks(steps)
    axes[2].grid(color="#DDDDDD", linewidth=0.7)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, loc="upper left")
    for step, value, corr in zip(steps, chi_squared, correlation):
        axes[2].annotate(
            f"$r_1$={corr:.2f}",
            (step, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#555555",
        )

    figure.suptitle(
        "SR-RAW chain diagnostic: inconsistency accumulates, but real-pair residuals are not noise-limited",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.01,
        -0.02,
        "12 held-out scenes; adjacent low-band affine registrations; direct path uses their exact composition. "
        "Approximate noise from demosaiced luminance, not calibrated repeated RAW exposures. Source: SR-RAW (Zhang et al., 2020).",
        fontsize=7.3,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / "zoom_chain_diagnostic.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    final_step = str(int(steps[-1]))
    stretched = aggregate[final_step]["stretched_aa"]
    consistent = aggregate[final_step]["gaussian_scale_space"]
    path_ratio = stretched["mean_direct_chain_disagreement_nrmse"] / consistent[
        "mean_direct_chain_disagreement_nrmse"
    ]
    true_improvement = (
        stretched["mean_chain_nrmse"] - consistent["mean_chain_nrmse"]
    ) / stretched["mean_chain_nrmse"]
    paired_lines = []
    raw_pvalues = []
    for step in steps[1:]:
        selected = [row for row in result["rows"] if row["chain_steps"] == int(step)]
        aa = np.array([row["methods"]["stretched_aa"]["chain_to_truth"]["nrmse"] for row in selected])
        consistent_values = np.array(
            [row["methods"]["gaussian_scale_space"]["chain_to_truth"]["nrmse"] for row in selected]
        )
        relative = (aa - consistent_values) / aa
        rng = np.random.default_rng(800 + int(step))
        bootstrap = relative[rng.integers(0, len(relative), size=(30000, len(relative)))].mean(axis=1)
        interval = np.quantile(bootstrap, [0.025, 0.975])
        pvalue = float(wilcoxon(aa, consistent_values, alternative="greater").pvalue)
        raw_pvalues.append(pvalue)
        paired_lines.append(
            f"{int(step)} steps: {100*relative.mean():.1f}% "
            f"[{100*interval[0]:.1f}, {100*interval[1]:.1f}]"
        )
    order = np.argsort(raw_pvalues)
    holm = np.empty(len(raw_pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, raw_pvalues[index] * (len(raw_pvalues) - rank))
        running = max(running, adjusted)
        holm[index] = running
    paired_summary = "; ".join(paired_lines)
    holm_summary = ", ".join(f"{value:.3g}" for value in holm)
    report = f"""# SR-RAW chain diagnostic

## Design

- Twelve held-out SR-RAW scenes and five adjacent zoom steps (240→154→101→69→50→35 mm).
- Each adjacent target-to-source affine map is estimated by SIFT/RANSAC and strongly low-pass ECC.
- The one-step endpoint grid is the exact composition of those adjacent maps, so registration cannot explain direct-versus-chain disagreement.
- One low-band gain and bias is shared by every method and both paths for each endpoint.
- Compared direct bicubic, a frozen stretched-Gaussian AA convention, and the corresponding consistent Gaussian variance-increment family.

## Main result

After five steps, direct-versus-chain disagreement is:

- bicubic: {100*aggregate[final_step]['bicubic']['mean_direct_chain_disagreement_nrmse']:.2f}% NRMSE;
- stretched Gaussian AA: {100*stretched['mean_direct_chain_disagreement_nrmse']:.2f}% NRMSE;
- consistent variance increment: {100*consistent['mean_direct_chain_disagreement_nrmse']:.2f}% NRMSE.

The stretched convention therefore has {path_ratio:.1f}× the path disagreement of the consistent family. Its chained error against the real endpoint capture is {100*stretched['mean_chain_nrmse']:.2f}% versus {100*consistent['mean_chain_nrmse']:.2f}%, a {100*true_improvement:.1f}% relative reduction for the consistent family.

Paired per-scene relative reductions and bootstrap 95% intervals are: {paired_summary}. The corresponding one-sided Wilcoxon p-values after Holm correction are {holm_summary}. These are exploratory statistics because this chain protocol was designed after inspecting the earlier pairwise pilot.

## Convolution/model-adequacy diagnostic

An oracle 11×11 shift-invariant kernel is fitted separately for each pair on alternating 16-pixel spatial tiles and evaluated on the held-out tiles. Even this oracle has median approximate reduced chi-squared values of about {min(aggregate[str(step)]['oracle_convolution']['median_approx_reduced_chi_squared'] for step in steps):.1f}–{max(aggregate[str(step)]['oracle_convolution']['median_approx_reduced_chi_squared'] for step in steps):.1f}, not near one. Median residual lag-one correlation is {min(aggregate[str(step)]['oracle_convolution']['median_residual_lag1_correlation'] for step in steps):.2f}–{max(aggregate[str(step)]['oracle_convolution']['median_residual_lag1_correlation'] for step in steps):.2f}, not near zero. Even with a conservative image-based noise estimate, residuals are therefore much larger and more structured than independent sensor noise.

This means the SR-RAW pairs, after one global affine registration, are not adequately described by a single spatially invariant convolution plus sensor noise. Plausible causes include parallax, lens distortion, spatially varying PSF, demosaicing, and imperfect radiometry.

## Interpretation

The chain does amplify the paper's central inconsistency: path disagreement grows to several percent for stretched AA while the consistent variance-increment family remains near the numerical interpolation floor. The real-capture advantage also grows from roughly the pairwise half-percent effect to several percent in the middle of the chain.

However, this is evidence for the composition principle using the Gaussian special case, not yet a validation of the full non-Gaussian camera-derived family. The directly fitted real-data PSF has near-zero transfer values and becomes unstable under an unthresholded pseudoinverse; choosing its working band from the same data would add another uncontrolled degree of freedom.

The approximate chi-squared values must not be presented as calibrated sensor statistics: SR-RAW lacks repeated exposures, and the noise estimate comes from demosaiced luminance. Their distance from one and the strong residual correlation are nevertheless large enough to reject a noise-limited interpretation.
"""
    (OUTPUT / "zoom_chain_report.md").write_text(report)


if __name__ == "__main__":
    main()
