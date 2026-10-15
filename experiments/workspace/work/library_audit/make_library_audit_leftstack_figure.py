from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optical_cameras import spatial_kernel_slice


ROOT = Path(__file__).resolve().parents[2]
X4 = ROOT / "work/library_audit/full"
X8 = ROOT / "work/library_audit/full_x8"
OPTICAL_X4 = ROOT / "work/library_audit/optical_x4"
OPTICAL_X8 = ROOT / "work/library_audit/optical_x8"
ANNULAR_X4 = ROOT / "work/library_audit/annular_x4"
ANNULAR_X8 = ROOT / "work/library_audit/annular_x8"
OUTPUT = ROOT / "outputs"


GROUPS = [
    ("OpenCV Lanczos-4", "OpenCV Lanczos-4"),
    ("OpenCV bicubic", "OpenCV bicubic"),
    ("scikit-image cubic AA", "scikit-image cubic AA"),
    ("OpenCV area", "OpenCV area"),
    ("linear AA\n(Pillow/Torch/JAX)", "Pillow bilinear"),
    ("scikit-image linear AA", "scikit-image linear AA"),
    ("bicubic AA\n(Pillow/Torch)", "Pillow bicubic"),
    ("Lanczos-3 AA\n(Pillow/JAX)", "Pillow Lanczos"),
    ("Gaussian spectral control", "camera-consistent Gaussian"),
    ("annular-pupil family", "consistent annular pupil"),
    ("astigmatic-pupil family", "consistent astigmatic pupil"),
]


COLORS = {
    "OpenCV Lanczos-4": "#A21D21",
    "OpenCV bicubic": "#D55E00",
    "scikit-image cubic AA": "#CC79A7",
    "OpenCV area": "#E69F00",
    "linear AA\n(Pillow/Torch/JAX)": "#7A7A26",
    "scikit-image linear AA": "#009E73",
    "bicubic AA\n(Pillow/Torch)": "#56B4E9",
    "Lanczos-3 AA\n(Pillow/JAX)": "#0072B2",
    "Gaussian spectral control": "#666666",
    "annular-pupil family": "#111111",
    "astigmatic-pupil family": "#7A3E9D",
}


MARKERS = ["o", "s", "^", "v", "P", "X", "D", "<", "*", "h", "d"]


def gaussian_kernel_slice(scale: float = 0.5, samples: int = 1025) -> np.ndarray:
    frequencies = 2.0 * np.pi * np.fft.fftfreq(samples)
    horizontal, vertical = np.meshgrid(frequencies, frequencies)
    radius_squared = horizontal * horizontal + vertical * vertical
    sigma = 0.75
    symbol = np.exp(-0.5 * sigma * sigma * (scale ** -2 - 1.0) * radius_squared)
    return np.fft.fftshift(np.fft.ifft2(symbol).real)


def plot_kernel_strip(axes: list[plt.Axes]) -> None:
    names = ["Gaussian idealization", "annular pupil", "astigmatic pupil"]
    kernels = [
        gaussian_kernel_slice(),
        spatial_kernel_slice("annular pupil"),
        spatial_kernel_slice("astigmatic pupil"),
    ]
    for index, (axis, name, kernel) in enumerate(zip(axes, names, kernels)):
        center = kernel.shape[0] // 2
        window = kernel[center - 6 : center + 7, center - 6 : center + 7]
        signed = window / np.max(np.abs(window))
        axis.imshow(
            signed,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            origin="lower",
            interpolation="nearest",
            extent=(-6.5, 6.5, -6.5, 6.5),
        )
        axis.set_title(f"{chr(97 + index)}) {name}", loc="left", fontweight="bold")
        axis.set_xticks([-6, 0, 6])
        axis.set_yticks([-6, 0, 6])
        axis.grid(visible=False)
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        if index == 0:
            axis.set_ylabel("input pixels")
        else:
            axis.set_yticklabels([])
        axis.set_xlabel("input pixels", labelpad=0)
        if index > 0:
            axis.text(
                0.04, 0.96, f"min/peak = {signed.min():.2f}",
                transform=axis.transAxes, ha="left", va="top", fontsize=6.0,
                color="#263238",
            )


def bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def plot_figure(image: pd.DataFrame, models: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.0,
            "legend.fontsize": 5.4,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "savefig.bbox": "tight",
        }
    )
    figure = plt.figure(figsize=(7.05, 4.0))
    kernel_axes = [
        figure.add_axes([0.04, 0.75, 0.09, 0.17]),
        figure.add_axes([0.04, 0.53, 0.09, 0.17]),
        figure.add_axes([0.04, 0.31, 0.09, 0.17]),
    ]
    plot_kernel_strip(kernel_axes)
    titles = ["a) Gaussian idealization", "b) annular pupil", "c) astigmatic pupil"]
    for index, axis in enumerate(kernel_axes):
        axis.set_title(titles[index], loc="left", fontsize=6.0, pad=1)
        if index < 2:
            axis.set_xticklabels([])
            axis.set_xlabel("")
    axis_error = figure.add_axes([0.26, 0.27, 0.245, 0.65])
    axis_flip = figure.add_axes([0.685, 0.27, 0.295, 0.65])
    for index, (label, method) in enumerate(reversed(GROUPS)):
        means, lows, highs = [1e-6], [1e-6], [1e-6]
        for steps in range(2, 6):
            values = 100.0 * image[(image.method == method) & (image.steps == steps)].nrmse.to_numpy()
            mean, low, high = bootstrap(values, 1000 + 31 * index + steps)
            means.append(max(mean, 1e-6))
            lows.append(max(low, 1e-6))
            highs.append(max(high, 1e-6))
        color = COLORS[label]
        control = "family" in label or "control" in label
        linewidth = 1.85 if control else 1.2
        zorder = 8 if control else 3
        line, = axis_error.plot(
            range(1, 6),
            means,
            color=color,
            marker=MARKERS[(len(GROUPS) - 1 - index) % len(MARKERS)],
            markersize=3.7,
            linewidth=linewidth,
            label=label.replace("\n", " "),
            zorder=zorder,
        )
        axis_error.fill_between(range(1, 6), lows, highs, color=color, alpha=0.08, linewidth=0)
    axis_error.set_yscale("log")
    axis_error.set_xlim(1, 5)
    axis_error.set_xticks(range(1, 6))
    axis_error.set_ylim(5e-7, 10)
    axis_error.set_xlabel("number of equivalent resize steps")
    axis_error.set_ylabel("direct--chain disagreement (NRMSE, %)")
    axis_error.set_title("(d) Path dependence changes the image", loc="left", fontweight="bold", fontsize=6.6)

    y_positions = np.arange(len(GROUPS))
    for model_index, (model_name, marker, color) in enumerate(
        [("ResNet-18", "o", "#0072B2"), ("MobileNetV3-Small", "s", "#D55E00")]
    ):
        offsets = -0.12 if model_index == 0 else 0.12
        for row_index, (label, method) in enumerate(GROUPS):
            values = 100.0 * models[
                (models.method == method) & (models.steps == 5) & (models.model == model_name)
            ].top1_flip.astype(float).to_numpy()
            mean, low, high = bootstrap(values, 4000 + 53 * model_index + row_index)
            axis_flip.errorbar(
                mean,
                row_index + offsets,
                xerr=[[mean - low], [high - mean]],
                color=color,
                marker=marker,
                markersize=4.0,
                capsize=1.5,
                elinewidth=0.8,
                linewidth=0,
                label=model_name if row_index == 0 else None,
                zorder=4,
            )
    axis_flip.set_yticks(y_positions)
    axis_flip.set_yticklabels([label for label, _ in GROUPS])
    axis_flip.set_xlim(-1, 51)
    axis_flip.set_xlabel("top-1 prediction changes after five steps (%)")
    axis_flip.set_title("(e) The difference reaches vision models", loc="left", fontweight="bold", fontsize=6.6)
    axis_flip.grid(axis="x", alpha=0.18)
    axis_flip.grid(axis="y", visible=False)
    axis_flip.legend(loc="upper right", frameon=False, fontsize=6.2)
    figure.text(
        0.5,
        0.99,
        "Signed 2D relative kernels at $s=1/2$ (left); 50-scene $2047\\to255$ direct-vs-chain audit (right)",
        ha="center",
        va="top",
        fontsize=7.1,
    )
    handles, labels = axis_error.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045),
        ncol=4, frameon=False, handlelength=1.45, labelspacing=0.32,
        columnspacing=1.0, borderaxespad=0.0,
    )
    figure.savefig(OUTPUT / "library_path_audit_leftstack.pdf")
    figure.savefig(OUTPUT / "library_path_audit_leftstack.png", dpi=300)
    plt.close(figure)


def compact_table(image: pd.DataFrame, models: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, method in GROUPS:
        selected_image = image[(image.method == method) & (image.steps == 5)]
        selected_models = models[(models.method == method) & (models.steps == 5)]
        grid_row = grid[grid.method == method].iloc[0]
        rows.append(
            {
                "configuration": label.replace("\n", " "),
                "image_nrmse_percent": 100.0 * selected_image.nrmse.mean(),
                "resnet_top1_flip_percent": 100.0
                * selected_models[selected_models.model == "ResNet-18"].top1_flip.mean(),
                "mobilenet_top1_flip_percent": 100.0
                * selected_models[selected_models.model == "MobileNetV3-Small"].top1_flip.mean(),
                "affine_ramp_rms_source_pixels": float(grid_row.rms_source_pixels),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    table_x4: pd.DataFrame,
    table_x8: pd.DataFrame,
    models_x8: pd.DataFrame,
    metadata_x8: dict,
) -> None:
    high_confidence = models_x8[(models_x8.steps == 5) & (models_x8.direct_confidence >= 0.25)]
    high_rows = []
    for label, method in GROUPS:
        selected = high_confidence[high_confidence.method == method]
        high_rows.append(
            {
                "configuration": label.replace("\n", " "),
                "ResNet-18": 100.0
                * selected[selected.model == "ResNet-18"].top1_flip.mean(),
                "MobileNetV3-Small": 100.0
                * selected[selected.model == "MobileNetV3-Small"].top1_flip.mean(),
                "n ResNet": int((selected.model == "ResNet-18").sum()),
                "n MobileNet": int((selected.model == "MobileNetV3-Small").sum()),
            }
        )
    high_table = pd.DataFrame(high_rows)
    def markdown_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
        formatted = frame.copy()
        for column, format_string in formats.items():
            formatted[column] = formatted[column].map(lambda value: format_string.format(value))
        return formatted.to_markdown(index=False)

    report = f"""# Audit of equivalent resize programs

## Headline

Across 50 natural SR-RAW scenes, a direct $2047\\to255$ resize was compared with two-to-five-step programs having exactly the same endpoint and approximately equal log-scale steps. After five steps, the largest native pipelines disagree with their own direct result by about 5% NRMSE and change the top-1 prediction of two pretrained ImageNet models on 36--38% of crops. Three declared consistent families--a Gaussian idealization and annular- and astigmatic-pupil cameras--compose to floating-point precision and change no predictions. The figure displays their signed, two-dimensional $K_{{1/2}}$ relative kernels, not the camera PSFs. The annular example has a negative ring reaching about 13% of its positive peak, although its underlying incoherent camera PSF is nonnegative.

This is a path-dependence audit, not an accuracy comparison against ground truth. Prediction flips show that the numerical ambiguity can propagate into a vision system; they do not say which native path predicts the correct class.

## Five-step results, total x1/8

{markdown_table(table_x8, {
    'image_nrmse_percent': '{:.3f}',
    'resnet_top1_flip_percent': '{:.1f}',
    'mobilenet_top1_flip_percent': '{:.1f}',
    'affine_ramp_rms_source_pixels': '{:.4f}',
})}

The affine-ramp column is reported in original source pixels. It audits the complete discrete operator's phase/first-moment behavior, so kernel and grid effects are not silently conflated. It is not used to subtract or correct the image results.

## Sensitivity, total x1/4

{markdown_table(table_x4, {
    'image_nrmse_percent': '{:.3f}',
    'resnet_top1_flip_percent': '{:.1f}',
    'mobilenet_top1_flip_percent': '{:.1f}',
    'affine_ramp_rms_source_pixels': '{:.4f}',
})}

The ordering and conclusion persist at x1/4. Pillow/PyTorch/JAX linear AA agree with each other to approximately $1.5\\times10^{{-7}}$ NRMSE; Pillow/PyTorch bicubic AA and Pillow/JAX Lanczos-3 similarly form cross-library equivalence classes in this protocol.

## Prediction flips above 0.25 direct confidence, x1/8

{markdown_table(high_table, {
    'ResNet-18': '{:.1f}',
    'MobileNetV3-Small': '{:.1f}',
    'n ResNet': '{:d}',
    'n MobileNet': '{:d}',
})}

At confidence at least 0.5, no top-1 labels change in this 50-scene sample, although probability-vector and feature differences remain. Therefore the defensible paper claim is that equivalent resize programs can change model outputs and low-to-moderate-confidence decisions, not that they routinely overturn high-confidence recognition.

## Protocol

- Dataset: 50 scene folders below `{metadata_x8['protocol']['dataset']}`; `00001.JPG` from each.
- Source: centered 2047x2047 float RGB crop. Endpoint: 255x255; model input: central 224x224 crop.
- Paths: {metadata_x8['protocol']['paths']}.
- Quantization: none between steps; all resizers receive float32 values.
- Boundaries: only the central 224x224 of the 255x255 output is measured.
- Models: ImageNet-pretrained ResNet-18 and MobileNetV3-Small from torchvision; no model-owned resizing occurs.
- Uncertainty: the figure uses scene-level nonparametric 95% bootstrap intervals.
- Native implementations: Pillow, OpenCV, PyTorch, scikit-image, and JAX. Output sizes are explicit; their settings are recorded in `library_path_audit_results_x8.json`.
- Consistent controls: odd-grid centered Fourier realizations of three declared camera families, with the ratio $H_{{target}}(\\omega)/H_{{source}}(\\omega)$ on retained physical modes. The Gaussian transfer function has sigma 0.75 input pixels; the other two are normalized autocorrelations of annular and astigmatic circular pupils, include square-pixel integration, and have finite optical cutoff $0.9\\pi$ in reference-camera pixel units. The annular pupil has a 50%-radius central obstruction; the astigmatic pupil phase is $0.25$ waves. The numerical controls use periodic Fourier synthesis rather than an enumerated finite-neighbor implementation of Eq. (11). Their settings are recorded in `library_path_audit_optical_x8.json` and `library_path_audit_annular_x8.json`. They test path composition, not whether the SR-RAW JPEGs were captured with any of these three PSFs.

## Interpretation for the paper

The strongest result is not that every library is equally inconsistent. It is that path dependence varies over four orders of magnitude across widely used choices, reaches several percent for common pipelines, and can alter downstream predictions, whereas three different declared consistent families remove the path ambiguity by construction. This demonstrates that consistency is not peculiar to the Gaussian; it does not establish camera matching or recognition accuracy. Lanczos-3 is an important near-consistent empirical exception in this finite protocol, but its residual is still about 0.14% NRMSE at x1/8 rather than an operator identity. The native methods' different magnitudes are plausible consequences of their different kernels, anti-alias filters, grid and boundary conventions; this audit does not uniquely assign a numerical share to each cause.

The clean main-paper figure is `library_path_audit.pdf`. The x1/4 sensitivity, confidence-conditioned flips, all per-scene measurements, versions, and grid audit belong in the appendix or supplementary material.
"""
    (OUTPUT / "library_path_audit_report.md").write_text(report)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_x4 = pd.concat(
        [pd.read_csv(X4 / "image_metrics.csv"), pd.read_csv(OPTICAL_X4 / "image_metrics.csv"), pd.read_csv(ANNULAR_X4 / "image_metrics.csv")],
        ignore_index=True,
    )
    models_x4 = pd.concat(
        [pd.read_csv(X4 / "model_metrics.csv"), pd.read_csv(OPTICAL_X4 / "model_metrics.csv"), pd.read_csv(ANNULAR_X4 / "model_metrics.csv")],
        ignore_index=True,
    )
    grid_x4 = pd.concat(
        [pd.read_csv(X4 / "grid_audit.csv"), pd.read_csv(OPTICAL_X4 / "grid_audit.csv"), pd.read_csv(ANNULAR_X4 / "grid_audit.csv")],
        ignore_index=True,
    )
    image_x8 = pd.concat(
        [pd.read_csv(X8 / "image_metrics.csv"), pd.read_csv(OPTICAL_X8 / "image_metrics.csv"), pd.read_csv(ANNULAR_X8 / "image_metrics.csv")],
        ignore_index=True,
    )
    models_x8 = pd.concat(
        [pd.read_csv(X8 / "model_metrics.csv"), pd.read_csv(OPTICAL_X8 / "model_metrics.csv"), pd.read_csv(ANNULAR_X8 / "model_metrics.csv")],
        ignore_index=True,
    )
    grid_x8 = pd.concat(
        [pd.read_csv(X8 / "grid_audit.csv"), pd.read_csv(OPTICAL_X8 / "grid_audit.csv"), pd.read_csv(ANNULAR_X8 / "grid_audit.csv")],
        ignore_index=True,
    )
    metadata_x8 = json.loads((X8 / "results.json").read_text())
    plot_figure(image_x8, models_x8)
    table_x4 = compact_table(image_x4, models_x4, grid_x4)
    table_x8 = compact_table(image_x8, models_x8, grid_x8)
    table_x4.to_csv(OUTPUT / "library_path_audit_x4_summary.csv", index=False)
    table_x8.to_csv(OUTPUT / "library_path_audit_x8_summary.csv", index=False)
    write_report(table_x4, table_x8, models_x8, metadata_x8)
    for source, destination in [
        (X4 / "results.json", OUTPUT / "library_path_audit_results_x4.json"),
        (X8 / "results.json", OUTPUT / "library_path_audit_results_x8.json"),
        (X8 / "image_metrics.csv", OUTPUT / "library_path_audit_image_metrics_x8.csv"),
        (X8 / "model_metrics.csv", OUTPUT / "library_path_audit_model_metrics_x8.csv"),
        (X8 / "grid_audit.csv", OUTPUT / "library_path_audit_grid_x8.csv"),
        (OPTICAL_X4 / "results.json", OUTPUT / "library_path_audit_optical_x4.json"),
        (OPTICAL_X8 / "results.json", OUTPUT / "library_path_audit_optical_x8.json"),
        (OPTICAL_X8 / "image_metrics.csv", OUTPUT / "library_path_audit_optical_image_x8.csv"),
        (OPTICAL_X8 / "model_metrics.csv", OUTPUT / "library_path_audit_optical_models_x8.csv"),
        (ANNULAR_X4 / "results.json", OUTPUT / "library_path_audit_annular_x4.json"),
        (ANNULAR_X8 / "results.json", OUTPUT / "library_path_audit_annular_x8.json"),
        (ANNULAR_X8 / "image_metrics.csv", OUTPUT / "library_path_audit_annular_image_x8.csv"),
        (ANNULAR_X8 / "model_metrics.csv", OUTPUT / "library_path_audit_annular_models_x8.csv"),
        (ROOT / "work/library_audit/optical_cameras.py", OUTPUT / "library_path_audit_optical_cameras.py"),
        (ROOT / "work/library_audit/library_model_audit.py", OUTPUT / "library_path_audit_experiment.py"),
        (Path(__file__), OUTPUT / "library_path_audit_make_report.py"),
    ]:
        shutil.copy2(source, destination)
    print("wrote publication figure, report, summaries, and raw outputs")


if __name__ == "__main__":
    image_x8 = pd.concat(
        [pd.read_csv(X8 / "image_metrics.csv"),
         pd.read_csv(OPTICAL_X8 / "image_metrics.csv"),
         pd.read_csv(ANNULAR_X8 / "image_metrics.csv")],
        ignore_index=True,
    )
    models_x8 = pd.concat(
        [pd.read_csv(X8 / "model_metrics.csv"),
         pd.read_csv(OPTICAL_X8 / "model_metrics.csv"),
         pd.read_csv(ANNULAR_X8 / "model_metrics.csv")],
        ignore_index=True,
    )
    plot_figure(image_x8, models_x8)
    print("wrote left-stack PDF and PNG figure")
