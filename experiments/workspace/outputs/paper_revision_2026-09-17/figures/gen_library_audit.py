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


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "figs"


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
    figure = plt.figure(figsize=(8.0, 2.7))
    kernel_axes = [
        figure.add_axes([0.035, 0.735, 0.055, 0.163]),
        figure.add_axes([0.035, 0.440, 0.055, 0.163]),
        figure.add_axes([0.035, 0.145, 0.055, 0.163]),
    ]
    plot_kernel_strip(kernel_axes)
    titles = ["a) Gaussian idealization", "b) annular pupil", "c) astigmatic pupil"]
    for index, axis in enumerate(kernel_axes):
        axis.set_title(titles[index], loc="left", fontsize=6.0, pad=3)
        axis.tick_params(length=2, pad=1, labelsize=5.8)
        axis.set_ylabel("")
        for annotation in axis.texts:
            annotation.set_position((0.5, -0.12))
            annotation.set_ha("center")
            annotation.set_va("top")
            annotation.set_fontsize(5.8)
        if index < 2:
            axis.set_xticklabels([])
            axis.set_xlabel("")
        elif axis.texts:
            axis.texts[0].set_position((0.5, -0.67))
    axis_error = figure.add_axes([0.350, 0.20, 0.230, 0.69])
    axis_flip = figure.add_axes([0.760, 0.20, 0.225, 0.69])
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
        handles, labels, loc="center left", bbox_to_anchor=(0.145, 0.545),
        ncol=1, frameon=False, handlelength=1.45, labelspacing=0.85,
        handletextpad=0.5, borderaxespad=0.0, fontsize=6.0,
    )
    figure.savefig(OUTPUT / "library_path_audit.pdf")
    figure.savefig(OUTPUT / "library_path_audit.png", dpi=300)
    plt.close(figure)



if __name__ == "__main__":
    image = pd.concat([pd.read_csv(DATA / name) for name in [
        "library_path_audit_image_metrics_x8.csv",
        "library_path_audit_optical_image_x8.csv",
        "library_path_audit_annular_image_x8.csv"]], ignore_index=True)
    models = pd.concat([pd.read_csv(DATA / name) for name in [
        "library_path_audit_model_metrics_x8.csv",
        "library_path_audit_optical_models_x8.csv",
        "library_path_audit_annular_models_x8.csv"]], ignore_index=True)
    plot_figure(image, models)
