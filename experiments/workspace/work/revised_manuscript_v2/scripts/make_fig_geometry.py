"""Geometric round-trip test for camera-conjugated rigid actions.

The camera has a circular diffraction-limited passband.  The tested rigid
motion is a 90-degree rotation followed by a fractional translation, which
preserves that passband exactly.  We compare two applications of ordinary
bicubic warping with K_0 T K_0^dagger followed by its inverse.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import data


ROOT = Path(__file__).resolve().parents[1]
N = 256
SHIFT = (0.37, -0.44)  # rows, columns, in pixels


def airy_otf(radius: np.ndarray, cutoff: float = 0.82 * np.pi) -> np.ndarray:
    """Incoherent circular-pupil OTF, hard-thresholded for stable inversion."""
    q = np.clip(radius / cutoff, 0.0, 1.0)
    out = np.where(
        radius < cutoff,
        (2.0 / np.pi) * (np.arccos(q) - q * np.sqrt(1.0 - q * q)),
        0.0,
    )
    return np.where(out >= 0.035, out, 0.0)


w = 2.0 * np.pi * np.fft.fftfreq(N)
wy, wx = np.meshgrid(w, w, indexing="ij")
H = airy_otf(np.hypot(wx, wy))
H_DAG = np.zeros_like(H)
H_DAG[H > 0] = 1.0 / H[H > 0]


def filter_symbol(image: np.ndarray, symbol: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(image) * symbol).real


def fourier_shift(image: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    shifted = ndimage.fourier_shift(np.fft.fftn(image), shift=shift)
    return np.fft.ifftn(shifted).real


def rigid(image: np.ndarray) -> np.ndarray:
    return fourier_shift(np.rot90(image, 1), SHIFT)


def rigid_inverse(image: np.ndarray) -> np.ndarray:
    return np.rot90(fourier_shift(image, (-SHIFT[0], -SHIFT[1])), -1)


def camera_action(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    visible_scene = filter_symbol(image, H_DAG)
    moved = rigid_inverse(visible_scene) if inverse else rigid(visible_scene)
    return filter_symbol(moved, H)


def bicubic_action(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        shifted = ndimage.shift(
            image, (-SHIFT[0], -SHIFT[1]), order=3, mode="grid-wrap", prefilter=True
        )
        return np.rot90(shifted, -1)
    rotated = np.rot90(image, 1)
    return ndimage.shift(
        rotated, SHIFT, order=3, mode="grid-wrap", prefilter=True
    )


scene = data.camera()[128:384, 128:384].astype(np.float64) / 255.0
capture = filter_symbol(scene, H)
capture = (capture - capture.min()) / (capture.max() - capture.min())

classical_forward = bicubic_action(capture)
classical_back = bicubic_action(classical_forward, inverse=True)
camera_forward = camera_action(capture)
camera_back = camera_action(camera_forward, inverse=True)

err_classical = classical_back - capture
err_camera = camera_back - capture
rel_classical = np.linalg.norm(err_classical) / np.linalg.norm(capture - capture.mean())
rel_camera = np.linalg.norm(err_camera) / np.linalg.norm(capture - capture.mean())
print(f"classical relative RMSE: {rel_classical:.6e}")
print(f"camera-conjugated relative RMSE: {rel_camera:.6e}")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.2,
        "axes.titlesize": 7.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

fig, axes = plt.subplots(2, 4, figsize=(6.75, 2.65))
fig.subplots_adjust(left=0.055, right=0.995, bottom=0.035, top=0.90, wspace=0.055, hspace=0.24)

rows = [
    (capture, classical_forward, classical_back, err_classical),
    (capture, camera_forward, camera_back, err_camera),
]
row_names = ["classical bicubic", r"camera coordinates  $K_0 T K_\infty$"]
column_names = ["input capture", r"rigid action $g$", r"inverse action $g^{-1}$", "round-trip error"]
error_scale = float(np.max(np.abs(err_classical)))

for r, row in enumerate(rows):
    for c, image in enumerate(row):
        ax = axes[r, c]
        if c < 3:
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        else:
            ax.imshow(image, cmap="RdBu_r", vmin=-error_scale, vmax=error_scale, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#666666")
        if r == 0:
            ax.set_title(column_names[c], pad=2.5)

axes[0, 3].text(
    0.5,
    0.035,
    rf"rel. RMSE $={rel_classical:.1e}$",
    transform=axes[0, 3].transAxes,
    ha="center",
    va="bottom",
    fontsize=6.5,
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
)
axes[1, 3].text(
    0.5,
    0.035,
    rf"rel. RMSE $={rel_camera:.0e}$",
    transform=axes[1, 3].transAxes,
    ha="center",
    va="bottom",
    fontsize=6.5,
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
)

for r, name in enumerate(row_names):
    axes[r, 0].text(
        -0.055,
        0.5,
        name,
        transform=axes[r, 0].transAxes,
        rotation=90,
        ha="right",
        va="center",
        fontsize=7.0,
        fontweight="bold" if r == 1 else "normal",
        color="#0072B2" if r == 1 else "#555555",
    )

for r in range(2):
    for c in range(3):
        left = axes[r, c].get_position()
        right = axes[r, c + 1].get_position()
        axes[r, c].annotate(
            "",
            xy=(right.x0 - 0.004, (right.y0 + right.y1) / 2),
            xytext=(left.x1 + 0.004, (left.y0 + left.y1) / 2),
            xycoords=fig.transFigure,
            textcoords=fig.transFigure,
            arrowprops={"arrowstyle": "-|>", "lw": 0.65, "color": "#333333", "mutation_scale": 6},
        )

fig.savefig(ROOT / "figs" / "fig_geometry.pdf")
fig.savefig(ROOT / "figs" / "fig_geometry.png")
