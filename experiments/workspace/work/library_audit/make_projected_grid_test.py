"""Grid-only companion to the path-composition experiment.

This diagnostic fixes the prescribed physical zoom s and output count N.
It samples one already band-limited continuous target field on either
the paper's projected grid or a grid that silently replaces s with N/N0.
No filtering, cropping, padding, or boundary rule differs between arms.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"
SOURCE_SIZE = 1001
PHYSICAL_SCALE = 0.47
OUTPUT_SIZE = round(PHYSICAL_SCALE * SOURCE_SIZE)


def projected_grid(scale: float) -> np.ndarray:
    indices = np.arange(OUTPUT_SIZE, dtype=np.float64)
    return (indices - (OUTPUT_SIZE - 1) / 2.0) / scale + (SOURCE_SIZE - 1) / 2.0


def target_field(horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    """Continuous target field with every output-grid frequency below Nyquist."""
    x = PHYSICAL_SCALE * (horizontal - (SOURCE_SIZE - 1) / 2.0)
    y = PHYSICAL_SCALE * (vertical - (SOURCE_SIZE - 1) / 2.0)
    return (
        0.5
        + 0.21 * np.cos(0.70 * np.pi * x)
        + 0.19 * np.cos(0.60 * np.pi * y)
        + 0.10 * np.cos(0.30 * np.pi * (x + y))
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    physical_grid = projected_grid(PHYSICAL_SCALE)
    inferred_scale = OUTPUT_SIZE / SOURCE_SIZE
    count_grid = projected_grid(inferred_scale)
    displacement = count_grid - physical_grid
    true_x, true_y = np.meshgrid(physical_grid, physical_grid)
    wrong_x, wrong_y = np.meshgrid(count_grid, count_grid)
    true_image = target_field(true_x, true_y)
    wrong_image = target_field(wrong_x, wrong_y)
    difference = wrong_image - true_image
    nrmse = np.sqrt(np.mean(difference**2) / np.mean(true_image**2)) * 100.0

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    figure, (axis_grid, axis_image) = plt.subplots(
        1, 2, figsize=(7.05, 2.8), gridspec_kw={"wspace": 0.38, "width_ratios": [1.05, 1.0]}
    )
    axis_grid.plot(np.arange(OUTPUT_SIZE), displacement, color="#0072B2", linewidth=1.8)
    axis_grid.axhline(0, color="#777777", linewidth=0.7)
    axis_grid.set_xlim(0, OUTPUT_SIZE - 1)
    axis_grid.set_ylim(-0.57, 0.57)
    axis_grid.set_xlabel("output index $n$")
    axis_grid.set_ylabel("grid displacement (input pixels)")
    axis_grid.set_title("(a) Replacing $s$ by $N/N_0$", loc="left", fontweight="bold")
    axis_grid.grid(alpha=0.18, linewidth=0.5)

    crop = difference[:128, :128]
    maximum = float(np.max(np.abs(crop)))
    display = axis_image.imshow(
        crop, origin="lower", cmap="RdBu_r", vmin=-maximum, vmax=maximum,
        interpolation="nearest", extent=(0, 128, 0, 128),
    )
    axis_image.set_xlabel("output $x$ index")
    axis_image.set_ylabel("output $y$ index")
    axis_image.set_title("(b) Same target field, different samples", loc="left", fontweight="bold")
    axis_image.grid(visible=False)
    colorbar = figure.colorbar(display, ax=axis_image, shrink=0.75, pad=0.03)
    colorbar.set_label("inferred minus projected")
    figure.suptitle(
        "$N_0=1001$, prescribed $s=0.47$, common $N=470$: "
        f"grid-only NRMSE {nrmse:.1f}\\%",
        y=0.99, fontsize=8.3,
    )
    figure.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.95)
    figure.savefig(OUTPUT / "projected_grid_phase_test.pdf")
    figure.savefig(OUTPUT / "projected_grid_phase_test.png", dpi=300)
    plt.close(figure)
    print(
        f"s={PHYSICAL_SCALE}; N0={SOURCE_SIZE}; N={OUTPUT_SIZE}; "
        f"N/N0={inferred_scale:.8f}; max grid drift={np.max(np.abs(displacement)):.6f} "
        f"input pixels; grid-only NRMSE={nrmse:.3f}%"
    )


if __name__ == "__main__":
    main()
