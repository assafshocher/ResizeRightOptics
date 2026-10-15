"""Plot retained-band reruns on ordinary bicubic inputs, without altering old figures.

Read verified summary files only; never aggregate a subset of Set5 images.
The PyTorch-grid control changes downsampling coordinates only: upsampling still
uses ResizeRight, so this is not an unmodified end-to-end PyTorch pipeline.
"""
import argparse
import json
from pathlib import Path

from gen_fig_matched_modes import IMAGES, METHODS, STEPS, panel, save, plt, np


HERE = Path(__file__).resolve().parent
SCALES = (2, 3, 4)
MODES = ("bandlimited", "literal_projected")
ANCHOR_CAVEAT = (
    "On ordinary bicubic inputs, ours uses the original retained-band kernels; "
    "its anchor is approximate and its one-step network is independently trained."
)
GRID_CAVEAT = (
    "AA (PyTorch grid) changes downsampling to the floor/half-pixel grid while "
    "retaining the matched filter and ResizeRight upsampling; it is not an "
    "unmodified end-to-end torch.interpolate pipeline."
)


def load_summary(mode, scale, allow_partial=False):
    prefix = "matched_bandlimited" if mode == "bandlimited" else mode
    path = HERE / f"{prefix}_x{scale}_summary.json"
    if not path.exists():
        if not allow_partial:
            raise FileNotFoundError(path)
        return {
            "mode": mode, "scale": scale, "m": STEPS, "complete": False,
            "missing_records": len(IMAGES) * len(METHODS) * len(STEPS),
            "mean_psnr_y": {key: [None] * len(STEPS) for key in METHODS},
            "baseline_mean_y": None,
        }
    summary = json.loads(path.read_text())
    if summary["scale"] != scale or summary["m"] != STEPS:
        raise ValueError(f"Wrong scale or step range in {path}")
    if len(summary["images"]) != len(IMAGES) or set(summary["images"]) != set(IMAGES):
        raise ValueError(f"Expected all five Set5 images in {path}")
    missing = 0
    for method in METHODS:
        means = summary["mean_psnr_y"][method]
        per_image = summary["per_image_psnr_y"][method]
        if len(means) != len(STEPS) or any(len(per_image[i]) != len(STEPS) for i in IMAGES):
            raise ValueError(f"Wrong step count for {method} in {path}")
        for k, mean in enumerate(means):
            values = [per_image[image][k] for image in IMAGES]
            missing += sum(v is None for v in values)
            if any(v is not None and not np.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite per-image value in {path}, {method}, m={k + 1}")
            if any(v is None for v in values):
                if mean is not None:
                    raise ValueError(f"Partial-image mean forbidden in {path}, {method}, m={k + 1}")
            elif mean is None or not np.isfinite(mean) or not np.isclose(
                mean, np.mean(values), rtol=0, atol=1e-8
            ):
                raise ValueError(f"Incorrect five-image mean in {path}, {method}, m={k + 1}")
    if summary["missing_records"] != missing or summary["complete"] != (missing == 0):
        raise ValueError(f"Completion metadata disagrees with per-image records in {path}")
    baseline = summary["baseline_mean_y"]
    if baseline is not None and not np.isfinite(baseline):
        raise ValueError(f"Nonfinite baseline in {path}")
    if summary["complete"] and baseline is None:
        raise ValueError(f"Complete panel lacks its baseline in {path}")
    if not summary["complete"] and not allow_partial:
        raise ValueError(f"Incomplete summary: {path} ({missing} missing records)")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    summaries = {
        (mode, scale): load_summary(mode, scale, args.allow_partial)
        for mode in MODES for scale in SCALES
    }
    outputs = []
    for scale in SCALES:
        summary = summaries["literal_projected", scale]
        if not summary["complete"]:
            continue
        fig, ax = plt.subplots(figsize=(3.45, 2.8))
        panel(ax, summary)
        ax.set_ylabel("Mean Y-PSNR (dB)")
        ax.set_title(rf"Ordinary bicubic input — $\times {scale}$", fontsize=8.5)
        fig.text(.56, .99, "Ours: retained-band; approximate anchor", ha="center",
                 va="top", fontsize=7.5)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
                   handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(.53, .01))
        fig.subplots_adjust(left=.18, right=.98, bottom=.35, top=.85)
        name = f"fig_literal_projected_x{scale}"
        save(fig, name)
        outputs.append(str(HERE / f"{name}.png"))

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9))
    for i, mode in enumerate(MODES):
        for j, scale in enumerate(SCALES):
            panel(axes[i, j], summaries[mode, scale])
        axes[i, 0].set_ylabel("Mean Y-PSNR (dB)")
    fig.text(.54, .98, "Band-limited bicubic input", ha="center", va="top", fontsize=10)
    fig.text(.54, .535, "Ordinary bicubic input (ours: retained-band)",
             ha="center", va="top", fontsize=10)
    fig.text(.54, .505, "Ours: approximate anchor; independently trained one-step model",
             ha="center", va="top", fontsize=7.5, color=".35")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4,
               handlelength=1.6, columnspacing=1.15, bbox_to_anchor=(.53, .005))
    fig.subplots_adjust(left=.09, right=.99, bottom=.13, top=.91, wspace=.32, hspace=.78)
    complete = all(s["complete"] for s in summaries.values())
    name = "fig_set5_zssr_projected_six_panel" + ("" if complete else "_progress")
    save(fig, name)
    outputs.append(str(HERE / f"{name}.png"))
    print(json.dumps({
        "complete": complete, "figure": outputs[-1], "outputs": outputs,
        "anchor_caveat": ANCHOR_CAVEAT, "grid_caveat": GRID_CAVEAT,
        "aggregation": "Arithmetic mean of all five per-image Y-PSNR values; one seed",
        "panels": [{"mode": mode, "scale": scale, "missing": s["missing_records"]}
                   for (mode, scale), s in summaries.items()],
    }), flush=True)


if __name__ == "__main__":
    main()
