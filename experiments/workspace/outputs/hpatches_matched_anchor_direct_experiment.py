"""Direct-only matched-anchor HPatches audit, including an optional Gaussian.

Every compared pair of laws shares exactly the same 1/2 kernel and Fourier
grid. Gaussian width is fixed from the linear/cubic anchor Nyquist MTFs, not
tuned on HPatches scores. This is a kernel-model audit, not native resize calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import hpatches_geometry_experiment as geometry
import hpatches_matched_anchor_experiment as matched


LAWS = ("AA stretch", "consistent telescope")


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    dataset = args.dataset.resolve()
    sequences = sorted(
        directory for directory in dataset.iterdir()
        if directory.is_dir() and (directory / "1.ppm").exists()
    )
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    args.output.mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.01)
    task_rows: list[dict] = []
    anchor_rows: list[dict] = []

    for sequence_index, directory in enumerate(sequences, start=1):
        originals = {
            number: matched.trim_to_exact_half(
                geometry.load_grayscale(directory / f"{number}.ppm")
            )
            for number in range(1, 7)
        }
        for base in args.bases:
            extracted: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray | None]] = {}
            final_shapes: dict[tuple[int, int], tuple[int, int]] = {}
            for number, original in originals.items():
                half_shape = tuple(dimension // 2 for dimension in original.shape)
                anchor_aa = matched.spectral_resize(original, half_shape, base, "AA stretch")
                anchor_consistent = matched.spectral_resize(
                    original, half_shape, base, "consistent telescope"
                )
                anchor_rows.append(
                    {
                        "sequence": directory.name,
                        "image": number,
                        "base": base,
                        "anchor_nrmse_percent": geometry.image_disagreement(
                            anchor_aa, anchor_consistent
                        ),
                    }
                )
                extracted[(number, 2, "shared anchor")] = matched.features(
                    anchor_aa, original.shape, sift
                )
                for factor in args.factors:
                    final_shape = tuple(
                        geometry.nearest_odd(dimension / factor)
                        for dimension in original.shape
                    )
                    final_shapes[(number, factor)] = final_shape
                    for law in LAWS:
                        output = matched.spectral_resize(original, final_shape, base, law)
                        extracted[(number, factor, law)] = matched.features(
                            output, original.shape, sift
                        )

            for target_number in range(2, 7):
                truth = np.loadtxt(
                    directory / f"H_1_{target_number}", dtype=np.float64
                ).reshape(3, 3)
                for factor in (2, *args.factors):
                    cases = ("shared anchor",) if factor == 2 else LAWS
                    for law in cases:
                        reference = extracted[(1, factor, law)]
                        target = extracted[(target_number, factor, law)]
                        final_shape = (
                            tuple(dimension // 2 for dimension in originals[target_number].shape)
                            if factor == 2 else final_shapes[(target_number, factor)]
                        )
                        metrics = geometry.score_pair(
                            reference,
                            target,
                            truth,
                            originals[1].shape,
                            originals[target_number].shape,
                            final_shape,
                            matched.ransac_threshold(
                                originals[target_number].shape, final_shape
                            ),
                        )
                        task_rows.append(
                            {
                                "sequence": directory.name,
                                "type": directory.name[0],
                                "target": target_number,
                                "base": base,
                                "factor": factor,
                                "law": law,
                                "path": "full direct" if factor != 2 else "anchor",
                                **metrics,
                            }
                        )

        if sequence_index % 5 == 0 or sequence_index == len(sequences):
            write_rows(args.output / "task_results.csv", task_rows)
            write_rows(args.output / "anchor_results.csv", anchor_rows)
            print(f"processed {sequence_index}/{len(sequences)} sequences", flush=True)

    anchor_symbol = np.sqrt(
        float(matched.base_transfer("linear tent", np.array(np.pi)))
        * float(matched.base_transfer("Keys cubic a=-0.5", np.array(np.pi)))
    )
    protocol = {
        "dataset": str(dataset),
        "sequences": len(sequences),
        "bases": args.bases,
        "factors": args.factors,
        "anchor_factor": 2,
        "paths": ["full direct"],
        "gaussian_base_symbol": "exp(-alpha omega^2)",
        "gaussian_alpha": matched.gaussian_alpha(),
        "gaussian_sigma_pixels": float(np.sqrt(2 * matched.gaussian_alpha())),
        "gaussian_anchor_nyquist_mtf": float(anchor_symbol),
        "gaussian_width_rule": "geometric mean of linear/cubic x2 Nyquist MTFs; fixed before task scores",
        "model_only": "same periodic Fourier grid and retained-band projection, not literal library calls",
        "homography_min_matches": 8,
        "homography_success_threshold_output_px": 2.5,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", nargs="+", type=int, required=True)
    parser.add_argument(
        "--bases", nargs="+", default=[*matched.BASES, matched.GAUSSIAN_BASE],
        choices=[*matched.BASES, matched.GAUSSIAN_BASE]
    )
    parser.add_argument("--max-sequences", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
