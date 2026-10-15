"""HPatches audit with a declared unit-width interpolation-kernel baseline.

This is a kernel-law model, not a bitwise emulation of an image library or a
claim that signed bicubic/Lanczos kernels are physical intensity PSFs. The AA
downscale branch is deliberately identical to the earlier matched-x2 audit.
The consistent branch now chooses the base kernel's Fourier symbol as the
reference camera symbol and uses H(r omega)/H(omega), rather than completing
an AA-stretched x2 slice by the dyadic telescope. The x1 resize is identity
at sample centers for all three cardinal interpolation kernels.

Two distinct tasks are scored: (1) direct low-resolution homography estimation
and (2) downscale, interpolate back to the original dimensions, then estimate
the homography there. The latter is down-then-up, never an inverse of the
frequency-discarding downscale.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

import hpatches_geometry_experiment as geometry
import hpatches_matched_anchor_experiment as old


BASES = ("linear tent", "Keys cubic a=-0.5", "Lanczos-3")
LAWS = ("AA stretch", "unit-kernel consistent")
FACTORS = (4, 8, 12, 16, 20, 24, 28, 32)


@lru_cache(maxsize=1)
def lanczos_symbol_table() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 3, 2049, dtype=np.float64)
    kernel = np.sinc(x) * np.sinc(x / 3)
    frequencies = np.linspace(0, np.pi, 2049, dtype=np.float64)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    values = 2 * integrate(
        np.cos(frequencies[:, None] * x[None, :]) * kernel[None, :],
        x,
        axis=1,
    )
    values /= values[0]
    if np.min(values) <= 0:
        raise ValueError("Lanczos-3 symbol crosses zero on the retained band")
    return frequencies, values


def symbol(base: str, omega: np.ndarray) -> np.ndarray:
    if base == "Lanczos-3":
        frequencies, values = lanczos_symbol_table()
        return np.interp(np.abs(omega), frequencies, values)
    return old.base_transfer(base, omega)


def cardinal_kernel(base: str, x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    if base == "linear tent":
        return np.maximum(1 - a, 0)
    if base == "Keys cubic a=-0.5":
        parameter = -0.5
        return np.where(
            a < 1,
            (parameter + 2) * a**3 - (parameter + 3) * a**2 + 1,
            np.where(
                a < 2,
                parameter * a**3 - 5 * parameter * a**2
                + 8 * parameter * a - 4 * parameter,
                0,
            ),
        )
    if base == "Lanczos-3":
        return np.where(a < 3, np.sinc(x) * np.sinc(x / 3), 0)
    raise KeyError(base)


def support_radius(base: str) -> int:
    return {"linear tent": 1, "Keys cubic a=-0.5": 2, "Lanczos-3": 3}[base]


def interpolate_axis(
    image: np.ndarray, output_length: int, base: str, axis: int
) -> np.ndarray:
    """Unit-width cardinal interpolation on the same centered periodic grid."""
    input_length = image.shape[axis]
    positions = (np.arange(output_length) + 0.5) * input_length / output_length - 0.5
    left = np.floor(positions).astype(np.int64)
    radius = support_radius(base)
    offsets = range(1 - radius, radius + 1)
    indices = [left + offset for offset in offsets]
    weights = [cardinal_kernel(base, positions - index) for index in indices]
    normalization = np.sum(weights, axis=0)
    if np.any(np.abs(normalization) < 1e-9):
        raise ValueError("cardinal interpolation has vanishing local normalization")
    result = np.zeros(
        (output_length, image.shape[1]) if axis == 0 else (image.shape[0], output_length),
        dtype=np.float32,
    )
    for index, weight in zip(indices, weights):
        picked = np.take(image, index % input_length, axis=axis)
        normalized = (weight / normalization).astype(np.float32)
        result += picked * (normalized[:, None] if axis == 0 else normalized[None, :])
    return result


def conventional_up(image: np.ndarray, shape: tuple[int, int], base: str) -> np.ndarray:
    if shape[0] < image.shape[0] or shape[1] < image.shape[1]:
        raise ValueError("conventional_up only interpolates to a larger shape")
    return interpolate_axis(
        interpolate_axis(image, shape[1], base, axis=1),
        shape[0], base, axis=0,
    )


@lru_cache(maxsize=8192)
def down_axis(
    input_length: int, output_length: int, base: str, law: str
) -> tuple[np.ndarray, np.ndarray]:
    if output_length > input_length or output_length % 2 != 1:
        raise ValueError("downscaling requires an odd output Fourier grid")
    modes = np.arange(-(output_length // 2), output_length // 2 + 1)
    omega = 2 * np.pi * modes / input_length
    ratio = input_length / output_length
    numerator = symbol(base, ratio * omega)
    denominator = symbol(base, omega)
    if np.min(denominator) <= 0:
        raise ValueError(f"unstable base symbol for {base}")
    if law == "AA stretch":
        multiplier = numerator
    elif law == "unit-kernel consistent":
        multiplier = numerator / denominator
    else:
        raise KeyError(law)
    input_offset = -(input_length - 1) / 2
    output_offset = -ratio * (output_length - 1) / 2
    phase = np.exp(1j * omega * (output_offset - input_offset))
    return multiplier, phase


def spectral_down(
    image: np.ndarray, shape: tuple[int, int], base: str, law: str
) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    return spectral_down_from_spectrum(spectrum, image.shape, shape, base, law)


def spectral_down_from_spectrum(
    spectrum: np.ndarray,
    input_shape: tuple[int, int],
    shape: tuple[int, int],
    base: str,
    law: str,
) -> np.ndarray:
    input_height, input_width = input_shape
    output_height, output_width = shape
    multiplier_y, phase_y = down_axis(input_height, output_height, base, law)
    multiplier_x, phase_x = down_axis(input_width, output_width, base, law)
    y_start = input_height // 2 - output_height // 2
    x_start = input_width // 2 - output_width // 2
    cropped = spectrum[
        y_start : y_start + output_height,
        x_start : x_start + output_width,
    ]
    transfer = (
        multiplier_y[:, None] * multiplier_x[None, :]
        * phase_y[:, None] * phase_x[None, :]
        * (output_height / input_height) * (output_width / input_width)
    )
    return np.fft.ifft2(np.fft.ifftshift(cropped * transfer)).real.astype(np.float32)


def consistent_down(
    image: np.ndarray, shape: tuple[int, int], base: str
) -> np.ndarray:
    return spectral_down(image, shape, base, "unit-kernel consistent")


def consistent_up(
    image: np.ndarray, shape: tuple[int, int], base: str
) -> np.ndarray:
    """Retained-band pseudo-inverse of consistent_down on matching dimensions."""
    low_height, low_width = image.shape
    high_height, high_width = shape
    multiplier_y, phase_y = down_axis(
        high_height, low_height, base, "unit-kernel consistent"
    )
    multiplier_x, phase_x = down_axis(
        high_width, low_width, base, "unit-kernel consistent"
    )
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    embedded = np.zeros(shape, dtype=np.complex128)
    y_start = high_height // 2 - low_height // 2
    x_start = high_width // 2 - low_width // 2
    embedded[y_start : y_start + low_height, x_start : x_start + low_width] = (
        spectrum
        / (multiplier_y[:, None] * multiplier_x[None, :])
        / (phase_y[:, None] * phase_x[None, :])
        * (high_height / low_height) * (high_width / low_width)
    )
    return np.fft.ifft2(np.fft.ifftshift(embedded)).real.astype(np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
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
    image_rows: list[dict] = []

    for sequence_index, directory in enumerate(sequences, start=1):
        originals = {
            number: old.trim_to_exact_half(
                geometry.load_grayscale(directory / f"{number}.ppm")
            )
            for number in range(1, 7)
        }
        spectra = {
            number: np.fft.fftshift(np.fft.fft2(original))
            for number, original in originals.items()
        }
        for base in args.bases:
            features: dict[tuple[int, int, str, str], tuple[np.ndarray, np.ndarray | None]] = {}
            final_shapes: dict[tuple[int, int], tuple[int, int]] = {}
            for number, original in originals.items():
                features[(number, 1, "shared", "original")] = old.features(
                    original, original.shape, sift
                )
                for factor in args.factors:
                    final_shape = tuple(
                        geometry.nearest_odd(dimension / factor)
                        for dimension in original.shape
                    )
                    final_shapes[(number, factor)] = final_shape
                    low_images = {
                        "AA stretch": spectral_down_from_spectrum(
                            spectra[number], original.shape,
                            final_shape, base, "AA stretch"
                        ),
                        "unit-kernel consistent": spectral_down_from_spectrum(
                            spectra[number], original.shape,
                            final_shape, base, "unit-kernel consistent"
                        ),
                    }
                    for law, low in low_images.items():
                        high = (
                            conventional_up(low, original.shape, base)
                            if law == "AA stretch"
                            else consistent_up(low, original.shape, base)
                        )
                        features[(number, factor, law, "direct low")] = old.features(
                            low, original.shape, sift
                        )
                        features[(number, factor, law, "down then up")] = old.features(
                            high, original.shape, sift
                        )
                        image_rows.append(
                            {
                                "sequence": directory.name,
                                "image": number,
                                "base": base,
                                "factor": factor,
                                "law": law,
                                "roundtrip_nrmse_percent": geometry.image_disagreement(
                                    original, high
                                ),
                                "low_height": final_shape[0],
                                "low_width": final_shape[1],
                            }
                        )

            for target_number in range(2, 7):
                truth = np.loadtxt(
                    directory / f"H_1_{target_number}", dtype=np.float64
                ).reshape(3, 3)
                reference_shape = originals[1].shape
                target_shape = originals[target_number].shape
                cases = [(1, "shared", "original")]
                cases.extend(
                    (factor, law, path)
                    for factor in args.factors for law in LAWS
                    for path in ("direct low", "down then up")
                )
                for factor, law, path in cases:
                    reference = features[(1, factor, law, path)]
                    target = features[(target_number, factor, law, path)]
                    output_shape = (
                        target_shape if path != "direct low"
                        else final_shapes[(target_number, factor)]
                    )
                    ransac_threshold = (
                        old.ransac_threshold(target_shape, output_shape)
                        if path == "direct low" else 5.0
                    )
                    metrics = geometry.score_pair(
                        reference, target, truth,
                        reference_shape, target_shape, output_shape,
                        ransac_threshold,
                    )
                    task_rows.append(
                        {
                            "sequence": directory.name,
                            "type": directory.name[0],
                            "target": target_number,
                            "base": base,
                            "factor": factor,
                            "law": law,
                            "path": path,
                            **metrics,
                        }
                    )
        if sequence_index % 5 == 0 or sequence_index == len(sequences):
            write_csv(args.output / "task_results.csv", task_rows)
            write_csv(args.output / "image_results.csv", image_rows)
            print(f"processed {sequence_index}/{len(sequences)} sequences", flush=True)

    protocol = {
        "dataset": str(dataset),
        "sequences": len(sequences),
        "bases": args.bases,
        "factors": args.factors,
        "anchor_factor": 1,
        "unit_kernel_meaning": "cardinal base interpolant at unit pixel width; x1 sampled action is identity",
        "consistent_model_choice": "camera Fourier symbol H equals the declared unit-width interpolant symbol; down H(r omega)/H(omega)",
        "AA_down": "identical periodic spectral AA-stretched base filter to earlier matched-x2 experiment",
        "AA_up": "periodic, DC-normalized unit-width cardinal interpolation evaluated on the centered high grid",
        "consistent_up": "retained-band pseudo-inverse on the matching low/high Fourier grids",
        "gaussian_exclusion": "ordinary Gaussian is not cardinal at integer samples; no width or cardinalization declared",
        "model_only": "not bitwise Pillow/OpenCV; signed bases are formal kernel families, not physical intensity PSFs",
        "feature_extractor": "SIFT nfeatures=1500 contrastThreshold=0.01, mutual Lowe ratio=0.8",
        "homography_min_matches": 8,
        "direct_score": "success_2_5_output_px; RANSAC 1.25 output pixels",
        "roundtrip_score": "success_5px; RANSAC 5 original pixels; both images restored to original dimensions",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", nargs="+", type=int, default=list(FACTORS))
    parser.add_argument("--bases", nargs="+", default=list(BASES), choices=BASES)
    parser.add_argument("--max-sequences", type=int, default=0)
    args = parser.parse_args()
    if any(factor < 2 for factor in args.factors):
        parser.error("hard factors must be >=2")
    run(args)


if __name__ == "__main__":
    main()
