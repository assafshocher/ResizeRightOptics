"""Verify anchor-invariance of down-then-up and literal x2 downscale agreement.

The matched-x2 formal camera is the dyadic telescope of the shared AA x2
slice. Its coarsening multiplier is C(r omega)/C(omega), except that the
declared x2 slice is applied literally at r=2. The upscaler is its retained-
band pseudo-inverse. The product must be the same Fourier projection as the
unit-kernel experiment, independently of the base family.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import hpatches_geometry_experiment as geometry
import hpatches_matched_anchor_experiment as old
import hpatches_unit_kernel_experiment as unit


@lru_cache(maxsize=8192)
def camera_axis(base: str, omega_bytes: bytes, size: int) -> np.ndarray:
    omega = np.frombuffer(omega_bytes, dtype=np.float64, count=size)
    result = np.ones_like(omega)
    for index in range(16):
        result *= unit.symbol(base, omega / 2**index)
    return result


def camera(base: str, omega: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(omega, dtype=np.float64)
    return camera_axis(base, contiguous.tobytes(), contiguous.size)


@lru_cache(maxsize=8192)
def down_axis(
    input_length: int, output_length: int, base: str
) -> tuple[np.ndarray, np.ndarray]:
    modes = np.arange(-(output_length // 2), output_length // 2 + 1)
    omega = 2 * np.pi * modes / input_length
    ratio = input_length / output_length
    if input_length == 2 * output_length:
        transfer = unit.symbol(base, ratio * omega)
    else:
        transfer = camera(base, ratio * omega) / camera(base, omega)
    _, phase = unit.down_axis(
        input_length, output_length, base, "AA stretch"
    )
    return transfer, phase


def matched2_down(
    image: np.ndarray, shape: tuple[int, int], base: str
) -> np.ndarray:
    high_height, high_width = image.shape
    low_height, low_width = shape
    transfer_y, phase_y = down_axis(high_height, low_height, base)
    transfer_x, phase_x = down_axis(high_width, low_width, base)
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    y_start = high_height // 2 - low_height // 2
    x_start = high_width // 2 - low_width // 2
    cropped = spectrum[
        y_start : y_start + low_height,
        x_start : x_start + low_width,
    ]
    multiplier = (
        transfer_y[:, None] * transfer_x[None, :]
        * phase_y[:, None] * phase_x[None, :]
        * (low_height / high_height) * (low_width / high_width)
    )
    return np.fft.ifft2(np.fft.ifftshift(cropped * multiplier)).real.astype(np.float32)


def matched2_up(
    image: np.ndarray, shape: tuple[int, int], base: str
) -> np.ndarray:
    low_height, low_width = image.shape
    high_height, high_width = shape
    transfer_y, phase_y = down_axis(high_height, low_height, base)
    transfer_x, phase_x = down_axis(high_width, low_width, base)
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    embedded = np.zeros(shape, dtype=np.complex128)
    y_start = high_height // 2 - low_height // 2
    x_start = high_width // 2 - low_width // 2
    embedded[y_start : y_start + low_height, x_start : x_start + low_width] = (
        spectrum
        / (transfer_y[:, None] * transfer_x[None, :])
        / (phase_y[:, None] * phase_x[None, :])
        * (high_height / low_height) * (high_width / low_width)
    )
    return np.fft.ifft2(np.fft.ifftshift(embedded)).real.astype(np.float32)


def projection(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    high_height, high_width = image.shape
    low_height, low_width = shape
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    embedded = np.zeros(image.shape, dtype=np.complex128)
    y_start = high_height // 2 - low_height // 2
    x_start = high_width // 2 - low_width // 2
    embedded[y_start : y_start + low_height, x_start : x_start + low_width] = (
        spectrum[y_start : y_start + low_height, x_start : x_start + low_width]
    )
    return np.fft.ifft2(np.fft.ifftshift(embedded)).real.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=5)
    parser.add_argument("--factors", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    args = parser.parse_args()
    sequences = sorted(
        directory for directory in args.dataset.iterdir()
        if directory.is_dir() and (directory / "1.ppm").exists()
    )[: args.max_sequences]
    records: list[dict] = []
    for directory in sequences:
        for number in range(1, 7):
            original = old.trim_to_exact_half(
                geometry.load_grayscale(directory / f"{number}.ppm")
            )
            for factor in args.factors:
                shape = (
                    tuple(dimension // 2 for dimension in original.shape)
                    if factor == 2
                    else tuple(
                        geometry.nearest_odd(dimension / factor)
                        for dimension in original.shape
                    )
                )
                aa_low = None
                target_projection = projection(original, shape)
                for base in unit.BASES:
                    low = matched2_down(original, shape, base)
                    high = matched2_up(low, original.shape, base)
                    unit_high = unit.consistent_up(
                        unit.consistent_down(original, shape, base),
                        original.shape, base,
                    )
                    if factor == 2:
                        aa_low = unit.spectral_down(
                            original, shape, base, "AA stretch"
                        )
                    records.append(
                        {
                            "sequence": directory.name,
                            "image": number,
                            "factor": factor,
                            "base": base,
                            "x2_low_vs_AA_max_abs": (
                                float(np.max(np.abs(low - aa_low)))
                                if factor == 2 else None
                            ),
                            "matched2_high_vs_projection_max_abs": float(
                                np.max(np.abs(high - target_projection))
                            ),
                            "matched2_high_vs_unit_high_max_abs": float(
                                np.max(np.abs(high - unit_high))
                            ),
                            "matched2_high_vs_projection_uint8_pixels": int(
                                np.count_nonzero(
                                    geometry.to_uint8(high)
                                    != geometry.to_uint8(target_projection)
                                )
                            ),
                        }
                    )
    summary = {
        "dataset": str(args.dataset.resolve()),
        "sequences_checked": len(sequences),
        "images_checked": 6 * len(sequences),
        "bases": unit.BASES,
        "factors": args.factors,
        "x2_max_abs_low_disagreement": max(
            record["x2_low_vs_AA_max_abs"]
            for record in records if record["factor"] == 2
        ),
        "max_abs_high_vs_projection": max(
            record["matched2_high_vs_projection_max_abs"] for record in records
        ),
        "max_abs_high_vs_unit_anchor": max(
            record["matched2_high_vs_unit_high_max_abs"] for record in records
        ),
        "max_uint8_pixel_disagreement_per_image_vs_projection": max(
            record["matched2_high_vs_projection_uint8_pixels"] for record in records
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
