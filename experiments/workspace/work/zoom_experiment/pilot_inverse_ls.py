from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def solve_constrained_quadratic(
    gram: np.ndarray,
    coordinates: np.ndarray,
    ridge: float,
) -> np.ndarray:
    count = gram.shape[0]
    constraints = np.stack(
        [
            np.ones(count),
            coordinates[:, 0],
            coordinates[:, 1],
        ],
        axis=1,
    )
    target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    system = np.block(
        [
            [gram + ridge * np.eye(count), constraints],
            [constraints.T, np.zeros((constraints.shape[1], constraints.shape[1]))],
        ]
    )
    rhs = np.concatenate([np.zeros(count), target])
    return np.linalg.solve(system, rhs)[:count]


def fit_gain_bias(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    design = np.stack([prediction[mask], np.ones(mask.sum())], axis=1)
    gain, bias = np.linalg.lstsq(design, target[mask], rcond=None)[0]
    return float(gain), float(bias)


def fft_filter(image: np.ndarray, transfer_builder, padding: int = 32) -> np.ndarray:
    padded = np.pad(image, ((padding, padding), (padding, padding)), mode="reflect")
    transfer = transfer_builder(padded.shape)
    result = np.fft.ifft2(np.fft.fft2(padded) * transfer).real
    return result[padding:-padding, padding:-padding].astype(np.float32)


def kernel_transfer(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    kh, kw = kernel.shape
    centered = np.zeros(shape, dtype=np.float64)
    y0 = height // 2 - kh // 2
    x0 = width // 2 - kw // 2
    centered[y0 : y0 + kh, x0 : x0 + kw] = kernel
    return np.fft.fft2(np.fft.ifftshift(centered))


def capture_prediction(
    source: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    inverse_kernel: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    # cv2.filter2D is correlation; flip the taps to implement convolution.
    latent_source = cv2.filter2D(
        source,
        cv2.CV_32F,
        np.flip(inverse_kernel).astype(np.float32),
        borderType=cv2.BORDER_REFLECT101,
    )
    transformed = cv2.remap(
        latent_source,
        source_x.astype(np.float32),
        source_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT101,
    )

    def camera_transfer(shape: tuple[int, int]) -> np.ndarray:
        inverse_transfer = kernel_transfer(inverse_kernel, shape)
        return np.conj(inverse_transfer) / (np.abs(inverse_transfer) ** 2 + eta**2)

    prediction = fft_filter(transformed, camera_transfer)
    probe_shape = (256, 256)
    h_transfer = camera_transfer(probe_shape)
    psf = np.fft.fftshift(np.fft.ifft2(h_transfer).real)
    return prediction, psf


def rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("work/zoom_experiment/results/00007_psf_pilot_v2"),
    )
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/zoom_experiment/results/00007_inverse_ls"),
    )
    args = parser.parse_args()

    arrays = np.load(args.pilot / "arrays.npz")
    metadata = json.loads((args.pilot / "results.json").read_text())
    source = arrays["source_crop"].astype(np.float32)
    target = arrays["target_crop"].astype(np.float32)
    train_mask = arrays["train_mask"].copy()
    validation_mask = arrays["validation_mask"].copy()
    source_x0, source_y0, _, _ = metadata["geometry"]["source_crop_xyxy"]
    source_x = arrays["source_x"].astype(np.float32) - source_x0
    source_y = arrays["source_y"].astype(np.float32) - source_y0

    radius = args.radius
    train_mask[:radius, :] = False
    train_mask[-radius:, :] = False
    train_mask[:, :radius] = False
    train_mask[:, -radius:] = False
    validation_mask[:radius, :] = False
    validation_mask[-radius:, :] = False
    validation_mask[:, :radius] = False
    validation_mask[:, -radius:] = False

    # Radiometric normalization is a nuisance estimate from the training tiles.
    direct = arrays["direct"].astype(np.float32)
    direct_gain = metadata["photometric"]["direct_gain"]
    direct_bias = metadata["photometric"]["direct_bias"]
    unadjusted_direct = (direct - direct_bias) / direct_gain
    gain, bias = fit_gain_bias(unadjusted_direct, target, train_mask)
    source_adjusted = gain * source + bias

    offsets = np.array(
        [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)],
        dtype=np.float64,
    )
    train_columns: list[np.ndarray] = []
    validation_columns: list[np.ndarray] = []
    full_columns: list[np.ndarray] = []
    for dx, dy in offsets.astype(int):
        target_shift = np.roll(target, shift=(dy, dx), axis=(0, 1))
        source_sample = cv2.remap(
            source_adjusted,
            (source_x - dx).astype(np.float32),
            (source_y - dy).astype(np.float32),
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT101,
        )
        column = target_shift - source_sample
        train_columns.append(column[train_mask])
        validation_columns.append(column[validation_mask])
        full_columns.append(column)

    design_train = np.stack(train_columns, axis=1).astype(np.float64)
    design_validation = np.stack(validation_columns, axis=1).astype(np.float64)
    gram = design_train.T @ design_train / design_train.shape[0]

    candidates: list[dict] = []
    best = None
    for ridge in np.logspace(-10, -2, 17):
        coefficients = solve_constrained_quadratic(gram, offsets, float(ridge))
        train_latent_rmse = float(np.sqrt(np.mean((design_train @ coefficients) ** 2)))
        validation_latent_rmse = float(np.sqrt(np.mean((design_validation @ coefficients) ** 2)))
        candidate = {
            "ridge": float(ridge),
            "train_latent_rmse": train_latent_rmse,
            "validation_latent_rmse": validation_latent_rmse,
            "coefficient_l2": float(np.linalg.norm(coefficients)),
        }
        candidates.append(candidate)
        if best is None or validation_latent_rmse < best[0]:
            best = (validation_latent_rmse, coefficients, candidate)

    assert best is not None
    coefficients = best[1]
    inverse_kernel = coefficients.reshape(2 * radius + 1, 2 * radius + 1)

    capture_candidates: list[dict] = []
    best_capture = None
    for eta in [0.003, 0.01, 0.025, 0.05, 0.1]:
        prediction, psf = capture_prediction(
            source_adjusted,
            source_x,
            source_y,
            inverse_kernel,
            eta,
        )
        output_gain, output_bias = fit_gain_bias(prediction, target, train_mask)
        prediction = output_gain * prediction + output_bias
        candidate = {
            "eta": eta,
            "train_rmse": rmse(prediction, target, train_mask),
            "validation_rmse": rmse(prediction, target, validation_mask),
            "output_gain": output_gain,
            "output_bias": output_bias,
            "psf_negative_mass_fraction": float(
                np.abs(psf[psf < 0]).sum() / (np.abs(psf).sum() + 1e-12)
            ),
        }
        capture_candidates.append(candidate)
        if best_capture is None or candidate["validation_rmse"] < best_capture[0]:
            best_capture = (candidate["validation_rmse"], prediction, psf, candidate)

    assert best_capture is not None
    result = {
        "formulation": "constrained linear least squares for the inverse camera kernel",
        "basis": "integer translates of a fixed Lanczos-4 reconstruction function",
        "grid": "frozen low-band homography; target pixel centers mapped directly into source coordinates",
        "constraints": ["sum(g)=1", "first moment x=0", "first moment y=0"],
        "best_latent_solution": best[2],
        "ridge_candidates": candidates,
        "capture_candidates": capture_candidates,
        "direct_bicubic_validation_rmse": rmse(direct, target, validation_mask),
        "best_capture_solution": best_capture[3],
        "warning": (
            "The homogeneous latent objective can favor a smoothing representation. "
            "Physical admissibility must be checked after recovering the camera PSF."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        args.output / "arrays.npz",
        inverse_kernel=inverse_kernel,
        prediction=best_capture[1],
        recovered_psf=best_capture[2],
        target=target,
        train_mask=train_mask,
        validation_mask=validation_mask,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
