from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

from pilot_psf_fit import filter_image, fit_gain_bias, metrics, torch_grid


def affine_approximation(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x0: int,
    target_y0: int,
) -> np.ndarray:
    height, width = source_x.shape
    target_y, target_x = np.mgrid[
        target_y0 : target_y0 + height,
        target_x0 : target_x0 + width,
    ]
    design = np.stack([target_x.ravel(), target_y.ravel(), np.ones(target_x.size)], axis=1)
    coefficients_x = np.linalg.lstsq(design, source_x.ravel(), rcond=None)[0]
    coefficients_y = np.linalg.lstsq(design, source_y.ravel(), rcond=None)[0]
    return np.stack([coefficients_x, coefficients_y], axis=0)


def symmetric_orbits(radius: int) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    unvisited = {(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)}
    orbits: list[list[tuple[int, int]]] = []
    while unvisited:
        point = min(unvisited)
        reflected = (-point[0], -point[1])
        orbit = [point] if reflected == point else [point, reflected]
        for member in orbit:
            unvisited.discard(member)
        orbits.append(orbit)
    moments = np.array(
        [np.mean([dx * dx + dy * dy for dx, dy in orbit]) for orbit in orbits],
        dtype=np.float64,
    )
    return orbits, moments


def basis_transfer(
    omega_x: np.ndarray,
    omega_y: np.ndarray,
    dx: float,
    dy: float,
    sigma: float,
) -> np.ndarray:
    return np.exp(-0.5 * sigma**2 * (omega_x**2 + omega_y**2)) * np.exp(
        -1j * (omega_x * dx + omega_y * dy)
    )


def build_design(
    target: np.ndarray,
    transformed_source: np.ndarray,
    linear_map: np.ndarray,
    orbits: list[list[tuple[int, int]]],
    basis_sigma: float,
    padding: int,
) -> np.ndarray:
    target_padded = np.pad(target, ((padding, padding), (padding, padding)), mode="reflect")
    source_padded = np.pad(transformed_source, ((padding, padding), (padding, padding)), mode="reflect")
    target_fft = np.fft.fft2(target_padded)
    source_fft = np.fft.fft2(source_padded)
    height, width = target_padded.shape
    omega_y = 2 * np.pi * np.fft.fftfreq(height)
    omega_x = 2 * np.pi * np.fft.fftfreq(width)
    grid_x, grid_y = np.meshgrid(omega_x, omega_y)
    frequencies = np.stack([grid_x, grid_y], axis=0).reshape(2, -1)
    inverse_transpose = np.linalg.inv(linear_map).T
    transformed_frequencies = np.stack(
        [
            inverse_transpose[0, 0] * frequencies[0] + inverse_transpose[0, 1] * frequencies[1],
            inverse_transpose[1, 0] * frequencies[0] + inverse_transpose[1, 1] * frequencies[1],
        ],
        axis=0,
    )
    transformed_x = transformed_frequencies[0].reshape(height, width)
    transformed_y = transformed_frequencies[1].reshape(height, width)

    columns: list[np.ndarray] = []
    for orbit in orbits:
        orbit_column = np.zeros_like(target, dtype=np.float64)
        for dx, dy in orbit:
            target_kernel = basis_transfer(
                transformed_x,
                transformed_y,
                dx,
                dy,
                basis_sigma,
            )
            source_kernel = basis_transfer(grid_x, grid_y, dx, dy, basis_sigma)
            residual = np.fft.ifft2(target_fft * target_kernel - source_fft * source_kernel).real
            orbit_column += residual[padding:-padding, padding:-padding]
        columns.append(orbit_column / len(orbit))
    return np.stack(columns, axis=-1)


def solve_simplex_qp(
    gram: np.ndarray,
    radius_moment: np.ndarray,
    ridge: float,
    radius_penalty: float,
) -> np.ndarray:
    count = gram.shape[0]
    initial = np.zeros(count, dtype=np.float64)
    initial[np.argmin(radius_moment)] = 1.0
    regularized = gram + ridge * np.eye(count)

    def objective(weights: np.ndarray) -> float:
        quadratic = np.einsum("i,ij,j->", weights, regularized, weights, optimize=False)
        return float(quadratic + radius_penalty * np.dot(radius_moment, weights))

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2 * np.einsum("ij,j->i", regularized, weights, optimize=False) + radius_penalty * radius_moment

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
        options={"ftol": 1e-14, "maxiter": 2000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def expand_orbits(
    orbit_weights: np.ndarray,
    orbits: list[list[tuple[int, int]]],
    radius: int,
) -> np.ndarray:
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for mass, orbit in zip(orbit_weights, orbits):
        for dx, dy in orbit:
            kernel[dy + radius, dx + radius] = mass / len(orbit)
    return kernel


def predict_camera_sandwich(
    source: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    psf_weights: np.ndarray,
    basis_sigma: float,
    eta: float,
) -> np.ndarray:
    source_tensor = torch.from_numpy(source.astype(np.float32))[None, None]
    weights_tensor = torch.from_numpy(psf_weights.astype(np.float32))
    grid = torch_grid(source_x, source_y, source.shape)
    with torch.no_grad():
        scene = filter_image(source_tensor, weights_tensor, basis_sigma, True, eta, padding=48)
        transformed = F.grid_sample(
            scene,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
        prediction = filter_image(transformed, weights_tensor, basis_sigma, False, eta, padding=32)
    return prediction[0, 0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("work/zoom_experiment/results/00007_psf_pilot_v2"),
    )
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--basis-sigma", type=float, default=0.28)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/zoom_experiment/results/00007_physical_ls"),
    )
    args = parser.parse_args()

    arrays = np.load(args.pilot / "arrays.npz")
    metadata = json.loads((args.pilot / "results.json").read_text())
    source = arrays["source_crop"].astype(np.float32)
    target = arrays["target_crop"].astype(np.float32)
    train_mask = arrays["train_mask"].copy()
    validation_mask = arrays["validation_mask"].copy()
    target_x0, target_y0, _, _ = metadata["geometry"]["target_roi_xywh"]
    source_crop_x0, source_crop_y0, _, _ = metadata["geometry"]["source_crop_xyxy"]

    affine = affine_approximation(arrays["source_x"], arrays["source_y"], target_x0, target_y0)
    target_y, target_x = np.mgrid[
        target_y0 : target_y0 + target.shape[0],
        target_x0 : target_x0 + target.shape[1],
    ].astype(np.float64)
    source_x_full = affine[0, 0] * target_x + affine[0, 1] * target_y + affine[0, 2]
    source_y_full = affine[1, 0] * target_x + affine[1, 1] * target_y + affine[1, 2]
    source_x = source_x_full - source_crop_x0
    source_y = source_y_full - source_crop_y0
    transformed_source = cv2.remap(
        source,
        source_x.astype(np.float32),
        source_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT101,
    )

    gain, bias = fit_gain_bias(transformed_source, target, train_mask)
    source_adjusted = gain * source + bias
    transformed_source = gain * transformed_source + bias

    orbits, radius_moment = symmetric_orbits(args.radius)
    design = build_design(
        target,
        transformed_source,
        affine[:, :2],
        orbits,
        args.basis_sigma,
        padding=64,
    )
    design_train = design[train_mask].astype(np.float64)
    design_validation = design[validation_mask].astype(np.float64)
    gram = design_train.T @ design_train / design_train.shape[0]

    configurations: list[dict] = []
    best = None
    for ridge in [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        for radial in [0.0, 1e-8, 1e-7, 1e-6, 1e-5]:
            orbit_weights = solve_simplex_qp(gram, radius_moment, ridge, radial)
            train_residual = design_train @ orbit_weights
            validation_residual = design_validation @ orbit_weights
            configuration = {
                "ridge": ridge,
                "radius_penalty": radial,
                "train_cross_rmse": float(np.sqrt(np.mean(train_residual**2))),
                "validation_cross_rmse": float(np.sqrt(np.mean(validation_residual**2))),
                "second_moment_coefficients": float(radius_moment @ orbit_weights),
            }
            configurations.append(configuration)
            if best is None or configuration["validation_cross_rmse"] < best[0]:
                best = (configuration["validation_cross_rmse"], orbit_weights, configuration)

    assert best is not None
    psf_weights = expand_orbits(best[1], orbits, args.radius)
    capture_results: list[dict] = []
    best_capture = None
    for eta in [0.003, 0.01, 0.025, 0.05, 0.1]:
        prediction = predict_camera_sandwich(
            source_adjusted,
            source_x,
            source_y,
            psf_weights,
            args.basis_sigma,
            eta,
        )
        output_gain, output_bias = fit_gain_bias(prediction, target, train_mask)
        prediction = output_gain * prediction + output_bias
        validation = metrics(prediction, target, validation_mask)
        training = metrics(prediction, target, train_mask)
        capture_result = {
            "eta": eta,
            "training": training,
            "validation": validation,
            "output_gain": output_gain,
            "output_bias": output_bias,
        }
        capture_results.append(capture_result)
        if best_capture is None or validation["rmse"] < best_capture[0]:
            best_capture = (validation["rmse"], prediction, capture_result)

    direct_gain, direct_bias = fit_gain_bias(transformed_source, target, train_mask)
    direct = direct_gain * transformed_source + direct_bias
    result = {
        "formulation": "affine cross-convolution constrained least squares for the physical PSF",
        "identity": "Y(omega) H(A^{-T} omega) = Fourier[Qx](omega) H(omega)",
        "affine_target_to_source": affine.tolist(),
        "psf_basis": {
            "type": "centrosymmetric mixture of continuous unit-mass Gaussian translates",
            "radius": args.radius,
            "basis_sigma_pixels": args.basis_sigma,
            "coefficient_weights": psf_weights.tolist(),
            "sum": float(psf_weights.sum()),
            "minimum": float(psf_weights.min()),
        },
        "best_cross_equation": best[2],
        "cross_equation_configurations": configurations,
        "capture_results": capture_results,
        "direct_affine_lanczos": {
            "training": metrics(direct, target, train_mask),
            "validation": metrics(direct, target, validation_mask),
        },
        "best_capture_result": best_capture[2],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        args.output / "arrays.npz",
        target=target,
        transformed_source=transformed_source,
        direct=direct,
        prediction=best_capture[1],
        psf_weights=psf_weights,
        train_mask=train_mask,
        validation_mask=validation_mask,
        affine=affine,
        source_x=source_x_full,
        source_y=source_y_full,
    )
    print(json.dumps({
        "best_cross_equation": result["best_cross_equation"],
        "direct_affine_lanczos": result["direct_affine_lanczos"],
        "best_capture_result": result["best_capture_result"],
        "psf_second_moment": result["best_cross_equation"]["second_moment_coefficients"],
    }, indent=2))


if __name__ == "__main__":
    main()
