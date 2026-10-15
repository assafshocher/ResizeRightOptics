from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from multiscene_physical_ls import (
    feature_affine,
    gaussian_prefilter_prediction,
    project_affine,
    refine_low_band_affine,
    robust_gain_bias,
    sample_bicubic,
)
from pilot_psf_fit import load_linear_raw


DATASET = Path(str(__import__('os').environ.get('SCALING_CAPTURE_DATA', 'data/captures')))


def as_homogeneous(affine: np.ndarray) -> np.ndarray:
    return np.vstack([affine, [0.0, 0.0, 1.0]])


def estimate_adjacent_geometry(
    scene: str,
    images: dict[int, np.ndarray],
    cache_root: Path,
    last_index: int,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    affines: dict[int, np.ndarray] = {}
    correlations: dict[int, float] = {}
    for target_index in range(2, last_index + 1):
        cache = cache_root / f"{scene}_{target_index-1:05d}_to_{target_index:05d}.json"
        if cache.exists():
            record = json.loads(cache.read_text())
            affine = np.array(record["affine_target_to_source"], dtype=np.float64)
            correlation = float(record["low_band_ecc_correlation"])
        else:
            initial, feature_stats = feature_affine(images[target_index - 1], images[target_index])
            affine, correlation = refine_low_band_affine(
                images[target_index - 1],
                images[target_index],
                initial,
            )
            record = {
                "scene": scene,
                "source_index": f"{target_index-1:05d}",
                "target_index": f"{target_index:05d}",
                "affine_target_to_source": affine.tolist(),
                "low_band_ecc_correlation": correlation,
                "feature_geometry": feature_stats,
                "target_to_source_linear_scale": float(np.sqrt(abs(np.linalg.det(affine[:, :2])))),
            }
            cache.write_text(json.dumps(record, indent=2))
        affines[target_index] = affine
        correlations[target_index] = correlation
    return affines, correlations


def transform_box(box: tuple[int, int, int, int], affine: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    corners = np.array(
        [[x0, y0], [x1 - 1, y0], [x0, y1 - 1], [x1 - 1, y1 - 1]],
        dtype=np.float64,
    )
    mapped = cv2.transform(corners[:, None, :].astype(np.float32), affine.astype(np.float32))[:, 0]
    return (
        int(np.floor(mapped[:, 0].min())) - margin,
        int(np.floor(mapped[:, 1].min())) - margin,
        int(np.ceil(mapped[:, 0].max())) + margin + 1,
        int(np.ceil(mapped[:, 1].max())) + margin + 1,
    )


def build_boxes(
    final_index: int,
    image_shape: tuple[int, int],
    affines: dict[int, np.ndarray],
    roi_size: int,
    margin: int,
) -> dict[int, tuple[int, int, int, int]]:
    height, width = image_shape
    x0 = (width - roi_size) // 2
    y0 = (height - roi_size) // 2
    boxes = {final_index: (x0, y0, x0 + roi_size, y0 + roi_size)}
    for index in range(final_index, 1, -1):
        box = transform_box(boxes[index], affines[index], margin)
        x0, y0, x1, y1 = box
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise RuntimeError(f"required chain support leaves frame at step {index}: {box}")
        boxes[index - 1] = box
    return boxes


def grid_for_step(
    target_box: tuple[int, int, int, int],
    source_box: tuple[int, int, int, int],
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tx0, ty0, tx1, ty1 = target_box
    sx0, sy0, _, _ = source_box
    yy, xx = np.mgrid[ty0:ty1, tx0:tx1].astype(np.float64)
    source_x, source_y = project_affine(affine, xx, yy)
    return source_x - sx0, source_y - sy0


def composite_affine(affines: dict[int, np.ndarray], final_index: int) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    for index in range(2, final_index + 1):
        result = result @ as_homogeneous(affines[index])
    return result[:2]


def valid_mask(target: np.ndarray, direct_sample: np.ndarray, border: int = 12) -> np.ndarray:
    mask = np.ones_like(target, dtype=bool)
    mask[:border] = mask[-border:] = False
    mask[:, :border] = mask[:, -border:] = False
    mask &= (target > 0.003) & (target < 0.97)
    mask &= (direct_sample > 0.003) & (direct_sample < 0.97)
    return mask


def residual_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    residual = prediction[mask] - target[mask]
    dynamic = float(np.quantile(target[mask], 0.995) - np.quantile(target[mask], 0.005))
    rmse = float(np.sqrt(np.mean(residual**2)))
    noise_kernel = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    noise_response = cv2.filter2D(target, cv2.CV_32F, noise_kernel.astype(np.float32), borderType=cv2.BORDER_REFLECT101)
    noise_sigma = float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(noise_response[1:-1, 1:-1])) / 6.0)
    noise_sigma = max(noise_sigma, 1e-6)
    residual_image = prediction - target
    interior = mask[:, :-1] & mask[:, 1:]
    horizontal = np.corrcoef(residual_image[:, :-1][interior], residual_image[:, 1:][interior])[0, 1]
    interior = mask[:-1, :] & mask[1:, :]
    vertical = np.corrcoef(residual_image[:-1, :][interior], residual_image[1:, :][interior])[0, 1]
    return {
        "rmse": rmse,
        "nrmse": rmse / max(dynamic, 1e-8),
        "psnr_db_robust_range": float(20 * np.log10(max(dynamic, 1e-8) / max(rmse, 1e-12))),
        "approx_noise_sigma": noise_sigma,
        "approx_reduced_chi_squared": float(np.mean(residual**2) / (noise_sigma**2)),
        "residual_lag1_horizontal": float(horizontal),
        "residual_lag1_vertical": float(vertical),
    }


def disagreement_metrics(direct: np.ndarray, chain: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    difference = direct[mask] - chain[mask]
    dynamic = float(np.quantile(target[mask], 0.995) - np.quantile(target[mask], 0.005))
    rmse = float(np.sqrt(np.mean(difference**2)))
    return {"rmse": rmse, "nrmse": rmse / max(dynamic, 1e-8)}


def fit_oracle_convolution(
    transformed_source: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    kernel_size: int = 11,
) -> tuple[np.ndarray, np.ndarray, float]:
    radius = kernel_size // 2
    padded = np.pad(transformed_source, ((radius, radius), (radius, radius)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    design = windows.reshape(-1, kernel_size * kernel_size).astype(np.float64)
    response = target.reshape(-1).astype(np.float64)
    selected = train_mask.reshape(-1)
    train_design = design[selected]
    train_response = response[selected]
    augmented = np.concatenate([train_design, np.ones((train_design.shape[0], 1))], axis=1)
    normal = np.einsum("ni,nj->ij", augmented, augmented, optimize=False)
    rhs = np.einsum("ni,n->i", augmented, train_response, optimize=False)
    ridge = 1e-6 * float(np.trace(normal[:-1, :-1]) / (kernel_size * kernel_size))
    regularizer = np.zeros_like(normal)
    regularizer[:-1, :-1] = ridge * np.eye(kernel_size * kernel_size)
    constraint = np.zeros(augmented.shape[1], dtype=np.float64)
    constraint[:-1] = 1.0
    kkt = np.block(
        [
            [normal + regularizer, constraint[:, None]],
            [constraint[None, :], np.zeros((1, 1))],
        ]
    )
    solution = np.linalg.solve(kkt, np.concatenate([rhs, [1.0]]))[:-1]
    kernel = solution[:-1].reshape(kernel_size, kernel_size)
    bias = float(solution[-1])
    prediction = (
        np.einsum("ij,j->i", design, solution[:-1], optimize=False) + bias
    ).reshape(target.shape).astype(np.float32)
    return prediction, kernel, bias


def continuous_psf_transfer(
    omega_x: np.ndarray,
    omega_y: np.ndarray,
    weights: np.ndarray,
    basis_sigma: float,
) -> np.ndarray:
    radius_y = weights.shape[0] // 2
    radius_x = weights.shape[1] // 2
    transfer = np.zeros_like(omega_x, dtype=np.complex128)
    for row in range(weights.shape[0]):
        for column in range(weights.shape[1]):
            mass = float(weights[row, column])
            if mass == 0.0:
                continue
            dx = column - radius_x
            dy = row - radius_y
            transfer += mass * np.exp(-1j * (omega_x * dx + omega_y * dy))
    transfer *= np.exp(-0.5 * basis_sigma**2 * (omega_x**2 + omega_y**2))
    return transfer


def relative_camera_prediction(
    source: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    linear_map: np.ndarray,
    weights: np.ndarray,
    basis_sigma: float,
    eta: float = 1e-6,
    padding: int = 32,
) -> np.ndarray:
    transformed = sample_bicubic(source, source_x, source_y)
    padded = np.pad(transformed, ((padding, padding), (padding, padding)), mode="reflect")
    height, width = padded.shape
    omega_y = 2 * np.pi * np.fft.fftfreq(height)
    omega_x = 2 * np.pi * np.fft.fftfreq(width)
    grid_x, grid_y = np.meshgrid(omega_x, omega_y)
    frequencies = np.stack([grid_x.ravel(), grid_y.ravel()], axis=0)
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
    target_transfer = continuous_psf_transfer(grid_x, grid_y, weights, basis_sigma)
    source_transfer = continuous_psf_transfer(transformed_x, transformed_y, weights, basis_sigma)
    multiplier = target_transfer * source_transfer.conj() / (np.abs(source_transfer) ** 2 + eta**2)
    prediction = np.fft.ifft2(np.fft.fft2(padded) * multiplier).real
    return prediction[padding:-padding, padding:-padding].astype(np.float32)


def apply_method(
    method: str,
    source: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    linear_map: np.ndarray,
    linear_scale: float,
    estimated_psf: np.ndarray,
    estimated_basis_sigma: float,
    estimated_eta: float,
    gaussian_sigma: float,
    gaussian_eta: float,
    scale_space_alpha: float,
    stretched_alpha: float,
) -> np.ndarray:
    if method == "bicubic":
        return sample_bicubic(source, source_x, source_y)
    if method == "stretched_aa":
        sigma = stretched_alpha * linear_scale
        blurred = cv2.GaussianBlur(
            source,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT101,
        )
        return sample_bicubic(blurred, source_x, source_y)
    if method == "gaussian_scale_space":
        sigma = scale_space_alpha * np.sqrt(max(linear_scale * linear_scale - 1.0, 0.0))
        blurred = cv2.GaussianBlur(
            source,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT101,
        )
        return sample_bicubic(blurred, source_x, source_y)
    if method == "gaussian_kernel":
        return relative_camera_prediction(
            source,
            source_x,
            source_y,
            linear_map,
            np.ones((1, 1), dtype=np.float32),
            gaussian_sigma,
            eta=1e-6,
        )
    if method == "estimated_kernel":
        return relative_camera_prediction(
            source,
            source_x,
            source_y,
            linear_map,
            estimated_psf,
            estimated_basis_sigma,
            eta=1e-6,
        )
    raise ValueError(method)


def run_scene(
    scene: str,
    images: dict[int, np.ndarray],
    affines: dict[int, np.ndarray],
    correlations: dict[int, float],
    final_index: int,
    roi_size: int,
    margin: int,
    model: dict[str, Any],
) -> dict[str, Any]:
    boxes = build_boxes(final_index, images[1].shape, affines, roi_size, margin)
    composite = composite_affine(affines, final_index)
    source_box = boxes[1]
    final_box = boxes[final_index]
    sx0, sy0, sx1, sy1 = source_box
    fx0, fy0, fx1, fy1 = final_box
    source = images[1][sy0:sy1, sx0:sx1]
    target = images[final_index][fy0:fy1, fx0:fx1]
    direct_x, direct_y = grid_for_step(final_box, source_box, composite)
    direct_unadjusted = sample_bicubic(source, direct_x, direct_y)
    mask = valid_mask(target, direct_unadjusted)
    yy, xx = np.mgrid[: target.shape[0], : target.shape[1]]
    train_mask = mask & ((xx // 16 + yy // 16) % 2 == 0)
    evaluation_mask = mask & ~((xx // 16 + yy // 16) % 2 == 0)
    gain, bias = robust_gain_bias(direct_unadjusted, target, train_mask)
    source = gain * source + bias

    methods = ["bicubic", "stretched_aa", "gaussian_scale_space", "gaussian_kernel", "estimated_kernel"]
    method_results: dict[str, Any] = {}
    total_scale = float(np.sqrt(abs(np.linalg.det(composite[:, :2]))))
    for method in methods:
        direct = apply_method(
            method,
            source,
            direct_x,
            direct_y,
            composite[:, :2],
            total_scale,
            model["estimated_psf"],
            model["estimated_basis_sigma"],
            model["estimated_eta"],
            model["gaussian_sigma"],
            model["gaussian_eta"],
            model["scale_space_alpha"],
            model["stretched_alpha"],
        )
        current = source
        current_box = source_box
        for index in range(2, final_index + 1):
            output_box = boxes[index]
            source_x, source_y = grid_for_step(output_box, current_box, affines[index])
            step_scale = float(np.sqrt(abs(np.linalg.det(affines[index][:, :2]))))
            current = apply_method(
                method,
                current,
                source_x,
                source_y,
                affines[index][:, :2],
                step_scale,
                model["estimated_psf"],
                model["estimated_basis_sigma"],
                model["estimated_eta"],
                model["gaussian_sigma"],
                model["gaussian_eta"],
                model["scale_space_alpha"],
                model["stretched_alpha"],
            )
            current_box = output_box
        chain = current
        method_results[method] = {
            "direct_to_truth": residual_metrics(direct, target, evaluation_mask),
            "chain_to_truth": residual_metrics(chain, target, evaluation_mask),
            "direct_chain_disagreement": disagreement_metrics(direct, chain, target, evaluation_mask),
        }
    transformed_source = sample_bicubic(source, direct_x, direct_y)
    oracle_prediction, oracle_kernel, oracle_bias = fit_oracle_convolution(
        transformed_source,
        target,
        train_mask,
    )
    oracle_metrics = residual_metrics(oracle_prediction, target, evaluation_mask)
    method_results["oracle_convolution"] = {
        "direct_to_truth": oracle_metrics,
        "chain_to_truth": oracle_metrics,
        "direct_chain_disagreement": {"rmse": 0.0, "nrmse": 0.0},
        "kernel_sum": float(oracle_kernel.sum()),
        "kernel_l1": float(np.abs(oracle_kernel).sum()),
        "bias": oracle_bias,
    }
    return {
        "scene": scene,
        "final_index": final_index,
        "chain_steps": final_index - 1,
        "total_linear_scale": total_scale,
        "minimum_adjacent_ecc": float(min(correlations[index] for index in range(2, final_index + 1))),
        "photometric_gain": gain,
        "photometric_bias": bias,
        "source_box": source_box,
        "target_box": final_box,
        "methods": method_results,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for steps in sorted({row["chain_steps"] for row in rows}):
        selected = [row for row in rows if row["chain_steps"] == steps]
        step_result: dict[str, Any] = {"number_of_scenes": len(selected)}
        for method in [
            "bicubic",
            "stretched_aa",
            "gaussian_scale_space",
            "gaussian_kernel",
            "estimated_kernel",
            "oracle_convolution",
        ]:
            direct_nrmse = np.array([row["methods"][method]["direct_to_truth"]["nrmse"] for row in selected])
            chain_nrmse = np.array([row["methods"][method]["chain_to_truth"]["nrmse"] for row in selected])
            disagreement = np.array(
                [row["methods"][method]["direct_chain_disagreement"]["nrmse"] for row in selected]
            )
            chi_squared = np.array(
                [row["methods"][method]["chain_to_truth"]["approx_reduced_chi_squared"] for row in selected]
            )
            correlation = np.array(
                [
                    0.5
                    * (
                        row["methods"][method]["chain_to_truth"]["residual_lag1_horizontal"]
                        + row["methods"][method]["chain_to_truth"]["residual_lag1_vertical"]
                    )
                    for row in selected
                ]
            )
            step_result[method] = {
                "mean_direct_nrmse": float(direct_nrmse.mean()),
                "mean_chain_nrmse": float(chain_nrmse.mean()),
                "mean_direct_chain_disagreement_nrmse": float(disagreement.mean()),
                "median_approx_reduced_chi_squared": float(np.median(chi_squared)),
                "median_residual_lag1_correlation": float(np.median(correlation)),
            }
        result[str(steps)] = step_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-count", type=int, default=12)
    parser.add_argument("--last-index", type=int, default=6)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--margin", type=int, default=28)
    parser.add_argument("--geometry-cache", type=Path)
    parser.add_argument(
        "--model-results",
        type=Path,
        default=Path("work/zoom_experiment/results/multiscene_physical_ls_d4/results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/zoom_experiment/results/chain_diagnostic"),
    )
    args = parser.parse_args()

    model_results = json.loads(args.model_results.read_text())
    model = {
        "estimated_psf": np.array(model_results["physical_psf"]["weights"], dtype=np.float32),
        "estimated_basis_sigma": float(model_results["physical_psf"]["basis_sigma_pixels"]),
        "estimated_eta": float(model_results["physical_psf"]["selected"]["eta"]),
        "gaussian_sigma": float(model_results["gaussian_camera_baseline"]["selected_sigma_pixels"]),
        "gaussian_eta": float(model_results["gaussian_camera_baseline"]["selected_eta"]),
        "scale_space_alpha": float(model_results["gaussian_scale_space_sanity_check"]["selected_alpha"]),
        "stretched_alpha": float(model_results["stretched_gaussian_aa_baseline"]["selected_alpha"]),
    }
    scenes = model_results["split"]["test"][: args.scene_count]
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.geometry_cache if args.geometry_cache is not None else args.output / "geometry"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scene_number, scene in enumerate(scenes, 1):
        print(f"scene {scene_number}/{len(scenes)}: {scene}", flush=True)
        try:
            images = {
                index: load_linear_raw(DATASET / scene / f"{index:05d}.ARW")
                for index in range(1, args.last_index + 1)
            }
            affines, correlations = estimate_adjacent_geometry(
                scene,
                images,
                cache_root,
                args.last_index,
            )
            for final_index in range(2, args.last_index + 1):
                if min(correlations[index] for index in range(2, final_index + 1)) < 0.94:
                    raise RuntimeError(
                        f"adjacent low-band ECC below 0.94 by endpoint {final_index}: "
                        f"{min(correlations[index] for index in range(2, final_index + 1)):.4f}"
                    )
                rows.append(
                    run_scene(
                        scene,
                        images,
                        affines,
                        correlations,
                        final_index,
                        args.roi_size,
                        args.margin,
                        model,
                    )
                )
        except Exception as error:
            failures.append({"scene": scene, "error": str(error)})
            print(f"FAILED {scene}: {error}", flush=True)

    result = {
        "experiment": "real RAW chain-length diagnostic with direct and chained paths sharing the identical composed adjacent affine grid",
        "scenes_requested": scenes,
        "rows": rows,
        "aggregate": aggregate(rows),
        "failures": failures,
        "model": {
            "estimated_psf_source": str(args.model_results),
            "estimated_basis_sigma": model["estimated_basis_sigma"],
            "estimated_eta": model["estimated_eta"],
            "gaussian_sigma": model["gaussian_sigma"],
            "gaussian_eta": model["gaussian_eta"],
            "scale_space_alpha": model["scale_space_alpha"],
            "stretched_alpha": model["stretched_alpha"],
        },
        "important_qualification": "Reduced chi-squared is approximate: sigma is estimated by a wavelet estimator from demosaiced linear luminance, not from repeated RAW exposures or a calibrated Poisson-Gaussian sensor model.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"aggregate": result["aggregate"], "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
