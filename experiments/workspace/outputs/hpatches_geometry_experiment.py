"""Resize-path audit on HPatches with independently supplied homographies.

This is an outer-pyramid/SIFT test, not a PSF-calibration experiment. Every
method receives the same grayscale input, final shape, final quantization,
detector, descriptor matcher, and RANSAC settings. The supplied HPatches
homographies score absolute geometric quality; direct-vs-chain differences
score path sensitivity. The Gaussian spectral arm is a declared camera family
and is *not* estimated from HPatches.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def nearest_odd(value: float) -> int:
    integer = int(round(value))
    if integer % 2 == 0:
        lower, upper = integer - 1, integer + 1
        integer = lower if abs(value - lower) <= abs(value - upper) else upper
    return max(integer, 3)


def path_shapes(source: tuple[int, int], fraction: float, steps: int) -> list[tuple[int, int]]:
    final = tuple(nearest_odd(dimension * fraction) for dimension in source)
    if steps == 1:
        return [source, final]
    shapes = [source]
    for index in range(1, steps):
        portion = index / steps
        shapes.append(
            tuple(
                nearest_odd(dimension * (target / dimension) ** portion)
                for dimension, target in zip(source, final)
            )
        )
    shapes.append(final)
    if any(not (a[0] > b[0] and a[1] > b[1]) for a, b in zip(shapes, shapes[1:])):
        raise ValueError(f"nondecreasing path {shapes}")
    return shapes


def resize_opencv(image: np.ndarray, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    height, width = shape
    return cv2.resize(image, (width, height), interpolation=interpolation).astype(np.float32)


def resize_pillow(image: np.ndarray, shape: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    height, width = shape
    plane = Image.fromarray(np.ascontiguousarray(image, dtype=np.float32))
    return np.asarray(plane.resize((width, height), resample=resample), dtype=np.float32)


def resize_consistent_gaussian(
    image: np.ndarray, shape: tuple[int, int], sigma_camera: float = 0.75
) -> np.ndarray:
    """Centered periodic Fourier camera family, generalized to odd rectangles.

    This arm takes the scale in each axis to be output-size/input-size. That
    removes the separate grid-size mismatch and makes same-endpoint direct and
    multistep paths algebraically comparable. Input and output dimensions are
    odd to avoid ambiguous singleton Nyquist modes.
    """
    input_height, input_width = image.shape
    output_height, output_width = shape
    if any(size % 2 != 1 for size in (input_height, input_width, output_height, output_width)):
        raise ValueError("spectral realization requires odd dimensions")
    if output_height > input_height or output_width > input_width:
        raise ValueError("coarsening paths only")

    y_modes = np.arange(-(output_height // 2), output_height // 2 + 1)
    x_modes = np.arange(-(output_width // 2), output_width // 2 + 1)
    omega_y, omega_x = np.meshgrid(
        2 * np.pi * y_modes / input_height,
        2 * np.pi * x_modes / input_width,
        indexing="ij",
    )
    ratio_y = input_height / output_height
    ratio_x = input_width / output_width
    relative_camera = np.exp(
        -0.5
        * sigma_camera**2
        * ((ratio_y**2 - 1) * omega_y**2 + (ratio_x**2 - 1) * omega_x**2)
    )
    input_y0 = -(input_height - 1) / 2
    input_x0 = -(input_width - 1) / 2
    output_y0 = -ratio_y * (output_height - 1) / 2
    output_x0 = -ratio_x * (output_width - 1) / 2
    phase = np.exp(
        1j
        * (omega_y * (output_y0 - input_y0) + omega_x * (output_x0 - input_x0))
    )
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    y_start = (input_height - output_height) // 2
    x_start = (input_width - output_width) // 2
    cropped = spectrum[
        y_start : y_start + output_height,
        x_start : x_start + output_width,
    ]
    multiplier = relative_camera * phase * (
        output_height / input_height
    ) * (output_width / input_width)
    return np.fft.ifft2(np.fft.ifftshift(cropped * multiplier)).real.astype(np.float32)


METHODS = OrderedDict(
    [
        ("OpenCV Lanczos-4", lambda image, shape: resize_opencv(image, shape, cv2.INTER_LANCZOS4)),
        ("OpenCV bicubic", lambda image, shape: resize_opencv(image, shape, cv2.INTER_CUBIC)),
        ("OpenCV area", lambda image, shape: resize_opencv(image, shape, cv2.INTER_AREA)),
        (
            "Pillow Lanczos-3 AA",
            lambda image, shape: resize_pillow(image, shape, Image.Resampling.LANCZOS),
        ),
        ("Camera-consistent Gaussian", resize_consistent_gaussian),
    ]
)


def follow_path(
    image: np.ndarray, shapes: list[tuple[int, int]], method: str
) -> np.ndarray:
    result = image
    for shape in shapes[1:]:
        result = METHODS[method](result, shape)
    return result


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    # Drop at most the last row/column. Original HPatches coordinates are
    # unchanged, so supplied homographies still apply to the retained pixels.
    height, width = image.shape
    image = image[: height - (height % 2 == 0), : width - (width % 2 == 0)]
    return image.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def keypoints_to_original(
    keypoints: list[cv2.KeyPoint],
    source_shape: tuple[int, int],
    final_shape: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = source_shape
    final_height, final_width = final_shape
    points = np.array([point.pt for point in keypoints], dtype=np.float64).reshape(-1, 2)
    points[:, 0] = (points[:, 0] + 0.5) * source_width / final_width - 0.5
    points[:, 1] = (points[:, 1] + 0.5) * source_height / final_height - 0.5
    return points


def project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(homography)) or np.max(np.abs(homography)) > 1e12:
        return np.full((len(points), 2), np.nan)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        projected = (homography @ homogeneous.T).T
        return projected[:, :2] / projected[:, 2:3]


def mutual_ratio_matches(
    first: np.ndarray | None,
    second: np.ndarray | None,
    ratio: float = 0.8,
) -> list[tuple[int, int]]:
    if first is None or second is None or len(first) < 2 or len(second) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(first, second, k=2)
    reverse = matcher.knnMatch(second, first, k=2)
    reverse_ok = {
        match.queryIdx: match.trainIdx
        for pair in reverse
        if len(pair) == 2
        for match, competitor in [pair]
        if match.distance < ratio * competitor.distance
    }
    return [
        (match.queryIdx, match.trainIdx)
        for pair in forward
        if len(pair) == 2
        for match, competitor in [pair]
        if match.distance < ratio * competitor.distance
        and reverse_ok.get(match.trainIdx) == match.queryIdx
    ]


def homography_grid_error(
    estimate: np.ndarray | None,
    truth: np.ndarray,
    first_shape: tuple[int, int],
    second_shape: tuple[int, int],
    second_final_shape: tuple[int, int],
) -> tuple[float, float]:
    if estimate is None:
        return float("nan"), float("nan")
    h1, w1 = first_shape
    h2, w2 = second_shape
    xx, yy = np.meshgrid(
        np.linspace(0.1 * w1, 0.9 * w1, 5),
        np.linspace(0.1 * h1, 0.9 * h1, 5),
    )
    points = np.column_stack([xx.ravel(), yy.ravel()])
    truth_points = project(truth, points)
    keep = (
        (truth_points[:, 0] >= 0)
        & (truth_points[:, 0] < w2)
        & (truth_points[:, 1] >= 0)
        & (truth_points[:, 1] < h2)
    )
    if not np.any(keep):
        return float("nan"), float("nan")
    estimate_points = project(estimate, points[keep])
    difference = estimate_points - truth_points[keep]
    final_height, final_width = second_final_shape
    output_difference = difference * np.array(
        [final_width / w2, final_height / h2], dtype=np.float64
    )
    return (
        float(np.median(np.linalg.norm(difference, axis=1))),
        float(np.median(np.linalg.norm(output_difference, axis=1))),
    )


def score_pair(
    reference: tuple[np.ndarray, np.ndarray | None],
    target: tuple[np.ndarray, np.ndarray | None],
    truth: np.ndarray,
    reference_shape: tuple[int, int],
    target_shape: tuple[int, int],
    target_final_shape: tuple[int, int],
    ransac_threshold_original_pixels: float,
) -> dict[str, float | int]:
    points1, descriptors1 = reference
    points2, descriptors2 = target
    matches = mutual_ratio_matches(descriptors1, descriptors2)
    if not matches:
        return {
            "matches": 0,
            "gt_inliers_5px": 0,
            "gt_precision_5px": float("nan"),
            "homography_grid_error_px": float("nan"),
            "homography_grid_error_output_px": float("nan"),
            "success_5px": 0,
            "success_10px": 0,
            "success_2_output_px": 0,
            "success_2_5_output_px": 0,
        }
    indices1 = np.array([first for first, _ in matches], dtype=np.int64)
    indices2 = np.array([second for _, second in matches], dtype=np.int64)
    matched1 = points1[indices1]
    matched2 = points2[indices2]
    geometric_error = np.linalg.norm(project(truth, matched1) - matched2, axis=1)
    inliers = int(np.sum(geometric_error < 5.0))
    estimate = None
    if len(matches) >= 8:
        cv2.setRNGSeed(0)
        estimate, _ = cv2.findHomography(
            matched1.astype(np.float32),
            matched2.astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_original_pixels,
            maxIters=2000,
            confidence=0.995,
        )
    grid_error, output_grid_error = homography_grid_error(
        estimate, truth, reference_shape, target_shape, target_final_shape
    )
    return {
        "matches": len(matches),
        "gt_inliers_5px": inliers,
        "gt_precision_5px": inliers / len(matches),
        "homography_grid_error_px": grid_error,
        "homography_grid_error_output_px": output_grid_error,
        "success_5px": int(np.isfinite(grid_error) and grid_error < 5.0),
        "success_10px": int(np.isfinite(grid_error) and grid_error < 10.0),
        "success_2_output_px": int(
            np.isfinite(output_grid_error) and output_grid_error < 2.0
        ),
        "success_2_5_output_px": int(
            np.isfinite(output_grid_error) and output_grid_error < 2.5
        ),
    }


def image_disagreement(direct: np.ndarray, chain: np.ndarray) -> float:
    # Exclude an eight-pixel border to reduce method-specific extension effects.
    interior = np.s_[8:-8, 8:-8] if min(direct.shape) > 20 else np.s_[:, :]
    difference = chain[interior] - direct[interior]
    scale = max(float(np.sqrt(np.mean(direct[interior] ** 2))), 1e-12)
    return 100 * float(np.sqrt(np.mean(difference**2))) / scale


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    METHODS["Camera-consistent Gaussian"] = lambda image, shape: resize_consistent_gaussian(
        image, shape, sigma_camera=args.camera_sigma
    )
    dataset = args.dataset.resolve()
    sequences = sorted(
        directory
        for directory in dataset.iterdir()
        if directory.is_dir() and (directory / "1.ppm").exists()
    )
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    args.output.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.01)

    for index, directory in enumerate(sequences, start=1):
        originals = {
            number: load_grayscale(directory / f"{number}.ppm")
            for number in range(1, 7)
        }
        for method in METHODS:
            extracted: dict[tuple[str, int], tuple[np.ndarray, np.ndarray | None]] = {}
            for number, original in originals.items():
                direct_shapes = path_shapes(original.shape, args.fraction, 1)
                chain_shapes = path_shapes(original.shape, args.fraction, args.steps)
                direct = follow_path(original, direct_shapes, method)
                chain = follow_path(original, chain_shapes, method)
                image_rows.append(
                    {
                        "sequence": directory.name,
                        "type": directory.name[0],
                        "image": number,
                        "method": method,
                        "fraction": args.fraction,
                        "steps": args.steps,
                        "path_nrmse_percent": image_disagreement(direct, chain),
                        "source_height": original.shape[0],
                        "source_width": original.shape[1],
                        "final_height": direct.shape[0],
                        "final_width": direct.shape[1],
                    }
                )
                for path_name, image in (("direct", direct), ("chain", chain)):
                    keypoints, descriptors = sift.detectAndCompute(to_uint8(image), None)
                    points = keypoints_to_original(
                        keypoints, original.shape, image.shape
                    )
                    extracted[(path_name, number)] = (points, descriptors)
            for target_number in args.targets:
                truth = np.loadtxt(
                    directory / f"H_1_{target_number}", dtype=np.float64
                ).reshape(3, 3)
                target_final_shape = path_shapes(
                    originals[target_number].shape, args.fraction, 1
                )[-1]
                if args.ransac_output_px is None:
                    ransac_threshold = 5.0
                else:
                    final_height, final_width = target_final_shape
                    original_height, original_width = originals[target_number].shape
                    effective_scale = np.sqrt(
                        (final_height / original_height)
                        * (final_width / original_width)
                    )
                    ransac_threshold = args.ransac_output_px / effective_scale
                for path_name in ("direct", "chain"):
                    metrics = score_pair(
                        extracted[(path_name, 1)],
                        extracted[(path_name, target_number)],
                        truth,
                        originals[1].shape,
                        originals[target_number].shape,
                        target_final_shape,
                        ransac_threshold,
                    )
                    task_rows.append(
                        {
                            "sequence": directory.name,
                            "type": directory.name[0],
                            "target": target_number,
                            "method": method,
                            "path": path_name,
                            "fraction": args.fraction,
                            "steps": args.steps,
                            **metrics,
                        }
                    )
        if index % 5 == 0 or index == len(sequences):
            write_csv(args.output / "task_results.csv", task_rows)
            write_csv(args.output / "image_results.csv", image_rows)
            print(f"processed {index}/{len(sequences)} sequences", flush=True)

    metadata = {
        "dataset": str(dataset),
        "sequences": len(sequences),
        "targets": args.targets,
        "fraction": args.fraction,
        "steps": args.steps,
        "methods": list(METHODS),
        "sift": {"nfeatures": 1500, "contrastThreshold": 0.01},
        "matcher": "mutual Lowe ratio 0.8",
        "ransac": {
            "threshold_original_pixels": 5.0 if args.ransac_output_px is None else None,
            "threshold_output_pixels": args.ransac_output_px,
            "maxIters": 2000,
        },
        "geometry": "source trimmed at bottom/right to odd dimensions; half-pixel endpoint map",
        "quantization": "float32 between resizes, uint8 at final SIFT input only",
        "camera_control": {
            "family": "spectral Gaussian",
            "sigma_source_pixels": args.camera_sigma,
            "not_fitted_to_hpatches": True,
        },
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
    }
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--targets", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--ransac-output-px", type=float, default=None)
    parser.add_argument("--camera-sigma", type=float, default=0.75)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
