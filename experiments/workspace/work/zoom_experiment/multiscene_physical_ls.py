from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.stats import wilcoxon
import torch
import torch.nn.functional as F

from pilot_physical_ls import (
    build_design,
    expand_orbits,
    predict_camera_sandwich,
    solve_simplex_qp,
    symmetric_orbits,
)
from pilot_psf_fit import filter_image, load_linear_raw, metrics, torch_grid


DATASET = Path(str(__import__('os').environ.get('SCALING_CAPTURE_DATA', 'data/captures')))


def robust_gain_bias(source: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    source_low = cv2.GaussianBlur(source, (0, 0), 6.0)
    target_low = cv2.GaussianBlur(target, (0, 0), 6.0)
    keep = valid.copy()
    gain, bias = 1.0, 0.0
    for _ in range(4):
        x = source_low[keep].astype(np.float64)
        y = target_low[keep].astype(np.float64)
        design = np.stack([x, np.ones_like(x)], axis=1)
        gain, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = target_low - (gain * source_low + bias)
        center = np.median(residual[keep])
        scale = 1.4826 * np.median(np.abs(residual[keep] - center)) + 1e-8
        keep = valid & (np.abs(residual - center) < 3.5 * scale)
    return float(gain), float(bias)


def coefficient_orbits(radius: int, symmetry: str) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    if symmetry == "centrosymmetric":
        return symmetric_orbits(radius)
    if symmetry != "d4":
        raise ValueError(symmetry)
    unvisited = {(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)}
    orbits: list[list[tuple[int, int]]] = []
    while unvisited:
        dx, dy = min(unvisited)
        orbit = sorted(
            {
                (sx * a, sy * b)
                for a, b in [(dx, dy), (dy, dx)]
                for sx in [-1, 1]
                for sy in [-1, 1]
            }
        )
        for point in orbit:
            unvisited.discard(point)
        orbits.append(orbit)
    moments = np.array(
        [np.mean([dx * dx + dy * dy for dx, dy in orbit]) for orbit in orbits],
        dtype=np.float64,
    )
    return orbits, moments


def feature_affine(source: np.ndarray, target: np.ndarray, max_side: int = 1600) -> tuple[np.ndarray, dict[str, Any]]:
    scale = min(1.0, max_side / max(source.shape))
    source_small = cv2.resize(
        np.sqrt(np.clip(source, 0.0, 1.0)),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    target_small = cv2.resize(
        np.sqrt(np.clip(target, 0.0, 1.0)),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    source_u8 = np.uint8(255 * np.clip(source_small, 0.0, 1.0))
    target_u8 = np.uint8(255 * np.clip(target_small, 0.0, 1.0))
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.015, edgeThreshold=12)
    key_source, desc_source = sift.detectAndCompute(source_u8, None)
    key_target, desc_target = sift.detectAndCompute(target_u8, None)
    if desc_source is None or desc_target is None:
        raise RuntimeError("SIFT found no descriptors")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_target, desc_source, k=2)
    matches = [first for first, second in matches if first.distance < 0.72 * second.distance]
    if len(matches) < 20:
        raise RuntimeError(f"Only {len(matches)} ratio-test matches")
    points_target = np.float32([key_target[m.queryIdx].pt for m in matches]) / scale
    points_source = np.float32([key_source[m.trainIdx].pt for m in matches]) / scale
    affine, inliers = cv2.estimateAffine2D(
        points_target,
        points_source,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=20000,
        confidence=0.999,
        refineIters=20,
    )
    if affine is None or inliers is None:
        raise RuntimeError("Robust affine estimation failed")
    inliers = inliers.ravel().astype(bool)
    projected = cv2.transform(points_target[:, None, :], affine)[:, 0]
    residual = np.linalg.norm(projected - points_source, axis=1)
    return affine.astype(np.float64), {
        "ratio_test_matches": len(matches),
        "inliers": int(inliers.sum()),
        "median_reprojection_px": float(np.median(residual[inliers])),
    }


def project_affine(affine: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        affine[0, 0] * x + affine[0, 1] * y + affine[0, 2],
        affine[1, 0] * x + affine[1, 1] * y + affine[1, 2],
    )


def refine_low_band_affine(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    down: float = 0.25,
) -> tuple[np.ndarray, float]:
    source_small = cv2.resize(source, None, fx=down, fy=down, interpolation=cv2.INTER_AREA)
    target_small = cv2.resize(target, None, fx=down, fy=down, interpolation=cv2.INTER_AREA)
    scale = np.diag([down, down, 1.0])
    initial_h = np.vstack([initial, [0.0, 0.0, 1.0]])
    affine = (scale @ initial_h @ np.linalg.inv(scale))[:2].astype(np.float32)

    yy, xx = np.mgrid[: target_small.shape[0], : target_small.shape[1]].astype(np.float32)
    source_x, source_y = project_affine(affine, xx, yy)
    mask = (
        (source_x > 20)
        & (source_x < source_small.shape[1] - 21)
        & (source_y > 20)
        & (source_y < source_small.shape[0] - 21)
    )
    source_low = cv2.GaussianBlur(np.sqrt(np.clip(source_small, 0, 1)), (0, 0), 3.5)
    target_low = cv2.GaussianBlur(np.sqrt(np.clip(target_small, 0, 1)), (0, 0), 3.5)
    for image in (source_low, target_low):
        values = image[mask]
        image -= np.median(values)
        image /= np.std(values) + 1e-8
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
    correlation, refined = cv2.findTransformECC(
        target_low,
        source_low,
        affine,
        cv2.MOTION_AFFINE,
        criteria,
        inputMask=np.uint8(mask) * 255,
        gaussFiltSize=5,
    )
    refined_h = np.vstack([refined, [0.0, 0.0, 1.0]])
    return (np.linalg.inv(scale) @ refined_h @ scale)[:2], float(correlation)


def sample_bicubic(source: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    source_tensor = torch.from_numpy(source.astype(np.float32))[None, None]
    grid = torch_grid(source_x, source_y, source.shape)
    with torch.no_grad():
        result = F.grid_sample(
            source_tensor,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
    return result[0, 0].numpy()


def prepare_pair(
    scene: str,
    target_index: str,
    roi_size: int,
    cache_root: Path,
    force: bool,
) -> dict[str, Any]:
    cache = cache_root / f"{scene}_00001_to_{target_index}_{roi_size}.npz"
    meta_path = cache.with_suffix(".json")
    if cache.exists() and meta_path.exists() and not force:
        arrays = dict(np.load(cache))
        return {"arrays": arrays, "metadata": json.loads(meta_path.read_text())}

    folder = DATASET / scene
    source = load_linear_raw(folder / "00001.ARW")
    target = load_linear_raw(folder / f"{target_index}.ARW")
    initial, feature_stats = feature_affine(source, target)
    affine, ecc = refine_low_band_affine(source, target, initial)

    target_y0 = (target.shape[0] - roi_size) // 2
    target_x0 = (target.shape[1] - roi_size) // 2
    target_y, target_x = np.mgrid[
        target_y0 : target_y0 + roi_size,
        target_x0 : target_x0 + roi_size,
    ].astype(np.float64)
    source_x_full, source_y_full = project_affine(affine, target_x, target_y)
    margin = 96
    source_x0 = max(0, int(np.floor(source_x_full.min())) - margin)
    source_x1 = min(source.shape[1], int(np.ceil(source_x_full.max())) + margin + 1)
    source_y0 = max(0, int(np.floor(source_y_full.min())) - margin)
    source_y1 = min(source.shape[0], int(np.ceil(source_y_full.max())) + margin + 1)
    if source_x1 - source_x0 < roi_size or source_y1 - source_y0 < roi_size:
        raise RuntimeError("Projected central ROI is not contained in the source")

    source_crop = source[source_y0:source_y1, source_x0:source_x1]
    target_crop = target[target_y0 : target_y0 + roi_size, target_x0 : target_x0 + roi_size]
    source_x = source_x_full - source_x0
    source_y = source_y_full - source_y0
    direct_unadjusted = sample_bicubic(source_crop, source_x, source_y)
    border = 24
    valid = np.ones_like(target_crop, dtype=bool)
    valid[:border] = valid[-border:] = False
    valid[:, :border] = valid[:, -border:] = False
    valid &= (target_crop > 0.003) & (target_crop < 0.97)
    valid &= (direct_unadjusted > 0.003) & (direct_unadjusted < 0.97)
    gain, bias = robust_gain_bias(direct_unadjusted, target_crop, valid)
    source_adjusted = gain * source_crop + bias
    transformed_source = gain * direct_unadjusted + bias

    dynamic = float(np.quantile(target_crop[valid], 0.995) - np.quantile(target_crop[valid], 0.005))
    scale = float(np.sqrt(abs(np.linalg.det(affine[:, :2]))))
    arrays = {
        "source": source_adjusted.astype(np.float32),
        "target": target_crop.astype(np.float32),
        "transformed_source": transformed_source.astype(np.float32),
        "source_x": source_x.astype(np.float32),
        "source_y": source_y.astype(np.float32),
        "valid": valid,
        "affine": affine.astype(np.float64),
    }
    metadata = {
        "scene": scene,
        "source_index": "00001",
        "target_index": target_index,
        "feature_geometry": feature_stats,
        "low_band_ecc_correlation": ecc,
        "target_to_source_area_scale": float(abs(np.linalg.det(affine[:, :2]))),
        "target_to_source_linear_scale": scale,
        "photometric_gain": gain,
        "photometric_bias": bias,
        "robust_dynamic_range": dynamic,
        "target_roi_xywh": [target_x0, target_y0, roi_size, roi_size],
        "source_crop_xyxy": [source_x0, source_y0, source_x1, source_y1],
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    meta_path.write_text(json.dumps(metadata, indent=2))
    return {"arrays": arrays, "metadata": metadata}


def normalized_metrics(prediction: np.ndarray, pair: dict[str, Any]) -> dict[str, float]:
    arrays = pair["arrays"]
    result = metrics(prediction, arrays["target"], arrays["valid"])
    result["nrmse"] = result["rmse"] / pair["metadata"]["robust_dynamic_range"]
    return result


def camera_prediction(
    pair: dict[str, Any],
    psf: np.ndarray,
    basis_sigma: float,
    eta: float,
) -> np.ndarray:
    arrays = pair["arrays"]
    return predict_camera_sandwich(
        arrays["source"],
        arrays["source_x"],
        arrays["source_y"],
        psf,
        basis_sigma,
        eta,
    )


def evaluate_configuration(
    pairs: list[dict[str, Any]],
    psf: np.ndarray,
    basis_sigma: float,
    eta: float,
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for pair in pairs:
        prediction = camera_prediction(pair, psf, basis_sigma, eta)
        rows.append(normalized_metrics(prediction, pair))
    return float(np.mean([row["nrmse"] for row in rows])), rows


def solve_candidates(
    train_pairs: list[dict[str, Any]],
    validation_pairs: list[dict[str, Any]],
    radius: int,
    basis_sigma: float,
    symmetry: str,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    orbits, moments = coefficient_orbits(radius, symmetry)
    grams = []
    for pair in train_pairs:
        arrays = pair["arrays"]
        design = build_design(
            arrays["target"],
            arrays["transformed_source"],
            arrays["affine"][:, :2],
            orbits,
            basis_sigma,
            padding=48,
        )
        values = design[arrays["valid"]].astype(np.float64)
        values /= max(pair["metadata"]["robust_dynamic_range"], 1e-8)
        scene_gram = np.einsum("ni,nj->ij", values, values, optimize=False) / values.shape[0]
        if not np.isfinite(scene_gram).all():
            raise RuntimeError(f"Non-finite design matrix for scene {pair['metadata']['scene']}")
        grams.append(scene_gram)
    gram = np.mean(grams, axis=0)
    gram_scale = float(np.trace(gram) / gram.shape[0])
    candidates: list[dict[str, Any]] = []
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    for radius_fraction in [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        radius_penalty = radius_fraction * gram_scale
        orbit_weights = solve_simplex_qp(
            gram,
            moments,
            ridge=1e-6 * gram_scale,
            radius_penalty=radius_penalty,
        )
        psf = expand_orbits(orbit_weights, orbits, radius)
        for eta in [0.01, 0.025, 0.05, 0.1]:
            score, scene_metrics = evaluate_configuration(
                validation_pairs,
                psf,
                basis_sigma,
                eta,
            )
            row = {
                "radius_fraction": radius_fraction,
                "radius_penalty": radius_penalty,
                "eta": eta,
                "validation_mean_nrmse": score,
                "second_moment_coefficients": float(moments @ orbit_weights),
                "validation_scene_metrics": scene_metrics,
            }
            candidates.append(row)
            if best is None or score < best[0]:
                best = (score, psf, row)
    assert best is not None
    return best[1], best[2], candidates


def select_gaussian(
    validation_pairs: list[dict[str, Any]],
) -> tuple[float, float, list[dict[str, Any]]]:
    delta = np.ones((1, 1), dtype=np.float32)
    candidates = []
    best: tuple[float, float, float] | None = None
    for sigma in [0.28, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]:
        for eta in [0.01, 0.025, 0.05, 0.1]:
            score, scene_metrics = evaluate_configuration(
                validation_pairs,
                delta,
                sigma,
                eta,
            )
            row = {
                "sigma": sigma,
                "eta": eta,
                "validation_mean_nrmse": score,
                "validation_scene_metrics": scene_metrics,
            }
            candidates.append(row)
            if best is None or score < best[0]:
                best = (score, sigma, eta)
    assert best is not None
    return best[1], best[2], candidates


def gaussian_prefilter_prediction(pair: dict[str, Any], alpha: float, law: str) -> np.ndarray:
    arrays = pair["arrays"]
    scale = pair["metadata"]["target_to_source_linear_scale"]
    if law == "variance_increment":
        sigma_source = alpha * np.sqrt(max(scale * scale - 1.0, 0.0))
    elif law == "stretched_kernel":
        sigma_source = alpha * scale
    else:
        raise ValueError(law)
    if sigma_source > 1e-6:
        source = cv2.GaussianBlur(
            arrays["source"],
            (0, 0),
            sigmaX=float(sigma_source),
            sigmaY=float(sigma_source),
            borderType=cv2.BORDER_REFLECT101,
        )
    else:
        source = arrays["source"]
    return sample_bicubic(source, arrays["source_x"], arrays["source_y"])


def select_prefilter(
    validation_pairs: list[dict[str, Any]],
    law: str,
) -> tuple[float, list[dict[str, Any]]]:
    candidates = []
    best: tuple[float, float] | None = None
    for alpha in [0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5]:
        scene_metrics = []
        for pair in validation_pairs:
            prediction = gaussian_prefilter_prediction(pair, alpha, law)
            scene_metrics.append(normalized_metrics(prediction, pair))
        score = float(np.mean([row["nrmse"] for row in scene_metrics]))
        row = {
            "alpha": alpha,
            "law": law,
            "validation_mean_nrmse": score,
            "validation_scene_metrics": scene_metrics,
        }
        candidates.append(row)
        if best is None or score < best[0]:
            best = (score, alpha)
    assert best is not None
    return best[1], candidates


def evaluate_methods(
    pairs: list[dict[str, Any]],
    psf: np.ndarray,
    basis_sigma: float,
    eta: float,
    gaussian_sigma: float,
    gaussian_eta: float,
    scale_space_alpha: float,
    stretched_aa_alpha: float,
) -> list[dict[str, Any]]:
    rows = []
    delta = np.ones((1, 1), dtype=np.float32)
    for pair in pairs:
        arrays = pair["arrays"]
        predictions = {
            "direct_bicubic": arrays["transformed_source"],
            "stretched_gaussian_aa": gaussian_prefilter_prediction(pair, stretched_aa_alpha, "stretched_kernel"),
            "gaussian_scale_space": gaussian_prefilter_prediction(pair, scale_space_alpha, "variance_increment"),
            "gaussian_camera": camera_prediction(pair, delta, gaussian_sigma, gaussian_eta),
            "estimated_camera": camera_prediction(pair, psf, basis_sigma, eta),
        }
        row: dict[str, Any] = {
            "scene": pair["metadata"]["scene"],
            "target_index": pair["metadata"]["target_index"],
            "low_band_ecc_correlation": pair["metadata"]["low_band_ecc_correlation"],
            "target_to_source_linear_scale": pair["metadata"]["target_to_source_linear_scale"],
        }
        for name, prediction in predictions.items():
            row[name] = normalized_metrics(prediction, pair)
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = [
        "direct_bicubic",
        "stretched_gaussian_aa",
        "gaussian_scale_space",
        "gaussian_camera",
        "estimated_camera",
    ]
    result: dict[str, Any] = {"number_of_pairs": len(rows)}
    for method in methods:
        values = np.array([row[method]["nrmse"] for row in rows])
        result[method] = {
            "mean_nrmse": float(values.mean()),
            "median_nrmse": float(np.median(values)),
            "mean_psnr_db": float(np.mean([row[method]["psnr_db_robust_range"] for row in rows])),
        }
    direct = np.array([row["direct_bicubic"]["nrmse"] for row in rows])
    stretched_aa = np.array([row["stretched_gaussian_aa"]["nrmse"] for row in rows])
    scale_space = np.array([row["gaussian_scale_space"]["nrmse"] for row in rows])
    gaussian = np.array([row["gaussian_camera"]["nrmse"] for row in rows])
    estimated = np.array([row["estimated_camera"]["nrmse"] for row in rows])

    def comparison(reference: np.ndarray, proposed: np.ndarray) -> dict[str, Any]:
        relative = (reference - proposed) / reference
        rng = np.random.default_rng(12345)
        samples = np.mean(
            relative[rng.integers(0, len(relative), size=(20000, len(relative)))],
            axis=1,
        )
        try:
            pvalue = float(wilcoxon(reference, proposed, alternative="greater").pvalue)
        except ValueError:
            pvalue = 1.0
        return {
            "mean_relative_rmse_reduction": float(relative.mean()),
            "bootstrap_95_percent_ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "wins": int(np.sum(proposed < reference)),
            "paired_wilcoxon_one_sided_p": pvalue,
        }

    result["estimated_vs_direct"] = comparison(direct, estimated)
    result["estimated_vs_stretched_gaussian_aa"] = comparison(stretched_aa, estimated)
    result["estimated_vs_gaussian_camera"] = comparison(gaussian, estimated)
    result["gaussian_camera_vs_direct"] = comparison(direct, gaussian)
    result["gaussian_camera_vs_stretched_gaussian_aa"] = comparison(stretched_aa, gaussian)
    result["gaussian_camera_vs_scale_space_prefilter"] = comparison(scale_space, gaussian)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-size", type=int, default=384)
    parser.add_argument("--basis-sigma", type=float, default=0.28)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--symmetry", choices=["centrosymmetric", "d4"], default="centrosymmetric")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/zoom_experiment/results/multiscene_physical_ls"),
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    scenes = sorted(path.name for path in DATASET.iterdir() if path.is_dir() and (path / "00001.ARW").exists())
    order = np.random.default_rng(args.seed).permutation(scenes).tolist()
    train_scenes = order[: args.train_count]
    validation_scenes = order[args.train_count : args.train_count + args.validation_count]
    test_scenes = order[
        args.train_count + args.validation_count :
        args.train_count + args.validation_count + args.test_count
    ]
    split = {"train": train_scenes, "validation": validation_scenes, "test": test_scenes}
    print(json.dumps(split, indent=2), flush=True)

    cache_root = args.cache_root if args.cache_root is not None else args.output / "cache"
    failures = []

    def prepare_many(scene_ids: list[str], target_index: str) -> list[dict[str, Any]]:
        prepared = []
        for number, scene in enumerate(scene_ids, 1):
            print(f"prepare {target_index} {number}/{len(scene_ids)}: {scene}", flush=True)
            try:
                pair = prepare_pair(scene, target_index, args.roi_size, cache_root, args.force)
                if pair["metadata"]["low_band_ecc_correlation"] < 0.94:
                    raise RuntimeError(
                        f"low-band ECC correlation {pair['metadata']['low_band_ecc_correlation']:.4f} < 0.94"
                    )
                prepared.append(pair)
            except Exception as error:
                failures.append({"scene": scene, "target_index": target_index, "error": str(error)})
                print(f"FAILED {scene} {target_index}: {error}", flush=True)
        return prepared

    train_pairs = prepare_many(train_scenes, "00002")
    validation_pairs = prepare_many(validation_scenes, "00002")
    test_seen_scale = prepare_many(test_scenes, "00002")
    test_unseen_scale = prepare_many(test_scenes, "00003")
    if len(train_pairs) < 3 or len(validation_pairs) < 2 or len(test_seen_scale) < 2 or len(test_unseen_scale) < 2:
        raise RuntimeError("Too few successfully registered pairs for a meaningful split")

    print("solve physical PSF candidates", flush=True)
    psf, best_physical, physical_candidates = solve_candidates(
        train_pairs,
        validation_pairs,
        args.radius,
        args.basis_sigma,
        args.symmetry,
    )
    print("select Gaussian-camera baseline", flush=True)
    gaussian_sigma, gaussian_eta, gaussian_candidates = select_gaussian(validation_pairs)
    print("select Gaussian scale-space prefilter sanity check", flush=True)
    scale_space_alpha, scale_space_candidates = select_prefilter(validation_pairs, "variance_increment")
    print("select stretched-kernel Gaussian AA baseline", flush=True)
    stretched_aa_alpha, stretched_aa_candidates = select_prefilter(validation_pairs, "stretched_kernel")
    print("evaluate held-out scenes and held-out zoom", flush=True)
    seen_rows = evaluate_methods(
        test_seen_scale,
        psf,
        args.basis_sigma,
        best_physical["eta"],
        gaussian_sigma,
        gaussian_eta,
        scale_space_alpha,
        stretched_aa_alpha,
    )
    unseen_rows = evaluate_methods(
        test_unseen_scale,
        psf,
        args.basis_sigma,
        best_physical["eta"],
        gaussian_sigma,
        gaussian_eta,
        scale_space_alpha,
        stretched_aa_alpha,
    )

    result = {
        "experiment": "one PSF estimated by affine cross-convolution LS on training scenes, selected on validation scenes, frozen on held-out scenes and zoom",
        "split": split,
        "successful_counts": {
            "train": len(train_pairs),
            "validation": len(validation_pairs),
            "test_seen_scale": len(test_seen_scale),
            "test_unseen_scale": len(test_unseen_scale),
        },
        "geometry": "full affine target-to-source map: SIFT/RANSAC initialization, then strongly low-pass ECC refinement; every method uses the identical projected target grid",
        "radiometry": "one robust gain and bias estimated per pair from strongly low-passed direct samples and applied before every method",
        "physical_psf": {
            "basis": f"{args.symmetry} nonnegative mixture of continuous unit-mass Gaussian translates",
            "basis_sigma_pixels": args.basis_sigma,
            "radius": args.radius,
            "weights": psf.tolist(),
            "sum": float(psf.sum()),
            "minimum": float(psf.min()),
            "selected": best_physical,
        },
        "gaussian_camera_baseline": {
            "selected_sigma_pixels": gaussian_sigma,
            "selected_eta": gaussian_eta,
        },
        "gaussian_scale_space_sanity_check": {
            "selected_alpha": scale_space_alpha,
            "sigma_source_formula": "alpha * sqrt(target_to_source_scale^2 - 1)",
        },
        "stretched_gaussian_aa_baseline": {
            "selected_alpha": stretched_aa_alpha,
            "sigma_source_formula": "alpha * target_to_source_scale",
        },
        "test_seen_scale": {
            "source_target": "00001 to 00002 (PSF training zoom ratio, unseen scenes)",
            "rows": seen_rows,
            "aggregate": aggregate(seen_rows),
        },
        "test_unseen_scale": {
            "source_target": "00001 to 00003 (unseen zoom ratio and unseen scenes)",
            "rows": unseen_rows,
            "aggregate": aggregate(unseen_rows),
        },
        "failures": failures,
        "selection_records": {
            "physical": physical_candidates,
            "gaussian_camera": gaussian_candidates,
            "gaussian_scale_space": scale_space_candidates,
            "stretched_gaussian_aa": stretched_aa_candidates,
        },
        "limitations": [
            "pilot uses a central luminance ROI from linearly demosaiced RAW rather than a CFA-native likelihood",
            "the affine scene model is exact for planar scenes or pure camera rotation, but real scenes contain parallax and lens distortion",
            "geometry is estimated from the image pair, although only after strong low-pass filtering and identically for every method",
            "focal-length changes may also change the real lens PSF, so frozen-PSF transfer is deliberately stringent",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.output / "estimated_psf.npz", weights=psf)
    print(json.dumps({
        "selected_physical": best_physical,
        "selected_gaussian": {"sigma": gaussian_sigma, "eta": gaussian_eta},
        "selected_scale_space_alpha": scale_space_alpha,
        "selected_stretched_aa_alpha": stretched_aa_alpha,
        "seen_scale": result["test_seen_scale"]["aggregate"],
        "unseen_scale": result["test_unseen_scale"]["aggregate"],
        "failures": failures,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
