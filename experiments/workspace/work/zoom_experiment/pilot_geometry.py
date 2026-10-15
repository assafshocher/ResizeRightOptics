from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rawpy


def load_gray(path: Path, max_side: int = 1800) -> tuple[np.ndarray, float]:
    if path.suffix.upper() == ".ARW":
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=False,
                output_bps=16,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
            ).astype(np.float32)
        rgb /= 65535.0
        linear_gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        image = np.uint8(np.clip(np.sqrt(linear_gray), 0, 1) * 255)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
    scale = min(1.0, max_side / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, scale


def estimate_homography(source_path: Path, target_path: Path) -> dict:
    source, source_scale = load_gray(source_path)
    target, target_scale = load_gray(target_path)

    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.015, edgeThreshold=12)
    key_source, desc_source = sift.detectAndCompute(source, None)
    key_target, desc_target = sift.detectAndCompute(target, None)
    if desc_source is None or desc_target is None:
        raise RuntimeError("SIFT found no descriptors")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(desc_target, desc_source, k=2)
    matches = [m for m, n in raw_matches if m.distance < 0.72 * n.distance]
    if len(matches) < 12:
        raise RuntimeError(f"Only {len(matches)} ratio-test matches")

    points_target_small = np.float32([key_target[m.queryIdx].pt for m in matches])
    points_source_small = np.float32([key_source[m.trainIdx].pt for m in matches])
    points_target = points_target_small / target_scale
    points_source = points_source_small / source_scale

    homography, mask_h = cv2.findHomography(
        points_target,
        points_source,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=2.0,
        confidence=0.999,
        maxIters=20000,
    )
    affine, mask_a = cv2.estimateAffinePartial2D(
        points_target,
        points_source,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=20000,
        confidence=0.999,
        refineIters=20,
    )
    if homography is None or affine is None:
        raise RuntimeError("Robust geometry estimation failed")

    mask_h = mask_h.ravel().astype(bool)
    mask_a = mask_a.ravel().astype(bool)
    projected_h = cv2.perspectiveTransform(points_target[:, None, :], homography)[:, 0, :]
    projected_a = cv2.transform(points_target[:, None, :], affine)[:, 0, :]
    residual_h = np.linalg.norm(projected_h - points_source, axis=1)
    residual_a = np.linalg.norm(projected_a - points_source, axis=1)

    linear = homography[:2, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    return {
        "source": str(source_path),
        "target": str(target_path),
        "source_shape": list(load_gray(source_path, max_side=100000)[0].shape),
        "target_shape": list(load_gray(target_path, max_side=100000)[0].shape),
        "homography_target_to_source": homography.tolist(),
        "similarity_target_to_source": affine.tolist(),
        "ratio_test_matches": len(matches),
        "homography_inliers": int(mask_h.sum()),
        "homography_median_reprojection_px": float(np.median(residual_h[mask_h])),
        "similarity_inliers": int(mask_a.sum()),
        "similarity_median_reprojection_px": float(np.median(residual_a[mask_a])),
        "homography_origin_singular_values": singular_values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="00007")
    parser.add_argument("--source-index", default="00001")
    parser.add_argument("--target-index", default="00002")
    parser.add_argument("--extension", default="JPG", choices=["JPG", "ARW"])
    parser.add_argument("--dataset", type=Path, default=Path(str(__import__('os').environ.get('SCALING_CAPTURE_DATA', 'data/captures'))))
    parser.add_argument("--output", type=Path, default=Path("work/zoom_experiment/results/00007_geometry.json"))
    args = parser.parse_args()

    scene = args.dataset / args.scene
    result = estimate_homography(
        scene / f"{args.source_index}.{args.extension}",
        scene / f"{args.target_index}.{args.extension}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
