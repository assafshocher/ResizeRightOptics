from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rawpy
import torch
import torch.nn.functional as F


def load_linear_raw(path: Path) -> np.ndarray:
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=False,
            output_bps=16,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
        ).astype(np.float32)
    rgb /= 65535.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def refine_low_band_homography(source: np.ndarray, target: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, float]:
    down = 0.25
    source_small = cv2.resize(source, None, fx=down, fy=down, interpolation=cv2.INTER_AREA)
    target_small = cv2.resize(target, None, fx=down, fy=down, interpolation=cv2.INTER_AREA)
    scale = np.diag([down, down, 1.0]).astype(np.float32)
    homography = scale @ initial.astype(np.float32) @ np.linalg.inv(scale)

    y, x = np.mgrid[0 : target_small.shape[0], 0 : target_small.shape[1]].astype(np.float32)
    denominator = homography[2, 0] * x + homography[2, 1] * y + homography[2, 2]
    source_x = (homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]) / denominator
    source_y = (homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]) / denominator
    mask = (
        (source_x > 20)
        & (source_x < source_small.shape[1] - 21)
        & (source_y > 20)
        & (source_y < source_small.shape[0] - 21)
    ).astype(np.uint8) * 255

    source_low = cv2.GaussianBlur(np.sqrt(np.clip(source_small, 0, 1)), (0, 0), 3.0)
    target_low = cv2.GaussianBlur(np.sqrt(np.clip(target_small, 0, 1)), (0, 0), 3.0)
    for image in (source_low, target_low):
        values = image[mask > 0]
        image -= np.median(values)
        image /= np.std(values) + 1e-8

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-8)
    correlation, refined = cv2.findTransformECC(
        target_low,
        source_low,
        homography,
        cv2.MOTION_HOMOGRAPHY,
        criteria,
        inputMask=mask,
        gaussFiltSize=5,
    )
    return np.linalg.inv(scale) @ refined @ scale, float(correlation)


def project_grid(homography: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denominator = homography[2, 0] * x + homography[2, 1] * y + homography[2, 2]
    source_x = (homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]) / denominator
    source_y = (homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]) / denominator
    return source_x, source_y


def torch_grid(source_x: np.ndarray, source_y: np.ndarray, source_shape: tuple[int, int]) -> torch.Tensor:
    height, width = source_shape
    grid_x = 2.0 * source_x / (width - 1) - 1.0
    grid_y = 2.0 * source_y / (height - 1) - 1.0
    return torch.from_numpy(np.stack([grid_x, grid_y], axis=-1).astype(np.float32))[None]


def transfer_from_weights(weights: torch.Tensor, shape: tuple[int, int], basis_sigma: float) -> torch.Tensor:
    height, width = shape
    kernel_height, kernel_width = weights.shape
    # Put the zero-offset coefficient exactly at the FFT origin after
    # ifftshift.  The usual symmetric padding is off by one when an odd
    # kernel is embedded in an even-sized image.
    top = height // 2 - kernel_height // 2
    bottom = height - kernel_height - top
    left = width // 2 - kernel_width // 2
    right = width - kernel_width - left
    centered = F.pad(weights, (left, right, top, bottom))
    coefficient_transfer = torch.fft.fft2(torch.fft.ifftshift(centered))
    omega_y = 2 * torch.pi * torch.fft.fftfreq(height, device=weights.device)
    omega_x = 2 * torch.pi * torch.fft.fftfreq(width, device=weights.device)
    basis_transfer = torch.exp(
        -0.5 * basis_sigma**2 * (omega_y[:, None] ** 2 + omega_x[None, :] ** 2)
    )
    return coefficient_transfer * basis_transfer


def filter_image(
    image: torch.Tensor,
    weights: torch.Tensor,
    basis_sigma: float,
    inverse: bool,
    eta: float,
    padding: int,
) -> torch.Tensor:
    padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    transfer = transfer_from_weights(weights, padded.shape[-2:], basis_sigma)
    if inverse:
        multiplier = transfer.conj() / (transfer.abs().square() + eta**2)
    else:
        multiplier = transfer
    result = torch.fft.ifft2(torch.fft.fft2(padded[0, 0]) * multiplier).real[None, None]
    return result[..., padding:-padding, padding:-padding]


def fit_gain_bias(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    x = prediction[mask].reshape(-1)
    y = target[mask].reshape(-1)
    design = np.stack([x, np.ones_like(x)], axis=1)
    gain, bias = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(gain), float(bias)


def metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    residual = prediction[mask] - target[mask]
    rmse = float(np.sqrt(np.mean(residual**2)))
    dynamic = float(np.quantile(target[mask], 0.995) - np.quantile(target[mask], 0.005))
    psnr = float(20 * np.log10(max(dynamic, 1e-8) / max(rmse, 1e-12)))
    return {"rmse": rmse, "psnr_db_robust_range": psnr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="00007")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--roi-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("work/zoom_experiment/results/00007_psf_pilot"))
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = Path(str(__import__('os').environ.get('SCALING_CAPTURE_DATA', 'data/captures'))) / args.scene
    source = load_linear_raw(dataset / "00001.ARW")
    target = load_linear_raw(dataset / "00002.ARW")
    geometry_path = Path(f"work/zoom_experiment/results/{args.scene}_geometry_raw.json")
    initial = np.array(json.loads(geometry_path.read_text())["homography_target_to_source"], dtype=np.float64)
    homography, ecc_correlation = refine_low_band_homography(source, target, initial)

    roi_height = roi_width = args.roi_size
    target_y0 = (target.shape[0] - roi_height) // 2
    target_x0 = (target.shape[1] - roi_width) // 2
    target_y, target_x = np.mgrid[
        target_y0 : target_y0 + roi_height,
        target_x0 : target_x0 + roi_width,
    ].astype(np.float64)
    source_x_full, source_y_full = project_grid(homography, target_x, target_y)

    source_margin = 96
    source_x0 = max(0, int(np.floor(source_x_full.min())) - source_margin)
    source_x1 = min(source.shape[1], int(np.ceil(source_x_full.max())) + source_margin + 1)
    source_y0 = max(0, int(np.floor(source_y_full.min())) - source_margin)
    source_y1 = min(source.shape[0], int(np.ceil(source_y_full.max())) + source_margin + 1)
    source_crop = source[source_y0:source_y1, source_x0:source_x1]
    target_crop = target[target_y0 : target_y0 + roi_height, target_x0 : target_x0 + roi_width]
    source_x = source_x_full - source_x0
    source_y = source_y_full - source_y0
    grid = torch_grid(source_x, source_y, source_crop.shape)

    device = torch.device("cpu")
    source_tensor = torch.from_numpy(source_crop.astype(np.float32))[None, None].to(device)
    target_tensor = torch.from_numpy(target_crop.astype(np.float32))[None, None].to(device)
    grid = grid.to(device)

    tile = 64
    yy, xx = np.mgrid[0:roi_height, 0:roi_width]
    train_mask_np = ((xx // tile + yy // tile) % 2 == 0)
    radiometric_mask = (target_crop > 0.005) & (target_crop < 0.95)
    train_mask_np &= radiometric_mask
    validation_mask_np = (~((xx // tile + yy // tile) % 2 == 0)) & radiometric_mask
    train_mask = torch.from_numpy(train_mask_np)[None, None].to(device)

    with torch.no_grad():
        direct = F.grid_sample(
            source_tensor,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )[0, 0].numpy()
    direct_gain, direct_bias = fit_gain_bias(direct, target_crop, train_mask_np)
    direct = direct_gain * direct + direct_bias

    radius = 4
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
    gy, gx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    initial_logits = -(gx.square() + gy.square()) / (2 * 1.0**2)
    logits = torch.nn.Parameter(initial_logits)
    log_gain = torch.nn.Parameter(torch.tensor(np.log(max(direct_gain, 1e-3)), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.tensor(direct_bias, dtype=torch.float32))
    optimizer = torch.optim.Adam([logits, log_gain, bias], lr=0.025)
    basis_sigma = 0.28
    eta = 0.05
    history: list[dict[str, float]] = []

    for iteration in range(args.iterations):
        optimizer.zero_grad()
        symmetric_logits = 0.5 * (logits + torch.flip(logits, dims=(0, 1)))
        weights = torch.softmax(symmetric_logits.reshape(-1), dim=0).reshape_as(logits)
        visible_scene = filter_image(source_tensor, weights, basis_sigma, True, eta, padding=48)
        transformed_scene = F.grid_sample(
            visible_scene,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
        prediction = filter_image(transformed_scene, weights, basis_sigma, False, eta, padding=32)
        prediction = torch.exp(log_gain) * prediction + bias
        residual = prediction - target_tensor
        data_loss = torch.sqrt(residual[train_mask].square().mean() + 1e-10)
        radius_penalty = (weights * (gx.square() + gy.square())).sum()
        smoothness = (
            (weights[1:, :] - weights[:-1, :]).square().mean()
            + (weights[:, 1:] - weights[:, :-1]).square().mean()
        )
        loss = data_loss + 5e-6 * radius_penalty + 1e-4 * smoothness
        loss.backward()
        optimizer.step()
        if iteration % 10 == 0 or iteration == args.iterations - 1:
            history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach()),
                    "data_loss": float(data_loss.detach()),
                    "psf_second_moment": float(radius_penalty.detach()),
                }
            )
            print(history[-1], flush=True)

    with torch.no_grad():
        symmetric_logits = 0.5 * (logits + torch.flip(logits, dims=(0, 1)))
        weights = torch.softmax(symmetric_logits.reshape(-1), dim=0).reshape_as(logits)
        visible_scene = filter_image(source_tensor, weights, basis_sigma, True, eta, padding=48)
        transformed_scene = F.grid_sample(
            visible_scene,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
        proposed = filter_image(transformed_scene, weights, basis_sigma, False, eta, padding=32)
        proposed = (torch.exp(log_gain) * proposed + bias)[0, 0].numpy()
        weights_np = weights.numpy()

    results = {
        "scene": args.scene,
        "source": "00001.ARW (240 mm)",
        "target": "00002.ARW (154 mm)",
        "geometry": {
            "homography_target_to_source": homography.tolist(),
            "low_band_ecc_correlation": ecc_correlation,
            "target_roi_xywh": [target_x0, target_y0, roi_width, roi_height],
            "source_crop_xyxy": [source_x0, source_y0, source_x1, source_y1],
        },
        "continuous_psf": {
            "coefficient_weights": weights_np.tolist(),
            "basis": "unit-mass isotropic Gaussian",
            "basis_sigma_pixels": basis_sigma,
            "tikhonov_eta": eta,
            "second_moment_coefficients": float((weights_np * (gx.numpy() ** 2 + gy.numpy() ** 2)).sum()),
        },
        "photometric": {
            "direct_gain": direct_gain,
            "direct_bias": direct_bias,
            "proposed_gain": float(torch.exp(log_gain).detach()),
            "proposed_bias": float(bias.detach()),
        },
        "validation": {
            "direct_bicubic": metrics(direct, target_crop, validation_mask_np),
            "camera_sandwich": metrics(proposed, target_crop, validation_mask_np),
        },
        "training": {
            "direct_bicubic": metrics(direct, target_crop, train_mask_np),
            "camera_sandwich": metrics(proposed, target_crop, train_mask_np),
        },
        "history": history,
        "important_limitations": [
            "one scene and one zoom pair",
            "linear RAW demosaiced with the same bilinear algorithm rather than CFA-native fitting",
            "continuous projected grid evaluated with bicubic sampling of the reconstructed scene",
            "geometry estimated from the same scene but only after strong low-pass filtering",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        args.output / "arrays.npz",
        source_crop=source_crop,
        target_crop=target_crop,
        direct=direct,
        proposed=proposed,
        train_mask=train_mask_np,
        validation_mask=validation_mask_np,
        psf_weights=weights_np,
        homography=homography,
        source_x=source_x_full,
        source_y=source_y_full,
    )
    print(json.dumps(results["validation"], indent=2))


if __name__ == "__main__":
    main()
