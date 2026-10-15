from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import cv2
import jax
import jax.numpy as jax_numpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import skimage
import torch
import torch.nn.functional as torch_functional
import torchvision
from PIL import Image, ImageOps
from scipy.special import rel_entr
from skimage.metrics import structural_similarity
from skimage.transform import resize as skimage_resize
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from optical_cameras import ANNULAR_INNER_RADIUS, ASTIGMATISM_WAVES, CUTOFF, PUPIL_SAMPLES, resize_optical


Array = np.ndarray
ResizeFunction = Callable[[Array, int], Array]


def centered_crop(image: Array, size: int) -> Array:
    height, width = image.shape[:2]
    if size > min(height, width):
        raise ValueError(f"crop {size} does not fit image {width}x{height}")
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size]


def load_scene_crops(dataset: Path, source_size: int, scene_count: int) -> tuple[list[str], list[Array]]:
    scene_directories = sorted(path for path in dataset.iterdir() if path.is_dir())[:scene_count]
    scene_names: list[str] = []
    crops: list[Array] = []
    for directory in scene_directories:
        path = directory / "00001.JPG"
        if not path.exists():
            continue
        with Image.open(path) as pil_image:
            rgb = ImageOps.exif_transpose(pil_image).convert("RGB")
            image = np.asarray(rgb, dtype=np.float32) / 255.0
        crops.append(centered_crop(image, source_size))
        scene_names.append(directory.name)
    if not crops:
        raise RuntimeError(f"no 00001.JPG files found below {dataset}")
    return scene_names, crops


def resize_pillow(image: Array, output_size: int, resample: Image.Resampling) -> Array:
    channels = []
    for channel in range(image.shape[2]):
        plane = Image.fromarray(np.ascontiguousarray(image[..., channel], dtype=np.float32))
        channels.append(np.asarray(plane.resize((output_size, output_size), resample=resample)))
    return np.stack(channels, axis=-1).astype(np.float32)


def resize_opencv(image: Array, output_size: int, interpolation: int) -> Array:
    return cv2.resize(image, (output_size, output_size), interpolation=interpolation).astype(np.float32)


def resize_torch(image: Array, output_size: int, mode: str) -> Array:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    result = torch_functional.interpolate(
        tensor,
        size=(output_size, output_size),
        mode=mode,
        align_corners=False,
        antialias=True,
    )
    return result[0].permute(1, 2, 0).numpy().astype(np.float32)


def resize_skimage(image: Array, output_size: int, order: int) -> Array:
    return skimage_resize(
        image,
        (output_size, output_size, image.shape[2]),
        order=order,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
        clip=False,
    ).astype(np.float32)


def resize_jax(image: Array, output_size: int, method: str) -> Array:
    result = jax.image.resize(
        jax_numpy.asarray(image),
        (output_size, output_size, image.shape[2]),
        method=method,
        antialias=True,
    )
    return np.asarray(result, dtype=np.float32)


def resize_consistent_gaussian(image: Array, output_size: int, sigma_camera: float = 0.75) -> Array:
    """Centered periodic Fourier realization of a Gaussian camera family.

    Odd input/output sizes avoid an ambiguous Nyquist singleton. The field of
    view is fixed, and the multiplier is H(D_out*w)/H(D_in*w). This makes
    same-direction compositions agree up to floating-point roundoff.
    """
    input_size = image.shape[0]
    if image.shape[1] != input_size or input_size % 2 != 1 or output_size % 2 != 1:
        raise ValueError("the spectral control requires odd square sizes")
    if output_size > input_size:
        raise ValueError("this audit uses the coarsening branch only")
    field_of_view = float(input_size)
    input_spacing = 1.0
    output_spacing = field_of_view / float(output_size)
    input_offset = -input_spacing * (input_size - 1) / 2.0
    output_offset = -output_spacing * (output_size - 1) / 2.0
    start = (input_size - output_size) // 2
    stop = start + output_size
    frequencies = 2.0 * np.pi * np.arange(-(output_size // 2), output_size // 2 + 1) / field_of_view
    omega_x, omega_y = np.meshgrid(frequencies, frequencies)
    omega_squared = omega_x**2 + omega_y**2
    relative_camera = np.exp(
        -0.5 * sigma_camera**2 * (output_spacing**2 - input_spacing**2) * omega_squared
    )
    phase = np.exp(1j * (omega_x + omega_y) * (output_offset - input_offset))
    scale = (output_size / input_size) ** 2
    result = np.empty((output_size, output_size, image.shape[2]), dtype=np.float32)
    for channel in range(image.shape[2]):
        spectrum = np.fft.fftshift(np.fft.fft2(image[..., channel]))
        cropped = spectrum[start:stop, start:stop]
        output_spectrum = cropped * relative_camera * phase * scale
        result[..., channel] = np.fft.ifft2(np.fft.ifftshift(output_spectrum)).real.astype(np.float32)
    return result


def method_registry() -> OrderedDict[str, ResizeFunction]:
    return OrderedDict(
        [
            ("Pillow bilinear", lambda image, size: resize_pillow(image, size, Image.Resampling.BILINEAR)),
            ("Pillow bicubic", lambda image, size: resize_pillow(image, size, Image.Resampling.BICUBIC)),
            ("Pillow Lanczos", lambda image, size: resize_pillow(image, size, Image.Resampling.LANCZOS)),
            ("OpenCV area", lambda image, size: resize_opencv(image, size, cv2.INTER_AREA)),
            ("OpenCV bicubic", lambda image, size: resize_opencv(image, size, cv2.INTER_CUBIC)),
            ("OpenCV Lanczos-4", lambda image, size: resize_opencv(image, size, cv2.INTER_LANCZOS4)),
            ("PyTorch bilinear AA", lambda image, size: resize_torch(image, size, "bilinear")),
            ("PyTorch bicubic AA", lambda image, size: resize_torch(image, size, "bicubic")),
            ("scikit-image linear AA", lambda image, size: resize_skimage(image, size, 1)),
            ("scikit-image cubic AA", lambda image, size: resize_skimage(image, size, 3)),
            ("JAX linear AA", lambda image, size: resize_jax(image, size, "linear")),
            ("JAX Lanczos-3 AA", lambda image, size: resize_jax(image, size, "lanczos3")),
            ("camera-consistent Gaussian", resize_consistent_gaussian),
            ("consistent clear pupil", lambda image, size: resize_optical(image, size, "clear pupil")),
            ("consistent annular pupil", lambda image, size: resize_optical(image, size, "annular pupil")),
            ("consistent astigmatic pupil", lambda image, size: resize_optical(image, size, "astigmatic pupil")),
        ]
    )


def nearest_odd(value: float) -> int:
    integer = int(round(value))
    if integer % 2 == 0:
        lower, upper = integer - 1, integer + 1
        integer = lower if abs(value - lower) <= abs(value - upper) else upper
    return max(integer, 3)


def path_sizes(source_size: int, target_size: int, steps: int) -> list[int]:
    ratio = target_size / source_size
    sizes = [source_size]
    for step in range(1, steps):
        sizes.append(nearest_odd(source_size * ratio ** (step / steps)))
    sizes.append(target_size)
    if any(a <= b for a, b in zip(sizes, sizes[1:])):
        raise ValueError(f"non-decreasing path: {sizes}")
    return sizes


def follow_path(image: Array, sizes: list[int], resize_function: ResizeFunction) -> Array:
    result = image
    for output_size in sizes[1:]:
        result = resize_function(result, output_size)
    return result


def image_metrics(direct: Array, chain: Array, evaluation_size: int) -> dict[str, float]:
    direct_eval = centered_crop(direct, evaluation_size)
    chain_eval = centered_crop(chain, evaluation_size)
    difference = chain_eval.astype(np.float64) - direct_eval.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference**2)))
    robust_range = float(np.quantile(direct_eval, 0.995) - np.quantile(direct_eval, 0.005))
    return {
        "rmse": rmse,
        "nrmse": rmse / max(robust_range, 1e-12),
        "psnr_db": float(20.0 * np.log10(1.0 / max(rmse, 1e-12))),
        "ssim": float(
            structural_similarity(
                direct_eval,
                chain_eval,
                data_range=1.0,
                channel_axis=2,
            )
        ),
    }


def grid_audit(
    resize_function: ResizeFunction,
    source_size: int,
    target_size: int,
    steps: int,
    evaluation_size: int,
) -> dict[str, float]:
    coordinate = np.arange(source_size, dtype=np.float32)
    x = np.broadcast_to(coordinate[None, :], (source_size, source_size))
    y = np.broadcast_to(coordinate[:, None], (source_size, source_size))
    ramp = np.stack([x, y, 0.37 * x + 0.63 * y], axis=-1) / (source_size - 1)
    direct = resize_function(ramp, target_size)
    chain = follow_path(ramp, path_sizes(source_size, target_size, steps), resize_function)
    difference_pixels = (centered_crop(chain, evaluation_size) - centered_crop(direct, evaluation_size)) * (
        source_size - 1
    )
    return {
        "rms_source_pixels": float(np.sqrt(np.mean(difference_pixels**2))),
        "max_source_pixels": float(np.max(np.abs(difference_pixels))),
    }


class VisionModel:
    def __init__(self, name: str, model: torch.nn.Module, device: torch.device):
        self.name = name
        self.model = model.eval().to(device)
        self.device = device
        self.features: list[torch.Tensor] = []
        self.handle = self.model.avgpool.register_forward_hook(self._capture_features)

    def _capture_features(self, _module, _inputs, output):
        self.features.append(torch.flatten(output.detach(), 1).cpu())

    def infer(self, images: list[Array], batch_size: int) -> tuple[Array, Array]:
        logits: list[Array] = []
        feature_rows: list[Array] = []
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        self.features = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                array = np.stack(images[start : start + batch_size]).astype(np.float32)
                tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(self.device)
                tensor = (tensor.clamp(0.0, 1.0) - mean) / std
                self.features = []
                output = self.model(tensor)
                logits.append(output.detach().cpu().numpy())
                feature_rows.append(torch.cat(self.features, dim=0).numpy())
        return np.concatenate(logits, axis=0), np.concatenate(feature_rows, axis=0)


def load_models(device: torch.device) -> list[VisionModel]:
    return [
        VisionModel("ResNet-18", resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), device),
        VisionModel(
            "MobileNetV3-Small",
            mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1),
            device,
        ),
    ]


def softmax(logits: Array) -> Array:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def model_comparison_rows(
    model_name: str,
    scene_names: list[str],
    direct_logits: Array,
    direct_features: Array,
    chain_logits: Array,
    chain_features: Array,
) -> list[dict[str, object]]:
    direct_probability = softmax(direct_logits)
    chain_probability = softmax(chain_logits)
    mixture = 0.5 * (direct_probability + chain_probability)
    js = 0.5 * (
        np.sum(rel_entr(direct_probability, mixture), axis=1)
        + np.sum(rel_entr(chain_probability, mixture), axis=1)
    )
    feature_denominator = np.linalg.norm(direct_features, axis=1) * np.linalg.norm(chain_features, axis=1)
    cosine_distance = 1.0 - np.sum(direct_features * chain_features, axis=1) / np.maximum(
        feature_denominator, 1e-12
    )
    direct_top1 = direct_logits.argmax(axis=1)
    chain_top1 = chain_logits.argmax(axis=1)
    direct_confidence = direct_probability.max(axis=1)
    rows: list[dict[str, object]] = []
    for index, scene in enumerate(scene_names):
        rows.append(
            {
                "scene": scene,
                "model": model_name,
                "top1_flip": bool(direct_top1[index] != chain_top1[index]),
                "direct_top1": int(direct_top1[index]),
                "chain_top1": int(chain_top1[index]),
                "direct_confidence": float(direct_confidence[index]),
                "jensen_shannon_nats": float(js[index]),
                "feature_cosine_distance": float(cosine_distance[index]),
            }
        )
    return rows


def bootstrap_mean(values: Array, rng: np.random.Generator, draws: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_results(image_frame: pd.DataFrame, model_frame: pd.DataFrame) -> dict[str, object]:
    rng = np.random.default_rng(20260915)
    aggregate: dict[str, object] = {"image": {}, "models": {}}
    for (method, steps), group in image_frame.groupby(["method", "steps"], sort=False):
        key = f"{method}|{int(steps)}"
        aggregate["image"][key] = {}
        for metric in ["nrmse", "psnr_db", "ssim"]:
            mean, low, high = bootstrap_mean(group[metric].to_numpy(), rng)
            aggregate["image"][key][metric] = {"mean": mean, "ci95": [low, high]}
    for (method, steps, model), group in model_frame.groupby(["method", "steps", "model"], sort=False):
        key = f"{method}|{int(steps)}|{model}"
        aggregate["models"][key] = {}
        for metric in ["top1_flip", "jensen_shannon_nats", "feature_cosine_distance"]:
            mean, low, high = bootstrap_mean(group[metric].astype(float).to_numpy(), rng)
            aggregate["models"][key][metric] = {"mean": mean, "ci95": [low, high]}
    return aggregate


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    methods = method_registry()
    if args.method:
        unknown = sorted(set(args.method) - set(methods))
        if unknown:
            raise ValueError(f"unknown method(s): {unknown}")
        methods = OrderedDict((name, methods[name]) for name in args.method)
    scene_names, source_images = load_scene_crops(args.dataset, args.source_size, args.scene_count)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loaded {len(source_images)} scenes; model device={device}", flush=True)
    models = load_models(device)
    image_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    paths = {steps: path_sizes(args.source_size, args.target_size, steps) for steps in range(1, args.max_steps + 1)}
    for method_index, (method_name, resize_function) in enumerate(methods.items(), 1):
        print(f"method {method_index}/{len(methods)}: {method_name}", flush=True)
        grid = grid_audit(
            resize_function,
            args.source_size,
            args.target_size,
            args.max_steps,
            args.evaluation_size,
        )
        grid_rows.append({"method": method_name, **grid})
        direct_outputs = [resize_function(image, args.target_size) for image in source_images]
        direct_model_inputs = [centered_crop(image, args.evaluation_size) for image in direct_outputs]
        direct_inference = {
            model.name: model.infer(direct_model_inputs, args.batch_size)
            for model in models
        }
        for steps in range(2, args.max_steps + 1):
            print(f"  {steps} steps: {paths[steps]}", flush=True)
            chain_outputs = [follow_path(image, paths[steps], resize_function) for image in source_images]
            chain_model_inputs = [centered_crop(image, args.evaluation_size) for image in chain_outputs]
            for scene, direct, chain in zip(scene_names, direct_outputs, chain_outputs):
                image_rows.append(
                    {
                        "scene": scene,
                        "method": method_name,
                        "steps": steps,
                        **image_metrics(direct, chain, args.evaluation_size),
                    }
                )
            for model in models:
                direct_logits, direct_features = direct_inference[model.name]
                chain_logits, chain_features = model.infer(chain_model_inputs, args.batch_size)
                rows = model_comparison_rows(
                    model.name,
                    scene_names,
                    direct_logits,
                    direct_features,
                    chain_logits,
                    chain_features,
                )
                for row in rows:
                    row["method"] = method_name
                    row["steps"] = steps
                model_rows.extend(rows)
    image_frame = pd.DataFrame(image_rows)
    model_frame = pd.DataFrame(model_rows)
    grid_frame = pd.DataFrame(grid_rows)
    image_frame.to_csv(args.output / "image_metrics.csv", index=False)
    model_frame.to_csv(args.output / "model_metrics.csv", index=False)
    grid_frame.to_csv(args.output / "grid_audit.csv", index=False)
    result = {
        "protocol": {
            "dataset": str(args.dataset),
            "scenes": scene_names,
            "source_file": "00001.JPG",
            "source_size": args.source_size,
            "target_size": args.target_size,
            "evaluation_size": args.evaluation_size,
            "paths": paths,
            "float_pipeline": True,
            "boundary_handling": f"evaluate center {args.evaluation_size}x{args.evaluation_size} of {args.target_size}x{args.target_size}",
            "models": [model.name for model in models],
            "device": str(device),
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "Pillow": PIL.__version__,
            "OpenCV": cv2.__version__,
            "JAX": jax.__version__,
            "PyTorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit-image": skimage.__version__,
        },
        "method_configurations": {
            "Pillow bilinear": "Image.resize, Resampling.BILINEAR, float32 per channel",
            "Pillow bicubic": "Image.resize, Resampling.BICUBIC, float32 per channel",
            "Pillow Lanczos": "Image.resize, Resampling.LANCZOS, float32 per channel",
            "OpenCV area": "cv2.resize, INTER_AREA, float32",
            "OpenCV bicubic": "cv2.resize, INTER_CUBIC, float32",
            "OpenCV Lanczos-4": "cv2.resize, INTER_LANCZOS4, float32",
            "PyTorch bilinear AA": "torch.nn.functional.interpolate, bilinear, antialias=True, align_corners=False",
            "PyTorch bicubic AA": "torch.nn.functional.interpolate, bicubic, antialias=True, align_corners=False",
            "scikit-image linear AA": "skimage.transform.resize, order=1, anti_aliasing=True, reflect",
            "scikit-image cubic AA": "skimage.transform.resize, order=3, anti_aliasing=True, reflect",
            "JAX linear AA": "jax.image.resize, linear, antialias=True",
            "JAX Lanczos-3 AA": "jax.image.resize, lanczos3, antialias=True",
            "camera-consistent Gaussian": "centered Fourier realization, Gaussian camera sigma=0.75 input pixels",
            "consistent clear pupil": f"normalized clear circular pupil OTF, {PUPIL_SAMPLES} pupil samples, cutoff {CUTOFF/np.pi:.3f}pi, square pixel box, centered Fourier realization",
            "consistent annular pupil": f"normalized annular circular pupil OTF, inner radius {ANNULAR_INNER_RADIUS:.3f} of outer radius, {PUPIL_SAMPLES} pupil samples, cutoff {CUTOFF/np.pi:.3f}pi, square pixel box, centered Fourier realization",
            "consistent astigmatic pupil": f"normalized astigmatic circular pupil OTF, phase {ASTIGMATISM_WAVES:.3f} waves (ux^2-uy^2), {PUPIL_SAMPLES} pupil samples, cutoff {CUTOFF/np.pi:.3f}pi, square pixel box, centered Fourier realization",
        },
        "grid_audit": grid_rows,
        "aggregate": aggregate_results(image_frame, model_frame),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(f"saved results to {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(str(__import__('os').environ.get('SCALING_CAPTURE_DATA', 'data/captures'))))
    parser.add_argument("--scene-count", type=int, default=50)
    parser.add_argument("--source-size", type=int, default=1023)
    parser.add_argument("--target-size", type=int, default=255)
    parser.add_argument("--evaluation-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--method", action="append", help="run only the named configuration; may be repeated")
    parser.add_argument("--output", type=Path, default=Path("work/library_audit/results"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
