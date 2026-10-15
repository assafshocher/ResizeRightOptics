"""Appendix audit: common size-only resizers against a prescribed optical zoom.

The underlying natural JPG crop is treated as a periodic continuous Fourier
scene. A clear-pupil camera produces both the input capture and one target
capture at independently specified physical scale s and output count N.
The target is simulated, not a real photograph or a ground-truth camera
measurement. A same-optics/size-inferred-grid arm isolates coordinate error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from library_model_audit import (
    centered_crop,
    follow_path,
    image_metrics,
    load_models,
    load_scene_crops,
    method_registry,
    model_comparison_rows,
    path_sizes,
)
from physical_zoom import source_capture, target_capture


NATIVE_METHODS = [
    "Pillow bilinear",
    "Pillow bicubic",
    "Pillow Lanczos",
    "OpenCV area",
    "OpenCV bicubic",
    "OpenCV Lanczos-4",
    "PyTorch bilinear AA",
    "PyTorch bicubic AA",
    "scikit-image linear AA",
    "scikit-image cubic AA",
    "JAX linear AA",
    "JAX Lanczos-3 AA",
]
WRONG_GRID = "same optics, N/N0 grid"


def compare_candidates(
    method: str,
    steps: int,
    scenes: list[str],
    reference: list[np.ndarray],
    candidates: list[np.ndarray],
    reference_inference: dict,
    models: list,
    batch_size: int,
    evaluation_size: int,
) -> tuple[list[dict], list[dict]]:
    image_rows = [
        {
            "scene": scene,
            "method": method,
            "steps": steps,
            **image_metrics(target, candidate, evaluation_size),
        }
        for scene, target, candidate in zip(scenes, reference, candidates)
    ]
    inputs = [centered_crop(image, evaluation_size) for image in candidates]
    model_rows = []
    for model in models:
        target_logits, target_features = reference_inference[model.name]
        candidate_logits, candidate_features = model.infer(inputs, batch_size)
        rows = model_comparison_rows(
            model.name,
            scenes,
            target_logits,
            target_features,
            candidate_logits,
            candidate_features,
        )
        for row in rows:
            row["method"] = method
            row["steps"] = steps
        model_rows.extend(rows)
    return image_rows, model_rows


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    scenes, source_images = load_scene_crops(args.dataset, args.source_size, args.scene_count)
    print(f"loaded {len(source_images)} natural scenes", flush=True)
    physical_scale = (args.target_size + args.rounding_offset) / args.source_size
    inferred_scale = args.target_size / args.source_size
    source_captures = []
    reference = []
    wrong_grid = []
    for index, scene in enumerate(source_images):
        reference.append(target_capture(scene, physical_scale, args.target_size, args.camera))
        wrong_grid.append(
            target_capture(
                scene,
                physical_scale,
                args.target_size,
                args.camera,
                grid_scale=inferred_scale,
            )
        )
        source_captures.append(source_capture(scene, args.camera))
        source_images[index] = None
        if (index + 1) % 10 == 0:
            print(f"simulated {index + 1}/{len(scenes)} paired captures", flush=True)
    source_images.clear()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = load_models(device)
    reference_inputs = [centered_crop(image, args.evaluation_size) for image in reference]
    reference_inference = {
        model.name: model.infer(reference_inputs, args.batch_size) for model in models
    }
    image_rows = []
    model_rows = []
    control_image, control_models = compare_candidates(
        WRONG_GRID, 0, scenes, reference, wrong_grid, reference_inference,
        models, args.batch_size, args.evaluation_size,
    )
    image_rows.extend(control_image)
    model_rows.extend(control_models)
    print("measured same-optics wrong-grid control", flush=True)

    registry = method_registry()
    methods = args.method or NATIVE_METHODS
    if unknown := sorted(set(methods) - set(registry)):
        raise ValueError(f"unknown methods: {unknown}")
    paths = {
        steps: path_sizes(args.source_size, args.target_size, steps)
        for steps in range(1, args.max_steps + 1)
    }
    for method_index, method in enumerate(methods, 1):
        resize = registry[method]
        print(f"method {method_index}/{len(methods)}: {method}", flush=True)
        for steps, sizes in paths.items():
            outputs = [follow_path(image, sizes, resize) for image in source_captures]
            rows_image, rows_model = compare_candidates(
                method, steps, scenes, reference, outputs,
                reference_inference, models, args.batch_size, args.evaluation_size,
            )
            image_rows.extend(rows_image)
            model_rows.extend(rows_model)
            print(f"  {steps} steps: {sizes}", flush=True)

    image_frame = pd.DataFrame(image_rows)
    model_frame = pd.DataFrame(model_rows)
    image_frame.to_csv(args.output / "image_metrics.csv", index=False)
    model_frame.to_csv(args.output / "model_metrics.csv", index=False)
    output_indices = np.arange(args.target_size) - (args.target_size - 1) / 2.0
    edge_drift = float(
        np.max(np.abs(output_indices * (1.0 / inferred_scale - 1.0 / physical_scale)))
    )
    result = {
        "protocol": {
            "dataset": str(args.dataset),
            "scenes": scenes,
            "source_size": args.source_size,
            "target_size": args.target_size,
            "physical_scale": physical_scale,
            "size_inferred_scale": inferred_scale,
            "N0_times_s": args.target_size + args.rounding_offset,
            "max_grid_drift_source_pixels": edge_drift,
            "camera": args.camera,
            "camera_reference": "periodic Shannon scene, computed pupil camera, direct optical capture",
            "grid_only_control": WRONG_GRID,
            "paths": paths,
            "evaluation_size": args.evaluation_size,
            "models": [model.name for model in models],
            "methods": methods,
            "model_device": str(device),
            "float_pipeline": True,
            "warning": "simulated camera, not a measured natural-image PSF or a recognition-accuracy benchmark",
        }
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(f"saved {len(image_rows)} image comparisons to {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(str(__import__('os').environ.get('SCALING_CAPTURE_DATA', 'data/captures'))))
    parser.add_argument("--scene-count", type=int, default=50)
    parser.add_argument("--source-size", type=int, default=2047)
    parser.add_argument("--target-size", type=int, default=255)
    parser.add_argument("--rounding-offset", type=float, default=0.49)
    parser.add_argument("--camera", choices=["clear pupil", "annular pupil", "astigmatic pupil"], default="clear pupil")
    parser.add_argument("--evaluation-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--method", action="append")
    parser.add_argument("--output", type=Path, default=Path("work/library_audit/grid_zoom_pilot"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
