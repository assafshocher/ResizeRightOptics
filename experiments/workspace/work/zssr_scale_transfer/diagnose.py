"""Check optimization and forward composition using only the observed LR.

This deliberately never opens the original HR image. Any training-recipe
repair must help the shared LR reconstruction task, not select a method's
best final-target PSNR.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiment import PlainCNN, load_image, degradation, train
from kernels import CachedResize, ContinuousKernel

ROOT = Path(__file__).parent


@torch.no_grad()
def reconstruction(model, image, method, m, radius=64, depth=8):
    factor = 2**(1/m)
    kernel, aa = degradation(method, factor, radius)
    down = CachedResize(image.shape[-2:], 1/factor, method=kernel, antialiasing=aa)
    up = CachedResize(down.out_shape, factor, image.shape[-2:])
    interp = up(down(image))
    output = model(F.pad(interp, [depth]*4, mode="reflect").contiguous(memory_format=torch.channels_last))[
        ..., depth:-depth, depth:-depth]
    inner = (..., slice(16, -16), slice(16, -16))
    cnn_delta = (output-image)[inner]
    interp_delta = (interp-image)[inner]
    return {"cnn_l1": cnn_delta.abs().mean().item(),
            "interpolation_l1": interp_delta.abs().mean().item(),
            "cnn_rmse": cnn_delta.square().mean().sqrt().item(),
            "interpolation_rmse": interp_delta.square().mean().sqrt().item(),
            "cnn_rgb_bias": cnn_delta.mean((0, 2, 3)).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    image = load_image(ROOT / "data" / "baby_lr_bandlimited.npy", "cuda")
    reports = []
    # Identical fixed replacement schedule in both laws; no HR-based search.
    args = SimpleNamespace(seed=0, width=64, depth=8, device="cuda", batch=1,
                           steps=10000, crop=64, radius=64, lr=1e-4, image="baby")
    for m in [1, 5]:
        for method in (["consistent"] if m == 1 else ["consistent", "aa_bandlimited"]):
            model = PlainCNN().to("cuda", memory_format=torch.channels_last).eval()
            path = ROOT / "runs" / "first_image_v1" / "baby_seed0" / f"m{m}_{method}.pt"
            model.load_state_dict(torch.load(path, weights_only=True))
            before = reconstruction(model, image, method, m)
            report = {"m": m, "method": method, "initial_schedule": before}
            if options.repair:
                model, training = train(image, method, m, args)
                report["fixed_slower_lr_schedule"] = reconstruction(model, image, method, m)
                report["training"] = training
            reports.append(report)
            print(json.dumps({"event": "lr_only_diagnostic", **report}), flush=True)
    # Forward-only numerical check: compare direct down x2 with m smaller steps.
    anchor = ContinuousKernel(2)
    direct = CachedResize(image.shape[-2:], .5, method=anchor, antialiasing=False)(image)
    composition = []
    for method in ["consistent", "aa_bandlimited"]:
        for m in [2, 3, 4, 5]:
            factor = 2**(1/m)
            kernel, aa = degradation(method, factor, 64)
            current = image
            for j in range(1, m+1):
                shape = tuple(int(np.ceil(n / 2**(j/m) - 1e-9)) for n in image.shape[-2:])
                current = CachedResize(current.shape[-2:], 1/factor, shape, method=kernel,
                                       antialiasing=aa)(current)
            diff = (current-direct)[..., 16:-16, 16:-16]
            composition.append({"method": method, "m": m,
                                "rmse": diff.square().mean().sqrt().item()})
    report = {"uses_original_hr": False, "optimization": reports, "forward_composition": composition}
    suffix = "repair" if options.repair else "initial"
    (ROOT / f"diagnostic_{suffix}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "composition", "values": composition}), flush=True)


if __name__ == "__main__":
    main()
