"""Pretraining checks for scales 2/3/4, 1--10 steps, and native torch AA."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiment import PlainCNN, TorchAADownsample, crop_size, infer, train
from kernels import CachedResize

ROOT = Path(__file__).parent


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(19)
    report = {"uses_hr_pixels": False, "checks": []}
    for scale in (2, 3, 4):
        data = ROOT / ("data" if scale == 2 else f"data_x{scale}")
        manifest = json.loads((data / "manifest.json").read_text())
        for image in manifest["images"]:
            shape = image["lr_shape"][-2:]
            core = crop_size(shape, 64, 16)
            assert core >= 8 and min(shape)-core >= 16
            if scale == 2:
                assert core == 64  # Existing x2 measurements are reusable.
            report["checks"].append({"scale": scale, "image": image["image"],
                                      "lr_shape": shape, "loss_crop": core})
        for m in range(1, 11):
            factor = scale**(1/m)
            shape = (2*(32*scale + math.ceil(2*factor) + 10)+64,)*2
            x = torch.rand(1, 3, *shape, device="cuda")
            down = TorchAADownsample(shape, factor)
            actual = down(x)
            expected = F.interpolate(x, scale_factor=1/factor, mode="bicubic",
                                     align_corners=False, antialias=True,
                                     recompute_scale_factor=False)
            assert torch.equal(actual, expected)
            assert tuple(actual.shape[-2:]) == down.out_shape
            assert (down(torch.ones_like(x))-1).abs().max().item() < 3e-6
            # The centered RR inverse stays common; it must accept torch's
            # native floor-sized output without changing the requested scale.
            up = CachedResize(down.out_shape, factor, shape,
                              weight_device="cuda" if scale >= 3 else "cpu")
            assert tuple(up(actual).shape[-2:]) == shape
            # Difference between native torch and centered grids for the SAME
            # output length: not an assumed random error, a known affine shift.
            native_shift = (down.out_shape[0]*factor-shape[0])/2
            report["checks"].append({"scale": scale, "m": m,
                "father_shape": shape, "torch_son_shape": down.out_shape,
                "native_grid_shift_in_father_pixels": native_shift})

    # Centered RR coordinate maps compose even for rounded intermediate sizes.
    max_grid_error = 0.0
    for scale in (2, 3, 4):
        initial, target = math.ceil(511/scale), 511
        for m in range(1, 11):
            ratio = scale**(1/m)
            sizes = [initial] + [math.ceil(initial*scale**(j/m)-1e-9)
                                 for j in range(1, m)] + [target]
            coords = np.arange(target, dtype=float)
            for j in range(m, 0, -1):
                coords = coords/ratio+(sizes[j-1]-1)/2-(sizes[j]-1)/(2*ratio)
            direct = np.arange(target)/scale+(initial-1)/2-(target-1)/(2*scale)
            max_grid_error = max(max_grid_error, float(np.max(np.abs(coords-direct))))
    assert max_grid_error < 1e-10
    report["max_grid_composition_error"] = max_grid_error

    # Zero residual gives the identical RR bicubic inference chain at m=10.
    model = PlainCNN(width=8, depth=2, residual=True).cuda().eval()
    x = torch.rand(1, 3, 43, 47, device="cuda")
    with torch.no_grad():
        for scale in (2, 3, 4):
            args = SimpleNamespace(scale=scale, device="cuda", depth=2)
            target = (43*scale-1, 47*scale-1)
            prediction, sizes = infer(model, x, 10, args, target)
            expected = x
            for size in sizes:
                expected = CachedResize(expected.shape[-2:], scale**.1, size,
                    weight_device="cuda" if scale >= 3 else "cpu")(expected)
            assert torch.equal(prediction, expected)

    # Smallest-image crop and native torch output exercised through optimization.
    # These are synthetic tensors; no GT image or score is consulted.
    hashes = []
    for method in ("consistent", "aa_bandlimited", "aa", "torch_aa"):
        args = SimpleNamespace(scale=4, seed=0, width=8, depth=2, residual=True,
            device="cuda", lr=1e-4, crop=64, crop_margin=16, radius=128,
            steps=2, batch=1, image="synthetic")
        _, info = train(torch.rand(1, 3, 86, 57, device="cuda"), method, 10, args)
        hashes.append(info["coordinate_sha256"])
        assert info["core_size"] == 41
    assert len(set(hashes)) == 1
    report["all_methods_crop_coordinate_sha256"] = hashes[0]
    report["passed"] = True
    (ROOT / "multiscale_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
