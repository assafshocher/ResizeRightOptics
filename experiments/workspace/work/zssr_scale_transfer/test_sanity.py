"""Bug checks: grid, online crop indexing, cached GPU RR and CNN alignment."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiment import PlainCNN, load_image, metrics
from kernels import CachedResize, ContinuousKernel, interp_methods

ROOT = Path(__file__).parent


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(93)
    report = {}
    image = load_image(ROOT / "data" / "baby_lr.npy", "cuda")
    halo, core, depth = 78, 64, 8
    size = core + 2*halo
    padded = F.pad(image, [halo]*4, mode="reflect")[0]
    y = torch.tensor([12, 47], device="cuda")[:, None, None]
    x = torch.tensor([31, 60], device="cuda")[:, None, None]
    offsets = torch.arange(size, device="cuda")
    actual = padded[:, y+offsets[None, :, None], x+offsets[None, None, :]].permute(1, 0, 2, 3)
    expected = torch.stack([padded[:, 12:12+size, 31:31+size], padded[:, 47:47+size, 60:60+size]])
    report["online_crop_indexing_max_error"] = (actual-expected).abs().max().item()
    assert report["online_crop_indexing_max_error"] == 0
    report["crop_target_matches_original"] = (actual[0, :, halo:halo+core, halo:halo+core]
                                                - image[0, :, 12:12+core, 31:31+core]).abs().max().item()
    assert report["crop_target_matches_original"] == 0
    grid_error = 0
    for m in range(1, 6):
        factor = 2**(1/m)
        for length in [217, 220, 237]:
            small = int(np.ceil(length / factor))
            k = np.arange(length, dtype=np.float64)
            # RR upsampling grid is in small-image coordinates, then mapped
            # back through the degradation grid to original coordinates.
            up_grid = k/factor + (small-1)/2 - (length-1)/(2*factor)
            composed_grid = factor*up_grid + (length-1)/2 - factor*(small-1)/2
            grid_error = max(grid_error, float(np.max(np.abs(composed_grid-k))))
    report["down_up_coordinate_max_error"] = grid_error
    assert grid_error < 1e-10
    gpu_checks = []
    for factor in [2, 2**(1/5)]:
        for method, aa, name in [(interp_methods.cubic, True, "bicubic_aa"),
                                 (interp_methods.cubic, False, "bicubic_noaa"),
                                 (ContinuousKernel(factor), False, "consistent")]:
            plan = CachedResize((size, size), 1/factor, method=method, antialiasing=aa)
            error = (plan(actual)-plan.direct(actual)).abs().max().item()
            # Device-specific float32 grid evaluation can differ by a few ulps.
            assert error < 2e-5, (factor, name, error)
            gpu_checks.append({"factor": factor, "method": name, "max_error": error})
    report["gpu_cached_vs_unmodified_RR"] = gpu_checks
    model = PlainCNN().to("cuda", memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(ROOT / "runs/first_image_v1/baby_seed0/m1_aa.pt", weights_only=True))
    down = CachedResize((size, size), .5)
    interp = CachedResize(down.out_shape, 2, (size, size))(down(actual))
    train_input = interp[..., halo-depth:halo+core+depth, halo-depth:halo+core+depth]
    train_output = model(train_input.contiguous(memory_format=torch.channels_last))[..., depth:-depth, depth:-depth]
    full = model(F.pad(interp, [depth]*4, mode="reflect").contiguous(memory_format=torch.channels_last))[
        ..., depth:-depth, depth:-depth]
    inference_output = full[..., halo:halo+core, halo:halo+core]
    error = (train_output-inference_output).abs().max().item()
    report["training_vs_inference_same_patch_max_error"] = error
    assert error < 2e-5
    for parameter in model.parameters():
        parameter.zero_()
    report["zero_weights_output_max_proves_no_skip"] = model(train_input).abs().max().item()
    assert report["zero_weights_output_max_proves_no_skip"] == 0
    # Actual evaluation truth is an original image in [0,1]. An interpolated
    # LR may overshoot this range; it is not a valid truth for a clipping test.
    report["identity_psnr"] = metrics(image.clamp(0, 1), image.clamp(0, 1))
    assert abs(report["identity_psnr"]["psnr_rgb"] - 300) < 1e-10
    (ROOT / "sanity_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
