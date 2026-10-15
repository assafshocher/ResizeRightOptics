"""No GT/PSNR: check filters, sampling coordinates, and unchanged old arms."""
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from kernels import CachedResize, ContinuousKernel, interp_methods, rr
from matched_kernels import MatchedTorchGridResize, grid_convention


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(402)
    original = rr.get_projected_grid
    checks = []
    for scale in (2, 3, 4):
        weight_device = "cpu" if scale == 2 else "cuda"
        for m in range(1, 11):
            factor = scale**(1/m)
            kernel = ContinuousKernel(factor, radius=32*scale,
                                      law="aa_bandlimited", anchor=scale)
            # Larger than the kernel halo, including non-divisible lengths.
            shape = (2*(32*scale + math.ceil(2*factor) + 10)+64,)*2
            x = torch.rand(1, 1, *shape, device=weight_device)
            for mode, filt, aa in (("bandlimited", kernel, False),
                                    ("literal", interp_methods.cubic, True)):
                down = MatchedTorchGridResize(shape, 1/factor, method=filt,
                            antialiasing=aa, device=weight_device, weight_device=weight_device)
                actual = down(x)
                direct = down.direct(x)
                maxerr = float((actual-direct).abs().max())
                assert maxerr < 4e-6, (scale, m, mode, maxerr)
                dc = float((down(torch.ones_like(x))-1).abs().max())
                assert dc < 4e-6, (scale, m, mode, dc)
                assert down.out_shape == tuple(math.floor(n/factor) for n in shape)
                assert rr.get_projected_grid is original
                with grid_convention("torch_half_pixel"):
                    grid = rr.get_projected_grid(shape[0], down.out_shape[0], 1/factor,
                                                  torch, False, weight_device)
                expected = (torch.arange(down.out_shape[0], device=weight_device)+.5)*factor-.5
                grid_error = float((grid-expected).abs().max())
                assert grid_error < 8e-5, (scale, m, grid_error)
                row = dict(scale=scale, m=m, mode=mode, direct_max_error=maxerr,
                           dc_max_error=dc, grid_max_error=grid_error)
                if mode == "literal":
                    native = F.interpolate(x, scale_factor=1/factor, mode="bicubic",
                        align_corners=False, antialias=True, recompute_scale_factor=False)
                    # Same cubic/grid in the interior; edge handling deliberately
                    # remains RR reflect to isolate only the grid convention.
                    margin = 6
                    interior = float((actual[..., margin:-margin, margin:-margin]
                                      - native[..., margin:-margin, margin:-margin]).abs().max())
                    assert interior < 1e-4, (scale, m, interior)
                    row["native_torch_interior_max_error"] = interior
                checks.append(row)
    # Re-entering this module has not changed standard RR operations.
    x = torch.rand(1, 3, 57, 65, device="cuda")
    standard = CachedResize(x.shape[-2:], .73, weight_device="cuda")
    assert (standard(x)-standard.direct(x)).abs().max().item() < 3e-6
    report = {"passed": True, "checks": checks, "uses_hr_pixels": False,
              "meaning": "Same RR filters/reflection/normalization; native floor + half-pixel grid only"}
    from experiment import save_json
    save_json(Path(__file__).parent / "matched_grid_checks.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
