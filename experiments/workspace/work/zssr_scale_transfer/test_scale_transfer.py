"""Scale-generalization checks using synthetic data, not HR score selection."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiment import PlainCNN, reflect_halo, infer
from kernels import CachedResize, ContinuousKernel, camera


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {}
    x = torch.rand(1, 3, 71, 76)
    report["ordinary_reflect_max_error"] = float((reflect_halo(x, 50)-F.pad(x, [50]*4, mode="reflect")).abs().max())
    assert report["ordinary_reflect_max_error"] == 0
    expected = np.pad(x.numpy(), ((0, 0), (0, 0), (112, 112), (112, 112)), mode="reflect")
    report["large_reflect_max_error"] = float(np.max(np.abs(reflect_halo(x, 112).numpy()-expected)))
    assert report["large_reflect_max_error"] == 0
    omega = np.linspace(0, np.pi/3, 101)
    assert np.array_equal(camera(omega), camera(omega, anchor=2))

    # Every RR grid is centered. Compose its affine coordinate maps through
    # arbitrary intermediate integer sizes, keeping the exact requested scale.
    grid_error = 0.0
    for m in range(1, 6):
        ratio = 3**(1/m)
        sizes = [171] + [int(np.ceil(171*3**(j/m)-1e-9)) for j in range(1, m)] + [512]
        coordinates = np.arange(512, dtype=np.float64)
        for j in range(m, 0, -1):
            coordinates = coordinates/ratio + (sizes[j-1]-1)/2 - (sizes[j]-1)/(2*ratio)
        direct = np.arange(512)/3 + (171-1)/2 - (512-1)/6
        grid_error = max(grid_error, float(np.max(np.abs(coordinates-direct))))
    report["x3_nondivisible_grid_composition_error"] = grid_error
    assert grid_error < 1e-10

    x = torch.rand(1, 3, 249, 253, device="cuda")
    kernel = ContinuousKernel(3**(1/5), radius=96, anchor=3)
    cpu_plan = CachedResize(x.shape[-2:], 3**(-1/5), method=kernel, antialiasing=False)
    report["CPU_weights_vs_GPU_RR_max_error"] = float((cpu_plan(x)-cpu_plan.direct(x)).abs().max())
    plan = CachedResize(x.shape[-2:], 3**(-1/5), method=kernel, antialiasing=False,
                        weight_device="cuda")
    report["x3_fractional_GPU_RR_max_error"] = float((plan(x)-plan.direct(x)).abs().max())
    assert report["x3_fractional_GPU_RR_max_error"] < 2e-5, report

    # A zero-initialized residual model is exactly the shared bicubic path.
    model = PlainCNN(width=8, depth=2, residual=True).cuda().eval()
    x = torch.rand(1, 3, 43, 47, device="cuda")
    args = SimpleNamespace(scale=3, device="cuda", depth=2)
    target = (128, 140)
    for m in (1, 5):
        prediction, sizes = infer(model, x, m, args, target_shape=target)
        assert tuple(prediction.shape[-2:]) == target
        expected = x
        for size in sizes:
            expected = CachedResize(expected.shape[-2:], 3**(1/m), size, weight_device="cuda")(expected)
        error = float((prediction-expected).abs().max())
        report[f"initial_residual_m{m}_vs_bicubic_chain_error"] = error
        assert error == 0
    path = Path(__file__).parent / "scale_transfer_checks.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
