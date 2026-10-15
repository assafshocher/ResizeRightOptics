"""Numerical gates run before any learning or HR-based scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kernels import CachedResize, ContinuousKernel, camera, cubic_spectrum, band_window, interp_methods, rr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--max-m", type=int, default=5)
    args = parser.parse_args()
    scale, radius = args.scale, args.radius
    torch.set_num_threads(4)
    torch.manual_seed(0)
    report = {"scale": scale, "radius": radius, "max_m": args.max_m, "uses_original_hr": False}
    w = np.linspace(0, np.pi / scale, 2049)
    report["anchor_telescope_max_error"] = float(np.max(np.abs(camera(scale*w, scale)/camera(w, scale) - cubic_spectrum(scale*w)*band_window(scale*w))))
    assert report["anchor_telescope_max_error"] < 1e-6
    ours = ContinuousKernel(scale, radius=radius, anchor=scale)
    aa = ContinuousKernel(scale, radius=radius, law="aa_bandlimited", anchor=scale)
    report["anchor_lut_max_error"] = float(np.max(np.abs(ours.values - aa.values)))
    assert report["anchor_lut_max_error"] == 0
    checks = []
    # Nondivisible, nonsquare shapes as well as the anchor; all phases exercised.
    for ratio, shape in [(1/scale, (249, 253)), (scale**(-1/3), (249, 253)), (scale**(1/3), (249, 253))]:
        for method, antialiasing, label in [(interp_methods.cubic, True, "aa"),
                                           (interp_methods.cubic, False, "noaa"),
                                           (ours, False, "lut")]:
            x = torch.rand(1, 3, *shape)
            plan = CachedResize(shape, ratio, method=method, antialiasing=antialiasing, device="cpu")
            err = (plan(x) - plan.direct(x)).abs().max().item()
            dc = (plan(torch.ones_like(x)) - 1).abs().max().item()
            assert err < 3e-6, (ratio, label, err)
            assert dc < 3e-6, (ratio, label, dc)
            checks.append({"scale": ratio, "shape": shape, "kernel": label, "cache_max_error": err, "constant_max_error": dc})
    report["rr_checks"] = checks
    # Support sensitivity is measured on the observed LR, not used to tune PSNR.
    data = Path(__file__).parent / ("data" if scale == 2 else f"data_x{scale}")
    source = np.load(data / "baby_lr_bandlimited.npy")
    convergence = []
    for m in range(1, args.max_m+1):
        factor = scale**(1/m)
        a = ContinuousKernel(factor, radius=radius, anchor=scale)
        b = ContinuousKernel(factor, radius=2*radius, anchor=scale)
        # RR's NumPy backend supports reflection beyond one image width,
        # needed for the doubled-support reference on small observed LRs.
        low = rr.resize(source, scale_factors=[1/factor]*2, interp_method=a, antialiasing=False, pad_mode="reflect")
        high = rr.resize(source, scale_factors=[1/factor]*2, interp_method=b, antialiasing=False, pad_mode="reflect")
        diff = (low - high)[8:-8, 8:-8]
        convergence.append({"m": m, "factor": factor, "radius_vs_double_rmse": float(np.mean(diff**2)**0.5),
                            "radius_vs_double_max": float(np.max(np.abs(diff)))})
    report["support_sensitivity"] = convergence
    original_lr = np.load(data / "baby_lr.npy")
    matched_lr = np.load(data / "baby_lr_bandlimited.npy")
    report["literal_vs_matched_baby_rmse"] = float(np.mean((original_lr-matched_lr)**2)**0.5)
    report["support_gate_passed"] = all(item["radius_vs_double_rmse"] < 2e-5 for item in convergence)
    report["notes"] = "Finite support and RR per-phase normalization approximate the canonical law; no exact finite-image composition claim. Training uses matched input unless explicitly stated."
    (Path(__file__).parent / f"audit_x{scale}_radius{radius}_m{args.max_m}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    assert report["support_gate_passed"], convergence


if __name__ == "__main__":
    main()
