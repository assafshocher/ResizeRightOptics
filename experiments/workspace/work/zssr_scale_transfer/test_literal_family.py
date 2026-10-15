"""Pre-score numerical gates for the literal-anchor regularized extension."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiment import degradation, save_json, train
from kernels import CachedResize, interp_methods
from literal_spectrum import anchored_ridge_spectrum, full_camera, full_cubic_spectrum
from matched_kernels import LiteralAnchorKernel


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2309)
    report = {"uses_hr_pixels": False, "eta_fixed_before_scores": .001, "checks": []}
    # Independent spatial quadrature checks the closed-form Fourier transform.
    x = np.linspace(0, 2, 16385)
    w = np.r_[0., .001, .04, .079, .08, .1, np.linspace(.2, 16*np.pi, 111)]
    u_numeric = 2*np.trapezoid(np.cos(w[:, None]*x)*interp_methods.cubic(x), x, axis=1)
    err = float(np.max(np.abs(full_cubic_spectrum(w)-u_numeric)))
    assert err < 2e-9, err
    report["analytic_transform_max_error"] = err
    for scale in (2, 3, 4):
        r0 = 32*scale
        for m in range(1, 11):
            factor = scale**(1/m)
            kernel, aa = degradation("consistent_literal", factor, r0, anchor=scale)
            if m == 1:
                assert kernel is interp_methods.cubic and aa is True
                # Real RR weights, all fractional phases induced by a nonmultiple
                # length, coincide bitwise, not only at integer kernel taps.
                shape = (191, 197)
                a = CachedResize(shape, 1/scale, method=kernel, antialiasing=aa,
                                 device="cpu", weight_device="cpu")
                b = CachedResize(shape, 1/scale, device="cpu", weight_device="cpu")
                assert all(torch.equal(v, z) for v, z in zip(a.axes, b.axes))
                report["checks"].append(dict(scale=scale, m=m, exact_anchor_weights=True))
                continue
            assert kernel.diagnostics["spatial_correction_l1_tail"] < 2e-5
            assert kernel.diagnostics["correction_at_frequency_boundary"] < 1e-15
            # Twice the radius and Fourier period: independent truncation check.
            large = LiteralAnchorKernel(factor, 2*r0, scale, length=1 << 21, omega_max=64*np.pi)
            assert np.max(np.abs(kernel.values-large.values[:len(kernel.values)])) < 1e-8
            # Raw phase coefficients: no GT or learned score enters this test.
            phases = np.linspace(0, 1, 21)
            taps = np.arange(-2*r0+1, 2*r0+1)
            offset = phases[:, None]-taps[None, :]
            a, b = kernel(offset), large(offset)
            phase_sum = a.sum(1)
            assert np.min(np.abs(phase_sum)) > .1, (scale, m, phase_sum)
            a /= phase_sum[:, None]
            b /= b.sum(1)[:, None]
            max_l1 = float(np.max(np.sum(np.abs(a-b), axis=1)))
            assert max_l1 < 5e-5, (scale, m, max_l1)
            # Ridge normal equation and its agreement with the exact quotient
            # on the input Nyquist band. Aliased/out-of-band content is retained.
            omega = np.linspace(0, 8*np.pi, 8193)
            u, corr = anchored_ridge_spectrum(omega, factor, scale)
            h, target = full_camera(omega, scale), full_camera(factor*omega, scale)
            lhs = (h*h+kernel.ETA**2)*(u+corr)
            rhs = h*target+kernel.ETA**2*u
            assert np.max(np.abs(lhs-rhs)) < 1e-13
            band = omega <= np.pi
            inband = float(np.max(np.abs((u+corr)[band]-target[band]/h[band])))
            assert inband < 3e-6, (scale, m, inband)
            outside = (omega > np.pi/factor) & (omega < np.pi)
            assert np.max(np.abs((u+corr)[outside])) > 1e-5
            row = dict(scale=scale, m=m, phase_sum_min=float(phase_sum.min()),
                       phase_sum_max=float(phase_sum.max()), double_support_l1=max_l1,
                       inband_ridge_bias=inband, **kernel.diagnostics)
            report["checks"].append(row)
    # Exercise optimization with the unchanged online crop protocol, synthetic
    # observations only. All three arms must share the random crop schedule.
    hashes = []
    image = torch.rand(1, 3, 86, 57)
    for method in ("consistent_literal", "aa", "aa_grid_literal"):
        args = SimpleNamespace(scale=4, seed=0, width=8, depth=2, residual=True,
            device="cpu", lr=1e-4, crop=64, crop_margin=16, radius=128,
            steps=2, batch=1, image="synthetic")
        _, info = train(image, method, 10, args)
        hashes.append(info["coordinate_sha256"])
        assert info["core_size"] == 41
    assert len(set(hashes)) == 1
    report.update(passed=True, coordinate_sha256=hashes[0])
    save_json(Path(__file__).parent / "literal_family_checks.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
