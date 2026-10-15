"""Pre-score sampled-operator gates for the unchanged retained-band kernels."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import degradation, save_json
from kernels import ContinuousKernel, band_window, camera, cubic_spectrum, rr

ROOT = Path(__file__).resolve().parent


def legacy_window(omega):
    """Pre-fix evaluation, retained only for numerical regression checks."""
    t = np.clip((np.abs(omega)/np.pi - .8)/.2, 0, 1)
    return np.clip(1-(126*t**5-420*t**6+540*t**7-315*t**8+70*t**9), 0, 1)


def legacy_camera(omega, anchor):
    result = np.ones_like(omega, dtype=np.float64)
    for j in range(18):
        result *= cubic_spectrum(omega / anchor**j)
    return result * legacy_window(omega)


def main():
    torch.set_num_threads(4)
    report = dict(uses_hr=False, new_kernel=False,
        numerical_fix="Same C4 window, symmetry-stable polynomial evaluation", checks=[])
    grid = np.linspace(0, np.pi, 10001)
    window_error = float(np.max(np.abs(band_window(grid)-legacy_window(grid))))
    assert window_error < 1e-12, window_error
    assert band_window(np.array([np.pi/1.0001]))[0] > 0
    assert np.array_equal(band_window(np.array([0., .8*np.pi, np.pi, 2*np.pi])),
                          np.array([1., 1., 0., 0.]))
    report["window_legacy_max_abs_error"] = window_error
    for scale in (2, 3, 4):
        data = ROOT / ("data" if scale == 2 else f"data_x{scale}")
        observed = np.load(data / "baby_lr.npy")
        for m in range(1, 11):
            factor = scale ** (1/m)
            radius = 32 * scale
            a, aa = degradation("consistent", factor, radius, anchor=scale)
            assert isinstance(a, ContinuousKernel) and not aa
            b = ContinuousKernel(factor, radius=2*radius, anchor=scale)
            omega_grid = np.linspace(0, np.pi/factor, 4097)
            if m == 1:
                legacy = cubic_spectrum(factor*omega_grid)*legacy_window(factor*omega_grid)
                stable = cubic_spectrum(factor*omega_grid)*band_window(factor*omega_grid)
            else:
                legacy = legacy_camera(factor*omega_grid, scale)/legacy_camera(omega_grid, scale)
                stable = camera(factor*omega_grid, scale)/camera(omega_grid, scale)
            spectrum_error = float(np.max(np.abs(stable-legacy)))
            assert np.isfinite(a.values).all() and np.isfinite(b.values).all()
            assert spectrum_error < 1e-10, (scale, m, spectrum_error)
            errors = []
            for omega in np.array([.1, .25, .6, .85]) * np.pi / factor:
                expected = float(camera(np.array([factor*omega]), scale)[0]
                                 / camera(np.array([omega]), scale)[0])
                for phase in np.linspace(0, 1, 9):
                    offsets = phase - np.arange(-radius, radius+1)
                    weights = a(offsets)
                    weights /= weights.sum()
                    gain = np.sum(weights * np.exp(-1j*omega*offsets))
                    errors.append(float(abs(gain-expected)))
            assert max(errors) < .001, (scale, m, max(errors))
            lo = rr.resize(observed, scale_factors=[1/factor]*2,
                interp_method=a, antialiasing=False, pad_mode="reflect")
            hi = rr.resize(observed, scale_factors=[1/factor]*2,
                interp_method=b, antialiasing=False, pad_mode="reflect")
            rmse = float(np.sqrt(np.mean((lo[8:-8,8:-8]-hi[8:-8,8:-8])**2)))
            assert rmse < 2e-5, (scale, m, rmse)
            report["checks"].append(dict(scale=scale, m=m,
                phase_response_max_error=max(errors), support_double_rmse=rmse,
                spectrum_legacy_max_abs_error=spectrum_error))
            print(json.dumps(dict(event="preflight_step_passed",
                                  **report["checks"][-1])), flush=True)
        # Near-unit actual RR call: do not rely on the exact-unit bypass.
        n, factor = 1025, 1.0001
        x = np.arange(n)-(n-1)/2
        wave = np.cos(np.pi*x/4)[None,:]
        kernel = ContinuousKernel(factor, radius=32*scale, anchor=scale)
        result = rr.resize(wave, scale_factors=[1,1/factor], out_shape=[1,n],
            interp_method=kernel, antialiasing=False, pad_mode="reflect")
        gain = float(result[0,n//2])
        assert abs(gain-1) < .002, (scale, gain)
        report.setdefault("near_identity", []).append(dict(scale=scale, gain=gain))
        print(json.dumps(dict(event="near_identity_passed", scale=scale, gain=gain)), flush=True)
    report["passed"] = True
    save_json(ROOT / "literal_projected_checks.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
