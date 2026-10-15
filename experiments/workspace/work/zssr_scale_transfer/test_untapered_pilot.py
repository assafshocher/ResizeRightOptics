"""Pre-training diagnostics for the direct, untapered x2 camera family.

Uses only the observed baby LR image and synthetic waves, never HR images or
PSNR. The fixed radius-64 kernel is not changed by this diagnostic. Radius 128
is a convergence probe, not an automatically selected replacement. Exact
identities are tested in the continuous retained band; finite sampled RR
responses are reported separately and are not claimed to be machine exact.
"""
from __future__ import annotations

import json
from pathlib import Path
import traceback

import numpy as np
import torch

from experiment import degradation, save_json
from kernels import ContinuousKernel, cubic_spectrum, interp_methods, rr
from untapered_kernels import (
    UntaperedKernel, untapered_camera, untapered_multiplier,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "untapered_pilot_checks.json"
ANCHOR = 2
RADIUS = 64
PHASES = np.linspace(0., 1., 9)
FREQUENCY_FRACTIONS = (.1, .25, .6, .9)


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def finite(*arrays):
    return all(bool(np.isfinite(a).all()) for a in arrays)


def sampled_response(kernel, omega, phase, radius=RADIUS):
    """Same finite taps and DC normalization used by separable RR, away from edges."""
    offsets = phase - np.arange(-radius, radius + 1)
    weights = np.asarray(kernel(offsets), dtype=np.float64)
    dc = float(weights.sum())
    if not finite(weights, dc) or dc <= 0:
        return dict(dc_before_normalization=dc, finite=False)
    weights = weights / dc
    gain = np.sum(weights * np.exp(-1j * omega * offsets))
    return dict(dc_before_normalization=dc,
                gain_real=float(gain.real), gain_imag=float(gain.imag),
                gain_magnitude=float(abs(gain)), finite=finite(gain))


def pair_norms(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    inner = diff[8:-8, 8:-8]
    if inner.size == 0:
        raise ValueError("Observed image is too small for the fixed interior diagnostic")
    return dict(finite=finite(a, b),
                all_rmse=float(np.sqrt(np.mean(diff**2))),
                all_max_abs=float(np.max(np.abs(diff))),
                interior_rmse=float(np.sqrt(np.mean(inner**2))),
                interior_max_abs=float(np.max(np.abs(inner))))


def resize_observed(observed, kernel, factor, aa=False):
    return rr.resize(observed, scale_factors=[1/factor]*2,
                     interp_method=kernel, antialiasing=aa, pad_mode="reflect")


def main():
    torch.set_num_threads(4)
    report = dict(
        uses_hr=False, uses_psnr=False, anchor=ANCHOR,
        observed_image="data/baby_lr.npy", radius=RADIUS,
        kernel="Untapered product/quotient, hard retained-band boundary; no smooth window or ridge",
        kernel_modified_by_test=False,
        thresholds=dict(continuous_anchor_max_error=1e-10,
                        continuous_telescope_max_error=1e-10,
                        sampled_complex_response_max_error=.1,
                        near_identity_gain_deviation=.01,
                        raw_sampled_dc_must_be_positive=True),
        support_probe="Radius 128 versus fixed radius 64, reported without a pass threshold",
        checks=[], failures=[], passed=False)

    def gate(condition, name, **details):
        if not condition:
            report["failures"].append(dict(check=name, **details))

    try:
        observed = np.load(ROOT / "data" / "baby_lr.npy")
        if observed.ndim != 3:
            raise ValueError(("Expected HWC observed LR image", observed.shape))
        gate(finite(observed), "observed_image_finite")
        report["observed_shape"] = list(observed.shape)
        report["observed_dtype"] = str(observed.dtype)

        # Within the continuous band, the product must recover the agreed
        # bicubic multiplier. Truncating the camera product at 18 factors
        # introduces only its explicitly measured last-factor discrepancy.
        omega_anchor = np.linspace(0, np.pi/ANCHOR, 4097)
        expected_anchor = cubic_spectrum(ANCHOR*omega_anchor)
        measured_anchor = untapered_multiplier(omega_anchor, ANCHOR, anchor=ANCHOR)
        anchor_error = float(np.max(np.abs(measured_anchor-expected_anchor)))
        report["continuous_anchor_max_error"] = anchor_error
        gate(finite(measured_anchor, expected_anchor) and anchor_error < 1e-10,
             "continuous_anchor", max_error=anchor_error)
        emit("untapered_anchor_checked", max_error=anchor_error)

        for m in range(1, 11):
            factor = ANCHOR**(1/m)
            kernel, aa = degradation("consistent_untapered", factor, RADIUS, anchor=ANCHOR)
            if not isinstance(kernel, UntaperedKernel) or aa:
                raise AssertionError("Training must use the untapered kernel without extra AA")
            gate(finite(kernel.values), "kernel_lut_finite", m=m)

            # The m-stage Fourier composition evaluates each multiplier in its
            # own pixel coordinates: K_r(w) K_r(rw) ... K_r(r^(m-1)w).
            # Keep just below the endpoint to avoid an r**m rounding excursion
            # across the hard support while still testing its one-sided limit.
            omega = np.linspace(0, (np.pi/ANCHOR)*(1-1e-12), 4097)
            chain = np.ones_like(omega)
            for stage in range(m):
                chain *= untapered_multiplier(
                    omega*factor**stage, factor, anchor=ANCHOR)
            direct = untapered_multiplier(omega, ANCHOR, anchor=ANCHOR)
            telescope_error = float(np.max(np.abs(chain-direct)))
            gate(finite(chain, direct) and telescope_error < 1e-10,
                 "continuous_telescope", m=m, max_error=telescope_error)

            responses = []
            for fraction in FREQUENCY_FRACTIONS:
                frequency = fraction*np.pi/factor
                expected = float(untapered_multiplier(
                    np.asarray([frequency]), factor, anchor=ANCHOR)[0])
                for phase in PHASES:
                    response = sampled_response(kernel, frequency, float(phase))
                    response.update(frequency_fraction_of_output_nyquist=fraction,
                                    input_omega=float(frequency), phase=float(phase),
                                    expected_continuous_gain=expected)
                    gate(response["finite"], "sampled_response_finite_positive_dc",
                         m=m, phase=float(phase), fraction=fraction)
                    if response["finite"]:
                        gain = response["gain_real"] + 1j*response["gain_imag"]
                        response["complex_response_error"] = float(abs(gain-expected))
                    responses.append(response)
            errors = [r["complex_response_error"] for r in responses if r["finite"]]
            max_error = max(errors) if errors else None
            gate(max_error is not None and max_error < .1,
                 "sampled_response_not_catastrophic", m=m, max_error=max_error)
            entry = dict(m=m, factor=factor, continuous_telescope_max_error=telescope_error,
                         sampled_response_max_error=max_error,
                         dc_before_normalization_range=[
                             min(r["dc_before_normalization"] for r in responses),
                             max(r["dc_before_normalization"] for r in responses)],
                         sampled_responses=responses)

            if m in (1, 2, 5, 10):
                longer = UntaperedKernel(factor, radius=2*RADIUS, anchor=ANCHOR)
                gate(finite(longer.values), "support_probe_lut_finite", m=m)
                lo = resize_observed(observed, kernel, factor)
                hi = resize_observed(observed, longer, factor)
                entry["radius64_vs128_observed"] = pair_norms(lo, hi)
                gate(entry["radius64_vs128_observed"]["finite"],
                     "support_probe_output_finite", m=m)

            if m == 1:
                ours = resize_observed(observed, kernel, factor)
                ordinary = resize_observed(observed, interp_methods.cubic, factor, aa=True)
                tapered_kernel = ContinuousKernel(factor, radius=RADIUS, anchor=ANCHOR)
                tapered = resize_observed(observed, tapered_kernel, factor)
                report["anchor_observed_same_rr_grid"] = dict(
                    untapered_vs_ordinary_bicubic_aa=pair_norms(ours, ordinary),
                    tapered_vs_ordinary_bicubic_aa=pair_norms(tapered, ordinary),
                    untapered_vs_tapered=pair_norms(ours, tapered))
                gate(finite(ours, ordinary, tapered), "anchor_comparison_finite")
                # Report the same phase/frequency responses for all three
                # kernels; AA is precisely the usual stretched cubic kernel.
                comparisons = []
                aa_kernel = lambda x: interp_methods.cubic(x/factor)/factor
                for fraction in FREQUENCY_FRACTIONS:
                    frequency = fraction*np.pi/factor
                    for phase in PHASES:
                        comparisons.append(dict(
                            frequency_fraction_of_output_nyquist=fraction,
                            phase=float(phase),
                            untapered=sampled_response(kernel, frequency, float(phase)),
                            ordinary_bicubic_aa=sampled_response(aa_kernel, frequency, float(phase)),
                            tapered=sampled_response(tapered_kernel, frequency, float(phase))))
                report["anchor_sampled_comparison"] = comparisons
                emit("untapered_anchor_observed_comparison",
                     **report["anchor_observed_same_rr_grid"])

            report["checks"].append(entry)
            emit("untapered_step_checked", m=m, factor=factor,
                 continuous_telescope_max_error=telescope_error,
                 sampled_response_max_error=max_error,
                 support_probe=entry.get("radius64_vs128_observed"))
            save_json(OUTPUT, report)

        # Exercise the real RR call close to one, not its exact-one bypass.
        n, factor = 1025, 1.0001
        x = np.arange(n)-(n-1)/2
        wave = np.cos(np.pi*x/4)[None, :]
        kernel = UntaperedKernel(factor, radius=RADIUS, anchor=ANCHOR)
        result = rr.resize(wave, scale_factors=[1, 1/factor], out_shape=[1, n],
                           interp_method=kernel, antialiasing=False, pad_mode="reflect")
        gain = float(result[0, n//2])
        expected_gain = float(untapered_multiplier(
            np.asarray([np.pi/4]), factor, anchor=ANCHOR)[0])
        report["near_identity"] = dict(factor=factor, input_omega=float(np.pi/4),
                                       actual_rr_center_gain=gain,
                                       expected_continuous_gain=expected_gain,
                                       deviation_from_one=abs(gain-1),
                                       finite=finite(kernel.values, result))
        gate(report["near_identity"]["finite"] and abs(gain-1) < .01,
             "near_identity", gain=gain)
        emit("untapered_near_identity_checked", **report["near_identity"])

        # Check the non-windowed camera itself through the retained-band edge.
        camera_grid = untapered_camera(np.linspace(0, np.pi, 4097), anchor=ANCHOR)
        gate(finite(camera_grid) and bool(np.all(camera_grid > 0)),
             "untapered_camera_finite_positive")
        report["camera_at_dc_and_band_edge"] = [float(camera_grid[0]), float(camera_grid[-1])]
        report["passed"] = not report["failures"]
        save_json(OUTPUT, report)
        emit("untapered_preflight_complete", passed=report["passed"],
             failures=report["failures"], output=str(OUTPUT))
        if not report["passed"]:
            raise AssertionError(report["failures"])
    except Exception as error:
        report["passed"] = False
        report["exception"] = dict(type=type(error).__name__, message=str(error),
                                   traceback=traceback.format_exc())
        save_json(OUTPUT, report)
        raise


if __name__ == "__main__":
    main()
