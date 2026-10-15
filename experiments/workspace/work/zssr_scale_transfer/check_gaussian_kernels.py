"""Synthetic CPU diagnostics: true unit gates and separately reported errors.

NumPy-only mode prints JSON and skips CachedResize checks. Full mode saves
with experiment.save_json. No images, training, PSNR, or kernel tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from gaussian_kernels import METHODS, TruncatedGaussian, gaussian_kernel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "ResizeRight"))
import resize_right as rr

TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def spectrum(omega, sigma):
    z = np.linspace(-4, 4, 4097)
    density = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return TRAPZ(np.cos(np.asarray(omega)[..., None] * sigma * z) * density, z, axis=-1) / TRAPZ(density, z)


def phase_report(kernel):
    phases = np.linspace(0, 1, 65)
    radius = math.ceil(kernel.radius) + 1
    offsets = phases[:, None] - np.arange(-radius, radius + 1)
    weights = kernel(offsets)
    sums = weights.sum(axis=1)
    normalized = np.divide(weights, sums[:, None], out=np.zeros_like(weights), where=sums[:, None] > 0)
    frequencies = np.pi * np.array([0.1, 0.4, 0.7, 0.95])
    gains = np.sum(normalized[None] * np.exp(-1j * frequencies[:, None, None] * offsets), axis=-1)
    expected = np.exp(-0.5 * (kernel.sigma * frequencies) ** 2)
    return dict(phases=phases.tolist(), raw_partition_sums=sums.tolist(),
                raw_partition_min=float(sums.min()), raw_partition_max=float(sums.max()),
                raw_partition_range=float(np.ptp(sums)), uncovered_phases=int(np.sum(sums == 0)),
                finite=bool(np.isfinite(weights).all() and np.isfinite(gains).all()),
                input_frequencies=frequencies.tolist(),
                max_phase_response_error_vs_ideal=np.max(np.abs(gains - expected[:, None]), axis=1).tolist())


def resize(image, factor, shape, kernel):
    return rr.resize(image, scale_factors=1 / factor, out_shape=shape,
                     interp_method=kernel, antialiasing=False, pad_mode="reflect")


def errors(actual, expected):
    delta = np.asarray(actual) - np.asarray(expected)
    return dict(max_abs=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta ** 2))))


def near_identity_rr_probe(kernel, factor, target):
    try:
        _, x = np.indices((65, 65))
        image = np.stack([np.ones((65, 65)), np.cos(np.pi * x / 2)], axis=-1)
        result = resize(image, factor, target, kernel)
        probe = dict(finite=bool(np.isfinite(result).all()), constant_max_error=errors(result[..., 0], 1)["max_abs"])
        if target == (65, 65):
            probe["sampled_wave_identity_max_error"] = errors(result[..., 1], image[..., 1])["max_abs"]
        return probe
    except ValueError as error:
        # NumPy RR cannot crop through negative padding. Keep this visible;
        # it is a backend limitation on this advisory grid, not a kernel fix.
        return dict(backend_error=type(error).__name__, message=str(error))


def run_checks(anchor=2, base_sigmas=(0.5, 1.0), m_max=10, numpy_only=False):
    if m_max < 1:
        raise ValueError("m_max must be positive")
    if not numpy_only:
        import torch
        from kernels import CachedResize
        torch.set_num_threads(4)
    report = dict(anchor=anchor, base_sigmas=list(base_sigmas), m_max=m_max,
                  numpy_only=numpy_only, cached_resize_checked=not numpy_only,
                  truncate_sigma=4, discarded_continuous_mass=math.erfc(4 / math.sqrt(2)),
                  continuum_quadrature_points=4097, continuum_frequency_range=[0, float(np.pi)],
                  synthetic_waves=["cos(0.2*pi*x)", "cos(0.8*pi*x)", "sin(0.37*pi*y+0.3)"],
                  uses_hr=False, uses_psnr=False, kernel_modified_by_test=False,
                  tolerances=dict(float64_identity=1e-12, float32_cached_and_constant=2e-5),
                  measurement_note="Composition and sampled response errors have no pass threshold. Near-one uncovered phases are reported limitations, outside the pilot sweep.",
                  cases=[], near_identity=[], failures=[])

    def gate(condition, name, **context):
        if not condition:
            report["failures"].append(dict(check=name, **context))

    invalid = [lambda: TruncatedGaussian(0), lambda: TruncatedGaussian(float("nan")),
               lambda: TruncatedGaussian(1, 0), lambda: gaussian_kernel("bad", 2),
               lambda: gaussian_kernel(METHODS[0], 1), lambda: gaussian_kernel(METHODS[0], 0),
               lambda: gaussian_kernel(METHODS[0], float("inf")),
               lambda: gaussian_kernel(METHODS[0], 2, anchor=1),
               lambda: gaussian_kernel(METHODS[0], 2, base_sigma=-1)]
    for index, operation in enumerate(invalid):
        try:
            operation()
            gate(False, "invalid_parameter_accepted", index=index)
        except ValueError:
            pass
    omega = np.linspace(0, np.pi, 129)
    for base in base_sigmas:
        anchor_kernel = gaussian_kernel(METHODS[0], anchor, anchor, base)
        direct_ideal = np.exp(-0.5 * (anchor_kernel.sigma * omega) ** 2)
        direct_truncated = spectrum(omega, anchor_kernel.sigma)
        for m in range(1, m_max + 1):
            factor = anchor ** (1 / m)
            for method in METHODS:
                kernel = gaussian_kernel(method, factor, anchor, base)
                context = dict(base_sigma=base, m=m, method=method)
                phase = phase_report(kernel)
                gate(phase["finite"] and phase["uncovered_phases"] == 0, "pilot_phase_coverage", **context)
                ideal_chain = np.exp(-0.5 * (kernel.sigma * omega) ** 2 * sum(factor ** (2 * j) for j in range(m)))
                truncated_chain = np.prod([spectrum(omega * factor ** j, kernel.sigma) for j in range(m)], axis=0)
                ideal_error = errors(ideal_chain, direct_ideal)["max_abs"]
                if method == METHODS[0]:
                    gate(ideal_error < 1e-12, "analytical_telescope", **context)
                entry = dict(**context, factor=factor, sigma=kernel.sigma, support_sz=kernel.support_sz,
                             phase_sampling=phase, ideal_continuum_chain_vs_anchor=ideal_error,
                             normalized_truncated_continuum_chain_vs_anchor=errors(truncated_chain, direct_truncated),
                             truncation_effect_on_chain=errors(truncated_chain, ideal_chain), grids=[])
                shapes = [(65, 67), (64, 70)] + ([(512, 512), (220, 220)] if m == 1 else [])
                for shape in shapes:
                    y, x = np.indices(shape)
                    image = np.stack([np.ones(shape), np.cos(0.2 * np.pi * x),
                                      np.cos(0.8 * np.pi * x), np.sin(0.37 * np.pi * y + 0.3)], axis=-1)
                    natural = tuple(math.ceil(n / factor) for n in shape)
                    for target in [natural, (natural[0] - 1, natural[1] + 1)]:
                        result = resize(image, factor, target, kernel)
                        gate(np.isfinite(result).all(), "rr_samples_finite", **context)
                        constant_error = float(np.max(np.abs(result[..., 0] - 1)))
                        gate(constant_error < 1e-12, "numpy_rr_constant", **context)
                        grid = dict(input_shape=shape, output_shape=target, constant_max_error=constant_error)
                        if m == 1:
                            reference = resize(image, anchor, target, anchor_kernel)
                            grid["anchor_operator_max_error"] = errors(result, reference)["max_abs"]
                            gate(np.array_equal(result, reference), "identical_anchor_operator", **context)
                        if not numpy_only:
                            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None]
                            plan = CachedResize(shape, 1 / factor, target, kernel, False, device="cpu")
                            cached, direct = plan(tensor), plan.direct(tensor)
                            grid["cached_vs_direct_max_error"] = (cached - direct).abs().max().item()
                            grid["torch_constant_max_error"] = (cached[:, 0] - 1).abs().max().item()
                            gate(all(torch.isfinite(w).all().item() for w in plan.axes) and torch.isfinite(direct).all().item(), "torch_weights_samples_finite", **context)
                            gate(grid["cached_vs_direct_max_error"] < 2e-5 and grid["torch_constant_max_error"] < 2e-5, "cached_rr_and_constant", **context)
                            if m == 1:
                                reference_plan = CachedResize(shape, 1 / anchor, target, anchor_kernel, False, device="cpu")
                                grid["anchor_weight_maps_identical"] = all(torch.equal(a, b) for a, b in zip(plan.axes, reference_plan.axes))
                                gate(grid["anchor_weight_maps_identical"], "identical_anchor_cached_maps", **context)
                        entry["grids"].append(grid)
                    target = tuple(math.ceil(n / anchor) for n in shape)
                    chain = image
                    for j in range(m):
                        stage_shape = target if j == m - 1 else tuple(math.ceil(n / factor ** (j + 1)) for n in shape)
                        chain = resize(chain, factor, stage_shape, kernel)
                    direct = resize(image, anchor, target, anchor_kernel)
                    gate(np.isfinite(chain).all(), "chain_samples_finite", **context)
                    gate(np.max(np.abs(chain[..., 0] - 1)) < 1e-12, "chain_constant", **context)
                    entry.setdefault("sampled_chain_vs_direct", []).append(dict(input_shape=shape,
                        output_shape=target, all_pixels=errors(chain[..., 1:], direct[..., 1:]),
                        interior=errors(chain[4:-4, 4:-4, 1:], direct[4:-4, 4:-4, 1:]),
                        per_wave_interior=[errors(chain[4:-4, 4:-4, i], direct[4:-4, 4:-4, i]) for i in range(1, 4)]))
                report["cases"].append(entry)
        for method in METHODS:
            for epsilon in [1e-2, 1e-3, 1e-4, 1e-6]:
                factor = 1 + epsilon
                kernel = gaussian_kernel(method, factor, anchor, base)
                phase = phase_report(kernel)
                aligned = near_identity_rr_probe(kernel, factor, (65, 65))
                off_grid = near_identity_rr_probe(kernel, factor, (64, 64))
                gate(phase["finite"] and aligned.get("finite", True) and off_grid.get("finite", True), "near_identity_finite")
                report["near_identity"].append(dict(base_sigma=base, method=method, factor=factor,
                    sigma=kernel.sigma, ideal_gain_at_pi_over_2=math.exp(-0.5 * (kernel.sigma * np.pi / 2) ** 2),
                    uncovered_phases=phase["uncovered_phases"], phase_count=len(phase["phases"]),
                    aligned_grid=aligned, off_grid=off_grid))
    report["passed"] = not report["failures"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy-only", action="store_true", help="Print JSON; skip PyTorch and saving")
    parser.add_argument("--output", type=Path, default=ROOT / "gaussian_kernel_checks.json")
    parser.add_argument("--anchor", type=float, default=2)
    parser.add_argument("--base-sigmas", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--m-max", type=int, default=10)
    args = parser.parse_args()
    report = run_checks(args.anchor, args.base_sigmas, args.m_max, args.numpy_only)
    if args.numpy_only:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        from experiment import save_json
        save_json(args.output, report)
        print(json.dumps(dict(passed=report["passed"], failures=report["failures"], output=str(args.output))))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
