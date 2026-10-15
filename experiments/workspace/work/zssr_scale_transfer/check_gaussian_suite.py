"""CUDA preflight for the fixed sigma=2 Gaussian Set5 scale-transfer suite.

Inspect HR array headers only to reproduce the actual observation/crop grids.
Every numerical input is synthetic; no HR image pixels, PSNR, training, or
parameter selection are used. Run explicitly; importing this module is inert.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


ROOT = Path(__file__).resolve().parent
IMAGES = ("baby", "bird", "butterfly", "head", "woman")
METHODS = ("gaussian_consistent", "gaussian_aa", "aa_grid_gaussian")
ANCHOR_SIGMA = 2.0
SOURCE_FILES = (
    "check_gaussian_suite.py", "check_gaussian_kernels.py",
    "gaussian_kernels.py", "kernels.py", "matched_kernels.py",
    "experiment.py", "vendor/ResizeRight/resize_right.py",
    "vendor/ResizeRight/interp_methods.py",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _finite_json(value):
    """Retain failed checks even when their measured error is nonfinite."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def save_report(path, report):
    """Atomically save even a missing-dependency or failed-CUDA report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(_finite_json(report), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_checks(scales=(2, 3, 4), device="cuda", progress=False):
    scales = tuple(int(scale) for scale in scales)
    if not scales or len(set(scales)) != len(scales) or any(s not in (2, 3, 4) for s in scales):
        raise ValueError("Choose distinct scales from 2, 3, and 4")
    start = time.perf_counter()
    report = dict(
        passed=False, scales=list(scales), anchor_sigma=ANCHOR_SIGMA,
        device=device, images=list(IMAGES), methods=list(METHODS), m=list(range(1, 11)),
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: digest(ROOT / name) for name in SOURCE_FILES},
        data_sha256={}, hr_shapes={}, per_scale={}, cases=[], failures=[],
        cached_resize_checked=False, cuda_checked=False, numpy_only=False,
        uses_hr_pixels=False, uses_hr_shapes=True, uses_psnr=False,
        kernel_modified_by_test=False,
        tolerances=dict(cached_vs_direct=2e-5, constant=2e-5,
                        cuda_vs_weight_backend=2e-5, projected_grid=1e-4),
        synthetic_probes=["1", "cos(0.2*pi*x)", "cos(0.8*pi*x)",
                          "sin(0.37*pi*y+0.3)"],
        measurement_note=("Phase partition, ideal sinusoid response, and continuum "
                          "composition/truncation errors are measurements, not "
                          "composition pass gates. Gaussian support is +/-4 sigma."),
        training_grid=dict(crop=64, crop_margin=16, depth=8,
                           halo="32*S + ceil(2*S**(1/m)) + 8 + 2",
                           rr_output="ceil(n/r)", torch_grid_output="floor(n/r)",
                           rr_grid="j/q + (n-1)/2 - (out-1)/(2*q)",
                           torch_grid="(j+0.5)/q-0.5", q="1/r"),
    )

    def gate(condition, check, **context):
        if not condition:
            report["failures"].append(dict(check=check, **context))

    def finish():
        report["gate_failures"] = report["failures"]
        report["elapsed_seconds"] = time.perf_counter() - start
        report["passed"] = not report["failures"]
        return report

    try:
        import numpy as np
        import torch
        from check_gaussian_kernels import phase_report, spectrum
        from experiment import crop_size
        from gaussian_kernels import gaussian_kernel
        from kernels import CachedResize, rr
        from matched_kernels import MatchedTorchGridResize, grid_convention
    except ImportError as error:
        gate(False, "required_dependency_unavailable", error=str(error))
        return finish()
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        gate(False, "cuda_required", requested_device=device,
             cuda_available=torch.cuda.is_available())
        return finish()

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    report["environment"] = dict(torch=torch.__version__, numpy=np.__version__,
                                  cuda=torch.version.cuda,
                                  gpu=torch.cuda.get_device_name(device))
    original_grid = rr.get_projected_grid
    for image in IMAGES:
        path = ROOT / "data" / f"{image}_hr.npy"
        try:
            header = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(n) for n in header.shape)
            del header
            if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 2:
                raise ValueError(f"Expected an HWC RGB HR array, got {shape}")
            report["hr_shapes"][image] = list(shape[:2])
            report["data_sha256"][image] = digest(path)
        except Exception as error:
            gate(False, "hr_header_unavailable", image=image, error=str(error))
    if report["failures"]:
        return finish()

    def probes(shape, backend):
        y = torch.arange(shape[0], dtype=torch.float32, device=backend)[:, None]
        x = torch.arange(shape[1], dtype=torch.float32, device=backend)[None, :]
        return torch.stack((torch.ones(shape, device=backend),
                            torch.cos(.2 * math.pi * x).expand(shape),
                            torch.cos(.8 * math.pi * x).expand(shape),
                            torch.sin(.37 * math.pi * y + .3).expand(shape)))[None]

    def error(actual, expected):
        delta = actual - expected
        return dict(max_abs=float(delta.abs().max()),
                    rmse=float(delta.square().mean().sqrt()))

    def grid_check(plan, convention, context):
        results = []
        for n, out in zip(plan.shape, plan.out_shape):
            with grid_convention(convention):
                actual = rr.get_projected_grid(n, out, plan.scale, torch, False,
                                               plan.weight_device)
            j = torch.arange(out, dtype=torch.float64, device=plan.weight_device)
            expected = ((j + .5) / plan.scale - .5 if convention == "torch_half_pixel"
                        else j / plan.scale + (n - 1) / 2 - (out - 1) / (2 * plan.scale))
            maximum = float((actual.double() - expected).abs().max())
            gate(maximum < report["tolerances"]["projected_grid"],
                 "projected_grid_formula", **context, input_length=n,
                 output_length=out, maximum_error=maximum)
            results.append(dict(input_length=n, output_length=out,
                                max_abs_error=maximum,
                                first_coordinate=float(actual[0]),
                                last_coordinate=float(actual[-1])))
        gate(rr.get_projected_grid is original_grid, "grid_context_restored", **context)
        return results

    def plan_check(plan, convention, context, gaussian=False):
        # The real plan computes on CUDA. Its CPU-weight x2 counterpart is
        # compared to direct RR on CPU, avoiding a false cross-backend gate.
        backend_plan = copy.copy(plan)
        backend_plan.axes = [axis.to(plan.weight_device) for axis in plan.axes]
        source = probes(plan.shape, plan.weight_device)
        cached = backend_plan(source)
        direct = backend_plan.direct(source)
        gpu_cached = plan(source.to(device))
        finite = all(bool(torch.isfinite(axis).all()) for axis in plan.axes)
        finite = finite and all(bool(torch.isfinite(v).all())
                                for v in (cached, direct, gpu_cached))
        delta = error(cached, direct)
        constant_error = float((cached[:, 0] - 1).abs().max())
        gpu_delta = error(gpu_cached, cached.to(device))
        partition_error = max(float((axis.sum(1) - 1).abs().max()) for axis in plan.axes)
        gate(finite, "finite_cached_weights_and_probes", **context)
        gate(delta["max_abs"] < report["tolerances"]["cached_vs_direct"],
             "cached_vs_direct", **context, **delta)
        gate(max(constant_error, partition_error) < report["tolerances"]["constant"],
             "constant_preservation", **context, constant_max_error=constant_error,
             weight_partition_max_error=partition_error)
        gate(gpu_delta["max_abs"] < report["tolerances"]["cuda_vs_weight_backend"],
             "cuda_application_matches_weight_backend", **context, **gpu_delta)
        gate(tuple(cached.shape[-2:]) == plan.out_shape, "actual_output_shape", **context)
        result = dict(input_shape=list(plan.shape), output_shape=list(plan.out_shape),
                      weight_device=str(plan.weight_device), compute_device=device,
                      finite=finite, cached_vs_direct=delta,
                      per_probe_cached_vs_direct=[error(cached[:, i], direct[:, i]) for i in range(4)],
                      constant_max_error=constant_error,
                      weight_partition_max_error=partition_error,
                      cuda_vs_weight_backend=gpu_delta,
                      grid=grid_check(plan, convention, context))
        if gaussian:
            # Compare infinite, untruncated continuous sinusoid filtering to
            # finite sampled filtering, excluding every reflected boundary.
            coords = []
            for n, out in zip(plan.shape, plan.out_shape):
                j = torch.arange(out, dtype=torch.float64, device=plan.weight_device)
                coords.append((j + .5) / plan.scale - .5 if convention == "torch_half_pixel"
                              else j / plan.scale + (n - 1) / 2 - (out - 1) / (2 * plan.scale))
            yy, xx = coords[0][:, None], coords[1][None, :]
            mask = ((yy >= plan.method.radius + 1) & (yy <= plan.shape[0] - 2 - plan.method.radius)
                    & (xx >= plan.method.radius + 1) & (xx <= plan.shape[1] - 2 - plan.method.radius))
            expected = [torch.cos(.2 * math.pi * xx).expand(plan.out_shape),
                        torch.cos(.8 * math.pi * xx).expand(plan.out_shape),
                        torch.sin(.37 * math.pi * yy + .3).expand(plan.out_shape)]
            frequencies = (.2 * math.pi, .8 * math.pi, .37 * math.pi)
            result["interior_sinusoid_error_vs_ideal_continuum"] = [
                error(cached[0, i + 1][mask].double(),
                      wave[mask] * math.exp(-.5 * (plan.method.sigma * frequency) ** 2))
                for i, (wave, frequency) in enumerate(zip(expected, frequencies))]
            result["interior_probe_pixels"] = int(mask.sum())
        return result

    def same_axes(left, right):
        return all(torch.equal(a, b) for a, b in zip(left.axes, right.axes))

    with torch.no_grad():
        for scale in scales:
            if progress:
                print(json.dumps(dict(event="gaussian_preflight_scale", scale=scale)), flush=True)
            failure_start = len(report["failures"])
            weight_device = "cpu" if scale == 2 else device
            base_sigma = ANCHOR_SIGMA / scale
            summary = dict(scale=scale, anchor_sigma=ANCHOR_SIGMA, base_sigma=base_sigma,
                           weight_device=weight_device, anchor_checks=[], kernel_diagnostics=[],
                           case_count=0, distinct_training_geometries=0,
                           sampling_difference_cases=0,
                           discarded_continuous_mass=math.erfc(4 / math.sqrt(2)))
            report["per_scale"][str(scale)] = summary
            try:
                anchor_ours = gaussian_kernel(METHODS[0], scale, anchor=scale, base_sigma=base_sigma)
                anchor_aa = gaussian_kernel(METHODS[1], scale, anchor=scale, base_sigma=base_sigma)
                gate(anchor_ours.sigma == anchor_aa.sigma == ANCHOR_SIGMA,
                     "fixed_anchor_sigma", scale=scale)
                for image in IMAGES:
                    shape = tuple(report["hr_shapes"][image])
                    kwargs = dict(antialiasing=False, device=device, weight_device=weight_device)
                    anchor = CachedResize(shape, 1 / scale, method=anchor_ours, **kwargs)
                    aa_anchor = CachedResize(shape, 1 / scale, method=anchor_aa, **kwargs)
                    exact = same_axes(anchor, aa_anchor)
                    gate(exact, "hr_anchor_weights_identical", scale=scale, image=image)
                    observed_shape = tuple(math.ceil(n / scale) for n in shape)
                    gate(anchor.out_shape == observed_shape, "shared_observation_shape", scale=scale, image=image)
                    row = plan_check(anchor, "rr_centered",
                                     dict(scale=scale, image=image, grid="shared_hr_anchor"), gaussian=True)
                    summary["anchor_checks"].append(dict(image=image,
                        anchor_weights_identical=exact, shared_observed_shape=list(observed_shape), **row))
                    del anchor, aa_anchor

                for m in range(1, 11):
                    factor = scale ** (1 / m)
                    ours = gaussian_kernel(METHODS[0], factor, anchor=scale, base_sigma=base_sigma)
                    aa = gaussian_kernel(METHODS[1], factor, anchor=scale, base_sigma=base_sigma)
                    kernels = {METHODS[0]: ours, METHODS[1]: aa, METHODS[2]: aa}
                    omega = np.linspace(0, np.pi, 65)
                    for method in METHODS[:2]:
                        kernel = kernels[method]
                        phase = phase_report(kernel)
                        gate(phase["finite"] and phase["uncovered_phases"] == 0,
                             "phase_coverage_and_finiteness", scale=scale, m=m, method=method)
                        ideal_chain = np.exp(-.5 * (kernel.sigma * omega) ** 2 *
                                            sum(factor ** (2 * j) for j in range(m)))
                        ideal_anchor = np.exp(-.5 * (ANCHOR_SIGMA * omega) ** 2)
                        truncated = spectrum(omega, kernel.sigma)
                        ideal = np.exp(-.5 * (kernel.sigma * omega) ** 2)
                        summary["kernel_diagnostics"].append(dict(m=m, method=method,
                            factor=factor, sigma=kernel.sigma, radius=kernel.radius,
                            support_sz=kernel.support_sz, truncate_sigma=kernel.truncate,
                            phase_sampling=phase,
                            continuous_truncation_max_error=float(np.max(np.abs(truncated - ideal))),
                            ideal_continuum_chain_vs_anchor_max_error=float(np.max(np.abs(ideal_chain - ideal_anchor)))))

                    geometry_results = {}
                    for image in IMAGES:
                        shape = tuple(report["hr_shapes"][image])
                        observed = tuple(math.ceil(n / scale) for n in shape)
                        core = crop_size(observed, 64, 16)
                        halo = 32 * scale + math.ceil(2 * factor) + 8 + 2
                        father = (core + 2 * halo,) * 2
                        reused = father in geometry_results
                        if not reused:
                            context = dict(scale=scale, m=m, father_shape=list(father))
                            plans = {}
                            for method in METHODS:
                                resize_class = MatchedTorchGridResize if method == METHODS[2] else CachedResize
                                plans[method] = resize_class(father, 1 / factor,
                                    method=kernels[method], antialiasing=False, device=device,
                                    weight_device=weight_device)
                            callable_shared = plans[METHODS[1]].method is plans[METHODS[2]].method
                            gate(callable_shared, "aa_grid_identical_filter_callable", **context)
                            exact_anchor = same_axes(plans[METHODS[0]], plans[METHODS[1]]) if m == 1 else None
                            if m == 1:
                                gate(exact_anchor, "m1_father_anchor_weights_identical", **context)
                            results = {}
                            up_results = {}
                            for method, plan in plans.items():
                                matched = method == METHODS[2]
                                expected = tuple((math.floor if matched else math.ceil)(n * (1 / factor)) for n in father)
                                gate(plan.out_shape == expected, "floor_or_ceil_son_shape", **context, method=method)
                                results[method] = plan_check(plan,
                                    "torch_half_pixel" if matched else "rr_centered",
                                    dict(context, method=method, grid="training_down"), gaussian=True)
                                if plan.out_shape not in up_results:
                                    up = CachedResize(plan.out_shape, factor, father, device=device,
                                                      weight_device=weight_device)
                                    up_results[plan.out_shape] = plan_check(up, "rr_centered",
                                        dict(context, method=method, grid="training_up"))
                                    del up
                                results[method]["upsampler"] = up_results[plan.out_shape]
                            rr_plan, grid_plan = plans[METHODS[1]], plans[METHODS[2]]
                            shape_differs = rr_plan.out_shape != grid_plan.out_shape
                            rr_first = (father[0] - 1) / 2 - (rr_plan.out_shape[0] - 1) * factor / 2
                            torch_first = (factor - 1) / 2
                            sampling_differs = shape_differs or abs(rr_first - torch_first) > 1e-10
                            if sampling_differs:
                                gate(not same_axes(rr_plan, grid_plan), "different_grid_weight_maps", **context)
                            geometry_results[father] = dict(arms=results,
                                aa_grid_filter_callable_identical=callable_shared,
                                m1_anchor_weights_identical=exact_anchor,
                                sampling_differs=sampling_differs,
                                aa_grid_shape_differs=shape_differs,
                                rr_minus_torch_first_coordinate=rr_first - torch_first)
                            summary["distinct_training_geometries"] += 1
                            del plans, rr_plan, grid_plan, plan
                        row = dict(scale=scale, image=image, m=m, factor=factor,
                                   hr_shape=list(shape), observed_shape=list(observed),
                                   core_size=core, halo=halo, father_size=father[0],
                                   reused_identical_geometry_check=reused, **geometry_results[father])
                        report["cases"].append(row)
                        summary["case_count"] += 1
                        summary["sampling_difference_cases"] += int(row["sampling_differs"])
                gate(summary["case_count"] == 50, "complete_scale_case_coverage", scale=scale)
                gate(summary["sampling_difference_cases"] > 0, "aa_grid_sampling_differs_in_sweep", scale=scale)
            except Exception as error:
                gate(False, "scale_preflight_exception", scale=scale,
                     error_type=type(error).__name__, error=str(error))
            diagnostics = summary["kernel_diagnostics"]
            summary["max_phase_partition_range"] = max(
                (d["phase_sampling"]["raw_partition_range"] for d in diagnostics), default=None)
            summary["max_phase_sinusoid_error_vs_ideal"] = max(
                (max(d["phase_sampling"]["max_phase_response_error_vs_ideal"]) for d in diagnostics), default=None)
            summary["max_continuous_truncation_error"] = max(
                (d["continuous_truncation_max_error"] for d in diagnostics), default=None)
            summary["passed"] = len(report["failures"]) == failure_start
            if progress:
                print(json.dumps(dict(event="gaussian_preflight_scale_complete", scale=scale,
                                      passed=summary["passed"], cases=summary["case_count"],
                                      elapsed_seconds=time.perf_counter() - start)), flush=True)
    gate(len(report["cases"]) == 50 * len(scales), "complete_suite_case_coverage",
         actual=len(report["cases"]), expected=50 * len(scales))
    gate(rr.get_projected_grid is original_grid, "final_grid_context_restored")
    report["cached_resize_checked"] = bool(report["cases"])
    report["cuda_checked"] = bool(report["cases"])
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, choices=(2, 3, 4), nargs="+", default=[2, 3, 4])
    parser.add_argument("--device", default="cuda", help="CUDA compute device; x2 weights remain on CPU")
    parser.add_argument("--output", type=Path, default=ROOT / "gaussian_suite_checks.json")
    args = parser.parse_args()
    report = run_checks(args.scale, args.device, progress=True)
    save_report(args.output, report)
    print(json.dumps(dict(passed=report["passed"], scales=report["scales"],
                          cases=len(report["cases"]), failures=report["failures"],
                          output=str(args.output))), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
