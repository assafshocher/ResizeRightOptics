"""Read-only model/kernel investigation; never trains or edits benchmark runs.

The cross-input check is deliberately NOT a new matched-training benchmark:
it feeds literal bicubic observations to previously bandlimited-trained models.
All outputs are diagnostic records in a separate, explicitly requested folder.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiment import PlainCNN, infer, load_image, metrics, save_json
from kernels import ContinuousKernel, interp_methods, rr
from literal_spectrum import anchored_ridge_spectrum, full_camera
from matched_kernels import LiteralAnchorKernel

ROOT = Path(__file__).resolve().parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_checks():
    rows = []
    for scale in (2, 3, 4):
        for m in (2, 5, 10):
            factor = scale ** (1 / m)
            radius = 32 * scale
            literal = LiteralAnchorKernel(factor, radius, scale)
            band = ContinuousKernel(factor, radius=radius, anchor=scale)
            for fraction in (.1, .25):
                omega = fraction * np.pi
                target = float(full_camera(np.array([factor * omega]), scale)[0]
                               / full_camera(np.array([omega]), scale)[0])
                for phase in (0., .25, .5, .75):
                    offsets = phase - np.arange(-radius, radius + 1)
                    for name, weights in (
                        ("literal_extension", literal(offsets)),
                        ("retained_band", band(offsets)),
                        ("ordinary_aa", interp_methods.cubic(offsets / factor) / factor),
                    ):
                        weights = weights / weights.sum()
                        gain = np.sum(weights * np.exp(-1j * omega * offsets))
                        rows.append(dict(scale=scale, m=m, factor=factor,
                            frequency_pi=fraction, phase=phase, method=name,
                            target_real=target, gain_real=float(gain.real),
                            gain_imag=float(gain.imag),
                            complex_error=float(abs(gain - target))))
    # Independent frequency-domain replica prediction for the decisive example.
    q = np.arange(-16, 17)
    omega = np.pi / 4
    k = sum(anchored_ridge_spectrum(omega + 2 * np.pi * q, 2., 4))
    dc = sum(anchored_ridge_spectrum(2 * np.pi * q, 2., 4))
    replica = dict(central=float(k[16]), minus_first=float(k[15]),
        plus_first=float(k[17]), integer_phase=float(k.sum() / dc.sum()),
        half_phase=float(np.sum(k * (-1.)**q) / np.sum(dc * (-1.)**q)))
    # Actual unmodified RR spatial calls, not just a kernel coefficient sum.
    actual = []
    literal = LiteralAnchorKernel(2., 128, 4)
    band = ContinuousKernel(2., radius=128, anchor=4)
    for n in (1024, 1025):
        target = int(np.ceil(n / 2))
        x = np.arange(n) - (n - 1) / 2
        y = 2 * (np.arange(target) - (target - 1) / 2)
        signal = np.stack([np.cos(omega * x), np.sin(omega * x)])
        for name, kernel, aa in [("literal_extension", literal, False),
                                 ("retained_band", band, False),
                                 ("ordinary_aa", interp_methods.cubic, True)]:
            z = rr.resize(signal, scale_factors=[1, .5], out_shape=[2, target],
                interp_method=kernel, antialiasing=aa, pad_mode="reflect")
            gain = ((z[0] + 1j * z[1]) * np.exp(-1j * omega * y))[100:-100]
            actual.append(dict(input_length=n, phase=float((y[0] + (n-1)/2) % 1),
                method=name, real=float(gain.real.mean()), imag=float(gain.imag.mean()),
                variation=float(gain.std())))
    near_identity = []
    for scale in (2, 3, 4):
        for factor in (1.0001, 1.01):
            kernel = LiteralAnchorKernel(factor, 32 * scale, scale)
            # Odd input/output dimensions give an integer-phase central row.
            n = 1025
            target = int(np.ceil(n / factor))
            if target % 2 == 0:
                target += 1
            x = np.arange(n) - (n - 1) / 2
            signal = np.stack([np.cos(omega * x), np.sin(omega * x)])
            for name, method, aa in [("literal_extension", kernel, False),
                                      ("ordinary_aa", interp_methods.cubic, True)]:
                z = rr.resize(signal, scale_factors=[1, 1/factor],
                    out_shape=[2, target], interp_method=method,
                    antialiasing=aa, pad_mode="reflect")
                near_identity.append(dict(scale=scale, factor=factor,
                    method=name, center_gain_real=float(z[0, target//2]),
                    center_gain_imag=float(z[1, target//2])))
    return dict(frequency_units="radians per input pixel", rows=rows,
                replica_prediction=replica, actual_rr=actual,
                near_identity=near_identity,
                identity_caveat="RR bypasses filtering at exactly scale=1; these tests use nonunit factors approaching one.")


def old_folder(scale, image, m):
    run = (f"set5_x{scale}_m1_10" if scale != 2 else
           "first_image_residual_v1" if m <= 5 else "set5_x2_m6_10")
    return ROOT / "runs" / run / f"{image}_seed0"


@torch.no_grad()
def cross_input(output):
    rows = []
    for scale in (2, 3, 4):
        data = ROOT / ("data" if scale == 2 else f"data_x{scale}")
        for image in IMAGES:
            lr_path = data / f"{image}_lr.npy"
            hr_path = data / f"{image}_hr.npy"
            observed = load_image(lr_path, "cuda")
            truth = load_image(hr_path, "cuda")
            native_path = ROOT / "runs" / f"matched_literal_x{scale}_v1" / f"{image}_seed0" / "results.json"
            native = json.loads(native_path.read_text())
            lookup = {(r["m"], r["method"]): r for r in native["results"]}
            for m in (2, 5, 10):
                folder = old_folder(scale, image, m)
                source = json.loads((folder / "results.json").read_text())
                config = dict(source["config"])
                config.update(scale=scale, device="cuda")
                model_path = folder / f"m{m}_consistent.pt"
                model = PlainCNN(config["width"], config["depth"], config["residual"])
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                model = model.to("cuda", memory_format=torch.channels_last).eval()
                prediction, shapes = infer(model, observed, m, SimpleNamespace(**config), truth.shape[-2:])
                score = metrics(prediction, truth, shave=scale)
                old = next(r for r in source["results"] if r["m"] == m and r["method"] == "consistent")
                row = dict(scale=scale, image=image, m=m,
                    cross_input_psnr_y=score["psnr_y"],
                    original_bandlimited_psnr_y=old["psnr_y"],
                    literal_extension_psnr_y=lookup[m, "consistent_literal"]["psnr_y"],
                    literal_aa_psnr_y=lookup[m, "aa"]["psnr_y"],
                    literal_bicubic_baseline_y=native["bicubic"]["psnr_y"],
                    checkpoint=str(model_path.relative_to(ROOT)), checkpoint_sha256=sha(model_path),
                    lr_sha256=sha(lr_path), hr_sha256=sha(hr_path),
                    output_shapes=shapes)
                rows.append(row)
                save_json(output / "cross_input_partial.json", dict(complete=False, records=rows))
                print(json.dumps(row), flush=True)
                del model, prediction
    means = []
    fields = ["cross_input_psnr_y", "original_bandlimited_psnr_y",
              "literal_extension_psnr_y", "literal_aa_psnr_y", "literal_bicubic_baseline_y"]
    for scale in (2, 3, 4):
        for m in (2, 5, 10):
            selected = [r for r in rows if r["scale"] == scale and r["m"] == m]
            assert len(selected) == 5
            means.append(dict(scale=scale, m=m, **{key: float(np.mean([r[key] for r in selected])) for key in fields}))
    result = dict(complete=True, training_performed=False,
        qualification="Existing bandlimited-trained models evaluated on literal bicubic inputs; diagnostic domain shift, NOT matched retraining or a fair new benchmark.",
        records=rows, means=means)
    save_json(output / "cross_input_complete.json", result)
    print(json.dumps({"means": means}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--part", choices=["kernels", "cross-input", "all"], default="all")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.part in {"kernels", "all"}:
        result = kernel_checks()
        save_json(args.output / "kernel_checks.json", result)
        print(json.dumps({"actual_rr": result["actual_rr"], "replicas": result["replica_prediction"],
                          "near_identity": result["near_identity"]}), flush=True)
    if args.part in {"cross-input", "all"}:
        cross_input(args.output)


if __name__ == "__main__":
    main()
