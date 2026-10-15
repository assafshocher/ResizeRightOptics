"""Plain PyTorch ZSSR: crop first, generate pairs online, repeat learned SR.

No pretrained weights, batch norm, self ensemble, back-projection,
HR-dependent stopping or HR-dependent model selection. A user-requested,
explicitly flagged shared bicubic residual ablation is also available.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from kernels import CachedResize, ContinuousKernel, interp_methods
from matched_kernels import MatchedTorchGridResize, LiteralAnchorKernel
from untapered_kernels import UntaperedKernel

ROOT = Path(__file__).parent
LABELS = {"consistent": "Camera law (band-limited)",
          "consistent_untapered": "Direct bicubic telescope (no smooth taper)",
          "aa_bandlimited": "AA stretch (same band limit)",
          "aa": "Literal RR bicubic-AA", "noaa": "Literal RR bicubic, no AA",
          "torch_aa": "PyTorch bicubic-AA (native grid)",
          "aa_grid_bandlimited": "Matched band-limited AA (PyTorch grid)",
          "aa_grid_literal": "Matched literal bicubic-AA (PyTorch grid)",
          "consistent_literal": "Camera law (literal anchor, regularized extension)"}


class TorchAADownsample:
    """Ordinary PyTorch scale-factor API; no manual shifts or grid degradation.

    Floor-sized half-pixel grid, align_corners=False, and explicit preservation
    of the requested scale factor. RR remains the common upsampler.
    """
    def __init__(self, shape, factor):
        self.scale = 1/factor
        self.out_shape = tuple(math.floor(n*self.scale) for n in shape)

    def __call__(self, image):
        return F.interpolate(image, scale_factor=self.scale, mode="bicubic",
                             align_corners=False, antialias=True,
                             recompute_scale_factor=False)


def crop_size(shape, requested, margin=0):
    core = min(requested, min(shape)-margin) if margin else requested
    if core < 8 or core > min(shape):
        raise ValueError(("Invalid loss crop", shape, requested, margin, core))
    return core


class PlainCNN(nn.Sequential):
    def __init__(self, width=64, depth=8, residual=False):
        layers = []
        for i in range(depth):
            conv = nn.Conv2d(3 if i == 0 else width,
                             3 if i == depth - 1 else width, 3, padding=1)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
            if i == depth - 1:
                with torch.no_grad():
                    if residual:
                        conv.weight.zero_()
                    else:
                        conv.weight.mul_(0.1)
            layers.append(conv)
            if i < depth - 1:
                layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)
        self.residual = residual

    def forward(self, image):
        output = super().forward(image)
        return image + output if self.residual else output


def load_image(path, device):
    return torch.from_numpy(np.load(path)).permute(2, 0, 1)[None].to(device)


def degradation(method, factor, radius, anchor=2):
    if method == "consistent_untapered":
        return UntaperedKernel(factor, radius=radius, anchor=anchor), False
    if method == "consistent_literal":
        if abs(factor-anchor) < 1e-12:
            return interp_methods.cubic, True  # Literal RR anchor, exactly.
        return LiteralAnchorKernel(factor, radius=radius, anchor=anchor), False
    if method in {"consistent", "aa_bandlimited", "aa_grid_bandlimited"}:
        law = "aa_bandlimited" if method == "aa_grid_bandlimited" else method
        return ContinuousKernel(factor, radius=radius, law=law, anchor=anchor), False
    return interp_methods.cubic, method in {"aa", "aa_grid_literal"}


def reflect_halo(image, halo):
    """Standard reflection, also defined when the halo exceeds image size.

    This only extends the observed image for crop extraction; all resampling
    remains ResizeRight. For ordinary halos it equals torch's reflect pad.
    """
    result = image
    for dim in (-2, -1):
        length = image.shape[dim]
        if length < 2:
            raise ValueError("Reflection needs at least two pixels")
        period = 2*(length-1)
        indices = torch.arange(-halo, length+halo, device=image.device) % period
        indices = torch.minimum(indices, period-indices)
        result = result.index_select(dim, indices)
    return result


def train(lr_image, method, m, args):
    """The training API receives ONLY the observed LR image."""
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    scale = getattr(args, "scale", 2)
    factor = scale**(1/m)
    network = PlainCNN(args.width, args.depth, getattr(args, "residual", False)).to(args.device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, fused=args.device.startswith("cuda"))
    n_h, n_w = lr_image.shape[-2:]
    core = crop_size((n_h, n_w), args.crop, getattr(args, "crop_margin", 0))
    # Degradation has a large support but the CNN sees only a small central crop.
    halo = args.radius + math.ceil(2*factor) + args.depth + 2
    father_size = core + 2*halo
    padded = reflect_halo(lr_image, halo)[0]
    # Preserve the original x2 implementation. For x3/x4, construct weights on
    # the compute device to avoid CPU/GPU fractional-grid rounding differences.
    weight_device = args.device if scale >= 3 else "cpu"
    if method == "torch_aa":
        down = TorchAADownsample((father_size, father_size), factor)
    else:
        kernel, aa = degradation(method, factor, args.radius, anchor=scale)
        resize_class = MatchedTorchGridResize if method.startswith("aa_grid_") else CachedResize
        down = resize_class((father_size, father_size), 1/factor,
                            method=kernel, antialiasing=aa, device=args.device,
                            weight_device=weight_device)
    up = CachedResize(down.out_shape, factor, out_shape=(father_size, father_size),
                      device=args.device, weight_device=weight_device)
    # Only coordinates are pre-generated; crops AND degradations are made anew.
    y = np_rng.integers(0, n_h - core + 1, size=(args.steps, args.batch))
    x = np_rng.integers(0, n_w - core + 1, size=(args.steps, args.batch))
    aug = np_rng.integers(0, 8, size=args.steps)
    y_indices = torch.as_tensor(y, device=args.device)[:, :, None, None]
    x_indices = torch.as_tensor(x, device=args.device)[:, :, None, None]
    offsets = torch.arange(father_size, device=args.device)
    coordinate_hash = hashlib.sha256(y.tobytes()+x.tobytes()+aug.tobytes()).hexdigest()
    losses, start = [], time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        start = time.perf_counter()
    network.train()
    for step in range(args.steps):
        father = padded[:, y_indices[step] + offsets[None, :, None],
                        x_indices[step] + offsets[None, None, :]].permute(1, 0, 2, 3)
        father = torch.rot90(father, int(aug[step] % 4), (-2, -1))
        if aug[step] >= 4:
            father = father.flip(-1)
        with torch.no_grad():
            son = down(father)
            interpolated = up(son)
            input_crop = interpolated[..., halo-args.depth:halo+core+args.depth,
                                      halo-args.depth:halo+core+args.depth].contiguous(memory_format=torch.channels_last)
            target = father[..., halo:halo+core, halo:halo+core]
        fraction = step / args.steps
        mult = 1 if fraction < .6 else (.1 if fraction < .85 else .01)
        if step < 20:
            mult *= (step + 1) / 20
        for group in optimizer.param_groups:
            group["lr"] = args.lr * mult
        optimizer.zero_grad(set_to_none=True)
        prediction = network(input_crop)[..., args.depth:-args.depth, args.depth:-args.depth]
        loss = F.l1_loss(prediction, target)
        loss.backward()
        optimizer.step()
        if (step + 1) % 250 == 0 or step == 0:
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError("Non-finite training loss")
            losses.append({"step": step+1, "l1": value})
            print(json.dumps({"event": "train", "image": args.image, "m": m,
                              "method": method, "step": step+1, "l1": value}), flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    return network.eval(), {"train_seconds": seconds, "losses": losses,
                             "coordinate_sha256": coordinate_hash,
                             "parameters": sum(p.numel() for p in network.parameters()),
                             "father_size": father_size, "core_size": core,
                             "training_son_shape": down.out_shape,
                             "training_downsampler": ("torch.interpolate" if method == "torch_aa"
                                 else "ResizeRight with PyTorch floor/half-pixel grid"
                                 if method.startswith("aa_grid_") else "ResizeRight")}


@torch.no_grad()
def infer(network, image, m, args, target_shape=None):
    scale = getattr(args, "scale", 2)
    factor = scale**(1/m)
    initial = image.shape[-2:]
    current = image
    sizes = []
    for j in range(1, m+1):
        shape = tuple(math.ceil(n * scale**(j/m) - 1e-9) for n in initial)
        if j == m and target_shape is not None:
            shape = tuple(target_shape)
        up = CachedResize(current.shape[-2:], factor, shape, device=args.device,
                          weight_device=args.device if scale >= 3 else "cpu")
        interpolated = up(current)
        # Reflection is common to every CNN at inference, with zero-padding
        # effects excluded. Optional residual is the identical bicubic input.
        padded = F.pad(interpolated, [args.depth]*4, mode="reflect")
        current = network(padded.contiguous(memory_format=torch.channels_last))[
            ..., args.depth:-args.depth, args.depth:-args.depth]
        sizes.append(shape)
    return current, sizes


def metrics(prediction, truth, shave=2):
    # Fixed scale-sized HR border shave. Also retain raw/unclipped scores.
    pred = prediction.double()[..., shave:-shave, shave:-shave]
    gt = truth.double()[..., shave:-shave, shave:-shave]
    weights = torch.tensor([65.481, 128.553, 24.966], device=pred.device, dtype=pred.dtype)[None, :, None, None]/255
    def score(delta):
        mse = delta.square().mean().item()
        return -10*math.log10(max(mse, 1e-30))
    return {"psnr_rgb": score(pred.clamp(0, 1)-gt),
            "psnr_y": score(((pred.clamp(0, 1)-gt)*weights).sum(1)),
            "psnr_rgb_unclipped": score(pred-gt),
            "psnr_y_unclipped": score(((pred-gt)*weights).sum(1))}


def durable_save(path, writer):
    """Commit one artifact atomically, flushing its bytes before its directory.

    The previous host reboot left zero-length model/prediction files despite
    readable score JSON. This changes persistence only, never computation.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    with temporary.open("wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_json(path, value):
    data = json.dumps(value, indent=2).encode("utf-8")
    durable_save(path, lambda stream: stream.write(data))


def save_output(path, tensor):
    array = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    durable_save(path.with_suffix(".npy"), lambda stream: np.save(stream, array))
    preview = Image.fromarray(np.rint(np.clip(array, 0, 1)*255).astype(np.uint8))
    durable_save(path.with_suffix(".png"), lambda stream: preview.save(stream, format="PNG"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="baby", choices=["baby", "bird", "butterfly", "head", "woman"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--crop-margin", type=int, default=0,
                   help="Shrink loss crop to min(64, shortest LR side minus this margin)")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--radius", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--m", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--methods", nargs="+", default=list(LABELS), choices=list(LABELS))
    p.add_argument("--input-kind", default="bandlimited", choices=["bandlimited", "literal"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-name", default="pilot")
    p.add_argument("--resume", action="store_true", help="Continue an interrupted run only with an identical configuration")
    p.add_argument("--residual", action="store_true", help="Shared bicubic skip ablation explicitly requested after the plain-CNN pilot")
    args = p.parse_args()
    torch.set_num_threads(4)
    # Keep resampling/metrics float32/64. TensorFloat32 is used only for CNN
    # convolutions, identically in all arms; no fp16/bf16 rounding of images.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    folder = ROOT / "runs" / args.run_name / f"{args.image}_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=args.resume)
    config = {k: v for k, v in vars(args).items() if k != "resume"} | {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                           "RR_commit": "510d4d5b67dccf4efdee9f311ed42609a71f17c5",
                           "ZSSR_commit": "d2948526fde799a0c111469b78cd9e1b8659cb0f",
                           "evaluation": f"Clipped float RGB; BT.601 Y with peak 1; {args.scale} HR pixel border; no registration; also unclipped scores",
                           "shared_anchor": ("Untapered bicubic anchor matches in-band symbol, not exact literal sampled AA; m1 independently trained"
                               if "consistent_untapered" in args.methods else
                               "No exact literal-AA anchor: retained-band consistent kernels on literal observations; m1 independently trained"
                               if args.input_kind == "literal" and "consistent" in args.methods else
                               "consistent_literal and aa: literal RR cubic-AA; grid arm shares continuous kernel, not all sample positions"
                               if "consistent_literal" in args.methods else
                               "consistent and aa_bandlimited only; literal AA and noAA are additional software references"),
                           "rr_weight_device": args.device if args.scale >= 3 else "cpu"}
    if "consistent_literal" in args.methods:
        config["literal_extension"] = {"ridge_eta": LiteralAnchorKernel.ETA,
            "ridge_reference": "literal AA", "fft_length": 1 << 20,
            "omega_max": float(32*np.pi), "density": 512,
            "exact_sampled_composition_claimed": False,
            "protocol": "MATCHED_PROTOCOL.md"}
    if "consistent_untapered" in args.methods:
        config["untapered_construction"] = {
            "camera": "product_j=0..17 U(omega/anchor**j)",
            "multiplier": "H(factor*omega)/H(omega)",
            "spectral_domain": "abs(omega)<=pi/factor",
            "smooth_window": False, "ridge": False,
            "density": 512, "quadrature": 4097,
            "exact_literal_anchor_claimed": False,
            "protocol": "UNTAPERED_PILOT_PROTOCOL.md"}
    if args.resume:
        previous = json.loads((folder / "config.json").read_text())
        previous.setdefault("residual", False)
        previous.setdefault("scale", 2)
        previous.setdefault("crop_margin", 0)
        previous.setdefault("rr_weight_device", "cpu")
        if previous != config:
            raise ValueError("Refusing to resume with a different configuration")
    else:
        save_json(folder / "config.json", config)
    # Separate execution provenance allows an interrupted experiment to resume
    # with the same scientific configuration after a kernel-only rollback.
    execution_path = folder / f"execution_{time.time_ns()}.json"
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    execution = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "kernel": platform.release(),
        "boot_id": boot_id_path.read_text().strip() if boot_id_path.exists() else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "resumed": args.resume,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (("experiment.py", "kernels.py", "matched_kernels.py", "literal_spectrum.py")
                         + (("untapered_kernels.py",) if "consistent_untapered" in args.methods else ()))
        },
    }
    save_json(execution_path, execution)
    print(json.dumps({"event": "execution", **execution}), flush=True)
    suffix = "lr_bandlimited" if args.input_kind == "bandlimited" else "lr"
    data = ROOT / ("data" if args.scale == 2 else f"data_x{args.scale}")
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("radius", 64) != args.radius:
        raise ValueError("Input-generation and training kernel radii differ")
    entry = next(item for item in manifest["images"] if item["image"] == args.image)
    # Requested output dimensions are metadata, not target pixel values.
    target_shape = tuple(entry["hr_shape"][-2:])
    lr_image = load_image(data / f"{args.image}_{suffix}.npy", args.device)
    # Truth is loaded by the evaluator only; never passed to train/infer.
    truth = load_image(data / f"{args.image}_hr.npy", args.device)
    baseline = CachedResize(lr_image.shape[-2:], args.scale, target_shape, device=args.device,
                            weight_device=args.device if args.scale >= 3 else "cpu")(lr_image)
    baseline_scores = metrics(baseline, truth, shave=args.scale)
    save_output(folder / "bicubic_baseline", baseline)
    records = (json.loads((folder / "results.json").read_text())["results"]
               if args.resume and (folder / "results.json").exists() else [])
    done = {(r["m"], r["method"]) for r in records}
    for m in args.m:
        anchor_method = "consistent_literal" if args.input_kind == "literal" else "consistent"
        anchor_partner = "aa" if args.input_kind == "literal" else "aa_bandlimited"
        common_anchor = next((r for r in records if r["m"] == 1 and r["method"] == anchor_method), None)
        for method in args.methods:
            if (m, method) in done:
                continue
            if m == 1 and method == anchor_partner and common_anchor is not None:
                record = dict(common_anchor) | {"method": method, "label": LABELS[method],
                                                "reused_identical_anchor_run": True}
            else:
                print(json.dumps({"event": "start", "image": args.image, "m": m, "method": method}), flush=True)
                model, training = train(lr_image, method, m, args)
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction, sizes = infer(model, lr_image, m, args, target_shape)
                torch.cuda.synchronize()
                timing = time.perf_counter()-start
                scores = metrics(prediction, truth, shave=args.scale)
                record = {"image": args.image, "m": m, "elementary_scale": args.scale**(1/m),
                          "method": method, "label": LABELS[method], **scores, **training,
                          "execution_provenance": execution_path.name,
                          "inference_seconds_including_RR_setup": timing, "inference_shapes": sizes,
                          "gain_over_bicubic_y": scores["psnr_y"]-baseline_scores["psnr_y"]}
                save_output(folder / f"m{m}_{method}", prediction)
                durable_save(folder / f"m{m}_{method}.pt",
                             lambda stream: torch.save(model.state_dict(), stream))
                if m == 1 and method == anchor_method:
                    common_anchor = record
                del model
            records.append(record)
            result = {"config": config, "bicubic": baseline_scores, "results": records}
            save_json(folder / "results.json", result)
            columns = ["image", "m", "method", "psnr_y", "psnr_rgb", "gain_over_bicubic_y", "train_seconds"]
            csv_buffer = io.StringIO(newline="")
            writer = csv.DictWriter(csv_buffer, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
            durable_save(folder / "results.csv",
                         lambda stream: stream.write(csv_buffer.getvalue().encode("utf-8")))
            print(json.dumps({"event": "result", **record}), flush=True)
    print(json.dumps({"event": "first_image_complete", "file": str(folder / "results.json"),
                      "bicubic": baseline_scores}), flush=True)


if __name__ == "__main__":
    main()
