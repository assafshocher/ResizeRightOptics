"""Obtain the five original Set5 images and prepare common RR inputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
from PIL import Image
import torch

from kernels import rr, ContinuousKernel

ROOT = Path(__file__).parent
URL = ("https://huggingface.co/datasets/eugenesiow/Set5/resolve/"
       "bcb97ac28c8a5f9113825f8d163564e16513f3ed/data/Set5_HR.tar.gz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--radius", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    archive = ROOT / "Set5_HR.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    folder = ROOT / ("data" if args.scale == 2 else f"data_x{args.scale}")
    folder.mkdir(exist_ok=True)
    records = []
    kernel = ContinuousKernel(args.scale, radius=args.radius, anchor=args.scale)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith((".png", ".bmp"))]
        if len(members) != 5:
            raise ValueError([m.name for m in members])
        for member in members:
            raw = tar.extractfile(member).read()
            name = Path(member.name).stem.lower().replace("_hr", "")
            if name not in {"baby", "bird", "butterfly", "head", "woman"}:
                raise ValueError(name)
            image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255
            hr = torch.from_numpy(image).permute(2, 0, 1)[None]
            lr = rr.resize(hr, scale_factors=1/args.scale, pad_mode="reflect")
            matched = rr.resize(hr, scale_factors=1/args.scale, pad_mode="reflect",
                                interp_method=kernel, antialiasing=False)
            # Float arrays are the data. PNGs are previews only (no LR requantization).
            np.save(folder / f"{name}_hr.npy", image)
            np.save(folder / f"{name}_lr.npy", lr[0].permute(1, 2, 0).numpy())
            np.save(folder / f"{name}_lr_bandlimited.npy", matched[0].permute(1, 2, 0).numpy())
            Image.fromarray(np.rint(image * 255).astype("uint8")).save(folder / f"{name}_hr.png")
            Image.fromarray(np.rint(lr[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")).save(folder / f"{name}_lr_preview.png")
            records.append({"image": name, "source_member": member.name,
                            "source_sha256": hashlib.sha256(raw).hexdigest(),
                            "hr_shape": list(hr.shape), "lr_shape": list(lr.shape),
                            "lr_min": lr.min().item(), "lr_max": lr.max().item()})
    manifest = {"url": URL, "academic_use_only": True, "scale": args.scale,
                "radius": args.radius,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "degradation": f"Unmodified ResizeRight cubic, antialiasing=True, scale=1/{args.scale}, reflect padding; unclipped float32; original HR dimensions retained",
                "matched_degradation": f"Cubic-AA symbol times fixed C4 lowpass (flat to 80% output Nyquist, zero at Nyquist); LUT density 512/pixel, radius {args.radius}; RR weights normalized per output; finite-support approximation, not exactly bandlimited",
                "images": sorted(records, key=lambda r: r["image"])}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
