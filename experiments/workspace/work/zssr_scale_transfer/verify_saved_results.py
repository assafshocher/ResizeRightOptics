"""Read-only verification of scored predictions/checkpoints; CPU evaluation only.

Never retrain, replace a prediction, or select a result by its score. Emit the
verification report to stdout so it can be saved with execution provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiment import save_json

ROOT = Path(__file__).resolve().parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]
METHODS = ["consistent", "aa_bandlimited", "aa", "torch_aa"]


def source_run(scale, method, m):
    if scale != 2:
        return f"set5_x{scale}_m1_10"
    if method == "torch_aa":
        return "set5_x2_torch_m1_10"
    return "first_image_residual_v1" if m <= 5 else "set5_x2_m6_10"


def score(prediction, truth, shave):
    pred = prediction[shave:-shave, shave:-shave].astype(np.float64)
    gt = truth[shave:-shave, shave:-shave].astype(np.float64)
    weights = np.array([65.481, 128.553, 24.966]) / 255

    def psnr(delta):
        return float(-10 * np.log10(max(float(np.mean(delta**2)), 1e-30)))

    clipped = np.clip(pred, 0, 1) - gt
    raw = pred - gt
    return {"psnr_rgb": psnr(clipped),
            "psnr_y": psnr((clipped * weights).sum(-1)),
            "psnr_rgb_unclipped": psnr(raw),
            "psnr_y_unclipped": psnr((raw * weights).sum(-1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", choices=[2, 3, 4], default=[2, 3, 4])
    parser.add_argument("--images", nargs="+", choices=IMAGES, default=IMAGES)
    parser.add_argument("--report", type=Path, help="Optional durable JSON verification report")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cache, checked, hashes = {}, set(), {}
    count, max_error = 0, 0.0
    for scale in args.scales:
        data = ROOT / ("data" if scale == 2 else f"data_x{scale}")
        for image in args.images:
            truth = np.load(data / f"{image}_hr.npy", allow_pickle=False)
            assert np.isfinite(truth).all()
            for m in range(1, 11):
                for method in METHODS:
                    folder = ROOT / "runs" / source_run(scale, method, m) / f"{image}_seed0"
                    if folder not in cache:
                        raw = (folder / "results.json").read_bytes()
                        result = json.loads(raw)
                        rows = {(r["m"], r["method"]): r for r in result["results"]}
                        assert len(rows) == len(result["results"]), folder
                        cache[folder] = rows
                        hashes[str(folder.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
                        base = np.load(folder / "bicubic_baseline.npy", allow_pickle=False)
                        assert base.shape == truth.shape and np.isfinite(base).all(), folder
                        for key, value in score(base, truth, scale).items():
                            error = abs(value - result["bicubic"][key])
                            assert error < 1e-7, (folder, "baseline", key, error)
                            max_error = max(max_error, error)
                    row = cache[folder][m, method]
                    artifact_method = "consistent" if row.get("reused_identical_anchor_run") else method
                    stem = folder / f"m{m}_{artifact_method}"
                    prediction = np.load(stem.with_suffix(".npy"), allow_pickle=False)
                    assert prediction.shape == truth.shape and np.isfinite(prediction).all(), stem
                    for key, value in score(prediction, truth, scale).items():
                        error = abs(value - row[key])
                        assert error < 1e-7, (stem, key, error)
                        max_error = max(max_error, error)
                    if stem not in checked:
                        state = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
                        assert sum(t.numel() for t in state.values()) == 225091, stem
                        assert all(torch.isfinite(t).all().item() for t in state.values()), stem
                        checked.add(stem)
                    count += 1
            print(json.dumps({"event": "verified_image", "scale": scale, "image": image,
                              "records": 40}), flush=True)
    report = {"passed": True, "scales": args.scales, "images": args.images,
              "records": count, "unique_model_prediction_pairs": len(checked),
              "max_recomputed_psnr_difference_db": max_error,
              "source_results_sha256": hashes}
    if args.report:
        save_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
