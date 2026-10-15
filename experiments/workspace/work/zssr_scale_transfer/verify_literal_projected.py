"""Verify the projected-kernel rerun on literal bicubic data, without training.

New ours records are independently trained at every m, including m=1. Existing
literal AA and PyTorch-grid controls are read in place, never modified. Missing
images propagate to null means; partial five-image averages are never reported.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiment import PlainCNN, durable_save, metrics, save_json

ROOT = Path(__file__).resolve().parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]
METHODS = ["ours", "aa", "grid"]
STEPS = list(range(1, 11))
EXPECTED = {"steps": 10000, "lr": 1e-4, "batch": 1, "crop": 64,
            "crop_margin": 16, "width": 64, "depth": 8, "seed": 0,
            "residual": True, "input_kind": "literal", "m": STEPS}
ANCHOR_CAVEAT = (
    "Literal bicubic-AA input; ours uses the earlier retained-band projected "
    "camera family unchanged. The anchor match is approximate, not exact. "
    "Ours at m=1 is independently trained, not copied from AA."
)
GRID_CAVEAT = (
    "RR filters/normalization/reflection on the PyTorch floor/half-pixel "
    "downsampling grid, with shared RR upsampling; not unmodified "
    "torch.interpolate end to end. Coordinate mismatch contributes to error."
)


def require(condition, *detail):
    if not condition:
        raise ValueError(detail)


def source(scale, image, method):
    if method == "ours":
        run, key = f"literal_projected_x{scale}_v1", "consistent"
    else:
        run = f"matched_literal_x{scale}_v1"
        key = "aa" if method == "aa" else "aa_grid_literal"
    return ROOT / "runs" / run / f"{image}_seed0" / "results.json", key


def tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    durable_save(path, lambda out: out.write(payload))


class Verifier:
    def __init__(self, allow_partial):
        self.allow_partial = allow_partial
        self.cache = {}
        self.checked = set()
        self.hashes = {}
        self.missing = []
        self.counts = {method: 0 for method in METHODS}
        self.max_error = 0.0
        self.coordinate_comparisons = 0
        self.shapes = {name: value.shape for name, value in
                       PlainCNN(64, 8, True).state_dict().items()}

    def scores(self, array, truth, scale, expected, context):
        require(array.shape == truth.shape and np.isfinite(array).all(),
                context, "invalid prediction shape or values")
        actual = metrics(tensor(array), tensor(truth), shave=scale)
        for name, value in actual.items():
            require(name in expected and math.isfinite(expected[name]),
                    context, "invalid stored metric", name)
            error = abs(value - expected[name])
            require(error < 1e-8, context, name, error)
            self.max_error = max(self.max_error, error)
        return actual

    def load(self, path, scale, image, method, truth):
        if path in self.cache:
            return self.cache[path]
        if not path.exists():
            return None
        raw = path.read_bytes()
        result = json.loads(raw)
        config = result["config"]
        for name, value in (EXPECTED | {"image": image, "scale": scale,
                                      "radius": 32*scale}).items():
            require(config.get(name) == value, path, name, config.get(name), value)
        require(config["rr_weight_device"] == ("cpu" if scale == 2 else "cuda"),
                path, "different RR weight device")
        if method == "ours":
            require(config["methods"] == ["consistent"], path, "unexpected ours methods")
            require("literal_extension" not in config, path, "unrestricted extension")
        rows = {(row["m"], row["method"]): row for row in result["results"]}
        require(len(rows) == len(result["results"]), path, "duplicate records")
        if method == "ours":
            require(set(rows) <= {(m, "consistent") for m in STEPS},
                    path, "unexpected ours records")
        baseline = np.load(path.parent / "bicubic_baseline.npy", allow_pickle=False)
        self.scores(baseline, truth, scale, result["bicubic"], path)
        self.hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        self.cache[path] = (result, rows)
        return self.cache[path]

    def row(self, scale, image, method, m, truth):
        path, key = source(scale, image, method)
        loaded = self.load(path, scale, image, method, truth)
        if loaded is None or (m, key) not in loaded[1]:
            missing = dict(scale=scale, image=image, method=method, m=m,
                           source=str(path.relative_to(ROOT)))
            self.missing.append(missing)
            require(self.allow_partial, "missing record", missing)
            return None
        result, rows = loaded
        row = rows[m, key]
        require(row["image"] == image and row["parameters"] == 225091,
                path, m, "invalid image or parameter count")
        require(abs(row["elementary_scale"] - scale**(1/m)) < 1e-12,
                path, m, "wrong elementary scale")
        require(len(row["coordinate_sha256"]) == 64, path, m, "missing crop hash")
        require(row["losses"][-1]["step"] == EXPECTED["steps"] and
                all(math.isfinite(loss["l1"]) for loss in row["losses"]),
                path, m, "incomplete or nonfinite training")
        reused = bool(row.get("reused_identical_anchor_run"))
        if method == "ours":
            require(not reused, path, m, "ours must train independently, including m=1")
        if reused:
            require(method == "aa" and m == 1, path, m, "unexpected reused control")
        artifact = "consistent_literal" if reused else key
        stem = path.parent / f"m{m}_{artifact}"
        prediction = np.load(stem.with_suffix(".npy"), allow_pickle=False)
        actual = self.scores(prediction, truth, scale, row, stem)
        if stem not in self.checked:
            state = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
            require(set(state) == set(self.shapes), stem, "wrong checkpoint keys")
            require(all(value.shape == self.shapes[name] and
                        torch.isfinite(value).all().item()
                        for name, value in state.items()), stem, "invalid checkpoint")
            require(sum(value.numel() for value in state.values()) == 225091,
                    stem, "wrong checkpoint parameter count")
            self.checked.add(stem)
        self.counts[method] += 1
        return row, dict(mode="literal_projected", scale=scale, image=image, m=m,
                         method=method, **actual, bicubic_y=result["bicubic"]["psnr_y"],
                         source=str(path.relative_to(ROOT)), source_method=key,
                         shared_anchor=reused)

    def aggregate(self, scale):
        data = ROOT / ("data" if scale == 2 else f"data_x{scale}")
        values = np.full((len(IMAGES), len(METHODS), len(STEPS)), np.nan)
        baselines = np.full(len(IMAGES), np.nan)
        records, start_missing = [], len(self.missing)
        for i, image in enumerate(IMAGES):
            truth = np.load(data / f"{image}_hr.npy", allow_pickle=False)
            require(truth.ndim == 3 and truth.shape[-1] == 3 and np.isfinite(truth).all(),
                    scale, image, "invalid truth")
            hashes = set()
            for k, m in enumerate(STEPS):
                crop_hashes = {}
                for j, method in enumerate(METHODS):
                    checked = self.row(scale, image, method, m, truth)
                    if checked is None:
                        continue
                    row, record = checked
                    values[i, j, k] = record["psnr_y"]
                    if np.isfinite(baselines[i]):
                        require(abs(baselines[i] - record["bicubic_y"]) < 1e-8,
                                scale, image, "baseline mismatch")
                    baselines[i] = record["bicubic_y"]
                    crop_hashes[method] = row["coordinate_sha256"]
                    hashes.add(row["coordinate_sha256"])
                    records.append(record)
                if "ours" in crop_hashes and "aa" in crop_hashes:
                    require(crop_hashes["ours"] == crop_hashes["aa"],
                            scale, image, m, "ours/AA crop coordinates differ")
                    self.coordinate_comparisons += 1
            require(len(hashes) <= 1, scale, image, "crop schedule changed across arms/steps")
            print(json.dumps(dict(event="verified_image", mode="literal_projected",
                                  scale=scale, image=image, counts=self.counts)), flush=True)
        # Deliberately not nanmean: any missing image means no five-image mean.
        means = values.mean(axis=0)
        clean = lambda array: [float(value) if np.isfinite(value) else None for value in array]
        missing = self.missing[start_missing:]
        summary = dict(mode="literal_projected", scale=scale, m=STEPS,
            complete=not missing, missing_records=len(missing), images=IMAGES,
            mean_psnr_y={method: clean(means[j]) for j, method in enumerate(METHODS)},
            baseline_mean_y=float(baselines.mean()) if np.isfinite(baselines).all() else None,
            per_image_psnr_y={method: {image: clean(values[i, j])
                                      for i, image in enumerate(IMAGES)}
                              for j, method in enumerate(METHODS)},
            aggregation="Arithmetic mean of all five per-image Y-PSNR values; one seed",
            anchor_caveat=ANCHOR_CAVEAT, independent_ours_m1=True,
            grid_caveat=GRID_CAVEAT, missing=missing,
            source_results_sha256={name: digest for name, digest in self.hashes.items()
                                   if f"_x{scale}_v1/" in name})
        output = ROOT / "figures"
        output.mkdir(exist_ok=True)
        save_json(output / f"literal_projected_x{scale}_summary.json", summary)
        fields = ["mode", "scale", "image", "m", "method", "psnr_rgb", "psnr_y",
                  "psnr_rgb_unclipped", "psnr_y_unclipped", "bicubic_y", "source",
                  "source_method", "shared_anchor"]
        write_csv(output / f"literal_projected_x{scale}_per_image.csv", records, fields)
        mean_rows = [dict(m=m, **{method: summary["mean_psnr_y"][method][k]
                                 for method in METHODS}, bicubic=summary["baseline_mean_y"])
                     for k, m in enumerate(STEPS)]
        write_csv(output / f"literal_projected_x{scale}_means.csv", mean_rows,
                  ["m", *METHODS, "bicubic"])
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--scales", nargs="+", type=int, choices=[2, 3, 4], default=[2, 3, 4])
    args = parser.parse_args()
    require(len(set(args.scales)) == len(args.scales), "duplicate scales")
    torch.set_num_threads(2)
    verifier = Verifier(args.allow_partial)
    summaries = [verifier.aggregate(scale) for scale in args.scales]
    expected_ours = 50 * len(args.scales)
    missing_ours = sum(row["method"] == "ours" for row in verifier.missing)
    require(verifier.counts["ours"] + missing_ours == expected_ours, "incorrect ours count")
    require(sum(verifier.counts.values()) + len(verifier.missing) == 3*expected_ours,
            "incorrect total count")
    complete = not verifier.missing and set(args.scales) == {2, 3, 4}
    report = dict(passed=True, complete=complete, selected_scales_complete=not verifier.missing,
        scales=args.scales, expected_new_runs=150, expected_selected_new_runs=expected_ours,
        records=verifier.counts["ours"], missing_new_runs=missing_ours,
        reused_control_records=verifier.counts["aa"] + verifier.counts["grid"],
        counts=verifier.counts, missing_records=verifier.missing,
        unique_model_prediction_pairs=len(verifier.checked),
        crop_coordinate_comparisons=verifier.coordinate_comparisons,
        max_recomputed_psnr_difference_db=verifier.max_error,
        anchor_caveat=ANCHOR_CAVEAT, grid_caveat=GRID_CAVEAT,
        source_results_sha256=verifier.hashes,
        panels=[dict(scale=s["scale"], complete=s["complete"],
                     missing_records=s["missing_records"]) for s in summaries])
    suffix = "complete" if complete else "partial"
    save_json(ROOT / f"verification_literal_projected_{suffix}.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
