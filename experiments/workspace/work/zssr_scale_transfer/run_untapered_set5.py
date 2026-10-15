"""Extend the unchanged untapered x2 pilot to Set5; retain baby in place."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from experiment import PlainCNN, metrics, save_json

ROOT = Path(__file__).resolve().parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]
STEPS = list(range(1, 11))
METHOD = "consistent_untapered"
RUN = "untapered_set5_x2_v1"
COMMON = dict(scale=2, m=STEPS, steps=10000, lr=1e-4, batch=1, crop=64,
              crop_margin=16, width=64, depth=8, seed=0, residual=True,
              radius=64, input_kind="literal", rr_weight_device="cpu")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_folder(image, scale=2):
    if scale != 2:
        return ROOT / "runs" / f"untapered_set5_x{scale}_v1" / f"{image}_seed0"
    run = "untapered_baby_x2_v1" if image == "baby" else RUN
    return ROOT / "runs" / run / f"{image}_seed0"


def tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]


def verify_image(image, scale=2):
    sources = {
        "untapered": (new_folder(image, scale), METHOD),
        "tapered": (ROOT/"runs"/f"literal_projected_x{scale}_v1"/f"{image}_seed0", "consistent"),
        "aa": (ROOT/"runs"/f"matched_literal_x{scale}_v1"/f"{image}_seed0", "aa"),
        "grid": (ROOT/"runs"/f"matched_literal_x{scale}_v1"/f"{image}_seed0", "aa_grid_literal"),
    }
    truth = np.load(ROOT/("data" if scale == 2 else f"data_x{scale}")/f"{image}_hr.npy", allow_pickle=False)
    truth_tensor = tensor(truth)
    shapes = {k: v.shape for k, v in PlainCNN(64, 8, True).state_dict().items()}
    by_method, hashes, base_scores, max_error = {}, {}, None, 0.
    baseline_array = None
    checked = set()
    for name, (folder, key) in sources.items():
        result = json.loads((folder/"results.json").read_text())
        config = result["config"]
        for field, value in (COMMON | {"image": image, "scale": scale, "radius": 32*scale,
                                      "rr_weight_device": "cpu" if scale == 2 else "cuda"}).items():
            assert config[field] == value, (image, name, field, config[field], value)
        if name == "untapered":
            assert config["methods"] == [METHOD]
            construction = config["untapered_construction"]
            assert not construction["smooth_window"] and not construction["ridge"]
            assert construction["density"] == 512 and construction["quadrature"] == 4097
        baseline = np.load(folder/"bicubic_baseline.npy", allow_pickle=False)
        assert baseline.shape == truth.shape and np.isfinite(baseline).all()
        measured_base = metrics(tensor(baseline), truth_tensor, shave=scale)
        for metric, value in measured_base.items():
            assert abs(value-result["bicubic"][metric]) < 1e-8
        if base_scores is None:
            base_scores, baseline_array = measured_base, baseline
        else:
            assert np.array_equal(baseline, baseline_array), (image, name, "baseline pixels")
        rows = {(r["m"], r["method"]): r for r in result["results"]}
        assert len(rows) == len(result["results"])
        if name == "untapered":
            assert set(rows) == {(m, METHOD) for m in STEPS}
        values, coordinates = [], []
        for m in STEPS:
            row = rows[m, key]
            assert row["image"] == image and row["parameters"] == 225091
            assert abs(row["elementary_scale"]-scale**(1/m)) < 1e-12
            assert row["losses"][-1]["step"] == 10000
            assert all(np.isfinite(x["l1"]) for x in row["losses"])
            reused = bool(row.get("reused_identical_anchor_run"))
            assert not reused or (name == "aa" and m == 1)
            artifact = "consistent_literal" if reused else key
            stem = folder/f"m{m}_{artifact}"
            prediction = np.load(stem.with_suffix(".npy"), allow_pickle=False)
            assert prediction.shape == truth.shape and np.isfinite(prediction).all()
            measured = metrics(tensor(prediction), truth_tensor, shave=scale)
            for metric, value in measured.items():
                error = abs(value-row[metric])
                assert error < 1e-8, (image, name, m, metric, error)
                max_error = max(max_error, error)
            if stem not in checked:
                state = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
                assert set(state) == set(shapes)
                assert all(x.shape == shapes[k] and torch.isfinite(x).all() for k, x in state.items())
                assert sum(x.numel() for x in state.values()) == 225091
                checked.add(stem)
            if name == "untapered":
                execution = json.loads((folder/row["execution_provenance"]).read_text())
                for filename, sha in execution["source_sha256"].items():
                    assert digest(ROOT/filename) == sha, (image, "source changed", filename)
            values.append(measured["psnr_y"])
            coordinates.append(row["coordinate_sha256"])
        by_method[name] = values
        hashes[name] = coordinates
    assert all(hashes[name] == hashes["untapered"] for name in hashes), (image, "crop mismatch")
    return dict(image=image, psnr_y=by_method, baseline_y=base_scores["psnr_y"],
                coordinate_comparisons=30, checkpoints_verified=len(checked),
                max_recomputed_psnr_difference_db=max_error,
                sources={k: str(v[0].relative_to(ROOT)) for k, v in sources.items()})


def aggregate(reports, scale=2):
    assert [r["image"] for r in reports] == IMAGES
    methods = ["untapered", "tapered", "aa", "grid"]
    means = {name: np.mean([r["psnr_y"][name] for r in reports], axis=0).tolist()
             for name in methods}
    return dict(passed=True, complete=True, scale=scale, images=IMAGES, m=STEPS, seed=0,
        mean_psnr_y=means, per_image=reports,
        baseline_mean_y=float(np.mean([r["baseline_y"] for r in reports])),
        delta_untapered_minus_aa=(np.array(means["untapered"])-means["aa"]).tolist(),
        delta_untapered_minus_tapered=(np.array(means["untapered"])-means["tapered"]).tolist(),
        max_recomputed_psnr_difference_db=max(r["max_recomputed_psnr_difference_db"] for r in reports),
        checkpoints_verified=sum(r["checkpoints_verified"] for r in reports),
        coordinate_comparisons=sum(r["coordinate_comparisons"] for r in reports),
        aggregation="Arithmetic mean of five image-level Y-PSNR scores; one seed; no selection.",
        qualification=f"Untapered removes smooth window and ridge, not output-Nyquist boundary or finite radius{32*scale} approximation. Literal AA anchor is approximate.",
        grid_qualification="Matched AA filter with PyTorch-style downsampling grid; shared RR upsampling, not stock torch.interpolate end to end.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    suite = ROOT/"runs/untapered_set5_suite_v1"
    suite.mkdir(exist_ok=args.resume)
    lock = (suite/"runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(state="verifying_baby", started_utc=now(), pid=os.getpid(),
        total_models=50, new_models=40, reused_baby_models=10, completed=[], current="baby",
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        affinity=sorted(os.sched_getaffinity(0)),
        source_sha256={n: digest(ROOT/n) for n in
            ("experiment.py", "kernels.py", "untapered_kernels.py", "run_untapered_set5.py", "UNTAPERED_SET5_PROTOCOL.md")},
        data_sha256={str(p.relative_to(ROOT)): digest(p) for image in IMAGES
                     for p in (ROOT/"data"/f"{image}_lr.npy", ROOT/"data"/f"{image}_hr.npy")})
    def save():
        status["updated_utc"] = now()
        save_json(suite/"status.json", status)
    save()
    reports = []
    try:
        assert json.loads((ROOT/"untapered_pilot_checks.json").read_text())["passed"]
        for image in IMAGES:
            folder = new_folder(image)
            if image != "baby":
                if folder.exists() and not args.resume:
                    raise FileExistsError(folder)
                status.update(state="training", current=image); save()
                command = [sys.executable, "-u", "experiment.py", "--image", image,
                    "--scale", "2", "--radius", "64", "--m", *map(str, STEPS),
                    "--methods", METHOD, "--input-kind", "literal", "--steps", "10000",
                    "--lr", ".0001", "--residual", "--crop-margin", "16", "--run-name", RUN]
                if folder.exists():
                    command.append("--resume")
                subprocess.run(command, cwd=ROOT, check=True)
            status.update(state="verifying", current=image); save()
            report = verify_image(image)
            reports.append(report)
            save_json(suite/f"verified_{image}.json", report)
            status["completed"].append(image)
            save()
            print(json.dumps(dict(event="verified_image", **report)), flush=True)
        summary = aggregate(reports)
        save_json(ROOT/"untapered_set5_x2_comparison.json", summary)
        status.update(state="complete", finished_utc=now(), current=None); save()
        print(json.dumps(dict(event="complete", **summary)), flush=True)
    except BaseException as error:
        status.update(state="failed", error=repr(error), stopped_utc=now()); save()
        raise


if __name__ == "__main__":
    main()
