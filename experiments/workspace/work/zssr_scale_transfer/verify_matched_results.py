"""Recompute all matched-panel scores from saved float predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiment import save_json
from figures.gen_fig_matched_modes import IMAGES, METHODS, MODES, ROOT, source
from verify_saved_results import score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    cache, checked, hashes = {}, set(), {}
    count, missing, max_error = 0, 0, 0.
    for mode in MODES:
        for scale in (2, 3, 4):
            data = ROOT / ("data" if scale == 2 else f"data_x{scale}")
            for image in IMAGES:
                truth = np.load(data / f"{image}_hr.npy", allow_pickle=False)
                assert np.isfinite(truth).all()
                for m in range(1, 11):
                    for method in METHODS:
                        path, key = source(mode, scale, image, method, m)
                        if not path.exists() and args.allow_partial:
                            missing += 1
                            continue
                        folder = path.parent
                        if path not in cache:
                            raw = path.read_bytes()
                            result = json.loads(raw)
                            rows = {(r["m"], r["method"]): r for r in result["results"]}
                            assert len(rows) == len(result["results"])
                            cache[path] = rows
                            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
                            base = np.load(folder / "bicubic_baseline.npy", allow_pickle=False)
                            assert base.shape == truth.shape and np.isfinite(base).all()
                            for metric, value in score(base, truth, scale).items():
                                error = abs(value-result["bicubic"][metric])
                                assert error < 1e-7, (folder, "baseline", metric, error)
                                max_error = max(max_error, error)
                        if (m, key) not in cache[path] and args.allow_partial:
                            missing += 1
                            continue
                        row = cache[path][m, key]
                        artifact = key
                        if row.get("reused_identical_anchor_run"):
                            artifact = "consistent_literal" if mode == "literal" else "consistent"
                        stem = folder / f"m{m}_{artifact}"
                        prediction = np.load(stem.with_suffix(".npy"), allow_pickle=False)
                        assert prediction.shape == truth.shape and np.isfinite(prediction).all(), stem
                        for metric, value in score(prediction, truth, scale).items():
                            error = abs(value-row[metric])
                            assert error < 1e-7, (stem, metric, error)
                            max_error = max(max_error, error)
                        if stem not in checked:
                            state = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
                            assert sum(t.numel() for t in state.values()) == 225091, stem
                            assert all(torch.isfinite(t).all().item() for t in state.values()), stem
                            checked.add(stem)
                        count += 1
                print(json.dumps(dict(event="verified", mode=mode, scale=scale,
                                      image=image, records_so_far=count)), flush=True)
    assert count+missing == 900
    if not args.allow_partial:
        assert count == 900 and len(checked) == 870
    report = dict(passed=True, complete=missing == 0, records=count, missing=missing,
                  unique_model_prediction_pairs=len(checked),
                  max_recomputed_psnr_difference_db=max_error,
                  source_results_sha256=hashes)
    name = "verification_matched_complete.json" if not missing else "verification_matched_partial.json"
    save_json(ROOT / name, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
