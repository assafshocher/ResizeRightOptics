"""Fixed Gaussian pilot; reuses the existing training/inference functions."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "runs/gaussian_baby_x2_v1"
METHODS = ("gaussian_consistent", "gaussian_aa")
STEPS = list(range(1, 11))
BASE_SIGMAS = (.5, 1.)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    if options.launch:
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        path = logs / f"gaussian_launch_{time.time_ns()}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve())]
        if options.resume:
            command.append("--resume")
        with path.open("xb") as output:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        print(json.dumps(dict(pid=proc.pid, log=str(path))))
        return

    import numpy as np
    import torch
    import experiment as exp
    from gaussian_kernels import gaussian_kernel
    from kernels import CachedResize

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    FOLDER.mkdir(parents=True, exist_ok=options.resume)
    lock = (FOLDER / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scientific_sources = ["experiment.py", "kernels.py", "matched_kernels.py",
                          "gaussian_kernels.py", "check_gaussian_kernels.py",
                          "gaussian_pilot.py", "GAUSSIAN_PILOT_PROTOCOL.md",
                          "figures/gen_fig_gaussian_pilot.py"]
    hashes = {n: digest(ROOT/n) for n in scientific_sources}
    args = SimpleNamespace(image="baby", scale=2, m=STEPS, steps=10000, batch=1,
        crop=64, crop_margin=16, width=64, depth=8, lr=1e-4, radius=64,
        seed=0, residual=True, device="cuda")
    config = dict(vars(args), base_sigmas=BASE_SIGMAS, truncate=4.,
                  methods=METHODS, source_sha256=hashes,
                  HR_sha256=digest(ROOT/"data/baby_hr.npy"),
                  qualification="Continuous Gaussian law; truncated spatial sampling is approximate; no frequency cutoff.")
    # Round-trip JSON canonicalizes tuples for strict resume comparisons.
    config = json.loads(json.dumps(config))
    if options.resume:
        assert json.loads((FOLDER/"config.json").read_text()) == config
    else:
        exp.save_json(FOLDER/"config.json", config)
    status = dict(state="preflight", pid=__import__("os").getpid(), started_utc=now(),
                  completed=[], current=None, config=config)
    def update(**fields):
        status.update(fields, updated_utc=now())
        exp.save_json(FOLDER/"status.json", status)
    update()
    original_degradation = exp.degradation
    try:
        subprocess.run([sys.executable, "check_gaussian_kernels.py", "--output",
                        "gaussian_kernel_checks.json"], cwd=ROOT, check=True)
        checks = json.loads((ROOT/"gaussian_kernel_checks.json").read_text())
        assert checks["passed"] and checks["cached_resize_checked"] and not checks["numpy_only"]
        exp.save_json(FOLDER/"kernel_checks.json", checks)
        truth = exp.load_image(ROOT/"data/baby_hr.npy", "cuda")
        target_shape = truth.shape[-2:]
        reports = []
        for base_sigma in BASE_SIGMAS:
            tag = f"base_sigma_{base_sigma:g}"
            folder = FOLDER/tag
            folder.mkdir(exist_ok=options.resume)
            def degrade(method, factor, radius, anchor=2):
                return gaussian_kernel(method, factor, anchor, base_sigma), False
            exp.degradation = degrade
            anchor_kernel, _ = degrade(METHODS[0], 2, 64)
            anchor = CachedResize(target_shape, .5, method=anchor_kernel,
                                  antialiasing=False, device="cuda", weight_device="cpu")
            observed = anchor(truth)
            exp.save_output(folder/"input", observed)
            baseline = CachedResize(observed.shape[-2:], 2, target_shape,
                                    device="cuda", weight_device="cpu")(observed)
            baseline_scores = exp.metrics(baseline, truth, shave=2)
            exp.save_output(folder/"bicubic_baseline", baseline)
            records = (json.loads((folder/"results.json").read_text())["results"]
                       if options.resume and (folder/"results.json").exists() else [])
            done = {(r["m"], r["method"]) for r in records}
            for m in STEPS:
                for method in METHODS:
                    if (m, method) in done:
                        continue
                    assert hashes == {n:digest(ROOT/n) for n in scientific_sources}
                    update(state="training", current=dict(base_sigma=base_sigma, m=m, method=method))
                    if m == 1 and method == METHODS[1]:
                        shared = next(r for r in records if r["m"]==1 and r["method"]==METHODS[0])
                        row = dict(shared, method=method, reused_identical_anchor_run=True)
                    else:
                        model, training = exp.train(observed, method, m, args)
                        prediction, sizes = exp.infer(model, observed, m, args, target_shape)
                        scores = exp.metrics(prediction, truth, shave=2)
                        stem = f"m{m}_{method}"
                        exp.save_output(folder/stem, prediction)
                        exp.durable_save(folder/f"{stem}.pt", lambda stream:torch.save(model.state_dict(),stream))
                        row = dict(image="baby", m=m, method=method, base_sigma=base_sigma,
                            anchor_sigma=2*base_sigma, elementary_scale=2**(1/m),
                            inference_shapes=sizes, artifact_stem=stem, **training, **scores)
                        del model
                    records.append(row)
                    exp.save_json(folder/"results.json", dict(config=config, base_sigma=base_sigma,
                                  bicubic=baseline_scores, results=records))
                    print(json.dumps(dict(event="result", base_sigma=base_sigma,
                                          m=m, method=method, psnr_y=row["psnr_y"])), flush=True)
            assert {(r["m"],r["method"]) for r in records} == {(m,k) for m in STEPS for k in METHODS}
            by_key = {(r["m"],r["method"]):r for r in records}
            for m in STEPS:
                assert by_key[m,METHODS[0]]["coordinate_sha256"] == by_key[m,METHODS[1]]["coordinate_sha256"]
            checked = set()
            maximum_error = 0.
            for row in records:
                stem = folder/row["artifact_stem"]
                saved = exp.load_image(stem.with_suffix(".npy"), "cuda")
                assert saved.shape==truth.shape and torch.isfinite(saved).all()
                remeasured = exp.metrics(saved, truth, shave=2)
                for key,val in remeasured.items():
                    error = abs(val-row[key]); maximum_error=max(maximum_error,error)
                    assert error<1e-8
                if stem not in checked:
                    state = torch.load(stem.with_suffix(".pt"),map_location="cpu",weights_only=True)
                    assert all(torch.isfinite(x).all() for x in state.values())
                    assert sum(x.numel() for x in state.values())==225091
                    checked.add(stem)
            report = dict(base_sigma=base_sigma, anchor_sigma=2*base_sigma,
                psnr_y={method:[by_key[m,method]["psnr_y"] for m in STEPS] for method in METHODS},
                baseline_y=baseline_scores["psnr_y"], checkpoints_verified=len(checked),
                max_recomputed_psnr_difference_db=maximum_error)
            report["delta_ours_minus_AA"]=[a-b for a,b in zip(report["psnr_y"][METHODS[0]],report["psnr_y"][METHODS[1]])]
            reports.append(report)
            exp.save_json(folder/"verified.json", report)
            status["completed"].append(tag)
            update(state="verified_width", current=None)
        summary = dict(passed=True, complete=True, image="baby", scale=2, m=STEPS,
                       seed=0, config=config, per_width=reports,
                       qualification="One image, one seed; spatially truncated Gaussian; not exact sampled composition.")
        assert digest(ROOT/"data/baby_hr.npy")==config["HR_sha256"]
        update(state="plotting", current=None)
        exp.save_json(FOLDER/"plot_input.json", summary)
        subprocess.run([sys.executable,"figures/gen_fig_gaussian_pilot.py",
                        "--input",str(FOLDER/"plot_input.json")],cwd=ROOT,check=True)
        exp.save_json(FOLDER/"summary.json", summary)
        update(state="complete", current=None, finished_utc=now())
    except BaseException as error:
        update(state="failed", error=repr(error), stopped_utc=now())
        raise
    finally:
        exp.degradation = original_degradation


if __name__ == "__main__":
    main()
