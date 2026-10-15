"""Detached one-image diagnostic pilot; never overwrites previous experiments."""
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
METHOD = "consistent_untapered"
RUN = "untapered_baby_x2_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def verify():
    folder = ROOT/"runs"/RUN/"baby_seed0"
    current = json.loads((folder/"results.json").read_text())
    aa = json.loads((ROOT/"runs/matched_literal_x2_v1/baby_seed0/results.json").read_text())
    tapered = json.loads((ROOT/"runs/literal_projected_x2_v1/baby_seed0/results.json").read_text())
    config = current["config"]
    for key, value in dict(image="baby",scale=2,m=list(range(1,11)),methods=[METHOD],
            steps=10000,lr=1e-4,batch=1,crop=64,crop_margin=16,width=64,depth=8,
            seed=0,residual=True,radius=64,input_kind="literal",rr_weight_device="cpu").items():
        assert config[key] == value, (key,config[key],value)
    assert not config["untapered_construction"]["smooth_window"]
    assert not config["untapered_construction"]["ridge"]
    rows = {(r["m"],r["method"]):r for r in current["results"]}
    assert set(rows) == {(m,METHOD) for m in range(1,11)}
    old = {(r["m"],r["method"]):r for r in aa["results"]}
    smooth = {r["m"]:r for r in tapered["results"]}
    def tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1)[None]
    truth = tensor(np.load(ROOT/"data/baby_hr.npy"))
    baseline = tensor(np.load(folder/"bicubic_baseline.npy"))
    baseline_scores = metrics(baseline,truth,shave=2)
    for name,value in baseline_scores.items():
        assert abs(value-current["bicubic"][name]) < 1e-8
        assert abs(value-aa["bicubic"][name]) < 1e-8
    expected = PlainCNN(64,8,True).state_dict()
    maximum_error = 0.
    compared = []
    for m in range(1,11):
        row = rows[m,METHOD]
        assert not row.get("reused_identical_anchor_run")
        assert row["losses"][-1]["step"] == 10000
        assert all(np.isfinite(x["l1"]) for x in row["losses"])
        assert row["coordinate_sha256"] == old[m,"aa"]["coordinate_sha256"] == smooth[m]["coordinate_sha256"]
        stem = folder/f"m{m}_{METHOD}"
        prediction = np.load(stem.with_suffix(".npy"))
        assert np.isfinite(prediction).all()
        measured = metrics(tensor(prediction),truth,shave=2)
        for name,value in measured.items():
            difference = abs(value-row[name])
            assert difference < 1e-8, (m,name,difference)
            maximum_error = max(maximum_error,difference)
        state = torch.load(stem.with_suffix(".pt"),map_location="cpu",weights_only=True)
        assert set(state) == set(expected)
        assert all(x.shape == expected[k].shape and torch.isfinite(x).all() for k,x in state.items())
        assert sum(x.numel() for x in state.values()) == 225091
        compared.append(dict(m=m,untapered=row["psnr_y"],aa=old[m,"aa"]["psnr_y"],
            tapered=smooth[m]["psnr_y"],failed_extension=old[m,"consistent_literal"]["psnr_y"],
            untapered_minus_aa=row["psnr_y"]-old[m,"aa"]["psnr_y"],
            untapered_minus_tapered=row["psnr_y"]-smooth[m]["psnr_y"]))
    report = dict(passed=True,image="baby",scale=2,models=10,rows=compared,
        baseline=baseline_scores["psnr_y"],max_recomputed_psnr_difference_db=maximum_error,
        qualification="One-image fixed pilot, not Set5 mean. No taper or ridge; hard output-Nyquist integration and finite spatial support remain.")
    save_json(ROOT/"untapered_pilot_comparison.json",report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(2)
    folder=ROOT/"runs/untapered_pilot_suite_v1"
    folder.mkdir(exist_ok=args.resume)
    lock=(folder/"runner.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(state="preflight",started_utc=now(),pid=os.getpid(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        affinity=sorted(os.sched_getaffinity(0)),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("experiment.py","kernels.py","untapered_kernels.py","test_untapered_pilot.py",
             "run_untapered_pilot.py","UNTAPERED_PILOT_PROTOCOL.md")})
    def save(): save_json(folder/"status.json",status)
    def call(command):
        subprocess.run([sys.executable,"-u",*command],cwd=ROOT,check=True)
    save()
    try:
        call(["test_untapered_pilot.py"])
        assert json.loads((ROOT/"untapered_pilot_checks.json").read_text())["passed"]
        status.update(state="training",updated_utc=now());save()
        run_folder=ROOT/"runs"/RUN/"baby_seed0"
        call(["experiment.py","--image","baby","--scale","2","--radius","64",
              "--m",*map(str,range(1,11)),"--methods",METHOD,"--input-kind","literal",
              "--steps","10000","--lr",".0001","--residual","--crop-margin","16",
              "--run-name",RUN,*(["--resume"] if args.resume and run_folder.exists() else [])])
        status.update(state="verifying",updated_utc=now());save()
        report=verify()
        status.update(state="complete",finished_utc=now(),models=10);save()
        print(json.dumps(report),flush=True)
    except BaseException as error:
        status.update(state="failed",error=repr(error),stopped_utc=now());save()
        raise


if __name__ == "__main__":
    main()
