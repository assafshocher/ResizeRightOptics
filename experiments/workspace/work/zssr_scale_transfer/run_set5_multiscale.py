"""Fixed Set5 x2/x3/x4, m=1..10, four-arm suite; stop on any failure.

Reuse the completed x2 m=1..5 RR measurements by reference, never overwrite
them. No automatic retries, no PSNR-based choice, no system modifications.
"""
from datetime import datetime, timezone
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]
METHODS = ["consistent", "aa_bandlimited", "aa", "torch_aa"]
SUITE = ROOT / "runs" / "set5_multiscale_m1_10"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Explicit, one-time continuation; retain every completed measurement")
    args = parser.parse_args()
    SUITE.mkdir(exist_ok=args.resume)
    lock = (SUITE / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = SUITE / "status.json"
    if args.resume:
        previous_text = status_path.read_text()
        status = json.loads(previous_text)
        archive = SUITE / f"status_before_resume_{time.time_ns()}.json"
        with archive.open("x") as stream:
            stream.write(previous_text)
        status.update(resumed_utc=datetime.now(timezone.utc).isoformat(),
                      previous_boot_id=status["boot_id"], previous_status_file=archive.name,
                      status="resume_preflight", kernel=platform.release(),
                      boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    else:
        status = {"started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight", "completed": [], "scales": [2, 3, 4],
        "m": list(range(1, 11)), "images": IMAGES, "methods": METHODS,
        "kernel": platform.release(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "reused": "first_image_residual_v1: x2 m1..5, consistent/aa_bandlimited/aa",
        "protocol": "MULTISCALE_PROTOCOL.md"}
    status["runner_pid"] = os.getpid()
    status["cpu_affinity"] = sorted(os.sched_getaffinity(0))

    def save_status():
        # A reboot must not leave the prior valid status truncated.
        temporary = status_path.with_suffix(".json.tmp")
        with temporary.open("w") as stream:
            json.dump(status, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(status_path)

    def mark_complete(name):
        if name not in status["completed"]:
            status["completed"].append(name)
        save_status()

    def execute(name, arguments):
        status["current"] = name
        save_status()
        print(json.dumps({"event": "suite_start", "job": name}), flush=True)
        code = subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT).returncode
        if code:
            status.update(status="failed", returncode=code,
                          stopped_utc=datetime.now(timezone.utc).isoformat())
            save_status()
            raise SystemExit(code)
        mark_complete(name)

    # Refuse to reuse an incomplete old x2 image. Aggregator additionally checks
    # all optimizer/crop/initialization fields and actual crop-coordinate hashes.
    for image in IMAGES:
        path = ROOT / "runs" / "first_image_residual_v1" / f"{image}_seed0" / "results.json"
        old = json.loads(path.read_text())
        keys = {(r["m"], r["method"]) for r in old["results"]}
        assert all((m, method) in keys for m in range(1, 6) for method in METHODS[:3])

    if not (ROOT / "data_x4" / "manifest.json").exists():
        execute("prepare_x4", ["prepare.py", "--scale", "4", "--radius", "128"])
    for scale in (2, 3, 4):
        audit_path = ROOT / f"audit_x{scale}_radius{32*scale}_m10.json"
        if args.resume and audit_path.exists():
            audit = json.loads(audit_path.read_text())
            assert audit["support_gate_passed"] and audit["max_m"] == 10
            assert audit["scale"] == scale and audit["radius"] == 32*scale
            mark_complete(f"audit_x{scale}")
        else:
            execute(f"audit_x{scale}", ["audit.py", "--scale", str(scale),
                                       "--radius", str(32*scale), "--max-m", "10"])
    # Recheck the new boot's compute path; do not rerun any scored case.
    execute("synthetic_checks_resume" if args.resume else "synthetic_checks",
            ["test_multiscale.py"])
    status["status"] = "training"
    for scale in (2, 3, 4):
        for image in IMAGES:
            jobs = [(f"set5_x{scale}_m1_10", list(range(1, 11)), METHODS)]
            if scale == 2:
                jobs = [("set5_x2_m6_10", list(range(6, 11)), METHODS[:3]),
                        ("set5_x2_torch_m1_10", list(range(1, 11)), ["torch_aa"])]
            for run, ms, methods in jobs:
                name = f"{run}/{image}"
                folder = ROOT / "runs" / run / f"{image}_seed0"
                resume_flags = []
                if args.resume and folder.exists():
                    config = json.loads((folder / "config.json").read_text())
                    expected = {"scale": scale, "image": image, "seed": 0,
                        "m": ms, "methods": methods, "radius": 32*scale,
                        "steps": 10000, "lr": .0001, "residual": True,
                        "crop_margin": 16, "crop": 64, "batch": 1,
                        "width": 64, "depth": 8, "input_kind": "bandlimited"}
                    assert all(config[k] == v for k, v in expected.items()), name
                    rows = (json.loads((folder / "results.json").read_text())["results"]
                            if (folder / "results.json").exists() else [])
                    keys = {(r["m"], r["method"]) for r in rows}
                    planned = {(m, method) for m in ms for method in methods}
                    assert len(keys) == len(rows) and keys <= planned, name
                    assert all(math.isfinite(r["psnr_y"]) for r in rows), name
                    if keys == planned:
                        mark_complete(name)
                        print(json.dumps({"event": "reuse_complete", "job": name}), flush=True)
                        continue
                    # experiment.py enforces exact full-config equality and
                    # skips completed (m, method) records, without score selection.
                    resume_flags = ["--resume"]
                execute(name, ["experiment.py", "--image", image,
                    "--scale", str(scale), "--radius", str(32*scale),
                    "--m", *map(str, ms), "--methods", *methods,
                    "--steps", "10000", "--lr", "0.0001", "--residual",
                    "--crop-margin", "16", "--run-name", run, *resume_flags])
        execute(f"plots_x{scale}", ["figures/gen_fig_set5_multiscale.py",
                                    "--scales", *map(str, range(2, scale+1))])
        execute(f"plots_x{scale}_no_pytorch", ["figures/gen_fig_set5_multiscale.py",
                    "--scales", *map(str, range(2, scale+1)), "--without-pytorch"])
    status.update(status="complete", current=None,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    save_status()
    print(json.dumps({"event": "suite_complete", **status}), flush=True)


if __name__ == "__main__":
    main()
