"""Detached, locked 150-model rerun; preserves every previous experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiment import save_json

ROOT = Path(__file__).resolve().parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    suite = ROOT / "runs" / "literal_projected_suite_v1"
    suite.mkdir(exist_ok=args.resume)
    lock = (suite / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = suite / "status.json"
    status = dict(state="preflight", runner_pid=os.getpid(), started_utc=now(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        affinity=sorted(os.sched_getaffinity(0)), completed=[], current=None,
        total_models=150, input_kind="literal", method="consistent",
        source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
            for n in ("experiment.py", "kernels.py", "run_literal_projected_suite.py",
                      "test_literal_projected.py", "LITERAL_PROJECTED_PROTOCOL.md")})
    save_json(status_path, status)

    def call(command):
        code = subprocess.run([sys.executable, "-u", *command], cwd=ROOT).returncode
        if code:
            raise RuntimeError(f"Command failed ({code}): {command}")

    try:
        call(["test_literal_projected.py"])
        assert json.loads((ROOT/"literal_projected_checks.json").read_text())["passed"]
        for scale in (2, 3, 4):
            for image in IMAGES:
                run = f"literal_projected_x{scale}_v1"
                name = f"{run}/{image}"
                folder = ROOT / "runs" / run / f"{image}_seed0"
                if folder.exists() and not args.resume:
                    raise FileExistsError(folder)
                status.update(state="training", current=name, updated_utc=now())
                save_json(status_path, status)
                call(["experiment.py", "--image", image, "--scale", str(scale),
                    "--radius", str(32*scale), "--m", *map(str,range(1,11)),
                    "--methods", "consistent", "--input-kind", "literal",
                    "--steps", "10000", "--lr", ".0001", "--residual",
                    "--crop-margin", "16", "--run-name", run,
                    *(["--resume"] if folder.exists() else [])])
                result = json.loads((folder/"results.json").read_text())
                assert {(r["m"],r["method"]) for r in result["results"]} == {
                    (m,"consistent") for m in range(1,11)}
                assert not any(r.get("reused_identical_anchor_run") for r in result["results"])
                status["completed"].append(name)
                if scale == 2 and image == "baby":
                    prior = json.loads((ROOT/"runs"/"matched_literal_x2_v1"/"baby_seed0"/"results.json").read_text())
                    lookup = {(r["m"],r["method"]): r for r in prior["results"]}
                    milestone = dict(image=image, scale=scale, finished_utc=now(),
                        qualification="Single-image early check, not a Set5 average; no tuning or stopping by score.",
                        rows=[dict(m=r["m"], ours=r["psnr_y"],
                            aa=lookup[r["m"],"aa"]["psnr_y"],
                            failed_extension=lookup[r["m"],"consistent_literal"]["psnr_y"])
                            for r in result["results"]], baseline=prior["bicubic"]["psnr_y"])
                    save_json(suite/"first_image_check.json", milestone)
                    status["first_image_ready"] = True
                    print(json.dumps({"event":"first_image_check",**milestone}), flush=True)
                save_json(status_path, status)
            status.update(state="verifying_and_plotting", current=f"x{scale}")
            save_json(status_path, status)
            call(["verify_literal_projected.py", "--allow-partial"])
            call(["figures/gen_fig_literal_projected.py", "--allow-partial"])
        status.update(state="final_verification", current=None)
        save_json(status_path, status)
        call(["verify_literal_projected.py"])
        call(["figures/gen_fig_literal_projected.py"])
        status.update(state="complete", finished_utc=now(), current=None)
        save_json(status_path, status)
    except BaseException as error:
        status.update(state="failed", error=repr(error), stopped_utc=now())
        save_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
