"""Missing-only, checkpointed matched-anchor suite, independent of SSH.

Old band-limited ours/AA results remain immutable and will be referenced by
the new figure. No outcome-based restarts, no changes to system settings.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment import save_json

ROOT = Path(__file__).parent
IMAGES = ["baby", "bird", "butterfly", "head", "woman"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["bandlimited", "literal"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--after-bandlimited", action="store_true")
    args = p.parse_args()
    suite = ROOT / "runs" / f"matched_{args.mode}_suite_v1"
    suite.mkdir(exist_ok=args.resume)
    lock = (suite / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = suite / "status.json"
    status = dict(mode=args.mode, state="preflight", runner_pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  affinity=sorted(os.sched_getaffinity(0)), completed=[])
    methods = (["aa_grid_bandlimited"] if args.mode == "bandlimited" else
               ["consistent_literal", "aa", "aa_grid_literal"])
    save_json(status_path, status)
    if args.after_bandlimited:
        if args.mode != "literal":
            raise ValueError("Only literal mode can wait behind band-limited mode")
        band_status = ROOT / "runs" / "matched_bandlimited_suite_v1" / "status.json"
        status.update(state="queued_after_bandlimited")
        save_json(status_path, status)
        while True:
            prior = json.loads(band_status.read_text())
            if prior["state"] == "complete":
                break
            if prior["state"] == "failed":
                raise RuntimeError("The preceding band-limited job failed")
            if prior["boot_id"] != status["boot_id"]:
                raise RuntimeError("The preceding job belongs to a different boot")
            os.kill(prior["runner_pid"], 0)
            time.sleep(15)
    for check_name in ("matched_grid_checks.json", "literal_family_checks.json"):
        check = json.loads((ROOT / check_name).read_text())
        assert check["passed"], check_name
    for scale in (2, 3, 4):
        for image in IMAGES:
            run = f"matched_{args.mode}_x{scale}_v1"
            name = f"{run}/{image}"
            folder = ROOT / "runs" / run / f"{image}_seed0"
            status.update(state="training", current=name,
                          updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(status_path, status)
            flags = ["--resume"] if folder.exists() else []
            if folder.exists() and not args.resume:
                raise FileExistsError(folder)
            command = [sys.executable, "-u", "experiment.py", "--image", image,
                       "--scale", str(scale), "--radius", str(32*scale),
                       "--m", *map(str, range(1, 11)), "--methods", *methods,
                       "--input-kind", args.mode, "--steps", "10000", "--lr", ".0001",
                       "--residual", "--crop-margin", "16", "--run-name", run, *flags]
            print(json.dumps({"event": "matched_start", "name": name}), flush=True)
            code = subprocess.run(command, cwd=ROOT).returncode
            if code:
                status.update(state="failed", returncode=code,
                              stopped_utc=datetime.now(timezone.utc).isoformat())
                save_json(status_path, status)
                raise SystemExit(code)
            result = json.loads((folder / "results.json").read_text())
            keys = {(r["m"], r["method"]) for r in result["results"]}
            assert keys == {(m, method) for m in range(1, 11) for method in methods}
            status["completed"].append(name)
            save_json(status_path, status)
        # Each output mean requires all five images; incomplete panels are
        # explicitly marked and never get partial-image averages.
        code = subprocess.run([sys.executable, "figures/gen_fig_matched_modes.py",
                               "--allow-partial"], cwd=ROOT).returncode
        if code:
            status.update(state="failed", returncode=code, current="plotting")
            save_json(status_path, status)
            raise SystemExit(code)
    if args.mode == "literal":
        for command in ([sys.executable, "verify_matched_results.py"],
                        [sys.executable, "figures/gen_fig_matched_modes.py"]):
            code = subprocess.run(command, cwd=ROOT).returncode
            if code:
                status.update(state="failed", returncode=code, current=command[-1])
                save_json(status_path, status)
                raise SystemExit(code)
    status.update(state="complete", current=None,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    save_json(status_path, status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
