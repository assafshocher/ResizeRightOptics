"""Finish the fixed x2 residual protocol, sequentially on the remaining Set5 images.

Baby is already complete and is not reselected or rerun. No x3 jobs or no-AA
arms are launched here. Stop on any experiment failure, without auto-retrying.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parent
IMAGES = ["bird", "butterfly", "head", "woman"]
RUN_NAME = "first_image_residual_v1"


def main():
    status_path = ROOT / "runs" / RUN_NAME / "set5_batch_status.json"
    status = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "images": IMAGES, "completed": [], "status": "running",
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    status_path.write_text(json.dumps(status, indent=2))
    for name in IMAGES:
        status["current"] = name
        status_path.write_text(json.dumps(status, indent=2))
        command = [sys.executable, "-u", "experiment.py", "--image", name,
                   "--scale", "2", "--radius", "64", "--m", "1", "2", "3", "4", "5",
                   "--methods", "consistent", "aa_bandlimited", "aa",
                   "--steps", "10000", "--lr", "0.0001", "--residual",
                   "--run-name", RUN_NAME]
        print(json.dumps({"event": "batch_image_start", "image": name}), flush=True)
        code = subprocess.run(command, cwd=ROOT).returncode
        if code:
            status.update(status="failed", returncode=code)
            status_path.write_text(json.dumps(status, indent=2))
            raise SystemExit(code)
        status["completed"].append(name)
        status_path.write_text(json.dumps(status, indent=2))
    status.update(status="complete", current=None,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps({"event": "set5_batch_complete", **status}), flush=True)


if __name__ == "__main__":
    main()
