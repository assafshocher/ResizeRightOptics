"""Compare the local backup to the independent server audit's SHA-256 manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = "runs/gaussian_sigma2_suite_v1"


def main():
    audit = json.loads((ROOT / "GAUSSIAN_SUITE_AUDIT.json").read_text())
    files = audit["files_sha256"]
    missing, changed = [], []
    for relative, expected in files.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != expected:
            changed.append(relative)
    expected_run = {p for p in files if p.startswith(RUN + "/")}
    actual_run = {str(p.relative_to(ROOT)) for p in (ROOT / RUN).rglob("*") if p.is_file()}
    extras = sorted(actual_run - expected_run)
    count_models = sum(p.endswith(".pt") for p in expected_run)
    count_arrays = sum(p.endswith(".npy") for p in expected_run)
    passed = not missing and not changed and not extras and count_models == 435 and count_arrays == 465
    report = dict(passed=passed, checked_files=len(files), run_files=len(expected_run),
                  model_files=count_models, image_arrays=count_arrays,
                  missing=missing, changed=changed, extra_run_files=extras,
                  audit_sha256=hashlib.sha256((ROOT / "GAUSSIAN_SUITE_AUDIT.json").read_bytes()).hexdigest())
    result = json.dumps(report, indent=2) + "\n"
    (ROOT / "GAUSSIAN_SUITE_BACKUP.json").write_text(result)
    print(result)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
