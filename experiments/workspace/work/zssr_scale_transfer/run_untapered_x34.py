"""User-authorized approximate-anchor x3/x4 extension; detached and resumable."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from run_untapered_set5 import ROOT, IMAGES, STEPS, METHOD, now, digest, verify_image, aggregate
from experiment import save_json
from figures.gen_fig_untapered_set5 import plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = ROOT/"runs/untapered_x34_suite_v1"
    folder.mkdir(exist_ok=args.resume)
    lock = (folder/"runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    core = ["experiment.py", "kernels.py", "matched_kernels.py", "literal_spectrum.py", "untapered_kernels.py"]
    frozen = {n: digest(ROOT/n) for n in core}
    state = dict(state="preflight", pid=os.getpid(), started_utc=now(), total_models=100,
        completed=[], current=None, milestones=[], source_sha256=frozen,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        data_sha256={str(p.relative_to(ROOT)):digest(p) for s in (3,4) for im in IMAGES
                     for p in (ROOT/f"data_x{s}"/f"{im}_lr.npy",ROOT/f"data_x{s}"/f"{im}_hr.npy")},
        qualification="User explicitly accepts literal-input/retained-band anchor mismatch; no new kernel modification.")
    def save():
        state["updated_utc"] = now()
        save_json(folder/"status.json", state)
    save()
    try:
        for scale in (3,4):
            reports = []
            for image in IMAGES:
                assert frozen == {n:digest(ROOT/n) for n in core}
                run = f"untapered_set5_x{scale}_v1"
                target = ROOT/"runs"/run/f"{image}_seed0"
                if target.exists() and not args.resume:
                    raise FileExistsError(target)
                state.update(state="training",current=dict(scale=scale,image=image)); save()
                command = [sys.executable,"-u","experiment.py","--image",image,
                    "--scale",str(scale),"--radius",str(32*scale),"--m",*map(str,STEPS),
                    "--methods",METHOD,"--input-kind","literal","--steps","10000",
                    "--lr",".0001","--residual","--crop-margin","16","--run-name",run]
                if target.exists(): command.append("--resume")
                subprocess.run(command,cwd=ROOT,check=True)
                state.update(state="verifying");save()
                report = verify_image(image,scale)
                reports.append(report)
                save_json(folder/f"verified_x{scale}_{image}.json",report)
                state["completed"].append(dict(scale=scale,image=image))
                if image == "baby":
                    paths=plot(report["psnr_y"],report["baseline_y"],
                        ROOT/"figures"/f"untapered_x{scale}_first_image",True,scale,
                        f"Baby ×{scale} — first image only")
                    state["milestones"].append(dict(kind="first_image",scale=scale,
                        paths=[str(p.relative_to(ROOT)) for p in paths],ready_utc=now()))
                save()
            summary=aggregate(reports,scale)
            save_json(ROOT/f"untapered_set5_x{scale}_comparison.json",summary)
            paths=plot(summary["mean_psnr_y"],summary["baseline_mean_y"],ROOT/"figures",True,scale)
            state["milestones"].append(dict(kind="scale_complete",scale=scale,
                paths=[str(p.relative_to(ROOT)) for p in paths],ready_utc=now()))
            save()
        for name,sha in state["data_sha256"].items(): assert digest(ROOT/name)==sha
        state.update(state="complete",current=None,finished_utc=now());save()
    except BaseException as error:
        state.update(state="failed",error=repr(error),stopped_utc=now());save()
        raise


if __name__ == "__main__": main()
