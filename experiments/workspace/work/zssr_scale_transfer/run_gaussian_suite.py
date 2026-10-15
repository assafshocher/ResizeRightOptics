"""Detached, resumable sigma-2 Set5 suite using the unchanged ZSSR implementation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "runs/gaussian_sigma2_suite_v1"
PILOT = ROOT / "runs/gaussian_baby_x2_v1"
METHODS = ("gaussian_consistent", "gaussian_aa", "aa_grid_gaussian")
IMAGES = ("baby", "bird", "butterfly", "head", "woman")
SCALES = (2, 3, 4)
STEPS = list(range(1, 11))
SOURCES = ("experiment.py", "kernels.py", "matched_kernels.py", "gaussian_kernels.py",
           "check_gaussian_suite.py", "run_gaussian_suite.py",
           "GAUSSIAN_SUITE_PROTOCOL.md", "figures/gen_fig_gaussian_suite.py",
           "check_gaussian_kernels.py", "vendor/ResizeRight/resize_right.py",
           "vendor/ResizeRight/interp_methods.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(image, scale):
    return SimpleNamespace(image=image, scale=scale, m=STEPS, steps=10000, batch=1,
        crop=64, crop_margin=16, width=64, depth=8, lr=1e-4, radius=32*scale,
        seed=0, residual=True, device="cuda")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    if options.launch:
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        path = logs / f"gaussian_suite_launch_{time.time_ns()}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve())]
        if options.resume:
            command.append("--resume")
        with path.open("xb") as output:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        print(json.dumps(dict(pid=proc.pid, log=str(path))), flush=True)
        return

    import numpy as np
    import torch
    import experiment as exp
    from gaussian_kernels import gaussian_kernel
    from kernels import CachedResize
    from figures.gen_fig_gaussian_suite import plot_scale, plot_all

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    FOLDER.mkdir(parents=True, exist_ok=options.resume)
    lock = (FOLDER / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = {name:digest(ROOT/name) for name in SOURCES}
    data_hashes = {im:digest(ROOT/f"data/{im}_hr.npy") for im in IMAGES}
    config = json.loads(json.dumps(dict(
        images=IMAGES, scales=SCALES, m=STEPS, methods=METHODS, anchor_sigma=2.,
        truncate=4., seed=0, training_by_scale={s:vars(arguments("IMAGE",s)) for s in SCALES},
        source_sha256=hashes, HR_sha256=data_hashes,
        qualification="Spatially truncated Gaussian; no frequency cutoff; approximate sampled composition.")))
    if options.resume:
        assert json.loads((FOLDER/"config.json").read_text()) == config, "Resume configuration/source mismatch"
    else:
        exp.save_json(FOLDER/"config.json", config)
    previous = (json.loads((FOLDER/"status.json").read_text())
                if options.resume and (FOLDER/"status.json").exists() else {})
    if previous.get("state") == "complete":
        print("Suite already complete", flush=True)
        return
    status = dict(state="preflight", pid=os.getpid(), started_utc=previous.get("started_utc",now()),
                  resumed_utc=now() if options.resume else None, completed=previous.get("completed",[]),
                  milestones=previous.get("milestones",[]), current=None,
                  boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    def update(**fields):
        status.update(fields, updated_utc=now())
        exp.save_json(FOLDER/"status.json", status)
    def check_sources():
        assert hashes == {name:digest(ROOT/name) for name in SOURCES}, "Scientific sources changed during run"
    def array_matches(path, tensor):
        return np.array_equal(np.load(path),tensor.detach().cpu()[0].permute(1,2,0).numpy())
    def range_stats(tensor):
        return dict(output_min=float(tensor.min()), output_max=float(tensor.max()),
                    output_out_of_range_fraction=float(((tensor<0)|(tensor>1)).float().mean()))
    def verify_records(folder, records, truth, scale, complete=True):
        keyed = {(r["m"],r["method"]):r for r in records}
        assert len(keyed)==len(records), "Duplicate result"
        if complete:
            assert set(keyed)=={(m,k) for m in STEPS for k in METHODS}
        checked, max_error = set(), 0.
        for row in records:
            assert row["parameters"]==225091 and row["losses"][-1]["step"]==10000
            assert all(math.isfinite(v["l1"]) for v in row["losses"])
            stem = folder/row["artifact_stem"]
            saved = exp.load_image(stem.with_suffix(".npy"),"cuda")
            assert saved.shape==truth.shape and bool(torch.isfinite(saved).all())
            actual = exp.metrics(saved,truth,shave=scale)
            for key,value in actual.items():
                error=abs(value-row[key]); max_error=max(max_error,error)
                assert math.isfinite(value) and error<1e-8, (stem,key,value,row[key])
            row.update(range_stats(saved))
            if stem not in checked:
                state=torch.load(stem.with_suffix(".pt"),map_location="cpu",weights_only=True)
                reference=exp.PlainCNN(64,8,True).state_dict()
                assert state.keys()==reference.keys()
                assert all(state[k].shape==v.shape and bool(torch.isfinite(state[k]).all()) for k,v in reference.items())
                assert sum(x.numel() for x in state.values())==225091
                checked.add(stem)
        if complete:
            for m in STEPS:
                assert len({keyed[m,k]["coordinate_sha256"] for k in METHODS})==1
            assert keyed[1,METHODS[0]]["artifact_stem"]==keyed[1,METHODS[1]]["artifact_stem"]
        return len(checked),max_error
    def reuse_pilot(folder, observed, baseline, truth):
        pilot_config=json.loads((PILOT/"config.json").read_text())
        expected=vars(arguments("baby",2))
        assert all(pilot_config[k]==v for k,v in expected.items())
        for name in ("experiment.py","kernels.py","matched_kernels.py","gaussian_kernels.py"):
            assert pilot_config["source_sha256"][name]==hashes[name]
        assert pilot_config["HR_sha256"]==data_hashes["baby"]
        assert json.loads((PILOT/"summary.json").read_text())["complete"]
        source=PILOT/"base_sigma_1"
        assert array_matches(source/"input.npy",observed), "Pilot degradation mismatch"
        assert array_matches(source/"bicubic_baseline.npy",baseline), "Pilot baseline mismatch"
        original=json.loads((source/"results.json").read_text())["results"]
        assert {(r["m"],r["method"]) for r in original}=={(m,k) for m in STEPS for k in METHODS[:2]}
        copied=set(); result=[]
        for row in original:
            assert row["base_sigma"]==1. and row["anchor_sigma"]==2.
            stem=row["artifact_stem"]
            if stem not in copied:
                for ext in (".npy",".png",".pt"):
                    shutil.copy2(source/(stem+ext),folder/(stem+ext))
                copied.add(stem)
            result.append(dict(row,scale=2,reused_pilot=True,
                pilot_source=str(source),pilot_results_sha256=digest(source/"results.json")))
        verify_records(folder,result,truth,2,complete=False)
        return result
    def summary_for(scale,reports,first=False):
        names=["baby"] if first else list(IMAGES)
        assert [r["image"] for r in reports]==names
        return dict(passed=True,complete=True,scale=scale,images=names,m=STEPS,seed=0,
            anchor_sigma=2.,config=config,per_image=reports,
            mean_psnr_y={k:np.mean([r["psnr_y"][k] for r in reports],axis=0).tolist() for k in METHODS},
            mean_psnr_y_unclipped={k:np.mean([r["psnr_y_unclipped"][k] for r in reports],axis=0).tolist() for k in METHODS},
            baseline_mean_y=float(np.mean([r["baseline_y"] for r in reports])))

    original_degradation=exp.degradation
    update()
    try:
        check_sources()
        subprocess.run([sys.executable,"-u","check_gaussian_suite.py","--output",
                        str(FOLDER/"kernel_checks.json")],cwd=ROOT,check=True)
        checks=json.loads((FOLDER/"kernel_checks.json").read_text())
        assert checks["passed"] and checks["scales"]==list(SCALES) and checks["anchor_sigma"]==2.
        assert checks["device"]=="cuda"
        assert all(checks["source_sha256"][name]==hashes[name]
                   for name in checks["source_sha256"])
        summaries=[]
        for scale in SCALES:
            reports=[]
            weight_device="cuda" if scale>=3 else "cpu"
            def degrade(method,factor,radius,anchor=scale):
                name="gaussian_aa" if method=="aa_grid_gaussian" else method
                return gaussian_kernel(name,factor,anchor,2./anchor),False
            exp.degradation=degrade
            for image in IMAGES:
                check_sources()
                args=arguments(image,scale)
                folder=FOLDER/f"x{scale}"/f"{image}_seed0"
                folder.mkdir(parents=True,exist_ok=True)
                truth=exp.load_image(ROOT/f"data/{image}_hr.npy","cuda")
                target_shape=truth.shape[-2:]
                kernel,_=degrade(METHODS[0],scale,args.radius)
                observed=CachedResize(target_shape,1./scale,method=kernel,antialiasing=False,
                    device="cuda",weight_device=weight_device)(truth)
                baseline=CachedResize(observed.shape[-2:],scale,target_shape,
                    device="cuda",weight_device=weight_device)(observed)
                baseline_scores=exp.metrics(baseline,truth,shave=scale)
                for name,tensor in (("input",observed),("bicubic_baseline",baseline)):
                    if (folder/f"{name}.npy").exists():
                        assert array_matches(folder/f"{name}.npy",tensor), "Resume input mismatch"
                    else:
                        exp.save_output(folder/name,tensor)
                results_path=folder/"results.json"
                if results_path.exists():
                    stored=json.loads(results_path.read_text())
                    assert stored["config"]==config
                    records=stored["results"]
                    verify_records(folder,records,truth,scale,complete=False)
                elif scale==2 and image=="baby":
                    records=reuse_pilot(folder,observed,baseline,truth)
                else:
                    records=[]
                def save_results():
                    exp.save_json(results_path,dict(config=config,image=image,scale=scale,
                        bicubic=baseline_scores,results=records))
                save_results()
                done={(r["m"],r["method"]) for r in records}
                for m in STEPS:
                    for method in METHODS:
                        if (m,method) in done:
                            continue
                        check_sources()
                        update(state="training",current=dict(scale=scale,image=image,m=m,method=method))
                        if m==1 and method==METHODS[1]:
                            shared=next(r for r in records if r["m"]==1 and r["method"]==METHODS[0])
                            row=dict(shared,method=method,reused_identical_anchor_run=True)
                        else:
                            model,training=exp.train(observed,method,m,args)
                            prediction,sizes=exp.infer(model,observed,m,args,target_shape)
                            assert bool(torch.isfinite(prediction).all()), "Nonfinite prediction"
                            scores=exp.metrics(prediction,truth,shave=scale)
                            stem=f"m{m}_{method}"
                            exp.save_output(folder/stem,prediction)
                            exp.durable_save(folder/f"{stem}.pt",lambda stream:torch.save(model.state_dict(),stream))
                            row=dict(image=image,scale=scale,m=m,method=method,base_sigma=2./scale,
                                anchor_sigma=2.,elementary_scale=scale**(1/m),inference_shapes=sizes,
                                artifact_stem=stem,**training,**scores,**range_stats(prediction))
                            del model,prediction
                        records.append(row); done.add((m,method)); save_results()
                        print(json.dumps(dict(event="result",scale=scale,image=image,m=m,
                                              method=method,psnr_y=row["psnr_y"])),flush=True)
                update(state="verifying",current=dict(scale=scale,image=image))
                count,error=verify_records(folder,records,truth,scale)
                save_results()
                keyed={(r["m"],r["method"]):r for r in records}
                report=dict(image=image,scale=scale,seed=0,m=STEPS,anchor_sigma=2.,
                    psnr_y={k:[keyed[m,k]["psnr_y"] for m in STEPS] for k in METHODS},
                    psnr_y_unclipped={k:[keyed[m,k]["psnr_y_unclipped"] for m in STEPS] for k in METHODS},
                    output_out_of_range_fraction={k:[keyed[m,k]["output_out_of_range_fraction"] for m in STEPS] for k in METHODS},
                    baseline_y=baseline_scores["psnr_y"],checkpoints_verified=count,
                    max_recomputed_psnr_difference_db=error)
                exp.save_json(folder/"verified.json",report); reports.append(report)
                tag=f"x{scale}/{image}"
                if tag not in status["completed"]:
                    status["completed"].append(tag)
                if image=="baby":
                    summary=summary_for(scale,reports,True)
                    paths=plot_scale(summary,ROOT/f"figures/gaussian_sigma2_x{scale}_first_image",first_image=True)
                    exp.save_json(FOLDER/f"first_image_x{scale}.json",summary)
                    milestone=f"first_image_x{scale}"
                    if milestone not in status["milestones"]:
                        status["milestones"].append(milestone)
                    print(json.dumps(dict(event="milestone",name=milestone,plots=[str(p) for p in paths])),flush=True)
                update(state="verified_image",current=None)
                del truth,observed,baseline
            summary=summary_for(scale,reports)
            paths=plot_scale(summary,ROOT/"figures")
            exp.save_json(FOLDER/f"summary_x{scale}.json",summary); summaries.append(summary)
            milestone=f"complete_x{scale}"
            if milestone not in status["milestones"]:
                status["milestones"].append(milestone)
            print(json.dumps(dict(event="milestone",name=milestone,plots=[str(p) for p in paths])),flush=True)
            update(state="complete_scale",current=None)
        check_sources()
        assert data_hashes=={im:digest(ROOT/f"data/{im}_hr.npy") for im in IMAGES}
        plot_all(summaries,ROOT/"figures")
        exp.save_json(FOLDER/"summary.json",dict(passed=True,complete=True,config=config,
            per_scale=summaries,finished_utc=now()))
        update(state="complete",current=None,finished_utc=now())
    except BaseException as error:
        update(state="failed",error=repr(error),stopped_utc=now())
        raise
    finally:
        exp.degradation=original_degradation


if __name__=="__main__":
    main()
