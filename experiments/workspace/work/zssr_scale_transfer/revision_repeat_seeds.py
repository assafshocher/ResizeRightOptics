"""Paired seeds 1/2 at m=1,5,10; existing sources/results stay untouched."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/revision_seed_checks_20260920'
NAMES = ['baby', 'bird', 'butterfly', 'head', 'woman']
FILES = ['experiment.py', 'kernels.py', 'matched_kernels.py', 'gaussian_kernels.py',
         'vendor/ResizeRight/resize_right.py', 'vendor/ResizeRight/interp_methods.py']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (OUT / 'run.log').open('ab') as f:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps({'pid': proc.pid, 'folder': str(OUT)}))
        return
    lock = (OUT / 'runner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np
    import torch
    import experiment as exp
    from gaussian_kernels import gaussian_kernel
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    source_hash = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES}
    config = dict(seeds=[1,2], steps=[1,5,10], families=['bandlimited','gaussian'],
        scales=[2,3,4], images=NAMES, training_updates=10000, source_sha256=source_hash,
        note='Paired repeats, unchanged algorithm; m1 uses one shared model.')
    if (OUT/'config.json').exists():
        assert json.loads((OUT/'config.json').read_text()) == config
    else:
        exp.save_json(OUT/'config.json', config)
    rows = json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
    done = {(r['family'],r['scale'],r['image'],r['seed'],r['m'],r['arm']) for r in rows}
    original = exp.degradation
    status = dict(pid=os.getpid(), state='starting', total_rows=360,
                  started_utc=datetime.now(timezone.utc).isoformat())
    def report(**kw):
        status.update(kw, rows=len(rows), updated_utc=datetime.now(timezone.utc).isoformat())
        exp.save_json(OUT/'status.json', status)
    report()
    try:
        for family in config['families']:
            methods = (['consistent','aa_bandlimited'] if family=='bandlimited'
                       else ['gaussian_consistent','gaussian_aa'])
            if family=='gaussian':
                def degrade(method, factor, radius, anchor=2):
                    return gaussian_kernel(method, factor, anchor, 2./anchor), False
                exp.degradation = degrade
            else:
                exp.degradation = original
            for scale in config['scales']:
                for name in NAMES:
                    truth = exp.load_image(ROOT/f'data/{name}_hr.npy', 'cuda')
                    observed_path = (ROOT/f'runs/gaussian_sigma2_suite_v1/x{scale}/{name}_seed0/input.npy'
                        if family=='gaussian' else
                        ROOT/('data' if scale==2 else f'data_x{scale}')/f'{name}_lr_bandlimited.npy')
                    observed = exp.load_image(observed_path, 'cuda')
                    for seed in config['seeds']:
                        a = SimpleNamespace(image=name, scale=scale, steps=10000, batch=1,
                            crop=64, crop_margin=16, width=64, depth=8, lr=1e-4,
                            radius=32*scale, seed=seed, residual=True, device='cuda')
                        for m in config['steps']:
                            for arm, method in zip(['ours','aa'],methods):
                                key = (family,scale,name,seed,m,arm)
                                if key in done:
                                    continue
                                assert source_hash == {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES}
                                report(state='training', current=list(key))
                                if m==1 and arm=='aa':
                                    shared = next(r for r in rows if
                                        (r['family'],r['scale'],r['image'],r['seed'],r['m'],r['arm'])
                                        == (family,scale,name,seed,m,'ours'))
                                    row = dict(shared, arm=arm, method=method, shared_anchor=True)
                                else:
                                    model, details = exp.train(observed, method, m, a)
                                    pred, shapes = exp.infer(model, observed, m, a, truth.shape[-2:])
                                    assert bool(torch.isfinite(pred).all())
                                    scores = exp.metrics(pred, truth, shave=scale)
                                    stem = f'{family}_x{scale}_{name}_seed{seed}_m{m}_{arm}'
                                    exp.save_output(OUT/stem, pred)
                                    exp.durable_save(OUT/(stem+'.pt'), lambda f:torch.save(model.state_dict(),f))
                                    row = dict(family=family, scale=scale, image=name, seed=seed, m=m,
                                        arm=arm, method=method, artifact_stem=stem, shared_anchor=False,
                                        input_sha256=hashlib.sha256(observed_path.read_bytes()).hexdigest(),
                                        inference_shapes=shapes, **details, **scores)
                                    assert details['losses'][-1]['step']==10000
                                    del pred,model
                                rows.append(row); done.add(key)
                                exp.save_json(OUT/'results.json', rows)
                                report(state='completed_run', current=None)
                                print(json.dumps(dict(event='result', key=key, psnr_y=row['psnr_y'])),flush=True)
                    del truth,observed
        assert len(rows)==360
        for family in config['families']:
            for scale in config['scales']:
                for name in NAMES:
                    for seed in config['seeds']:
                        for m in config['steps']:
                            pair = [r for r in rows if (r['family'],r['scale'],r['image'],r['seed'],r['m']) == (family,scale,name,seed,m)]
                            assert len(pair)==2
                            assert pair[0]['coordinate_sha256']==pair[1]['coordinate_sha256']
                            if m==1: assert pair[0]['psnr_y']==pair[1]['psnr_y']
        report(state='complete', current=None)
    except BaseException as e:
        report(state='failed', error=repr(e))
        raise
    finally:
        exp.degradation = original

if __name__=='__main__': main()
