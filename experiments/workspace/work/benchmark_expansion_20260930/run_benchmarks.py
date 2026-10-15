"""Frozen-protocol BSD100/Set14 expansion; orchestration, not a new method.

The original resizing, training, inference and scoring functions are imported
unchanged. Each image/family/scale runs in its own process. Results commit only
after predictions and weights, and completed cases are never selected by score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parent
ZROOT = ROOT.parent / 'zssr_scale_transfer'
sys.path[:0] = [str(ZROOT), str(ROOT.parents[1] / 'figures')]
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
from PIL import Image
import torch
import experiment as exp
from kernels import CachedResize, rr, interp_methods
from gaussian_kernels import gaussian_kernel
import gen_clean_upscaling_validation as clean
from check_revision_observations import regularized_refine

OUT = ROOT / 'results'
DATA = ROOT / 'data'
SOURCES = {
    'BSD100': ('7a20e0a3c51c5e5153a4416c8606a1476565fa74', 100),
    'Set14': ('5afcf80d267dba61cdfa9a32b1a6fe4cca57b6d7', 14),
}
METHODS = {
    'bandlimited': ('consistent', 'aa_bandlimited', 'aa_grid_bandlimited'),
    'gaussian': ('gaussian_consistent', 'gaussian_aa', 'aa_grid_gaussian'),
    'bicubic': ('consistent_untapered', 'aa', 'aa_grid_literal'),
}
ORIGINAL_DEGRADATION = exp.degradation


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    exp.save_json(Path(path), value)


def arguments(image, scale):
    return SimpleNamespace(image=image, scale=scale, steps=10000, batch=1,
        crop=64, crop_margin=16, width=64, depth=8, lr=1e-4,
        radius=32*scale, seed=0, residual=True, device='cuda')


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def prepare():
    """Read only original HR archives; make our own floating-point observations."""
    for dataset, (revision, count) in SOURCES.items():
        folder = DATA / dataset
        folder.mkdir(parents=True, exist_ok=True)
        archive = DATA / f'{dataset}_HR.tar.gz'
        url = (f'https://huggingface.co/datasets/eugenesiow/{dataset}/resolve/'
               f'{revision}/data/{dataset}_HR.tar.gz')
        if not archive.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read()
            exp.durable_save(archive, lambda f: f.write(payload))
        records = []
        with tarfile.open(archive) as tar:
            members = sorted((m for m in tar.getmembers() if m.isfile()
                and m.name.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg'))),
                key=lambda m: m.name)
            assert len(members) == count, (dataset, len(members))
            for member in members:
                raw = tar.extractfile(member).read()
                name = Path(member.name).stem.lower()
                array = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'),
                                   dtype=np.float32) / 255
                assert min(array.shape[:2]) >= 96 and np.isfinite(array).all()
                path = folder / f'{name}.npy'
                if path.exists():
                    assert np.array_equal(np.load(path), array), path
                else:
                    exp.durable_save(path, lambda f: np.save(f, array))
                records.append(dict(image=name, member=member.name,
                    shape=list(array.shape), source_sha256=hashlib.sha256(raw).hexdigest(),
                    array_sha256=sha(path)))
        assert len({r['image'] for r in records}) == count
        manifest = dict(dataset=dataset, url=url, revision=revision,
            archive_sha256=sha(archive), count=count, images=records,
            policy='All HR images, RGB conversion, no extra modcrop, no provided LR files.')
        previous = folder / 'manifest.json'
        if previous.exists():
            assert json.loads(previous.read_text()) == manifest
        write(previous, manifest)
        print(json.dumps(dict(event='dataset_ready', dataset=dataset, count=count)), flush=True)


@lru_cache(maxsize=128)
def degrade(method, factor, radius, anchor=2):
    if 'gaussian' in method:
        name = 'gaussian_aa' if method == 'aa_grid_gaussian' else method
        return gaussian_kernel(name, factor, anchor, 2./anchor), False
    return ORIGINAL_DEGRADATION(method, factor, radius, anchor)


def truth_for(dataset, image):
    path = DATA / dataset / f'{image}.npy'
    manifest = json.loads((DATA / dataset / 'manifest.json').read_text())
    record = next(r for r in manifest['images'] if r['image'] == image)
    assert sha(path) == record['array_sha256'], 'HR input changed'
    return np.load(path)


def observation(truth, family, scale):
    method = 'aa' if family == 'bicubic' else METHODS[family][0]
    kernel, aa = degrade(method, scale, 32*scale, scale)
    if family != 'gaussian':
        # Original prepare.py creates these observations directly with RR on
        # CPU. Training/inference still keep their original GPU weight policy.
        return rr.resize(truth.cpu(), scale_factors=1/scale, pad_mode='reflect',
                         interp_method=kernel, antialiasing=aa).to(truth.device)
    return CachedResize(truth.shape[-2:], 1/scale, method=kernel, antialiasing=aa,
        device=str(truth.device), weight_device='cuda' if scale>=3 else 'cpu')(truth)


def run_clean(dataset, image, family, scale):
    folder = OUT / 'one_shot' / dataset / family / f'x{scale}' / image
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'results.json'
    if result_path.exists():
        stored = json.loads(result_path.read_text())
        assert len(stored['results']) == 9 and stored['complete']
        return
    truth = truth_for(dataset, image).astype(float)
    shape = truth.shape[:2]
    small = tuple(math.ceil(n/scale) for n in shape)
    kernel = clean.TruncatedGaussian(2) if family == 'gaussian' else interp_methods.cubic
    observed = clean.resize(truth, 1/scale, small, kernel, family != 'gaussian')
    rows = []
    for mode, obs in [('float', observed), ('uint8', np.round(np.clip(observed, 0, 1)*255)/255)]:
        methods = ['bilinear', 'bicubic', 'lanczos3', 'fourier', 'canonical'] if mode == 'float' else [
            'bicubic', 'lanczos3', 'canonical', 'regularized']
        for method in methods:
            start = time.monotonic()
            eta = None
            if method == 'regularized':
                prediction, eta = regularized_refine(obs, scale, shape, family)
            elif method in ('canonical', 'fourier'):
                prediction = clean.spectrum_refine(obs, scale, shape,
                    family if method == 'canonical' else 'identity')
            else:
                name = {'bilinear': 'linear', 'bicubic': 'cubic', 'lanczos3': 'lanczos3'}[method]
                prediction = clean.resize(obs, scale, shape, getattr(interp_methods, name))
            assert prediction.shape == truth.shape and np.isfinite(prediction).all()
            rows.append(dict(dataset=dataset, image=image, family=family, scale=scale,
                mode=mode, method=method, eta=eta, **clean.metrics(prediction, truth, scale),
                seconds=time.monotonic()-start))
    write(result_path, dict(complete=True, results=rows))
    print(json.dumps(dict(event='one_shot_complete', dataset=dataset, image=image,
        family=family, scale=scale, psnr={f"{r['mode']}/{r['method']}":r['psnr_y'] for r in rows})), flush=True)


def run_zssr(dataset, image, family, scale, max_m=10):
    exp.degradation = degrade
    args = arguments(image, scale)
    folder = OUT / 'zssr' / dataset / family / f'x{scale}' / image
    folder.mkdir(parents=True, exist_ok=True)
    truth = torch.from_numpy(truth_for(dataset, image)).permute(2, 0, 1)[None].cuda()
    target_shape = tuple(truth.shape[-2:])
    weight_device = 'cuda' if scale >= 3 else 'cpu'
    # The ordinary bicubic condition keeps the literal AA observation. Its
    # retained-band ratio is approximate, including at m=1, exactly as published.
    observed = observation(truth, family, scale)
    baseline = CachedResize(observed.shape[-2:], scale, target_shape,
                            device='cuda', weight_device=weight_device)(observed)
    for name, tensor in [('input', observed), ('bicubic_baseline', baseline)]:
        path = folder / f'{name}.npy'
        array = tensor[0].permute(1, 2, 0).cpu().numpy()
        if path.exists():
            assert np.array_equal(np.load(path), array), 'Resume input mismatch'
        else:
            exp.save_output(folder/name, tensor)
    path = folder / 'results.json'
    stored = json.loads(path.read_text()) if path.exists() else {}
    rows = stored.get('results', [])
    keyed = {(r['m'], r['method']):r for r in rows}
    assert len(keyed) == len(rows)
    for row in rows:
        stem = folder / row['artifact_stem']
        assert stem.with_suffix('.pt').stat().st_size > 0
        saved = exp.load_image(stem.with_suffix('.npy'), 'cuda')
        assert bool(torch.isfinite(saved).all())
        measured = exp.metrics(saved, truth, shave=scale)
        assert max(abs(measured[k]-row[k]) for k in measured) < 1e-8
        assert row['losses'][-1]['step'] == 10000
    for m in range(1, max_m+1):
        for method in METHODS[family]:
            if (m, method) in keyed:
                continue
            write(OUT/'current.json', dict(updated_utc=now(), dataset=dataset,
                image=image, family=family, scale=scale, m=m, method=method))
            if m == 1 and family != 'bicubic' and method == METHODS[family][1]:
                row = dict(keyed[m, METHODS[family][0]], method=method,
                           reused_identical_anchor_run=True)
            else:
                model, training = exp.train(observed, method, m, args)
                prediction, shapes = exp.infer(model, observed, m, args, target_shape)
                assert bool(torch.isfinite(prediction).all()), 'Nonfinite prediction'
                scores = exp.metrics(prediction, truth, shave=scale)
                stem = f'm{m}_{method}'
                exp.save_output(folder/stem, prediction)
                exp.durable_save(folder/f'{stem}.pt', lambda f: torch.save(model.state_dict(), f))
                row = dict(dataset=dataset, image=image, family=family, scale=scale,
                    m=m, method=method, seed=0, artifact_stem=stem,
                    elementary_scale=scale**(1/m), inference_shapes=shapes,
                    **scores, **training)
                del model, prediction
            rows.append(row)
            keyed[m, method] = row
            peers = [r for r in rows if r['m'] == m]
            assert len({r['coordinate_sha256'] for r in peers}) == 1
            write(path, dict(complete=len(rows)==30, config=vars(args),
                bicubic=exp.metrics(baseline, truth, shave=scale), results=rows))
            print(json.dumps(dict(event='zssr_case_complete', dataset=dataset,
                image=image, family=family, scale=scale, m=m, method=method,
                psnr_y=row['psnr_y'], train_seconds=row['train_seconds'])), flush=True)


def preflight():
    """Check geometry and source-equivalence, never gate continuation on a win."""
    checks = []
    # Reproduce the previous clean baseline with unchanged Set5 inputs/functions.
    reference = json.loads((ROOT/'reference_clean_summary.json').read_text())
    for family in ['gaussian', 'bicubic']:
        for scale in [2, 3, 4]:
            values = []
            for name in clean.NAMES:
                truth = np.load(ZROOT/f'data/{name}_hr.npy').astype(float)
                shape = truth.shape[:2]
                k = clean.TruncatedGaussian(2) if family == 'gaussian' else interp_methods.cubic
                obs = clean.resize(truth, 1/scale, tuple(math.ceil(n/scale) for n in shape), k, family!='gaussian')
                values.append(clean.metrics(clean.spectrum_refine(obs, scale, shape, family), truth, scale)['psnr_y'])
            expected = next(r['psnr_y'] for r in reference if r['family']==family
                and r['scale']==scale and r['method']=='camera_derived')
            error = abs(float(np.mean(values))-expected)
            assert error < 1e-7, (family, scale, error)
            checks.append(dict(test='original_clean_reproduction', family=family, scale=scale, error=error))
    # Match the original ZSSR input arrays and exact m=1 anchors on every scale.
    for scale in [2, 3, 4]:
        truth = exp.load_image(ZROOT/'data/baby_hr.npy', 'cuda')
        wd = 'cuda' if scale>=3 else 'cpu'
        for family in METHODS:
            obs = observation(truth, family, scale)
            reference_input = np.load(ROOT/f'reference_inputs/{family}_x{scale}.npy')
            error = float(np.max(np.abs(obs[0].permute(1,2,0).cpu().numpy()-reference_input)))
            assert error < 3e-6, (family, scale, error)
            if family!='bicubic':
                k2, aa2 = degrade(METHODS[family][1], scale, 32*scale, scale)
                other = (CachedResize(truth.shape[-2:], 1/scale, method=k2,
                    antialiasing=aa2, device='cuda', weight_device=wd)(truth)
                    if family=='gaussian' else rr.resize(truth.cpu(), scale_factors=1/scale,
                    interp_method=k2, antialiasing=aa2, pad_mode='reflect').cuda())
                assert torch.equal(obs, other), 'One-step anchor is not identical'
            checks.append(dict(test='original_input_and_anchor', family=family, scale=scale, max_error=error))
    write(OUT/'preflight.json', dict(passed=True, checks=checks, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), kernel=platform.release(), timestamp=now()))


def summarize():
    """Report image means, keeping all cases including negative PSNR outcomes."""
    groups = {}
    for kind in ['one_shot', 'zssr']:
        for path in (OUT/kind).glob('*/*/x*/*/results.json'):
            for row in json.loads(path.read_text())['results']:
                key = (kind, row['dataset'], row['family'], row['scale'],
                       row.get('mode', 'float'), row.get('m', 1), row['method'])
                groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        kind, dataset, family, scale, mode, m, method = key
        assert len({r['image'] for r in rows}) == len(rows)
        result.append(dict(experiment=kind, dataset=dataset, family=family,
            scale=scale, mode=mode, m=m, method=method, n_images=len(rows),
            complete_dataset=len(rows)==SOURCES[dataset][1],
            mean_psnr_y=float(np.mean([r['psnr_y'] for r in rows]))))
    write(OUT/'summary.json', result)


def suite():
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'runner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = [Path(__file__), *ZROOT.glob('*.py'), *(ROOT/'figures').glob('*.py'),
             * (ZROOT/'vendor/ResizeRight').glob('*.py')]
    config = dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        datasets=SOURCES, scales=[2,3,4], m=list(range(1,11)), methods=METHODS,
        training=vars(arguments('IMAGE',2)), scope='One-shot + quantized; ZSSR; seed 0',
        ordering='BSD100 first, then Set14. Image names sorted; no outcome selection.')
    config = json.loads(json.dumps(config))
    config_path = OUT/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Changed experiment sources/config'
    else:
        write(config_path, config)
    def command(*args):
        return [sys.executable, '-u', str(Path(__file__).resolve()), *map(str,args)]
    def run(command_args, label):
        for attempt in range(1,4):
            write(OUT/'status.json', dict(state='running', pid=os.getpid(), task=label,
                attempt=attempt, updated_utc=now(), kernel=platform.release()))
            code = subprocess.run(command_args, cwd=ROOT).returncode
            if code == 0:
                return
            write(OUT/'status.json', dict(state='retry_wait', task=label, returncode=code,
                attempt=attempt, updated_utc=now()))
            if attempt < 3:
                time.sleep(30)
        raise RuntimeError(f'{label} failed three times; stopping without changing the protocol')
    try:
        run(command('prepare'), 'dataset preparation')
        run(command('preflight'), 'protocol preflight')
        first = sorted(r['image'] for r in json.loads(
            (DATA/'BSD100/manifest.json').read_text())['images'])[0]
        run(command('zssr-pilot','BSD100',first,'bandlimited',2), 'first-image training validation')
        summarize()
        for dataset in SOURCES:
            manifest = json.loads((DATA/dataset/'manifest.json').read_text())
            images = sorted(r['image'] for r in manifest['images'])
            # Complete the inexpensive table first. Then each full-image ZSSR
            # sweep can be inspected while the remaining images continue.
            for image in images:
                for family in ['gaussian','bicubic']:
                    for scale in [2,3,4]:
                        run(command('clean',dataset,image,family,scale),
                            f'one-shot/{dataset}/{image}/{family}/x{scale}')
                summarize()
            for scale in [2,3,4]:
                for image in images:
                    for family in METHODS:
                        run(command('zssr',dataset,image,family,scale),
                            f'zssr/{dataset}/{image}/{family}/x{scale}')
                        summarize()
        write(OUT/'status.json', dict(state='complete', updated_utc=now()))
    except Exception as error:
        write(OUT/'status.json', dict(state='failed', error=repr(error), updated_utc=now()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','preflight','clean','zssr','zssr-pilot','suite','summary'])
    parser.add_argument('case', nargs='*')
    options = parser.parse_args()
    setup()
    if options.mode == 'prepare': prepare()
    elif options.mode == 'preflight': preflight()
    elif options.mode == 'suite': suite()
    elif options.mode == 'summary': summarize()
    else:
        dataset,image,family,scale = options.case
        if options.mode=='clean': run_clean(dataset,image,family,int(scale))
        else: run_zssr(dataset,image,family,int(scale), max_m=1 if options.mode=='zssr-pilot' else 10)
