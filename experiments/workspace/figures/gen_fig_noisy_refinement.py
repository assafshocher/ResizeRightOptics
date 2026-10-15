"""Noise stress tests, not a new SR benchmark or a manuscript modification.

A: exact periodic, alias-free separable pupil camera, natural Set5 crops.
B: existing clean-trained one-step Gaussian Set5 ZSSR checkpoints, no retraining.
All noise is independent additive white RGB noise, added after capture.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ZROOT = ROOT / 'work/zssr_scale_transfer'
sys.path.insert(0, str(ZROOT))
import experiment as exp
from kernels import CachedResize

OUT = ROOT / 'outputs/noisy_refinement_2026-09-20'
NAMES = ['baby', 'bird', 'butterfly', 'head', 'woman']
LEVELS = [0., .25, .5, 1., 2., 4., 8., 16.]
METHODS = ['canonical', 'threshold', 'wiener', 'bicubic', 'zssr']
LABELS = {'canonical': 'Canonical, no regularization',
          'threshold': r'Canonical, $|H|\geq10^{-3}$',
          'wiener': 'Tikhonov-regularized refinement', 'bicubic': 'Bicubic interpolation',
          'zssr': 'ZSSR trained on clean input'}
COLORS = dict(zip(METHODS, ['#D55E00', '#E69F00', '#009E73', '#0072B2', '#CC79A7']))
STYLES = dict(zip(METHODS, ['-', ':', '-', '--', '-.']))
YWEIGHTS = np.array([65.481, 128.553, 24.966]) / 255


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def scores(pred, truth, shave=0):
    if shave:
        pred = pred[shave:-shave, shave:-shave]
        truth = truth[shave:-shave, shave:-shave]
    def db(delta):
        return float(-10*np.log10(max(float(np.mean(delta**2)), 1e-30)))
    return dict(psnr_y=db((np.clip(pred, 0, 1)-truth) @ YWEIGHTS),
                psnr_y_raw=db((pred-truth) @ YWEIGHTS),
                psnr_rgb_raw=db(pred-truth), rmse=float(np.sqrt(np.mean((pred-truth)**2))),
                min=float(pred.min()), max=float(pred.max()),
                outside=float(np.mean((pred < 0) | (pred > 1))))


def fft2(x):
    return np.fft.fft2(x, axes=(0, 1))


def ifft2(x):
    return np.fft.ifft2(x, axes=(0, 1)).real


def magnify(filtered, offset=-.25):
    """Exact finite periodic Fourier interpolation on the RR-centered 2x grid.

    scipy.signal.resample splits even-grid Nyquist terms. Shift only after
    interpolation, so the two Nyquist signs get their correct opposite phases.
    offset is measured in source pixels, not target pixels.
    """
    z = signal.resample(signal.resample(filtered, 2*filtered.shape[0], axis=0),
                        2*filtered.shape[1], axis=1)
    wy = 2*np.pi*np.fft.fftfreq(z.shape[0])[:, None]
    wx = 2*np.pi*np.fft.fftfreq(z.shape[1])[None, :]
    return ifft2(fft2(z)*np.exp(1j*(wy+wx)*2*offset)[..., None])


def lens_factory():
    # Pupil autocorrelation from a separable rectangular pupil. Its intensity
    # PSF is nonnegative; pixel integration is included in each dimension.
    u = np.linspace(-1., 1., 8193)
    p = np.exp(2j*np.pi*(.18*u*u + .10*u**4))
    ac = signal.fftconvolve(p, np.conj(p[::-1]), mode='full').real
    ac = ac[len(p)-1:] / ac[len(p)-1]
    shifts = np.arange(len(p))*(u[1]-u[0])
    def h(w):
        q = 2*np.abs(w)/(.9*np.pi)
        return np.where(q < 2, np.interp(q, shifts, ac, right=0), 0)*np.sinc(w/(2*np.pi))
    return h


def multipliers(hs, ht, level, std):
    live = np.abs(hs) > 0
    q = np.divide(ht, hs, out=np.zeros_like(hs), where=live)
    cut = q*(np.abs(hs) >= 1e-3)
    # Declared heuristic: scalar Tikhonov parameter from known noise and input
    # contrast, never selected with HR/target pixels. It is not optimal Wiener.
    eta = (level/255)/max(std, 1e-6)
    if eta == 0:
        reg = q.copy()
    else:
        reg = ht*hs/(hs*hs + eta*eta)
    # Preserve the observed RGB mean, i.e. regularize the zero-mean component.
    reg[0, 0] = 1.
    return {'canonical': q, 'threshold': cut, 'wiener': reg}, eta


def physical(images, repeats):
    h = lens_factory()
    n = 96
    wh = 4*np.pi*np.fft.fftfreq(2*n)
    hsource = h(wh)[:, None]*h(wh)[None, :]
    htarget = h(wh/2)[:, None]*h(wh/2)[None, :]
    sourceband = np.abs(hsource) > 0
    w = 2*np.pi*np.fft.fftfreq(n)
    hs = h(w)[:, None]*h(w)[None, :]
    ht = h(w/2)[:, None]*h(w/2)[None, :]
    rr = CachedResize((n,n), 2, device='cpu', pad_mode='circular')
    rows, checks = [], {}
    for index, name in enumerate(images):
        original = np.load(ZROOT/f'data/{name}_hr.npy').astype(float)
        y, x = [(k-2*n)//2 for k in original.shape[:2]]
        scene = original[y:y+2*n, x:x+2*n]
        sf = fft2(scene)
        full = ifft2(sf*htarget[..., None])
        target = ifft2(sf*(htarget*sourceband)[..., None])
        # Input pixel centers are halfway between pairs of target pixels.
        phase = np.exp(.25j*(wh[:, None]+wh[None, :]))
        observed = ifft2(sf*(hsource*phase)[..., None])[::2, ::2]
        clean_q, _ = multipliers(hs, ht, 0, float(observed.std()))
        reconstructed = magnify(ifft2(fft2(observed)*clean_q['canonical'][..., None]))
        error = float(np.max(np.abs(reconstructed-target)))
        assert error < 1e-8, (name, 'clean optical identity', error)
        checks[name] = dict(clean_canonical_max_error=error,
                            minimum_psf_sample=float(ifft2(hsource).min()),
                            canonical_white_noise_gain=float(np.sqrt(np.mean(clean_q['canonical']**2))),
                            threshold_white_noise_gain=float(np.sqrt(np.mean(clean_q['threshold']**2))))
        for level in LEVELS:
            for rep in range(1 if level == 0 else repeats):
                rng = np.random.default_rng(10100 + index*100 + rep)
                noisy = observed + level/255*rng.standard_normal(observed.shape)
                maps, eta = multipliers(hs, ht, level, float(noisy.std()))
                predictions = {key:magnify(ifft2(fft2(noisy)*m[..., None])) for key,m in maps.items()}
                inp = torch.from_numpy(noisy.astype('float32')).permute(2,0,1)[None]
                predictions['bicubic'] = rr(inp)[0].permute(1,2,0).numpy().astype(float)
                for method,pred in predictions.items():
                    for target_name,truth in [('input_determined',target),('full_optical',full)]:
                        rows.append(dict(experiment='physical', image=name, noise_255=level,
                                         seed=rep, method=method, target=target_name, eta=eta,
                                         **scores(pred,truth)))
                if name == 'baby' and level in (0, 1, 4) and rep == 0:
                    np.savez_compressed(OUT/f'physical_baby_noise{level:g}.npz',
                                        observed=noisy, target=target, full=full, **predictions)
        print(json.dumps(dict(event='physical_image', image=name, **checks[name])), flush=True)
    return rows, checks


def gaussian_refiners(noisy, level, repetitions=1):
    n,m = noisy.shape[:2]
    # Use one COMPLETE even-reflection period, not an arbitrary padded window:
    # closing a partial reflection at the FFT seam creates a discontinuity that
    # an inverse Gaussian magnifies. This is reflect (no endpoint repetition),
    # as in the shared RR boundary convention, not a PSNR-selected boundary.
    period = np.pad(noisy, ((0,n-2),(0,m-2),(0,0)), mode='reflect')
    padded = np.tile(period,(repetitions,repetitions,1))
    wy = 2*np.pi*np.fft.fftfreq(padded.shape[0])[:, None]
    wx = 2*np.pi*np.fft.fftfreq(padded.shape[1])[None, :]
    # Sigma=2 observation kernel at total factor 2 implies camera variance 4/3.
    hs = np.exp(-.5*(4/3)*(wy*wy+wx*wx))
    ht = np.exp(-.5*(4/3)*(wy*wy+wx*wx)/4)
    maps, eta = multipliers(hs, ht, level, float(noisy.std()))
    ff = fft2(padded)
    out = {}
    for key,multiplier in maps.items():
        p = magnify(ifft2(ff*multiplier[..., None]))
        out[key] = p[:2*n,:2*m]
    return out, eta


def learned(images, repeats, device):
    rows, checks = [], {}
    torch.set_num_threads(4)
    for index,name in enumerate(images):
        folder = ZROOT/f'runs/gaussian_sigma2_suite_v1/x2/{name}_seed0'
        observed = np.load(folder/'input.npy')
        truth = np.load(ZROOT/f'data/{name}_hr.npy').astype(float)
        model = exp.PlainCNN(64,8,True)
        checkpoint = folder/'m1_gaussian_consistent.pt'
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        model = model.eval().to(device)
        up = CachedResize(observed.shape[:2], 2, truth.shape[:2], device=device)
        saved = np.load(folder/'m1_gaussian_consistent.npy')
        for level in LEVELS:
            for rep in range(1 if level == 0 else repeats):
                rng = np.random.default_rng(20100 + index*100 + rep)
                noisy = observed.astype(float) + level/255*rng.standard_normal(observed.shape)
                # Quantize only arithmetic dtype, never intensities or ranges.
                inp = torch.from_numpy(noisy.astype('float32')).permute(2,0,1)[None].to(device)
                with torch.inference_mode():
                    interpolated = up(inp)
                    predicted = model(F.pad(interpolated,[8]*4,mode='reflect').contiguous(memory_format=torch.channels_last))[...,8:-8,8:-8]
                bc = interpolated[0].permute(1,2,0).cpu().numpy().astype(float)
                sr = predicted[0].permute(1,2,0).cpu().numpy().astype(float)
                predictions,eta = gaussian_refiners(noisy,level)
                predictions.update(bicubic=bc,zssr=sr)
                if level == 0:
                    d = abs(scores(sr,truth,2)['psnr_y']-scores(saved,truth,2)['psnr_y'])
                    assert d < .03, (name,'checkpoint reproduction',d)
                    alternate,_ = gaussian_refiners(noisy,level,repetitions=2)
                    repeat_error = {k:float(np.max(np.abs(alternate[k]-predictions[k]))) for k in alternate}
                    assert max(repeat_error.values()) < 1e-8, (name,'FFT period repetition',repeat_error)
                    checks[name] = dict(clean_checkpoint_psnr_difference=d,
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        gaussian_period_repetition_max_difference=repeat_error,
                        saved_clean_zssr_psnr=scores(saved,truth,2)['psnr_y'])
                for method,pred in predictions.items():
                    for shave in (2,32):
                        rows.append(dict(experiment='clean_trained_sr',image=name,noise_255=level,
                                         seed=rep,method=method,target=f'HR_shave{shave}',eta=eta,
                                         **scores(pred,truth,shave)))
                if name == 'baby' and level in (0,1,4) and rep == 0:
                    np.savez_compressed(OUT/f'gaussian_baby_noise{level:g}.npz',observed=noisy,truth=truth,**predictions)
        print(json.dumps(dict(event='sr_image',image=name,**checks[name])),flush=True)
        del model
    return rows,checks


def aggregate(rows):
    result=[]
    keys=sorted({(r['experiment'],r['target'],r['noise_255'],r['method']) for r in rows})
    for experiment,target,level,method in keys:
        relevant=[r for r in rows if (r['experiment'],r['target'],r['noise_255'],r['method'])==(experiment,target,level,method)]
        entry=dict(experiment=experiment,target=target,noise_255=level,method=method)
        for metric in ['psnr_y','psnr_y_raw','psnr_rgb_raw','rmse']:
            per_image=[np.mean([r[metric] for r in relevant if r['image']==im]) for im in sorted({r['image'] for r in relevant})]
            entry[metric]=float(np.mean(per_image))
        result.append(entry)
    return result


def plot(summary):
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(9.1,3.45))
    for ax,experiment,target,title in zip(axs,['physical','clean_trained_sr'],['input_determined','HR_shave2'],
          ['Physical camera: input-determined target','Clean-trained SR: original HR target']):
        for method in METHODS:
            # Noise-free identities are reported separately, not silently
            # clipped to a finite plotting ceiling.
            rows=[r for r in summary if r['experiment']==experiment and r['target']==target and r['method']==method and r['noise_255']>0]
            if not rows:continue
            ax.plot([r['noise_255'] for r in rows],[r['psnr_y_raw'] for r in rows],
                    color=COLORS[method],ls=STYLES[method],lw=1.8,marker='o',ms=3,label=LABELS[method])
        ax.set_xscale('log',base=2)
        ax.set_xticks([.25,.5,1,2,4,8,16],['0.25','0.5','1','2','4','8','16'])
        ax.set_xlabel(r'White-noise standard deviation ($/255$)')
        ax.set_ylabel('Unclipped Y-PSNR (dB)')
        ax.set_title(title,fontsize=10)
        ax.grid(alpha=.2)
    handles,labels=axs[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.07,right=.99,top=.89,bottom=.30,wspace=.25)
    fig.savefig(OUT/'noisy_refinement.png',dpi=240,bbox_inches='tight')
    fig.savefig(OUT/'noisy_refinement.pdf',bbox_inches='tight')
    # Fixed predeclared example: baby, noise 1/255; no image selection by score.
    p=OUT/'physical_baby_noise1.npz'
    if p.exists():
        data=np.load(p)
        fig,axs=plt.subplots(1,5,figsize=(10,2.3))
        for ax,key,title in zip(axs,['target','canonical','threshold','wiener','bicubic'],
                         ['Input-determined target','Canonical','Thresholded inverse','Tikhonov-regularized','Bicubic']):
            ax.imshow(np.clip(data[key],0,1));ax.set_title(title,fontsize=9);ax.axis('off')
        fig.tight_layout(pad=.3)
        fig.savefig(OUT/'physical_example.png',dpi=220,bbox_inches='tight')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--device',default='mps')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.plot_only:
        plot(json.loads((OUT/'results.json').read_text())['summary'])
        return
    images=NAMES[:1] if args.pilot else NAMES
    repeats=3
    rows,physical_checks=physical(images,repeats)
    dump('physical_checks.json',physical_checks)
    dump('physical_rows.json',rows)
    other,sr_checks=learned(images,repeats,args.device)
    rows+=other
    summary=aggregate(rows)
    dump('results.json',dict(images=images,noise_255=LEVELS,noise_seeds=repeats,
                            physical_checks=physical_checks,sr_checks=sr_checks,rows=rows,summary=summary))
    with (OUT/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    plot(summary)
    for row in summary:
        if row['target'] in ['input_determined','HR_shave2'] and row['noise_255'] in [0,1,4,16]:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
