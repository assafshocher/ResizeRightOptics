"""Audit supplied RealSR scores and clean-refinement quantization sensitivity."""
from pathlib import Path
import sys, json, hashlib, shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'figures'))
import gen_clean_upscaling_validation as clean
OUT=ROOT/'outputs/paper_revision_red_2026-09-20/validation'

def regularized_refine(obs,s,shape,family):
    n,m=obs.shape[:2]
    period=np.pad(obs,((0,n-2),(0,m-2),(0,0)),mode='reflect')
    wy=2*np.pi*np.fft.fftfreq(period.shape[0]);wx=2*np.pi*np.fft.fftfreq(period.shape[1])
    def h(w):
        if family=='gaussian':return np.exp(-.5*4/(s*s-1)*w*w)
        result=np.ones_like(w)
        for j in range(32):result*=clean.cubic_spectrum(w/s**j)
        return result
    source=h(wy)[:,None]*h(wx)[None,:]
    target=h(wy/s)[:,None]*h(wx/s)[None,:]
    eta=(1/(255*np.sqrt(12)))/np.std(obs)
    q=target*source/(source*source+eta*eta);q[0,0]=1
    filtered=np.fft.ifft2(np.fft.fft2(period,axes=(0,1))*q[:,:,None],axes=(0,1)).real
    z=clean.signal.resample(clean.signal.resample(filtered,s*period.shape[0],axis=0),s*period.shape[1],axis=1)
    offsets=[(a-1)/2-(b-1)/(2*s) for a,b in zip((n,m),shape)]
    oy=2*np.pi*np.fft.fftfreq(z.shape[0])[:,None];ox=2*np.pi*np.fft.fftfreq(z.shape[1])[None,:]
    phase=np.exp(1j*s*(oy*offsets[0]+ox*offsets[1]))
    return np.fft.ifft2(np.fft.fft2(z,axes=(0,1))*phase[:,:,None],axes=(0,1)).real[:shape[0],:shape[1]],float(eta)

def realsr():
    dest=OUT/'realsr'; dest.mkdir(parents=True,exist_ok=True)
    provenance={}; summary=[]
    for fn in ['RESULTS.md','sweep_canon.jsonl','sweep_nikon.jsonl','sweep2.py']:
        src=Path(__import__('os').environ.get('SCALING_CAPTURE_DATA', 'data/captures'))/fn
        shutil.copy2(src,dest/fn)
        provenance[fn]=hashlib.sha256(src.read_bytes()).hexdigest()
    rng=np.random.default_rng(200920)
    for camera in ['canon','nikon']:
        rows=[json.loads(x) for x in (dest/f'sweep_{camera}.jsonl').read_text().splitlines()]
        unique={}
        for row in rows:
            if row['scene'] in unique: assert unique[row['scene']]==row
            unique[row['scene']]=row
        n_raw=len(rows); rows=list(unique.values());n=len(rows)
        indices=rng.integers(0,n,size=(100000,n))
        for s in [3,4]:
            scores={a:-20*np.log10([r[f'{a}{s}'] for r in rows]) for a in ['fam','str','oracle']}
            delta=scores['fam']-scores['str']
            summary.append(dict(camera=camera,scale=s,n=n,raw_rows=n_raw,
                medians={a:float(np.median(v)) for a,v in scores.items()},
                means={a:float(v.mean()) for a,v in scores.items()},
                mean_paired_gain=float(delta.mean()),median_paired_gain=float(np.median(delta)),
                mean_gain_ci95=np.percentile(delta[indices].mean(1),[2.5,97.5]).tolist(),
                median_gain_ci95=np.percentile(np.median(delta[indices],axis=1),[2.5,97.5]).tolist(),
                family_wins=int((delta>0).sum())))
    (dest/'paired_summary.json').write_text(json.dumps(summary,indent=2))
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('RealSR',json.dumps(summary),flush=True)

def quantization():
    dest=OUT/'quantization'; dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for family in ['gaussian','bicubic']:
        for s in [2,3,4]:
            for im in clean.NAMES:
                truth=np.load(clean.ZROOT/f'data/{im}_hr.npy').astype(float)
                shape=truth.shape[:2]; small=tuple(int(np.ceil(n/s)) for n in shape)
                k=clean.TruncatedGaussian(2) if family=='gaussian' else clean.interp_methods.cubic
                y=clean.resize(truth,1/s,small,k,family!='gaussian')
                yq=np.round(np.clip(y,0,1)*255)/255
                for mode,obs in [('float',y),('uint8',yq)]:
                    for method in ['camera','bicubic','lanczos3']:
                        pred=(clean.spectrum_refine(obs,s,shape,family) if method=='camera' else
                            clean.resize(obs,s,shape,getattr(clean.interp_methods,'cubic' if method=='bicubic' else 'lanczos3')))
                        rows.append(dict(family=family,scale=s,image=im,mode=mode,method=method,**clean.metrics(pred,truth,s)))
                    if mode=='uint8':
                        pred,eta=regularized_refine(obs,s,shape,family)
                        rows.append(dict(family=family,scale=s,image=im,mode=mode,method='camera_tikhonov',eta=eta,**clean.metrics(pred,truth,s)))
            print('quantization',family,s,flush=True)
    summary=[]
    for f,s,m,method in sorted({(r['family'],r['scale'],r['mode'],r['method']) for r in rows}):
        rr=[r for r in rows if (r['family'],r['scale'],r['mode'],r['method'])==(f,s,m,method)]
        summary.append(dict(family=f,scale=s,mode=m,method=method,
            psnr_y=float(np.mean([r['psnr_y'] for r in rr])),raw=float(np.mean([r['psnr_y_raw'] for r in rr]))))
    (dest/'results.json').write_text(json.dumps(rows,indent=2))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    realsr();quantization()
