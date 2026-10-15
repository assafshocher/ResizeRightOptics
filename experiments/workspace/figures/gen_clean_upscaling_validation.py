"""Clean Set5 one-shot interpolation and model-aware inverse comparison.

No training, back-projection, score-selected regularizer, or input quantization.
Finite RR inverse and camera-derived refinement are deliberately separate.
"""
from pathlib import Path
import sys, json, csv, time, argparse
from functools import lru_cache
import numpy as np
from scipy import signal, linalg

ROOT = Path(__file__).resolve().parents[1]
ZROOT = ROOT/'work/zssr_scale_transfer'
sys.path.insert(0, str(ZROOT))
if not hasattr(np, 'trapezoid'): np.trapezoid = np.trapz
from kernels import rr, interp_methods, cubic_spectrum
from gaussian_kernels import TruncatedGaussian

OUT = ROOT/'outputs/refinement_validation_2026-09-20/clean'
NAMES = ['baby','bird','butterfly','head','woman']
Y = np.array([65.481,128.553,24.966])/255

def resize(x, scale, shape, method, aa=True):
    return rr.resize(x, scale_factors=[scale,scale,1],
                     out_shape=[*shape,3], interp_method=method,
                     antialiasing=aa, pad_mode='reflect')

@lru_cache(None)
def axis_map(n, m, scale, family):
    kernel = TruncatedGaussian(2) if family=='gaussian' else interp_methods.cubic
    return rr.resize(np.eye(n), scale_factors=[1,scale], out_shape=[n,m],
                     interp_method=kernel, antialiasing=family!='gaussian',
                     pad_mode='reflect').T

def apply_axes(a, b, x):
    return np.stack([a @ x[:,:,c] @ b.T for c in range(3)], axis=-1)

def spectrum_refine(y, scale, shape, family, repetitions=1):
    """Complete even-reflection period, evaluated on RR's centered grid.

    scipy resample splits even-grid Nyquist terms before the fractional shift.
    Gaussian variance is 4/(scale^2-1). For a cubic-AA anchor, telescoping
    H(w)=prod_j U(w/scale^j) gives the inverse slice H(w/scale)/H(w)=1/U(w).
    No roll-off, threshold, or PSNR-selected damping is used.
    """
    n,m = y.shape[:2]
    period = np.pad(y, ((0,n-2),(0,m-2),(0,0)), mode='reflect')
    period = np.tile(period, (repetitions,repetitions,1))
    wy = 2*np.pi*np.fft.fftfreq(period.shape[0])
    wx = 2*np.pi*np.fft.fftfreq(period.shape[1])
    def q(w):
        if family=='gaussian': return np.exp(2*w*w/scale**2)
        if family=='bicubic': return 1/cubic_spectrum(w)
        return np.ones_like(w)
    f = np.fft.fft2(period, axes=(0,1))*q(wy)[:,None,None]*q(wx)[None,:,None]
    filtered = np.fft.ifft2(f,axes=(0,1)).real
    z = signal.resample(signal.resample(filtered,scale*period.shape[0],axis=0),
                        scale*period.shape[1],axis=1)
    offsets = [(a-1)/2-(b-1)/(2*scale) for a,b in zip((n,m),shape)]
    oy = 2*np.pi*np.fft.fftfreq(z.shape[0])[:,None]
    ox = 2*np.pi*np.fft.fftfreq(z.shape[1])[None,:]
    phase = np.exp(1j*scale*(oy*offsets[0]+ox*offsets[1]))
    result = np.fft.ifft2(np.fft.fft2(z,axes=(0,1))*phase[:,:,None],axes=(0,1)).real
    return result[:shape[0],:shape[1]]

def metrics(x,t,shave):
    x=x[shave:-shave,shave:-shave]; t=t[shave:-shave,shave:-shave]
    def psnr(a): return float(-10*np.log10(np.mean(((a-t)@Y)**2)))
    return dict(psnr_y=psnr(np.clip(x,0,1)), psnr_y_raw=psnr(x),
                outside_fraction=float(np.mean((x<0)|(x>1))))

def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--pilot',action='store_true'); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    grid_checks=[]
    for s in [2,3,4]:
        n,m=17,15
        for ky,kx in [(3,4),(n-1,m-1)]:
            wy=np.pi*ky/(n-1);wx=np.pi*kx/(m-1)
            y=np.repeat((np.cos(wy*np.arange(n))[:,None]*np.cos(wx*np.arange(m))[None,:])[:,:,None],3,axis=2)
            for dy,dx in [(0,0),(1,2)]:
                shape=(s*n-dy,s*m-dx)
                yy=np.arange(shape[0])/s+(n-1)/2-(shape[0]-1)/(2*s)
                xx=np.arange(shape[1])/s+(m-1)/2-(shape[1]-1)/(2*s)
                truth=np.cos(wy*yy)[:,None]*np.cos(wx*xx)[None,:]
                prediction=spectrum_refine(y,s,shape,'identity')[:,:,0]
                error=float(np.max(abs(prediction-truth)))
                assert error<1e-12
                grid_checks.append(dict(scale=s,source=[n,m],target=shape,frequencies=[ky,kx],max_error=error))
    save('centered_grid_checks.json',grid_checks)
    rows=[]; checks=[]; rng=np.random.default_rng(260920)
    for family in ['gaussian','bicubic']:
        for scale in ([2] if args.pilot else [2,3,4]):
            for name in (NAMES[:1] if args.pilot else NAMES):
                truth=np.load(ZROOT/f'data/{name}_hr.npy').astype(float)
                shape=truth.shape[:2]; small=tuple(int(np.ceil(n/scale)) for n in shape)
                kernel=TruncatedGaussian(2) if family=='gaussian' else interp_methods.cubic
                observed=resize(truth,1/scale,small,kernel,family!='gaussian')
                a,b=[axis_map(n,m,1/scale,family) for n,m in zip(shape,small)]
                matrix_error=float(np.max(np.abs(apply_axes(a,b,truth)-observed)))
                assert matrix_error<1e-12
                x=rng.normal(size=truth.shape); v=rng.normal(size=observed.shape)
                lhs=np.sum(apply_axes(a,b,x)*v); rhs=np.sum(x*apply_axes(a.T,b.T,v))
                adjoint_error=float(abs(lhs-rhs)/max(abs(lhs),abs(rhs),1))
                assert adjoint_error<1e-12
                t0=time.perf_counter(); ap=linalg.pinv(a); bp=linalg.pinv(b)
                setup_ms=(time.perf_counter()-t0)*1000
                pinv_error=max(float(np.max(np.abs(a@ap-np.eye(a.shape[0])))),
                               float(np.max(np.abs(b@bp-np.eye(b.shape[0])))))
                assert pinv_error<1e-8
                funcs={
                    'bilinear':lambda:resize(observed,scale,shape,interp_methods.linear),
                    'bicubic':lambda:resize(observed,scale,shape,interp_methods.cubic),
                    'lanczos3':lambda:resize(observed,scale,shape,interp_methods.lanczos3),
                    'bandlimited_interp':lambda:spectrum_refine(observed,scale,shape,'identity'),
                    'camera_derived':lambda:spectrum_refine(observed,scale,shape,family),
                    'finite_RR_pinv':lambda:apply_axes(ap,bp,observed)}
                predictions={}
                for method,fn in funcs.items():
                    t0=time.perf_counter(); out=fn(); cold_ms=(time.perf_counter()-t0)*1000
                    timings=[]
                    for _ in range(3):
                        t0=time.perf_counter(); fn(); timings.append((time.perf_counter()-t0)*1000)
                    predictions[method]=out
                    residual=float(np.sqrt(np.mean((apply_axes(a,b,out)-observed)**2)))
                    rows.append(dict(family=family,scale=scale,image=name,method=method,
                                     **metrics(out,truth,scale),forward_rmse=residual,
                                     psnr_y_shave32=metrics(out,truth,32)['psnr_y'],
                                     psnr_y_raw_shave32=metrics(out,truth,32)['psnr_y_raw'],
                                     first_call_ms=cold_ms,median_repeat_ms=float(np.median(timings)),
                                     separate_setup_ms=setup_ms if method=='finite_RR_pinv' else 0))
                saved_error=None
                if family=='gaussian':
                    saved=np.load(ZROOT/f'runs/gaussian_sigma2_suite_v1/x{scale}/{name}_seed0/input.npy')
                    saved_error=float(np.max(np.abs(saved-observed)))
                    assert saved_error<3e-6, (scale,name,saved_error)
                checks.append(dict(family=family,scale=scale,image=name,shape=shape,
                                   input_shape=small,RR_matrix_max_error=matrix_error,
                                   adjoint_relative_error=adjoint_error,pinv_right_inverse_max_error=pinv_error,
                                   previous_float32_observation_max_difference=saved_error))
                # First image is chosen in advance, not selected by gain.
                if name=='baby':
                    np.savez_compressed(OUT/f'{family}_x{scale}_baby.npz',truth=truth,observed=observed,**predictions)
                    repeated=spectrum_refine(observed,scale,shape,family,repetitions=2)
                    period_error=float(np.max(np.abs(repeated-predictions['camera_derived'])))
                    checks[-1]['period_repetition_max_error']=period_error
                    assert period_error<1e-8
                save('results.json',rows); save('checks.json',checks)
                print(json.dumps(dict(family=family,scale=scale,image=name,
                    psnr={r['method']:round(r['psnr_y'],3) for r in rows if (r['family'],r['scale'],r['image'])==(family,scale,name)})),flush=True)
    summary=[]
    for family,scale,method in sorted({(r['family'],r['scale'],r['method']) for r in rows}):
        group=[r for r in rows if (r['family'],r['scale'],r['method'])==(family,scale,method)]
        summary.append(dict(family=family,scale=scale,method=method,n_images=len(group),
                            **{k:float(np.mean([r[k] for r in group])) for k in
                               ['psnr_y','psnr_y_raw','psnr_y_shave32','psnr_y_raw_shave32','forward_rmse','median_repeat_ms','separate_setup_ms']}))
    save('summary.json',summary)
    for filename,data in [('per_image.csv',rows),('summary.csv',summary)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
