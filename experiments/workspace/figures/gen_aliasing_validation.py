"""Independent optical captures: aliasing, optical fidelity, and algebra.

Aligned periodic grids isolate sampling from boundary/grid mismatch. Finite LS
kernels are fitted to an independent scene, never the test target. No assertion
that the full-source-band quotient is exact when the source aliases.
"""
from pathlib import Path
import json,csv
import numpy as np
from scipy import signal,linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/refinement_validation_2026-09-20/aliasing'
N=1536; F=16; CUTOFFS=[.9,1.5,2.,3.]
METHODS=['projected','full_band','matched_AA','cubic_AA','heldout_LS']
LABELS=['Projected camera law','Full-band ratio (heuristic)','Matched AA stretch','Bicubic AA (unmatched)','Held-out discrete LS']
COLORS=['#D55E00','#CC79A7','#E69F00','#0072B2','#009E73']
STYLES=['-','--',':','-.','--']

u=np.linspace(-1,1,8193)
p=np.exp(2j*np.pi*(.18*u*u+.10*u**4))
ac=signal.fftconvolve(p,np.conj(p[::-1]),mode='full').real[len(p)-1:]
ac/=ac[0]; lag=np.arange(len(p))*(u[1]-u[0])
def H(w,cutoff):
    return np.interp(2*np.abs(w)/(cutoff*np.pi),lag,ac,right=0)*np.sinc(w/(2*np.pi))

def scene(kind,seed,n=N,f=F):
    rng=np.random.default_rng(seed); w=2*np.pi*np.fft.fftfreq(n*f,d=1/f); a=np.abs(w)
    amp=1/np.maximum(a,2*np.pi/200) if kind=='1/f' else np.minimum(1,1.5/np.maximum(a,1e-15))
    x=np.fft.ifft(np.fft.fft(rng.normal(size=n*f))*amp).real
    z=(x-x.mean())/x.std()*.12
    z*=min(1.,.48/np.max(np.abs(z)))
    return z+.5

def capture(x,r,cutoff,f=F):
    w=2*np.pi*np.fft.fftfreq(len(x),d=1/f)
    return np.fft.ifft(np.fft.fft(x)*H(r*w,cutoff)).real[::f*r]

def ratio(w,r,cutoff,projected):
    denominator=H(w,cutoff)
    out=np.divide(H(r*w,cutoff),denominator,out=np.zeros_like(w),where=np.abs(denominator)>1e-14)
    if projected: out*=np.abs(r*w)<np.pi
    return out

def keys(x):
    x=np.abs(x);return np.where(x<1,1.5*x**3-2.5*x*x+1,
                                np.where(x<2,-.5*x**3+2.5*x*x-4*x+2,0))

def operate(y,r,cutoff,method,kernels=None):
    w=2*np.pi*np.fft.fftfreq(len(y))
    if method in ['projected','full_band']:
        q=ratio(w,r,cutoff,method=='projected')
    elif method=='matched_AA':
        # Agrees exactly with the projected camera at r=2; stretch that
        # same continuous anchor by r/2 for other factors.
        q=ratio(w*r/2,2,cutoff,True)
    else:
        offsets=np.arange(-2*r,2*r+1) if method=='cubic_AA' else kernels[r][0]
        k=keys(offsets/r) if method=='cubic_AA' else kernels[r][1]
        if method=='cubic_AA': k=k/k.sum()
        taps=np.zeros(len(y));np.add.at(taps,offsets%len(y),k)
        q=np.fft.fft(taps)
    return np.fft.ifft(np.fft.fft(y)*q).real[::r]

def fit_kernel(x,r,cutoff):
    source=capture(x,1,cutoff);target=capture(x,r,cutoff)
    offsets=np.arange(-12*r,12*r+1); center=12*r
    A=source[(np.arange(len(target))[:,None]*r-offsets[None,:])%len(source)]
    others=np.arange(len(offsets))!=center
    # Equality constraint sum(k)=1, solved by eliminating the central tap.
    z=linalg.lstsq(A[:,others]-A[:,center,None],target-A[:,center],cond=1e-8)[0]
    k=np.empty(len(offsets));k[others]=z;k[center]=1-z.sum()
    return offsets,k

def rel(p,t): return float(np.linalg.norm(p-t)/np.linalg.norm(t-t.mean()))
def write(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];checks=[]
    for kind in ['1/f','texture']:
        training=scene(kind,61000,n=4*N)
        for cutoff in CUTOFFS:
            kernels={r:fit_kernel(training,r,cutoff) for r in [2,3,4]}
            for seed in range(5):
                x=scene(kind,72000+seed)
                source=capture(x,1,cutoff)
                targets={r:capture(x,r,cutoff) for r in [2,3,4]}
                for sigma in [0,.005]:
                    rng=np.random.default_rng(83000+seed)
                    y=source+sigma*rng.standard_normal(len(source))
                    for method in METHODS:
                        predictions={r:operate(y,r,cutoff,method,kernels) for r in [2,3,4]}
                        chain=operate(operate(y,2,cutoff,method,kernels),2,cutoff,method,kernels)
                        chain_error=rel(chain,predictions[4])
                        for r in [2,3,4]:
                            rows.append(dict(kind=kind,cutoff_pi=cutoff,seed=seed,noise=sigma,
                                factor=r,method=method,prediction_nrmse=rel(predictions[r],targets[r]),
                                chain_vs_direct_nrmse=chain_error if r==4 else None))
                        if sigma==0 and cutoff==.9 and method=='projected':
                            err=max(rel(predictions[r],targets[r]) for r in [2,3,4])
                            assert err<1e-10,(kind,seed,err)
                        if method=='projected': assert chain_error<1e-10,(cutoff,chain_error)
                if seed==0:
                    # Repeat the SAME continuous trigonometric scene at 2x the
                    # fine-grid density, not a newly randomized scene.
                    xx=signal.resample(x,2*len(x))
                    diff=max(float(np.max(np.abs(capture(xx,r,cutoff,f=2*F)-targets[r]))) for r in [2,3,4])
                    assert diff<1e-12
                    checks.append(dict(kind=kind,cutoff_pi=cutoff,fine_grid_doubling_max_error=diff,
                        fine_scene_min=float(x.min()),fine_scene_max=float(x.max()),
                        mtf_at_sensor_nyquist=float(H(np.array([np.pi]),cutoff)[0]),
                        LS_training_seed=61000,test_seeds=list(range(72000,72005))))
            write('results.json',rows);write('checks.json',checks)
            print(json.dumps(dict(done_kind=kind,cutoff=cutoff,rows=len(rows))),flush=True)
    summary=[]
    keys=sorted({(r['kind'],r['cutoff_pi'],r['noise'],r['factor'],r['method']) for r in rows})
    for kind,cutoff,noise,factor,method in keys:
        data=[r for r in rows if (r['kind'],r['cutoff_pi'],r['noise'],r['factor'],r['method'])==(kind,cutoff,noise,factor,method)]
        summary.append(dict(kind=kind,cutoff_pi=cutoff,noise=noise,factor=factor,method=method,
            prediction_nrmse=float(np.mean([r['prediction_nrmse'] for r in data])),
            chain_vs_direct_nrmse=float(np.mean([r['chain_vs_direct_nrmse'] for r in data])) if factor==4 else None))
    write('summary.json',summary)
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    # Explicit indistinguishable-input example, w=.3pi and 1.7pi alias.
    w1=.3*np.pi;w2=1.7*np.pi;cutoff=3.
    h1=float(H(np.array([w1]),cutoff)[0]);h2=float(H(np.array([w2]),cutoff)[0])
    n=np.arange(256)
    # Two nonnegative continuous scenes (offset .5, amplitudes <= .1).
    i1=.5+.1*h2*h1*np.cos(w1*n);i2=.5+.1*h1*h2*np.cos(w2*n)
    t1=.5+.1*h2*float(H(np.array([2*w1]),cutoff)[0])*np.cos(2*w1*n)
    t2=.5+.1*h1*float(H(np.array([2*w2]),cutoff)[0])*np.cos(2*w2*n)
    collision=dict(cutoff_pi=3.,frequencies_pi=[.3,1.7],
        source_max_difference=float(np.max(np.abs(i1-i2))),
        target_max_difference=float(np.max(np.abs(t1-t2))))
    assert collision['source_max_difference']<1e-12 and collision['target_max_difference']>1e-4
    write('alias_collision.json',collision)
    plot(summary)

def plot(summary):
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    # Main appendix figure: prespecified clean textured condition, factor 4.
    for kind,noise,stem in [('texture',0,'fig_aliasing_sweep'),('1/f',.005,'fig_aliasing_sweep_noisy_1f')]:
        fig,axes=plt.subplots(1,2,figsize=(7,2.65),layout='constrained')
        for ax,metric,title in zip(axes,['prediction_nrmse','chain_vs_direct_nrmse'],
                                  ['Optical prediction, $\\times1/4$','Two half-steps vs. one quarter-step']):
            for method,label,color,style in zip(METHODS,LABELS,COLORS,STYLES):
                data=[r for r in summary if r['kind']==kind and r['noise']==noise and r['factor']==4 and r['method']==method]
                values=[r[metric] for r in data]
                # Numerical identities are at a disclosed plotting floor only.
                ax.plot([r['cutoff_pi'] for r in data],np.maximum(values,1e-13),
                        color=color,ls=style,marker='o',ms=3,lw=1.5,label=label)
            ax.axvline(1,color='.6',lw=.8,ls=':');ax.set_xscale('linear');ax.set_xticks(CUTOFFS)
            ax.set_yscale('log');ax.set_ylim(4e-14,2);ax.set_title(title,fontsize=10)
            ax.set_xlabel(r'Optical cutoff / sensor Nyquist');ax.set_ylabel('Relative RMSE')
            ax.grid(axis='y',alpha=.15)
        axes[0].legend(loc='lower right',fontsize=7,frameon=False)
        for ext in ['pdf','png']:fig.savefig(OUT/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)

if __name__=='__main__':main()
