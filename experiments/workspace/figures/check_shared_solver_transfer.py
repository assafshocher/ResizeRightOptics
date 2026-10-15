"""Same x2 anchor, different transfer laws, identical finite Tikhonov solver.

Fixed synthetic pilot/full protocol: five Set5 central 192-square scenes,
Gaussian and physical separable pupil cameras; factors 2,3,4; noise 0/1/255;
three paired noise draws; eta=.01 primary, .001/.003/.03/.1 sensitivity.
No HR-selected parameters, learned priors, or changes to older experiments.
"""
from pathlib import Path
import sys, json, csv, hashlib
import numpy as np
from scipy import signal, linalg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/paper_revision_red_2026-09-20/validation/shared_solver'
DATA = ROOT/'work/zssr_scale_transfer/data'
NAMES = ['baby','bird','butterfly','head','woman']
N = 192
ETAS = [.001,.003,.01,.03,.1]
Y = np.array([65.481,128.553,24.966])/255

def physical_camera():
    u=np.linspace(-1.,1.,8193)
    pupil=np.exp(2j*np.pi*(.18*u*u+.10*u**4))
    corr=signal.fftconvolve(pupil,np.conj(pupil[::-1]),mode='full').real
    ac=corr[len(pupil)-1:]/corr[len(pupil)-1]
    shift=np.arange(len(pupil))*(u[1]-u[0])
    def h(w):
        a=np.asarray(w); z=2*np.abs(a)/(.9*np.pi)
        return np.where(z<2,np.interp(z,shift,ac,right=0),0)*np.sinc(a/(2*np.pi))
    return h

def axis_map(q, factor):
    w=2*np.pi*np.fft.fftfreq(N)
    x=factor*np.arange(N//factor)+(factor-1)/2
    # Apply filter to each input delta and read centered output coordinates.
    synthesis=np.exp(1j*x[:,None]*w[None,:])*q(w)[None,:]
    return (synthesis@np.fft.fft(np.eye(N),axis=0)/N).real

def apply(d,x):
    return np.stack([d@x[:,:,c]@d.T for c in range(x.shape[2])],axis=-1)

def solve(y, svd, eta):
    u,d,vt=svd
    sig=d[:,None]*d[None,:]
    filt=sig/(sig*sig+eta*eta)
    # Leave DC unchanged in both arms; solve identical zero-mean objective.
    mean=y.mean((0,1)); z=y-mean
    return np.stack([vt.T@((u.T@z[:,:,c]@u)*filt)@vt for c in range(3)],axis=-1)+mean

def psnr(x,t,factor,clip=True):
    if clip: x=np.clip(x,0,1)
    delta=(x-t)[factor:-factor,factor:-factor]@Y
    return float(-10*np.log10(np.mean(delta*delta)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(images=NAMES,n=N,scales=[2,3,4],etas=ETAS,primary_eta=.01,
        noise_255=[0,1],noise_repeats=3,telescope_factors=40,
        boundaries='periodic',grid='centered, S*j+(S-1)/2',
        reference='independently rendered HR camera capture, not latent scene',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256={im:hashlib.sha256((DATA/f'{im}_hr.npy').read_bytes()).hexdigest() for im in NAMES})
    (OUT/'protocol.json').write_text(json.dumps(config,indent=2))
    w=2*np.pi*np.fft.fftfreq(N)
    rows=[];checks=[]
    for family,h in [('Gaussian',lambda z:np.exp(-.5*(4/3)*np.asarray(z)**2)),
                     ('Pupil',physical_camera())]:
        def anchor(z):
            den=h(z); num=h(2*np.asarray(z))
            return np.divide(num,den,out=np.zeros_like(den),where=np.abs(den)>1e-15)
        def recovered(z):
            z=np.asarray(z); result=np.ones_like(z,dtype=float)
            for j in range(1,41): result*=anchor(z/2**j)
            return result
        telescope_error=float(np.max(np.abs(recovered(w)-h(w))))
        assert telescope_error<1e-9
        for factor in [2,3,4]:
            def consistent(z):
                if factor==2: return anchor(z)
                den=recovered(z)
                return np.divide(recovered(factor*np.asarray(z)),den,
                    out=np.zeros_like(den),where=np.abs(den)>1e-15)
            aa=lambda z:anchor(factor*np.asarray(z)/2)
            d_ours=axis_map(consistent,factor); d_aa=axis_map(aa,factor)
            if factor==2: assert np.array_equal(d_ours,d_aa)
            svds={k:linalg.svd(d,full_matrices=False) for k,d in [('ours',d_ours),('aa',d_aa)]}
            rng=np.random.default_rng(22900+factor)
            x=rng.normal(size=(N,N,3)); v=rng.normal(size=(N//factor,N//factor,3))
            adjoint=np.sum(apply(d_ours,x)*v)-np.sum(x*apply(d_ours.T,v))
            assert abs(adjoint)<1e-10
            for ii,name in enumerate(NAMES):
                original=np.load(DATA/f'{name}_hr.npy').astype(float)
                cy,cx=[(n-N)//2 for n in original.shape[:2]]
                scene=original[cy:cy+N,cx:cx+N]
                sf=np.fft.fft2(scene,axes=(0,1))
                hh=h(w)[:,None]*h(w)[None,:]
                truth=np.fft.ifft2(sf*hh[:,:,None],axes=(0,1)).real
                hfar=h(factor*w)[:,None]*h(factor*w)[None,:]
                phase=np.exp(1j*(factor-1)/2*(w[:,None]+w[None,:]))
                observed=np.fft.ifft2(sf*(hfar*phase)[:,:,None],axes=(0,1)).real[::factor,::factor]
                mismatch=float(np.max(np.abs(apply(d_ours,truth)-observed)))
                # Gaussian Nyquist convention differences are negligible here.
                assert mismatch<1e-9,(family,factor,name,mismatch)
                checks.append(dict(family=family,scale=factor,image=name,
                    independent_forward_error=mismatch,telescope_error=telescope_error,
                    adjoint_absolute_error=float(abs(adjoint))))
                for noise in [0,1]:
                    for rep in range(1 if noise==0 else 3):
                        rng=np.random.default_rng(200926+100*ii+10*factor+rep)
                        noisy=observed+(noise/255)*rng.standard_normal(observed.shape)
                        for eta in ETAS:
                            for arm in ['ours','aa']:
                                pred=solve(noisy,svds[arm],eta)
                                rows.append(dict(family=family,scale=factor,image=name,
                                    noise_255=noise,draw=rep,eta=eta,arm=arm,
                                    psnr_y=psnr(pred,truth,factor),
                                    psnr_y_raw=psnr(pred,truth,factor,False)))
                primary=[r for r in rows if r['family']==family and r['scale']==factor and
                         r['image']==name and r['noise_255']==1 and r['eta']==.01]
                print(json.dumps(dict(family=family,scale=factor,image=name,
                    mean_noisy={a:float(np.mean([r['psnr_y'] for r in primary if r['arm']==a])) for a in ['ours','aa']})),flush=True)
    for file,obj in [('results.json',rows),('checks.json',checks)]:
        (OUT/file).write_text(json.dumps(obj,indent=2))
    with (OUT/'results.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    summary=[]
    rng=np.random.default_rng(200927)
    for family in ['Gaussian','Pupil']:
        for factor in [2,3,4]:
            for noise in [0,1]:
                for eta in ETAS:
                    scores={a:np.array([np.mean([r['psnr_y'] for r in rows if
                        (r['family'],r['scale'],r['noise_255'],r['eta'],r['arm'],r['image'])==
                        (family,factor,noise,eta,a,name)]) for name in NAMES]) for a in ['ours','aa']}
                    gain=scores['ours']-scores['aa']
                    draws=rng.integers(0,5,size=(10000,5))
                    ci=np.percentile(gain[draws].mean(1),[2.5,97.5])
                    item=dict(family=family,scale=factor,noise_255=noise,eta=eta,
                        ours=float(scores['ours'].mean()),aa=float(scores['aa'].mean()),
                        gain=float(gain.mean()),gain_ci=ci.tolist(),per_image_gain=gain.tolist())
                    summary.append(item)
                    if eta==.01: print(json.dumps(item),flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()
