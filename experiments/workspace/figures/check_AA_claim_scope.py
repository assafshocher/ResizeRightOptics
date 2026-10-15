"""Actual sampled AA chains versus their continuous-symbol diagnostic."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/refinement_validation_2026-09-20'

def box(x):return ((x>=-.5)&(x<.5)).astype(float)
def linear(x):return np.maximum(1-np.abs(x),0)
def cubic(x):
    t=np.abs(x)
    return np.where(t<1,1.5*t**3-2.5*t*t+1,np.where(t<2,-.5*t**3+2.5*t*t-4*t+2,0))
def lanczos(x):return np.sinc(x)*np.sinc(x/3)*(np.abs(x)<3)
def resize(x,r,k,R):
    m=len(x)//r
    g=np.arange(m)*r+(r-1)/2
    offsets=np.arange(-int(R*r)-2,int(R*r)+3)
    field=np.floor(g).astype(int)[:,None]+offsets[None,:]
    weights=k((g[:,None]-field)/r);weights/=weights.sum(axis=1,keepdims=True)
    return np.sum(x[field%len(x)]*weights,axis=1)

rows=[]
for kind in ['random','impulse']:
    x=np.random.default_rng(0).normal(size=256) if kind=='random' else np.eye(1,256,120)[0]
    for name,k,R in [('box',box,.5),('linear',linear,1),('cubic',cubic,2),('lanczos3',lanczos,3)]:
        one=resize(x,4,k,R);two=resize(resize(x,2,k,R),2,k,R)
        t=np.linspace(-4,4,200001)
        rows.append(dict(input=kind,kernel=name,max_absolute_difference=float(np.max(abs(one-two))),
            relative_chain_RMSE=float(np.linalg.norm(one-two)/np.linalg.norm(one)),
            continuous_diagnostic=float(1-np.trapz(k(t)*np.cos(np.pi*t/2),t))))
        if name=='box':assert np.max(abs(one-two))<1e-15
(OUT/'AA_scope_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
