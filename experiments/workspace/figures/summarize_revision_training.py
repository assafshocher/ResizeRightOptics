"""Summarize original full sweeps and additional paired training seeds."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
Z=ROOT/'work/zssr_scale_transfer'
OUT=ROOT/'outputs/paper_revision_red_2026-09-20/validation/zssr_repeats'
NAMES=['baby','bird','butterfly','head','woman']

def original(family,s):
    if family=='bandlimited':
        p=Z/f'figures/matched_bandlimited_x{s}_summary.json';d=json.loads(p.read_text())
        return d,{a:np.array([d['per_image_psnr_y'][a][name] for name in NAMES]) for a in ['ours','aa','grid']},p
    if family=='gaussian':
        p=Z/f'runs/gaussian_sigma2_suite_v1/summary_x{s}.json';d=json.loads(p.read_text())
        keys={'ours':'gaussian_consistent','aa':'gaussian_aa','grid':'aa_grid_gaussian'}
    else:
        p=Z/f'untapered_set5_x{s}_comparison.json';d=json.loads(p.read_text())
        keys={'ours':'untapered','aa':'aa','grid':'grid'}
    per={r['image']:r for r in d['per_image']}
    return d,{a:np.array([per[name]['psnr_y'][k] for name in NAMES]) for a,k in keys.items()},p

def main():
    OUT.mkdir(parents=True,exist_ok=True);best=[];checks=[];repeats=[]
    rng=np.random.default_rng(200928);indices=rng.integers(0,5,size=(10000,5))
    path=OUT/'results.json'
    rows=json.loads(path.read_text()) if path.exists() else []
    for family in ['bandlimited','gaussian','regular']:
      for s in [2,3,4]:
        d,scores,p=original(family,s)
        checks.append(dict(file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        best.append(dict(family=family,scale=s,baseline=d['baseline_mean_y'],
            **{a:dict(best_mean=float(scores[a].mean(0).max()),best_m=int(scores[a].mean(0).argmax()+1),
                 m1=float(scores[a][:,0].mean()),m10=float(scores[a][:,-1].mean())) for a in scores}))
        if family=='regular':continue
        subset=[r for r in rows if r['family']==family and r['scale']==s]
        if len(subset)!=60:continue
        for m in [1,5,10]:
          cube={a:np.array([[scores[a][i,m-1]]+[next(r['psnr_y'] for r in subset if (r['image'],r['seed'],r['m'],r['arm'])==(name,seed,m,a)) for seed in [1,2]] for i,name in enumerate(NAMES)]) for a in ['ours','aa']}
          for name in NAMES:
            for seed in [1,2]:
              pair=[r for r in subset if (r['image'],r['seed'],r['m'])==(name,seed,m)]
              assert pair[0]['coordinate_sha256']==pair[1]['coordinate_sha256']
              if m==1:assert pair[0]['psnr_y']==pair[1]['psnr_y']
          delta=(cube['ours']-cube['aa']).mean(1)
          ci=np.percentile(delta[indices].mean(1),[2.5,97.5])
          repeats.append(dict(family=family,scale=s,m=m,n_images=5,seeds=[0,1,2],
            ours_mean=float(cube['ours'].mean()),aa_mean=float(cube['aa'].mean()),
            gain=float(delta.mean()),gain_ci95=ci.tolist(),
            per_seed_gain=(cube['ours']-cube['aa']).mean(0).tolist(),
            seed_sd_of_mean_gain=float((cube['ours']-cube['aa']).mean(0).std(ddof=1)),
            per_image_gain=delta.tolist()))
    for fn,obj in [('best_over_full_sweep.json',best),('paired_repeats_summary.json',repeats),('original_sources.json',checks)]:
      (OUT/fn).write_text(json.dumps(obj,indent=2))
    print(json.dumps(dict(repeat_rows=len(rows),repeat_conditions=len(repeats),best=best,repeats=repeats),indent=2))

if __name__=='__main__':main()
