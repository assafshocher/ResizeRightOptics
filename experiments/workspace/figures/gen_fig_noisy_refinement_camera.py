"""Paper-only packaging of the completed physical-camera noise experiment."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/noisy_refinement_2026-09-20'
OUT=ROOT/'outputs/refinement_validation_2026-09-20/noise'
OUT.mkdir(parents=True,exist_ok=True)
rows=json.loads((SOURCE/'physical_rows.json').read_text())
rows=[r for r in rows if r['target']=='input_determined' and r['method'] in ['canonical','wiener','bicubic']]
summary=[]
for level,method in sorted({(r['noise_255'],r['method']) for r in rows}):
    data=[r for r in rows if r['noise_255']==level and r['method']==method]
    means=[np.mean([r['psnr_y_raw'] for r in data if r['image']==name]) for name in sorted({r['image'] for r in data})]
    summary.append(dict(noise_255=level,method=method,psnr_y_raw=float(np.mean(means)),n_images=len(means)))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(3.5,2.6),layout='constrained')
for method,label,color,style,marker in [
    ('canonical','Canonical (exact)','#D55E00','-','o'),
    ('wiener','Tikhonov','#009E73','-','s'),
    ('bicubic','Bicubic','#0072B2','--','^')]:
    data=[r for r in summary if r['method']==method and r['noise_255']>0]
    ax.plot([r['noise_255'] for r in data],[r['psnr_y_raw'] for r in data],
            label=label,color=color,ls=style,marker=marker,ms=3,lw=1.5)
ax.set_xscale('log',base=2); ax.set_xticks([.25,.5,1,2,4,8,16],['.25','.5','1','2','4','8','16'])
ax.set_xlabel(r'Noise standard deviation ($/255$)');ax.set_ylabel('Unclipped Y-PSNR (dB)')
ax.grid(axis='y',alpha=.18);ax.legend(fontsize=8.5,loc='lower left',frameon=False)
for ext in ['pdf','png']:fig.savefig(OUT/f'fig_noisy_refinement_camera.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
# Visual example is fixed in advance: baby, first realization, sigma=1/255.
example=np.load(SOURCE/'physical_baby_noise1.npz')
fig,axes=plt.subplots(1,4,figsize=(7,1.85),layout='constrained')
for ax,key,label in zip(axes,['target','canonical','wiener','bicubic'],['Optical target','Canonical','Tikhonov','Bicubic']):
    ax.imshow(np.clip(example[key],0,1));ax.set_title(label,fontsize=11);ax.axis('off')
for ext in ['pdf','png']:fig.savefig(OUT/f'fig_noisy_refinement_example.{ext}',dpi=300,bbox_inches='tight')
print(json.dumps(summary,indent=2))
