"""Compare the implemented float32 round trip with direct ideal projection.

Prespecified 116 reference images, factors 4/16/32, all three base families.
Pixel-level check, not a rerun of the SIFT/RANSAC benchmark.
"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs'))
import hpatches_unit_kernel_experiment as exp
OUT=ROOT/'outputs/paper_revision_red_2026-09-20/validation/hpatches_projection'

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    dataset=ROOT/'work/hpatches_geometry/hpatches-sequences-release'
    for seq in sorted(x for x in dataset.iterdir() if x.is_dir() and (x/'1.ppm').exists()):
        original=exp.old.trim_to_exact_half(exp.geometry.load_grayscale(seq/'1.ppm'))
        f=np.fft.fftshift(np.fft.fft2(original));n=original.shape
        for r in [4,16,32]:
            shape=tuple(exp.geometry.nearest_odd(s/r) for s in n)
            y=n[0]//2-shape[0]//2;x=n[1]//2-shape[1]//2
            ff=np.zeros_like(f);ff[y:y+shape[0],x:x+shape[1]]=f[y:y+shape[0],x:x+shape[1]]
            projected=np.fft.ifft2(np.fft.ifftshift(ff)).real
            q=lambda a:np.round(np.clip(a,0,1)*255).astype(np.uint8)
            for base in exp.BASES:
                result=exp.consistent_up(exp.spectral_down_from_spectrum(f,n,shape,base,'unit-kernel consistent'),n,base)
                rows.append(dict(sequence=seq.name,factor=r,base=base,
                    max_error=float(np.max(abs(result-projected))),
                    rmse=float(np.sqrt(np.mean((result-projected)**2))),
                    uint8_differences=int(np.sum(q(result)!=q(projected))),pixels=int(result.size)))
        print(seq.name,flush=True)
    summary=dict(reference_images=len(rows)//9,comparisons=len(rows),
        maximum_error=max(r['max_error'] for r in rows),max_rmse=max(r['rmse'] for r in rows),
        quantized_pixel_difference_fraction=sum(r['uint8_differences'] for r in rows)/sum(r['pixels'] for r in rows),
        comparisons_with_quantization_changes=sum(r['uint8_differences']>0 for r in rows))
    (OUT/'results.json').write_text(json.dumps(rows,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
