"""Supplementary path-flip reference: three crops, matched-RMS white noise.

Same 50 scenes and five-step path as the archived audit. No class labels,
accuracy claims, tuned noise levels, or edits to the archived figure.
"""
from pathlib import Path
import sys,json,hashlib,platform
import numpy as np
import cv2,torch,torchvision,PIL,skimage
from PIL import Image,ImageOps
from skimage.transform import resize as skresize
from torchvision.models import resnet18,mobilenet_v3_small,ResNet18_Weights,MobileNet_V3_Small_Weights
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/library_audit'))
from optical_cameras import resize_optical
OUT=ROOT/'outputs/paper_revision_red_2026-09-20/validation/classifier_control'

def gaussian(image,m):
    n=image.shape[0]; modes=np.arange(-(m//2),m//2+1)
    wi=2*np.pi*modes/n;wo=2*np.pi*modes/m
    q=np.exp(-.5*.75**2*(wo**2-wi**2))
    off=-(n/m)*(m-1)/2+(n-1)/2
    q=q*np.exp(1j*wi*off)*(m/n)
    f=np.fft.fftshift(np.fft.fft2(image,axes=(0,1)),axes=(0,1))
    start=n//2-m//2
    return np.fft.ifft2(np.fft.ifftshift(f[start:start+m,start:start+m]*q[:,None,None]*q[None,:,None],axes=(0,1)),axes=(0,1)).real.astype(np.float32)

def pillow(image,m,mode):
    return np.stack([np.asarray(Image.fromarray(image[:,:,c],mode='F').resize((m,m),resample=mode)) for c in range(3)],axis=-1)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    methods={
      'Pillow bilinear':lambda x,m:pillow(x,m,Image.Resampling.BILINEAR),
      'Pillow bicubic':lambda x,m:pillow(x,m,Image.Resampling.BICUBIC),
      'Pillow Lanczos':lambda x,m:pillow(x,m,Image.Resampling.LANCZOS),
      'OpenCV area':lambda x,m:cv2.resize(x,(m,m),interpolation=cv2.INTER_AREA),
      'OpenCV bicubic':lambda x,m:cv2.resize(x,(m,m),interpolation=cv2.INTER_CUBIC),
      'OpenCV Lanczos-4':lambda x,m:cv2.resize(x,(m,m),interpolation=cv2.INTER_LANCZOS4),
      'scikit-image linear AA':lambda x,m:skresize(x,(m,m,3),order=1,anti_aliasing=True,mode='reflect',preserve_range=True,clip=False).astype(np.float32),
      'scikit-image cubic AA':lambda x,m:skresize(x,(m,m,3),order=3,anti_aliasing=True,mode='reflect',preserve_range=True,clip=False).astype(np.float32),
      'consistent Gaussian':gaussian,
      'consistent annular pupil':lambda x,m:resize_optical(x,m,'annular pupil'),
      'consistent astigmatic pupil':lambda x,m:resize_optical(x,m,'astigmatic pupil')}
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    models={'ResNet-18':resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device),
      'MobileNetV3-Small':mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval().to(device)}
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    old=json.loads((ROOT/'work/library_audit/full_x8/results.json').read_text())['protocol']
    protocol=dict(scenes=old['scenes'],path=[2047,1349,889,587,387,255],
      crop_origins=[[0,0],[15,15],[31,31]],crop_size=224,
      noise_draws=3,seed=20260920,noise='iid Gaussian rescaled to exact pre-clipping path-error RMS per crop',
      bootstrap='10000 scene-cluster resamples; average crops and noise within each scene',
      versions={k:str(v) for k,v in dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,torchvision=torchvision.__version__,opencv=cv2.__version__,pillow=PIL.__version__,skimage=skimage.__version__).items()},
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]; rng=np.random.default_rng(20260920)
    for scene in old['scenes']:
      im=ImageOps.exif_transpose(Image.open(Path(__import__('os').environ.get('SCALING_CAPTURE_DATA', 'data/captures'))/scene/'00001.JPG')).convert('RGB')
      x=np.asarray(im,dtype=np.float32)/255
      y0,x0=[(v-2047)//2 for v in x.shape[:2]];x=x[y0:y0+2047,x0:x0+2047]
      for method,resize in methods.items():
        direct=resize(x,255); chain=x
        for m in protocol['path'][1:]:chain=resize(chain,m)
        batch=[]; metrics=[]
        for ci,(i,j) in enumerate(protocol['crop_origins']):
          a=direct[i:i+224,j:j+224];b=chain[i:i+224,j:j+224]
          rms=float(np.sqrt(np.mean((b.astype(float)-a)**2)))
          batch.extend([a,b]); metrics.append(rms)
          for rep in range(3):
            z=rng.normal(size=a.shape);z*=rms/np.sqrt(np.mean(z*z))
            batch.append((a+z).astype(np.float32))
        tensor=torch.from_numpy(np.stack(batch)).permute(0,3,1,2).to(device)
        tensor=(tensor.clamp(0,1)-mean)/std
        with torch.inference_mode():
          for model,net in models.items():
            logits=net(tensor).cpu().numpy();labels=logits.argmax(1)
            for ci,rms in enumerate(metrics):
              ix=ci*5
              rows.append(dict(scene=scene,method=method,model=model,crop=ci,rms=rms,
                chain_flip=int(labels[ix]!=labels[ix+1]),
                noise_flip=float(np.mean(labels[ix]!=labels[ix+2:ix+5])),
                direct_margin=float(np.sort(logits[ix])[-1]-np.sort(logits[ix])[-2])))
      (OUT/'results.json').write_text(json.dumps(rows,indent=2))
      print(scene,len(rows),flush=True)
    summary=[];rng=np.random.default_rng(20260921)
    idx=rng.integers(0,50,size=(10000,50))
    for method in methods:
      for model in models:
        a=np.array([[np.mean([r[k] for r in rows if r['scene']==scene and r['method']==method and r['model']==model]) for k in ['chain_flip','noise_flip']] for scene in old['scenes']])
        ci=np.percentile(a[idx].mean(1),[2.5,97.5],axis=0)
        d=a[:,0]-a[:,1]
        summary.append(dict(method=method,model=model,chain=float(a[:,0].mean()),noise=float(a[:,1].mean()),
          chain_ci95=ci[:,0].tolist(),noise_ci95=ci[:,1].tolist(),
          paired_delta=float(d.mean()),paired_delta_ci95=np.percentile(d[idx].mean(1),[2.5,97.5]).tolist()))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
