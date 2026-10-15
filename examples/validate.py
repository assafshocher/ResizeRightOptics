"""Reproduce accuracy and timing measurements; write JSON and Markdown reports.

Run from the installed project root: python examples/validate.py
This is a numerical validation, not a visual-quality or SR benchmark.
"""
import json
import math
import platform
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from resize_right_optics import resize, Gaussian, Cubic, Airy


def maxerr(a, b):
    return float(np.max(np.abs(a-b)))


def timed(function, device, repeats=5):
    function()
    if device == 'mps': torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(repeats): function()
    if device == 'mps': torch.mps.synchronize()
    return 1000*(time.perf_counter()-start)/repeats


def main():
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(123)
    report = {'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                             'numpy': np.__version__, 'torch': torch.__version__,
                             'torch_threads': torch.get_num_threads(),
                             'mps': torch.backends.mps.is_available(), 'cuda_tested': False}}

    # Exact identities use compatible periodic extents and valid camera bands.
    exact = []
    camera = Gaussian(.5)
    n = 240
    x = sum(a*np.cos(2*np.pi*k/n*(np.arange(n)-(n-1)/2)+p)
            for k,a,p in [(1,.8,.1),(3,.4,.7),(7,.2,.4)])
    for s,t in [(.5,.5),(2/3,.75),(1.5,2),(2,1.5),(1.5,2/3),(.75,4/3)]:
        kwargs = dict(camera=camera,method='fourier')
        chain = resize(resize(x,s,**kwargs),t,**kwargs)
        direct = resize(x,s*t,**kwargs)
        exact.append({'scales':[s,t], 'max_abs_error':maxerr(chain,direct)})
    report['exact_periodic_chains'] = exact

    # Independently generated data exercise centered grids and phase filters.
    paths = []
    for s in [.25,.4,.5,.6,2/3,.75,1,1.25,1.5,2,3]:
        data = rng.normal(size=(3,17,11))
        kwargs = dict(camera=Gaussian(.6),dims=[1,2],support_sz=12,density=4)
        ordinary = resize(data,[s,.75],**kwargs)
        conv = resize(data,[s,.75],by_convs=True,**kwargs)
        tensor = resize(torch.tensor(data),[s,.75],by_convs=True,**kwargs).numpy()
        paths.append({'scale':s,'numpy_polyphase_error':maxerr(ordinary,conv),
                      'torch_numpy_error':maxerr(ordinary,tensor)})
    report['execution_paths'] = paths

    # Finite support is an approximation: show multiple supports and cameras.
    finite = []
    source = np.cos(2*np.pi*5/256*(np.arange(256)-127.5))
    for name,cam in [('cubic anchor',Cubic()),('Gaussian sigma=.5',Gaussian(.5)),
                     ('Gaussian sigma=1',Gaussian(1))]:
        for support in [16,32,64]:
            kwargs = dict(camera=cam,pad_mode='wrap',support_sz=support,density=16,period_factor=8)
            small = resize(source,.5,**kwargs)
            back = resize(small,2,**kwargs)
            direct = resize(source,.5,camera=cam,method='fourier')
            finite.append({'camera':name,'support':support,'downscale_error':maxerr(small,direct),
                           'cycle_error':maxerr(back,source)})
    report['finite_spatial_smooth_mode'] = finite

    # Do not hide the hard case: unregularized finite Airy refinement.
    u,v = np.meshgrid(np.arange(64)-31.5,np.arange(64)-31.5,indexing='ij')
    scene = np.cos(2*np.pi*(u+v)/64)
    hard = []
    for support in [16,32,64]:
        options = dict(camera=Airy(2.7),support_sz=support,density=4,pad_mode='wrap')
        with warnings.catch_warnings(record=True) as notes:
            small = resize(scene,.5,**options)
            back = resize(small,2,**options)
        direct = resize(scene,.5,camera=Airy(2.7),method='fourier')
        hard.append({'support':support,'downscale_error':maxerr(small,direct),
                     'cycle_error':maxerr(back,scene),'warnings':[str(n.message) for n in notes]})
    spectral_back = resize(resize(scene,.5,camera=Airy(2.7),method='fourier'),
                           2,camera=Airy(2.7),method='fourier')
    report['airy_hard_case'] = hard
    report['airy_periodic_cycle_error'] = maxerr(spectral_back,scene)

    # Timings include kernel construction, not just application of a cached
    # camera. Small GPU jobs may be slower than CPU jobs; report both honestly.
    times = []
    devices = ['numpy','cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
    source = rng.normal(size=(192,192)).astype(np.float32)
    for device in devices:
        image = source if device == 'numpy' else torch.tensor(source,device=device)
        for name,cam in [('Gaussian sigma=.7',Gaussian(.7)),('cubic anchor',Cubic()),('Airy',Airy(2.7))]:
            routes = [('fourier',False)] if name=='Airy' else [('spatial',False),('spatial',True),('fourier',False)]
            for method,conv in routes:
                run = lambda: resize(image,.75,camera=cam,method=method,by_convs=conv)
                times.append({'device':device,'camera':name,'method':method,'polyphase':conv,
                              'milliseconds':timed(run,device)})
    report['timings_192x192_float32_scale_075'] = times
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')

    lines = ['# Validation report', '', 'Generated by `python examples/validate.py`.', '',
             'The separate pytest suite covers gradients, N-D execution, projectors, padding,',
             'rounding, camera construction, optional Torch, dtype/device and invalid inputs.', '',
             '## Environment', '', f"- Python {platform.python_version()}, NumPy {np.__version__}, Torch {torch.__version__}.",
             f"- {platform.platform()}; Torch CPU threads: {torch.get_num_threads()}.",
             f"- Apple MPS available/tested: {torch.backends.mps.is_available()}. CUDA was **not** tested.", '',
             '## Exact periodic reference', '',
             f"- Largest chain/direct error in these checks: **{max(r['max_abs_error'] for r in exact):.3g}**.",
             f"- Airy down-up on the retained test modes: **{report['airy_periodic_cycle_error']:.3g}**.",
             f"- Largest ordinary/polyphase discrepancy: **{max(r['numpy_polyphase_error'] for r in paths):.3g}**.",
             f"- Largest NumPy/PyTorch discrepancy: **{max(r['torch_numpy_error'] for r in paths):.3g}**.", '',
             'These are double-precision tests on compatible periodic grids, not a claim about',
             'arbitrary finite photographs. The test suite separately checks that down-up loses',
             'the appropriate band and that incompatible rounded extents do not form exact cycles.', '',
             '## Finite spatial approximation', '',
             'One unit-amplitude, low-frequency cosine; periodic padding, Hann window,',
             'density 16, period factor 8. Errors are maximum absolute sample errors, not PSNR.', '',
             '| Camera | Support width | Downscale vs Fourier | Down-up vs input |',
             '|---|---:|---:|---:|']
    for r in finite:
        lines.append(f"| {r['camera']} | {r['support']} | {r['downscale_error']:.3g} | {r['cycle_error']:.3g} |")
    lines += ['', 'These errors depend on signal spectrum and numerical settings. They are not',
              'worst-case bounds; increasing support is not always monotonic.', '',
              '## Important limitation: Airy inverse kernels', '',
              'Unregularized full 2-D Airy refinement is poorly conditioned near its cutoff.',
              'The spatial implementation warns about high inverse gain. Even this smooth',
              'test can have substantial finite-kernel error:', '',
              '| Support width | Downscale vs Fourier | Down-up vs input |', '|---:|---:|---:|']
    for r in hard:
        lines.append(f"| {r['support']} | {r['downscale_error']:.3g} | {r['cycle_error']:.3g} |")
    lines += ['', 'Use the periodic Fourier implementation when its boundary assumption is suitable,',
              'or choose explicit regularization and validate the finite approximation. A short',
              'unregularized Airy kernel is **not** presented as production-accurate.', '',
              '## Timings', '', '192×192 float32, scale 0.75, five measured calls after a warm-up.',
              'Includes kernel construction; no learned-parameter caching. GPU timings are synchronized.', '',
              '| Backend | Camera | Method | Polyphase | ms |', '|---|---|---|---|---:|']
    for r in times:
        lines.append(f"| {r['device']} | {r['camera']} | {r['method']} | {r['polyphase']} | {r['milliseconds']:.2f} |")
    lines += ['', '## Not established by these tests', '',
              '- CUDA or other hardware/framework versions.',
              '- Robustness to an unknown or wrongly estimated camera, or pre-existing aliasing.',
              '- Image-quality/SR superiority to other libraries.',
              '- Uniform accuracy of short kernels across arbitrary optical PSFs.',
              '- Globally differentiable hard support/size/band decisions.', '',
              'Raw measurements and configuration are in `validation.json`.']
    (root/'VALIDATION.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:21]))


if __name__ == '__main__':
    main()
