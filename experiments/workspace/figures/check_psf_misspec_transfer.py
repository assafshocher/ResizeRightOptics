"""PSF-calibration sensitivity of the frozen shared-solver transfer experiment.

The rendered image, x2 anchor, grid, solver, eta, and noise draws follow
check_shared_solver_transfer.py. Only the camera parameter used to *estimate*
the x2 anchor changes; both transfer arms receive that same erroneous anchor.
"""
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
from scipy import linalg, signal

import check_shared_solver_transfer as base


OUT = base.ROOT / 'outputs/psf_misspec_transfer_2026-10-01'
DELTAS = [-.30, -.20, -.10, 0., .10, .20, .30]
ETA = .01


def pupil_camera(width_multiplier):
    """Original pupil and pixel integration; perturb only optical OTF width."""
    u = np.linspace(-1., 1., 8193)
    pupil = np.exp(2j * np.pi * (.18 * u*u + .10 * u**4))
    corr = signal.fftconvolve(pupil, np.conj(pupil[::-1]), mode='full').real
    ac = corr[len(pupil)-1:] / corr[len(pupil)-1]
    shift = np.arange(len(pupil)) * (u[1] - u[0])

    def h(w):
        a = np.asarray(w)
        z = 2*np.abs(a) / (.9*np.pi*width_multiplier)
        return np.where(z < 2, np.interp(z, shift, ac, right=0), 0) * np.sinc(a/(2*np.pi))

    return h


def estimated_camera(family, delta):
    if family == 'Gaussian':
        # Spatial PSF standard deviation changes; pixel integration is absent
        # in this frozen Gaussian protocol.
        return lambda z: np.exp(-.5 * (4/3) * (1+delta)**2 * np.asarray(z)**2)
    if delta == 0:
        return base.physical_camera()
    return pupil_camera(1+delta)


def anchor_from(h):
    def anchor(z):
        den = h(z)
        num = h(2*np.asarray(z))
        return np.divide(num, den, out=np.zeros_like(den), where=np.abs(den) > 1e-15)
    return anchor


def recovered_from(anchor):
    def recovered(z):
        z = np.asarray(z)
        result = np.ones_like(z, dtype=float)
        for j in range(1, 41):
            result *= anchor(z/2**j)
        return result
    return recovered


def observations(family, true_h):
    """Independent true-camera captures, never changed by estimated PSF."""
    w = 2*np.pi*np.fft.fftfreq(base.N)
    result = {}
    hh = true_h(w)[:, None] * true_h(w)[None, :]
    for name in base.NAMES:
        original = np.load(base.DATA/f'{name}_hr.npy').astype(float)
        cy, cx = [(n-base.N)//2 for n in original.shape[:2]]
        scene = original[cy:cy+base.N, cx:cx+base.N]
        sf = np.fft.fft2(scene, axes=(0, 1))
        truth = np.fft.ifft2(sf*hh[:, :, None], axes=(0, 1)).real
        for factor in [2, 3, 4]:
            hfar = true_h(factor*w)[:, None] * true_h(factor*w)[None, :]
            phase = np.exp(1j*(factor-1)/2*(w[:, None]+w[None, :]))
            observed = np.fft.ifft2(sf*(hfar*phase)[:, :, None], axes=(0, 1)).real[::factor, ::factor]
            result[name, factor] = truth, observed
    return result


def summarize(rows):
    rng = np.random.default_rng(200927)
    summary = []
    for family in ['Gaussian', 'Pupil']:
        for delta in DELTAS:
            for factor in [2, 3, 4]:
                for noise in [0, 1]:
                    scores = {}
                    for arm in ['ours', 'aa']:
                        scores[arm] = np.array([
                            np.mean([r['psnr_y'] for r in rows if
                                     (r['family'], r['delta'], r['scale'], r['noise_255'], r['image'], r['arm'])
                                     == (family, delta, factor, noise, name, arm)])
                            for name in base.NAMES])
                    gain = scores['ours'] - scores['aa']
                    draws = rng.integers(0, len(base.NAMES), size=(10000, len(base.NAMES)))
                    ci = np.percentile(gain[draws].mean(1), [2.5, 97.5])
                    summary.append(dict(family=family, delta=delta, scale=factor,
                        noise_255=noise, eta=ETA, ours=float(scores['ours'].mean()),
                        aa=float(scores['aa'].mean()), gain=float(gain.mean()),
                        gain_ci=ci.tolist(), per_image_gain=gain.tolist()))
    return summary


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = base.OUT / 'results.json'
    old_rows = json.loads(frozen.read_text())
    protocol = dict(
        parent_script=str(Path(base.__file__).relative_to(base.ROOT)),
        parent_script_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_results=str(frozen.relative_to(base.ROOT)),
        frozen_results_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        images=base.NAMES, n=base.N, deltas=DELTAS, scales=[2, 3, 4],
        eta=ETA, noise_255=[0, 1], noise_repeats=3, telescope_factors=40,
        boundary='periodic', grid='centered, S*j+(S-1)/2',
        comparison='same erroneous x2 anchor for both camera law and AA stretch',
        truth='independent captures from unperturbed PSF; 8-bit rounding not applied',
        perturbation={'Gaussian': 'spatial PSF sigma multiplier 1+delta',
                      'Pupil': 'optical OTF width multiplier 1+delta; pixel sinc unchanged'},
        data_sha256={im: hashlib.sha256((base.DATA/f'{im}_hr.npy').read_bytes()).hexdigest()
                     for im in base.NAMES})
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    old_index = {(r['family'], r['scale'], r['image'], r['noise_255'], r['draw'], r['eta'], r['arm']): r
                 for r in old_rows if r['eta'] == ETA}
    rows, checks, forward = [], [], []
    w = 2*np.pi*np.fft.fftfreq(base.N)
    for family in ['Gaussian', 'Pupil']:
        true_h = estimated_camera(family, 0.)
        captures = observations(family, true_h)
        for delta in DELTAS:
            tic = time.monotonic()
            est_h = estimated_camera(family, delta)
            anchor = anchor_from(est_h)
            recovered = recovered_from(anchor)
            telescope_error = float(np.max(np.abs(recovered(w)-est_h(w))))
            assert telescope_error < 1e-9, (family, delta, telescope_error)
            for factor in [2, 3, 4]:
                def consistent(z):
                    if factor == 2:
                        return anchor(z)
                    den = recovered(z)
                    return np.divide(recovered(factor*np.asarray(z)), den,
                        out=np.zeros_like(den), where=np.abs(den) > 1e-15)
                aa = lambda z: anchor(factor*np.asarray(z)/2)
                d_ours = base.axis_map(consistent, factor)
                d_aa = base.axis_map(aa, factor)
                if factor == 2:
                    assert np.array_equal(d_ours, d_aa)
                svds = {arm: linalg.svd(d, full_matrices=False)
                        for arm, d in [('ours', d_ours), ('aa', d_aa)]}
                for ii, name in enumerate(base.NAMES):
                    truth, observed = captures[name, factor]
                    for arm, matrix in [('ours', d_ours), ('aa', d_aa)]:
                        error_y = (base.apply(matrix, truth)-observed) @ base.Y
                        forward_rmse_y = float(np.sqrt(np.mean(error_y*error_y)))
                        forward.append(dict(family=family, delta=delta, scale=factor,
                            image=name, arm=arm, rmse_y=forward_rmse_y))
                        if delta == 0 and arm == 'ours':
                            assert forward_rmse_y < 1e-9, (family, factor, name, forward_rmse_y)
                    for noise in [0, 1]:
                        for rep in range(1 if noise == 0 else 3):
                            rng = np.random.default_rng(200926+100*ii+10*factor+rep)
                            noisy = observed+(noise/255)*rng.standard_normal(observed.shape)
                            for arm in ['ours', 'aa']:
                                pred = base.solve(noisy, svds[arm], ETA)
                                row = dict(family=family, delta=delta, scale=factor, image=name,
                                    noise_255=noise, draw=rep, eta=ETA, arm=arm,
                                    psnr_y=base.psnr(pred, truth, factor),
                                    psnr_y_raw=base.psnr(pred, truth, factor, False))
                                rows.append(row)
                                if delta == 0:
                                    key = (family, factor, name, noise, rep, ETA, arm)
                                    old = old_index[key]
                                    for metric in ['psnr_y', 'psnr_y_raw']:
                                        assert abs(row[metric]-old[metric]) < 1e-8, (key, metric, row[metric], old[metric])
                    if factor == 2:
                        pair = [r for r in rows if r['family'] == family and r['delta'] == delta
                                and r['scale'] == 2 and r['image'] == name]
                        for noise in [0, 1]:
                            for rep in range(1 if noise == 0 else 3):
                                p = {r['arm']: r['psnr_y'] for r in pair if r['noise_255'] == noise and r['draw'] == rep}
                                assert abs(p['ours']-p['aa']) < 1e-12
                checks.append(dict(family=family, delta=delta, scale=factor,
                    telescope_error=telescope_error,
                    transfer_matrix_difference=float(np.linalg.norm(d_ours-d_aa))))
            print(json.dumps(dict(family=family, delta=delta, seconds=round(time.monotonic()-tic, 1))), flush=True)
    summary = summarize(rows)
    (OUT/'results.json').write_text(json.dumps(rows, indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT/'checks.json').write_text(json.dumps(checks, indent=2))
    (OUT/'forward.json').write_text(json.dumps(forward, indent=2))
    with (OUT/'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT/'runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-started), indent=2))


if __name__ == '__main__':
    main()
