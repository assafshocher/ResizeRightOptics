"""Unselected smooth x2-anchor noise in the frozen shared-solver experiment.

Twenty sequential RNG seeds, no screening. The seed distribution is symmetric
under sign reversal, but each realization has an independent random sign.
Only the estimated x2 anchor changes; both arms receive it identically.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy import linalg

import check_shared_solver_transfer as base
import check_psf_misspec_transfer as misspec
from check_smooth_anchor_error import direct_anchor, relative_errors, coefficients


OUT = base.ROOT / "outputs/unscreened_anchor_noise_2026-10-01"
LEVELS = [0., .01, .02, .05]
SEEDS = list(range(1000, 1020))  # No seed or outcome screening.
ETA = .01


def summarize(rows):
    group = {}
    for row in rows:
        key = (row["family"], row["target_error"], row["seed"], row["scale"],
               row["noise_255"], row["image"], row["arm"])
        group.setdefault(key, []).append(row["psnr_y"])
    out = []
    for family in ["Gaussian", "Pupil"]:
        for level in LEVELS:
            seeds = [-1] if level == 0 else SEEDS
            for scale in [3, 4]:
                for noise in [0, 1]:
                    gains = []
                    for seed in seeds:
                        per_image = []
                        for image in base.NAMES:
                            ours = np.mean(group[(family, level, seed, scale, noise, image, "ours")])
                            aa = np.mean(group[(family, level, seed, scale, noise, image, "aa")])
                            per_image.append(ours-aa)
                        gains.append(float(np.mean(per_image)))
                    out.append(dict(family=family, target_error=level, scale=scale,
                        noise_255=noise, seed_gains=gains, n_seeds=len(gains),
                        mean=float(np.mean(gains)), median=float(np.median(gains)),
                        seed_sd=float(np.std(gains, ddof=1)) if len(gains)>1 else 0.,
                        seed_range=[float(min(gains)), float(max(gains))],
                        negative_seeds=int(sum(g < 0 for g in gains))))
    return out


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = base.OUT / "results.json"
    frozen = json.loads(frozen_path.read_text())
    frozen_index = {(r["family"], r["scale"], r["image"], r["noise_255"],
                     r["draw"], r["eta"], r["arm"]): r for r in frozen}
    protocol = dict(parent_script=str(Path(base.__file__).relative_to(base.ROOT)),
        parent_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_results_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        families=["Gaussian", "Pupil"], images=base.NAMES, n=base.N,
        seed_selection="20 sequential seeds 1000..1019, selected before outcomes and not screened",
        coefficient_distribution="three independent zero-mean Gaussian coefficients normalized by L1; symmetric under global sign reversal",
        perturbation="U_est(w)=U_true(w)[1+a sum_k c_k(cos(k w)-1)], k=1,2,3",
        guarantees="smooth, even, real, U_est(0)=1; no guarantee of nonnegative underlying PSF",
        target_anchor_errors=LEVELS, error_metric="relative Fourier L2 weighted by |H_true(w)|² on 192-point grid",
        eta=ETA, noise_255=[0, 1], noise_repeats=3,
        boundaries="periodic", grid="centered, S*j+(S-1)/2",
        truth="independent captures with unperturbed true PSF",
        controls="same erroneous anchor to both arms; same finite forward maps, solver, and paired noise draws",
        data_sha256={im: hashlib.sha256((base.DATA / f"{im}_hr.npy").read_bytes()).hexdigest()
                     for im in base.NAMES})
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows, diagnostics = [], []
    for family in ["Gaussian", "Pupil"]:
        h = misspec.estimated_camera(family, 0.)
        true_anchor = misspec.anchor_from(h)
        captures = misspec.observations(family, h)
        for level in LEVELS:
            for seed in ([-1] if level == 0 else SEEDS):
                c = np.zeros(3) if seed == -1 else coefficients(seed)
                estimated, amplitude = direct_anchor(true_anchor, c, level, h)
                actual, spatial = relative_errors(true_anchor, estimated, h)
                assert abs(actual-level) < 1e-10
                assert abs(float(estimated(np.array([0.]))[0])-1.) < 1e-12
                x2_a = base.axis_map(estimated, 2)
                x2_b = base.axis_map(lambda z: estimated(2*np.asarray(z)/2), 2)
                assert np.array_equal(x2_a, x2_b)
                diagnostics.append(dict(family=family, target_error=level, seed=seed,
                    amplitude=amplitude, actual_weighted_error=actual,
                    spatial_kernel_l2_error=spatial, x2_max_matrix_diff=0.))
                recovered = misspec.recovered_from(estimated)
                for scale in [3, 4]:
                    def consistent(z):
                        den = recovered(z)
                        return np.divide(recovered(scale*np.asarray(z)), den,
                            out=np.zeros_like(den), where=abs(den)>1e-15)
                    aa = lambda z: estimated(scale*np.asarray(z)/2)
                    svds = {arm: linalg.svd(base.axis_map(q, scale), full_matrices=False)
                            for arm, q in [("ours", consistent), ("aa", aa)]}
                    for ii, image in enumerate(base.NAMES):
                        truth, observed = captures[image, scale]
                        for noise in [0, 1]:
                            for rep in range(1 if noise == 0 else 3):
                                rng = np.random.default_rng(200926 + 100*ii + 10*scale + rep)
                                noisy = observed + (noise/255)*rng.standard_normal(observed.shape)
                                for arm in ["ours", "aa"]:
                                    pred = base.solve(noisy, svds[arm], ETA)
                                    row = dict(family=family, target_error=level, seed=seed,
                                        scale=scale, image=image, noise_255=noise,
                                        draw=rep, eta=ETA, arm=arm,
                                        psnr_y=base.psnr(pred, truth, scale),
                                        psnr_y_raw=base.psnr(pred, truth, scale, False))
                                    rows.append(row)
                                    if level == 0:
                                        key = (family, scale, image, noise, rep, ETA, arm)
                                        for metric in ["psnr_y", "psnr_y_raw"]:
                                            assert abs(row[metric]-frozen_index[key][metric]) < 1e-8
        print(json.dumps(dict(family=family, completed=True)), flush=True)

    summary = summarize(rows)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "runtime.json").write_text(json.dumps(dict(seconds=time.monotonic()-t0), indent=2))
    lines = ["# Unselected smooth ×2-anchor noise", "",
        "Independent follow-up to the prescreened pilot: 20 sequential, unselected RNG seeds per nonzero level. The perturbation distribution is sign-symmetric, smooth, even, real, and DC-preserving, applied directly to the estimated ×2 anchor. It need not correspond to a nonnegative spatial PSF. True captures, five Set5 scenes, noise draws, centered grid, periodic boundaries, and η=0.01 solver are unchanged; both arms receive the same erroneous anchor.", "",
        "Calibrated error is relative Fourier L2 in U₂ weighted by |H_true(ω)|². The unweighted relative spatial circular-kernel L2 and every seed are in diagnostics.json. Zero-error rows reproduce the frozen baseline; ×2 arm matrices tie exactly for every seed.", "",
        "PSNR gain is camera law minus AA stretch, averaged over five scenes (and three paired draws for noise 1/255). SD and negative counts are across perturbation seeds, not scenes.", "",
        "| Camera | U₂ error | Noise /255 | ×3 mean; median; SD; negative | ×4 mean; median; SD; negative |",
        "|---|---:|---:|---:|---:|"]
    for family in ["Gaussian", "Pupil"]:
        for level in LEVELS:
            for noise in [0, 1]:
                pair = [next(s for s in summary if
                    (s["family"], s["target_error"], s["scale"], s["noise_255"]) ==
                    (family, level, scale, noise)) for scale in [3, 4]]
                fmt = lambda s: (f'{s["mean"]:+.2f}; {s["median"]:+.2f}; '
                                 f'{s["seed_sd"]:.2f}; {s["negative_seeds"]}/{s["n_seeds"]}')
                lines.append(f"| {family} | {level:.0%} | {noise} | {fmt(pair[0])} | {fmt(pair[1])} |")
    lines += ["", "This is a controlled sensitivity result, not a real-PSF-estimation benchmark. In particular, an error percentage alone cannot characterize an estimator: smooth perturbation shape and systematic bias matter.\n"]
    (OUT / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
