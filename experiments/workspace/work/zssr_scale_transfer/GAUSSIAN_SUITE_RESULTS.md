# Gaussian sigma-2 Set5 ZSSR: complete results

Training completed 19 September 2026 at 15:41:29 UTC. All five Set5 images,
three total factors (2, 3, 4), ten learned step counts and three methods are
present: 450 score rows and 435 distinct model/prediction pairs. The 19
previously verified baby x2 models were reused wholesale; 416 models were
newly trained. No runs were removed or parameters changed based on scores.

## Protocol

Sigma=2 is the actual shared total-scale degradation width, in input pixels,
at each total factor S. It is not the unstretched AA base width: base_sigma=2/S.
The Gaussian is truncated spatially at 4 sigma; there is no Fourier cutoff,
taper, ridge or minimum support. All three methods share this observed image.

Ours follows sigma(r)=2 sqrt((r^2-1)/(S^2-1)); matched AA uses sigma(r)=2r/S.
The third arm changes only AA's downsampling coordinates/output size to the
PyTorch-style floor/half-pixel convention. Filtering, reflection, normalization,
and all common bicubic upsampling remain ResizeRight. It is not unmodified
PyTorch end-to-end and not an isolated AA-law comparison.

Same eight-layer residual CNN, online crop-first pairs, seed0, 10000 updates,
initialization/crop schedule and inference protocol throughout. No
back-projection or HR-based selection. One-step ours/AA filters agree exactly
and share one trained model. Each other setting trains its own model, then
applies it repeatedly m times. See GAUSSIAN_SUITE_PROTOCOL.md for all settings.

Scores are arithmetic means of five image-level BT.601 Y-PSNR values, with S
HR border pixels excluded. Plotted scores clip final RGB to [0,1]; raw scores
and output ranges are retained. These are controlled one-seed experiments,
not claims of statistical significance or a published leaderboard protocol.

## Results at fixed step counts

| Total factor | Steps | Ours | Matched AA | AA, PyTorch grid | Bicubic interpolation |
|---|---:|---:|---:|---:|---:|
| x2 | 1 | 33.94 | 33.94 | 33.92 | 27.78 |
| x2 | 3 | 33.85 | 19.86 | 19.90 | 27.78 |
| x2 | 10 | 32.26 | 7.80 | 7.40 | 27.78 |
| x3 | 1 | 31.27 | 31.27 | 29.94 | 27.50 |
| x3 | 3 | 32.70 | 23.98 | 19.41 | 27.50 |
| x3 | 10 | 27.60 | 10.11 | 9.33 | 27.50 |
| x4 | 1 | 27.43 | 27.43 | 27.02 | 27.04 |
| x4 | 3 | 30.09 | 27.28 | 22.92 | 27.04 |
| x4 | 10 | 24.99 | 14.74 | 12.93 | 27.04 |

All values are dB, rounded only here. Full curves and per-image values are
saved in runs/gaussian_sigma2_suite_v1/summary_xS.json and summary.json.

## Supported finding and limits

The Gaussian scale law gives higher mean PSNR than matched AA at every tested
multi-step setting (m=2..10), at all three total scales. This is a downstream
learned-SR effect, not merely numerical verification of a composition identity.

It does **not** mean that more steps always improve ours. Its observed mean
peaks occur at m=2 (x2), m=3 (x3), and m=4 (x4), followed by degradation.
These are descriptive peaks in this sweep, not independently selected/tested
hyperparameters. At m=10 ours is only 0.096 dB above the x3 bicubic baseline;
at x4 it is 2.044 dB below the baseline, and below it on every image.

The continuum Gaussian law is exact; finite spatial support and sampling are
not an exact sampled semigroup. At fixed sigma2, the higher total factors have
less alias suppression. These experiments alone do not establish the cause of
ours' late-step decline; do not label it solely numerical error.

The AA arms show substantial output-amplitude instability during repeated
inference, not only blur. At ten steps, mean raw Y-PSNR is:

| Total factor | Ours | Matched AA | AA, PyTorch grid |
|---|---:|---:|---:|
| x2 | 32.26 | -12.25 | -13.13 |
| x3 | 27.59 | 2.44 | 1.31 |
| x4 | 24.98 | 12.00 | 9.88 |

Thus clipping conceals some of the competing arms' instability; all raw
outputs and both metric versions remain available. Do not interpret the
large late-step gaps as ordinary one-shot-resizing gains.

## Artifacts and verification

- Combined plot: figures/fig_gaussian_sigma2_all_scales.png and .pdf.
- Full scale plots: figures/fig_gaussian_sigma2_x2/x3/x4.png and .pdf.
- All artifacts: runs/gaussian_sigma2_suite_v1/.
- Kernel preflight: runs/gaussian_sigma2_suite_v1/kernel_checks.json.
- Independent final audit: GAUSSIAN_SUITE_AUDIT.json — passed.
- Local/server backup comparison: GAUSSIAN_SUITE_BACKUP.json — passed.

The runner checked all 150 actual training-grid configurations before training,
recomputed each saved prediction's four scores, and verified complete finite
checkpoints, crop-coordinate pairing, source/data hashes and full summaries.
An independent NumPy float64 audit recomputed all four scores for all 435 saved
predictions and 15 baselines, checked all 435 CPU-loaded checkpoints and all
450 score rows, independently reconstructed crop hashes, and verified frozen
source/data hashes. Maximum score discrepancy was 7.11e-15 dB; all summary
means matched exactly. Local SHA-256 comparison matched all 1431 server files
in the audit manifest, including all 1406 run-folder files: no missing, changed,
or extra run files. The final x4 and combined figures were visually inspected.
