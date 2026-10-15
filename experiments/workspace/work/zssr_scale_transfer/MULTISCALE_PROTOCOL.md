# Fixed multiscale ZSSR extension — 17 September 2026

Requested sweep: Set5 (all five images), total SR scales ×2, ×3, ×4;
learned step counts m=1,...,10 at EACH total scale. One fresh per-image,
per-method, per-m network is trained and applied m times. Each step has ratio
S^(1/m). This is not ten sequential training stages. Seed 0, no back-projection.

## Four training downsamplers

1. Camera-ratio continuous kernel (ours).
2. AA-stretched kernel with the identical fixed spectral window.
3. Ordinary ResizeRight bicubic with AA.
4. Ordinary PyTorch `interpolate` bicubic with AA: `scale_factor=1/r`,
   `align_corners=False`, `recompute_scale_factor=False`, `antialias=True`.

Only the fourth arm's training downsampler is changed to PyTorch. Every arm
uses the SAME RR bicubic training upsampler and inference path, so the CNN,
requested scales, loss, training budget, initialization, crop coordinates and
augmentation remain controlled. No shifts are manually inserted. PyTorch's
native floor-sized half-pixel grid differs naturally from RR's centered grid.
Its error cannot be attributed solely to the AA scale law; arms 1 vs 2 isolate
that law, while arms 3/4 are software references. No no-AA arm is included.

## Input and numerical gates

Each scale has its own common filtered bicubic-AA input and calibrated anchor.
At m=1, arms 1/2 coincide exactly and share one trained result. Ordinary RR and
PyTorch AA are NOT claimed to share that band-limited anchor. HR images retain
their original dimensions; rounding is handled explicitly, not by registration.

Kernel radius = 32*S (64,96,128); density 512/pixel, quadrature 4097, same C4
window and spectral construction as the previous x2 experiment. Before any new
training, compare each m=1..10 kernel against twice its support on observed LR.
Require RMSE < 2e-5; stop on failure before looking at new HR scores. This tests
spatial truncation, not all sources of numerical/learning error. Finite support
and phase normalization do not have exact continuum composition guarantees.

## Fixed training recipe and small-image crop rule

Eight 3×3 convolution layers, width64, ReLU, shared bicubic residual, zero last
layer, Adam (default betas), L1, 10,000 updates, peak LR1e-4, batch1, 20-update
warmup, decay ×0.1 at60% and ×0.01 at85%. No pretrained weights, self-ensemble,
HR-based selection or stopping. All pairs are constructed ONLINE after drawing
each fresh crop; only resampling weights and random coordinates are cached.

Loss-crop side = min(64, shortest LR dimension − 16). This predetermined rule
retains at least 17 translated crop positions along each axis and is identical
across methods. It leaves all x2 crops at64. At x3 only woman shrinks to60.
At x4: baby64, bird56, butterfly48, head54, woman41. A reflected halo supports
the filters/CNN, but the supervised crop itself lies inside the observed LR.

Preserve CPU-built RR weights for x2 to match completed results. For x3/x4,
build RR weights on GPU to match direct GPU RR rounding. Pixel operations
float32, metricsfloat64; TF32 only for CNN convolutions. cuDNN benchmarking
means runs are seeded, not guaranteed bitwise deterministic.

## Reuse, evaluation and reporting

Reuse ALL completed x2 m1..5 results from `first_image_residual_v1` for arms1–3;
do not choose the better of reruns. Add m6..10 and PyTorch m1..10 separately.
Run x3 and x4 afresh under this fixed protocol; retain the interrupted x3 baby
pilot separately, without selecting by its scores. Do not alter previous files.

Evaluate clipped float RGB predictions using BT.601 Y-PSNR, shave S HR pixels;
retain raw/unclipped and RGB metrics. Plot arithmetic mean of the five per-image
PSNR values (equal image weight) against m, with each scale's own one-shot
bicubic baseline. Absolute curves only; retain all requested points, including
any losses. One seed/five images: no significance claim or seed error bars.
These filtered-input scores are not standard literal-bicubic Set5 leaderboard
scores. Repeated learned reconstruction error can accumulate even for ours.

Batch is sequential on desk2 and independent of SSH lifetime. Record boot ID,
kernel and source hashes; stop on any numerical/training failure, without an
automatic retry or operating-system changes. Generate plots after each scale.

## Launched execution

Started 2026-09-17 at 18:19:38 UTC (21:19:38 Israel), runner PID7242.
Remote root: `<original-server>/codex-zssr-scale-transfer-20260917` on `desk2`.
Status: `runs/set5_multiscale_m1_10/status.json`.
Log: `logs/launch_1789669178831625243.log`.
Kernel: `7.0.0-30-generic`; boot ID `76156b9d-4242-407e-9da1-1359a1a4b920`.
600 requested result records; 75 already measured, 10 new matched-anchor
records reuse their exactly identical one-step counterpart, leaving515 new
network trainings. Completion is determined by the status file, not this note.

## User-requested continuation — 18 September 2026

The 17 September reboot interrupted x3/bird after all x2 measurements and
x3/baby had completed. On 18 September the user requested continuation.
The host remained on kernel7.0.0-31; boot ID
`c9ed6e39-3cdc-4ef1-baee-f5fbcf3e8bd0`. OS/driver settings were not changed.

All39 unique x3/baby model/prediction pairs were readable and finite. Bird's
three m1 model/prediction pairs were zero-length, although four score records
(including the shared anchor) survived. The ENTIRE interrupted bird folder is
preserved as `runs/set5_x3_m1_10/bird_seed0_interrupted_20260918`; bird is rerun
with unchanged settings, not selected by PSNR. All completed x2 and x3/baby
measurements are retained without retraining. No files were deleted.

Only persistence/runner logic changed: artifacts now use atomic replace plus
file/directory flush; resume skips verified complete cases and records the new
execution provenance. Training, kernels, inference, seeds and metrics are
unchanged. Read/write round trips and preservation of prior JSON on a simulated
write failure passed. Synthetic compute tests rerun before scored training.
This is one explicitly requested continuation, not an automatic reboot retry.
New runner PID21769; log `logs/launch_1789764175829494608.log`.

The user also requested plots without PyTorch. These are additional display
variants named `*_no_pytorch`, with the full data and four-method plots retained.

## CPU-affinity mitigation — 19 September 2026

Another host reboot interrupted x3/head after m6. All140 unique completed
x3 model/prediction pairs loaded successfully and were finite; they have been
backed up locally. Two supplied crash reports identify logical CPU6 at the
kernel instruction-fetch fault. Logical CPUs6/7 share physical core3.

Before continuing the suite, run synthetic checks and the remaining x3/head
cases with process affinity0–5,8–31. This excludes that core from the training
threads without changing system-wide settings or the scientific protocol.
Record and verify affinity in execution provenance. This does not establish
the root cause, and results are not selected by score. Completed measurements
are preserved; only missing cases run. See `INCIDENT_20260919.md`.

The bounded head continuation completed successfully. Its40 results and the
200 completed x2 results were re-evaluated from saved float predictions;
maximum PSNR discrepancy was7.2e-15 dB. All checkpoint tensors were finite.
The full missing-case continuation then started under affinity0–5,8–31,
runnerPID6435, log `logs/launch_1789766178902921942.log`, boot ID
`28b78966-4ce3-4084-96f6-627600b3e243`. All40 live training threads of the
initial resumed process had that mask. The root cause remains unresolved.

## Completed — 19 September 2026

The suite finished at 00:51:44 Israel (18 September, 21:51:44 UTC): all 600
requested records and all three complete five-image plots. No further reboot
occurred during the affinity-restricted continuation. All 585 unique saved
model/prediction pairs passed verification, with re-evaluated PSNRs agreeing
within 1.1e-14 dB. The 15 additional rows share identical one-step anchors.
See `verification_complete_20260919.json`; the data and figures are backed
up locally. The CPU workaround does not establish the machine's root cause.
