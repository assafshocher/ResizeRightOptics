# Ordinary-bicubic input, original retained-band kernels — 19 September 2026

User requested150 fresh ours-only trainings: five Set5 images, total scales
x2/x3/x4, learned step counts1..10. This replaces no files from earlier runs.

## Sole scientific change from the failed literal extension

Use the existing `consistent` kernel law in kernels.py: original
bicubic-derived camera with fixed C4 spectral window, original radius32*S,
density512 and quadrature4097. Remove no window and add no full-spectrum ridge.
Numerical repair authorized19September: evaluate the identical C4 polynomial
by symmetry near its cutoff to avoid cancellation to zero. Regression checks
compare all30 actual-step spectra with the prior evaluation (tolerance1e-10);
the near-identity gate and all original acceptance thresholds remain unchanged.
Train AND evaluate on unmodified literal RR bicubic-AA observations *_lr.npy.
This is fresh self-training on those observations, NOT the frozen-model
cross-input diagnostic from the investigation.

The earlier projected fractional kernels do not have an exact literal-bicubic
anchor. This mismatch is disclosed, not corrected by an extra mechanism.
All150 models, including15 m=1 models, are independently trained. Never reuse
the literal AA one-step score or claim the two one-step results must coincide.

Everything else remains fixed:8-layer width64 ReLU CNN, shared bicubic residual,
zero last layer, seed0, online crop-first training, same crop/augmentation
schedule, loss crop min(64,shortest LR dimension-16), radius-dependent common
halo,10000 Adam updates, L1, peak learning rate1e-4,20-update warmup and the
existing decay schedule. No back-projection, no pretraining, no GT tuning,
no score-based restarts. All resampling remains ResizeRight. x2 weights CPU,
x3/x4 weights GPU as previously. No system/driver/power modifications.

## Pre-score checks and early feedback

test_literal_projected.py checks actual kernel taps over fractional phases,
near-identity RR behavior, and radius-doubling convergence on observed literal
LR pixels (no original HR scoring). Failures stop before training; do not tune
these gates or kernels from reconstruction scores.

First complete job is baby,x2,m1..10. Its first_image_check.json compares all
ten scores with the existing literal AA and failed-extension scores, with no
partial Set5 average. Report this milestone promptly, favorable or unfavorable.
The fixed suite continues unless an actual numerical/execution failure occurs;
underperformance alone is reported, not used to silently retune or select runs.

## Reuse and evaluation

Existing ordinary-bicubic AA, grid arm, and interpolation-only baselines are
reused unchanged from matched_literal_x{S}_v1. The grid arm changes only
downsampling coordinates/output size; upsampling remains RR-centered, so it
is not a pure kernel-law comparison or unmodified PyTorch end-to-end pipeline.

Evaluate saved float predictions with the existing clipped BT.601 Y-PSNR,
shaveS pixels, and all-five-image mean. Retain all outcomes. One seed/five
images, no significance claim. Verify predictions and finite checkpoints,
crop hashes against old competitors, and score recomputation before final plots.

## Execution and outputs

New run folders: runs/literal_projected_x{2,3,4}_v1/{image}_seed0.
Suite: runs/literal_projected_suite_v1/status.json and first_image_check.json.
Launcher: launch.py --literal-projected-suite, detached on desk2, affinity0-5,8-31.
Resume only missing cases with identical configuration, after checking failure;
do not duplicate a live runner or overwrite completed scores.

verify_literal_projected.py generates new verification, summaries and CSVs.
figures/gen_fig_literal_projected.py writes new figure names, reusing the
unchanged completed band-limited top row. All old six-panel plots and the
diagnosed failed-extension results remain intact as historical artifacts.

No manuscript changes are part of this rerun.
