# Fixed Gaussian sigma-2 ZSSR suite

Frozen before launch. Extend the completed Gaussian pilot to all five Set5 images
(baby, bird, butterfly, head, woman), total factors 2, 3, 4, and learned step
counts 1 through 10. One seed (0), no parameter search or target-based selection.

## Degradation and three arms

Sigma=2 means the **actual shared total-scale degradation** has standard deviation
2 input pixels at each total factor S. It does not mean the unstretched AA base
has sigma 2: the base parameter is 2/S. Generate new observed images from the
original HR arrays using ResizeRight and this Gaussian, spatially truncated at
4 sigma. Do not clip or quantize observed images. No frequency cutoff, taper,
ridge, support floor, or band projection is added.

For an elementary downscale factor r=S^(1/m), use:

- Ours: sigma(r)=2 sqrt((r^2-1)/(S^2-1)).
- AA: sigma(r)=2r/S, on the ordinary ResizeRight centered grid.
- AA (PyTorch grid): exactly the AA filter, on the existing floor-sized,
  half-pixel PyTorch-style downsampling grid. Filtering remains ResizeRight.

All three use the same ordinary ResizeRight bicubic upsampling. The grid arm is
not stock PyTorch end-to-end. Ours and AA share the exact anchor callable and
matrix at m=1 and share that one trained checkpoint. The grid arm trains
independently, including m=1. Its grid need not agree at the anchor.

This is the continuous Gaussian law with a finite spatial implementation; it is
not an exact band-limited sampled semigroup. In particular sigma=2 suppresses
aliasing less at x4 than at x2. Check and report approximation diagnostics rather
than changing the degradation based on scores.

## Training and scoring

Reuse experiment.py without edits: online crop-then-downscale pairs, eight-layer
64-channel CNN with the previously adopted shared bicubic residual, Adam/L1,
10,000 iterations, batch 1, core crop 64 (existing small-image adjustment),
learning rate 1e-4 with existing schedule, radius/halo parameter 32S. Same seed,
initialization and crop-coordinate schedule within each comparison. No
back-projection, ensemble, extra training data, or HR-dependent stopping. Train
one fresh network for each elementary factor; apply it m times at inference.

Use the previous protocol's scale-sized HR border shave. Save RGB and BT.601 Y
PSNR, clipped and unclipped, and output min/max/out-of-range fraction. Average
the five image-level dB values with equal weights. One seed: no significance
claim or error bars. Keep all ten steps, favorable or not.

## Checks, provenance, and persistence

Preflight the actual crop dimensions, CPU/CUDA weight-construction paths, all
three arms, finite weights, constants, cached/direct equivalence, grid formulas,
and exact ours/AA anchor equality. Record phase and composition diagnostics.
Freeze source and original-image SHA-256 hashes. Recompute scores from every
saved prediction and verify finite complete checkpoints and paired crop hashes.

Reuse the completed baby x2 sigma-2 pilot's ours/AA records only after checking
its source hashes, training configuration, input and baseline pixel equality.
Preserve explicit pilot provenance; no selective reuse. This is 450 method-score
rows, 435 distinct trained models, of which 19 already exist in the pilot.

The runner uses a single-instance lock, atomic durable artifacts and a detached
process. Resume only the same frozen configuration; do not silently restart an
outcome-dependent failure. Save a first-baby plot and a complete-five-image plot
for each scale, then a combined three-panel plot. Never label a partial mean as
the full Set5 result.
