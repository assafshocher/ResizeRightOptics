# Gaussian ZSSR pilot — fixed before neural results, 19 September 2026

Scope: baby only, total scale S=2, learned step counts m=1..10, seed0.
Two predeclared Gaussian widths: base_sigma=.5 and1, so the actual x2 anchor
standard deviations are1 and2 input pixels. Both are reported, not selected
by their scores. This is a pilot, not a five-image Set5 average.

For elementary reduction r=S^(1/m), in that stage's input-pixel coordinates:

    sigma_consistent = base_sigma*S*sqrt((r*r-1)/(S*S-1))
    sigma_AA = base_sigma*r

Evaluate exp(-x*x/(2*sigma*sigma)) for |x|<=4*sigma and zero outside.
ResizeRight supplies its normal per-phase normalization. At r=S both methods
use the SAME kernel implementation and same RR grid. Verify their weight maps
before training and reuse the identical m1 network only after this check.
Create each observed input by applying that same anchor to the original HR.
No Fourier cutoff, spectral taper, ridge, back-projection, or phase correction.
The reference law is exact for continuous, untruncated Gaussians; sampled and
spatially truncated realization is only approximate. Small-sigma phase coverage
and chain errors must be measured and reported, not silently repaired.

Network/training: reuse experiment.train and experiment.infer unchanged via a
process-local degradation callback. Eight layers, width64, shared bicubic skip,
Adam lr1e-4,10000 updates, existing warmup/decay, batch1, crop64/margin16,
halo radius64, online crop-first pairs, identical seed/crop schedule for arms.
Every resize uses RR; bicubic interpolation is the shared upsampler and baseline.
No PyTorch-grid arm in this initial kernel-only test.

Metrics: existing clipped and unclipped RGB/Y PSNR, shave2 HR pixels; no target
pixels passed to training, no stopping/model selection using HR. Preserve all
outcomes. Record hashes and verify saved predictions/checkpoints/crop schedules.
Synthetic checks precede neural scores. Truncation beyond4sigma is numerical
approximation, not an assertion of band limitation or exact sampled composition.

New files only; previous bicubic experiments and their source remain unchanged.
Runner: gaussian_pilot.py --launch on desk2, optionally --resume after confirming
no live runner. It detaches from SSH, locks its output, and saves durable status.
Output: runs/gaussian_baby_x2_v1; figures/fig_gaussian_baby_x2_pilot.png/.pdf.
Do not automatically expand to all images or scales based on the pilot outcome.
