# Gaussian pilot: baby x2, one seed

Completed19September2026,14:30:45UTC. Both predeclared widths are retained.
The existing ZSSR training recipe is unchanged; see GAUSSIAN_PILOT_PROTOCOL.md.
All filtering and resampling uses ResizeRight. Gaussian support is +/-4sigma;
there is no added frequency cutoff, taper or regularization. Ours uses the
Gaussian variance-difference law; AA stretches the common base Gaussian.
The one-step maps agree exactly and the one-step network is shared in each pair.

## Y-PSNR (dB), clipped predictions

| Steps | sigma1 ours | sigma1 AA | sigma2 ours | sigma2 AA |
|---:|---:|---:|---:|---:|
|1|38.563|38.563|37.167|37.167|
|2|38.633|36.131|36.904|28.894|
|3|38.109|31.478|36.753|21.558|
|4|36.546|27.844|36.411|16.379|
|5|34.473|25.712|36.186|12.828|
|6|32.030|23.140|36.062|9.942|
|7|30.720|21.760|36.143|8.665|
|8|30.114|19.815|35.792|7.655|
|9|28.977|19.001|35.828|7.236|
|10|34.933|17.200|35.456|7.030|

Sigma denotes the common x2 anchor standard deviation in input pixels, not
the elementary-step sigma. Bicubic interpolation baselines:35.198(sigma1)
and31.126(sigma2). Different widths generate different inputs; do not compare
absolute PSNR across widths as if their reconstruction problem were identical.

## Interpretation and adverse findings

- Wider sigma2 is promising for repeated-step robustness: ours remains above
  its bicubic baseline throughout. Its best score is nevertheless at one step.
- Narrow sigma1 beats AA at each multistep count but is not stable: ours falls
  below baseline at steps5..10. The abrupt9->10 rise is present in saved arrays;
  no artifact/configuration discrepancy was found. Its cause is not established.
- Sigma2 AA deteriorates into severe output instability. At m10 its raw range
  is -13.334..17.821 and87.489% of RGB channel samples lie outside[0,1]. Its
  rawY-PSNR is -8.932dB; clipping gives the reported7.030dB. This is not merely
  a small loss of perceptual sharpness. Do not state a cause as verified.
- Same sigma2 ours m10 range is -.020..1.009, with.381% outside[0,1].
- One image/seed is preliminary evidence, not a Set5 mean or significance test.
- Full synthetic checks pass their actual unit gates (finite weights, constant
  preservation, identical anchor, cached/direct equivalence), but intentionally
  do not gate composition error. Gaussian continuum closure is exact; sampled,
  finite-support composition is not. The narrow Gaussian has substantial
  fractional-phase errors that remain explicitly reported.

## Verification and artifacts

38 unique checkpoints/predictions,40 records; both independent verification
passes reloaded all checkpoints and recomputed all four PSNR variants plus
baselines (max discrepancy7.11e-15dB). All predictions finite512x512; crop hashes
match for each paired m. All132 files in runs/gaussian_baby_x2_v1 have matching
local/server SHA256 manifests. Core experiment.py and kernels.py unchanged.

- Full-precision summary: runs/gaussian_baby_x2_v1/summary.json
- Plot: figures/fig_gaussian_baby_x2_pilot.png and .pdf
- Full numerical checks: gaussian_kernel_checks_full.json
- Source: gaussian_kernels.py,check_gaussian_kernels.py,gaussian_pilot.py
- Log: logs/gaussian_launch_1789827933685691823.log

Training-recipe checks preserved the existing comparison settings; the
academic-plotting workflow shows both fixed widths, small markers and a common
legend, with vectorPDF and300dpiPNG. No unfavorable points are omitted.
