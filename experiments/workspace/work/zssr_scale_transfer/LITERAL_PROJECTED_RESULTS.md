# Ordinary-bicubic ZSSR rerun — complete

All150 fresh ours models finished on19September2026. The original retained-band
kernel law was used on ordinary ResizeRight bicubic-AA observations. The only
numerical repair was stable evaluation of the same C4 window polynomial; all
30 actual-step spectra agree with the previous evaluation within4.19e-13.
All architecture, optimization, crop schedules, seeds, and evaluation settings
remained fixed. All15 one-step models were independently trained.

## Five-image Set5 means

Mean ours minus ordinary AA, in dB:

| Scale | 1 step | 2 steps | 5 steps | 10 steps | Sustained mean crossover |
|---|---:|---:|---:|---:|---:|
| ×2 | -0.780 | -1.010 | +1.178 | +3.788 | 4 steps |
| ×3 | -0.818 | -0.963 | +0.325 | +3.321 | 5 steps |
| ×4 | -0.830 | -0.831 | -0.826 | +1.940 | 6 steps |

At10 steps:

| Scale | Ours | AA | Bicubic interpolation |
|---|---:|---:|---:|
| ×2 | 33.751 | 29.963 | 33.692 |
| ×3 | 30.565 | 27.244 | 30.420 |
| ×4 | 28.316 | 26.376 | 28.435 |

Ours beats AA at10 steps on all5 images at every scale. This supports multistep
robustness relative to AA, not universal SR superiority: AA wins at low step
counts, and ours at10 steps is close to interpolation alone, slightly below it
on average at×4. These are one-seed, five-image descriptive comparisons, not
significance claims.

## Scope and figure

figures/fig_set5_zssr_projected_six_panel.pdf and .png contain all six panels.
The top row is the unchanged completed band-limited experiment. The bottom row
uses ordinary bicubic-AA input; ours has an approximate, not exact, anchor.
AA (PyTorch grid) changes downsampling to floor/half-pixel coordinates while
keeping the RR filters and RR upsampling. It is not an unmodified end-to-end
PyTorch pipeline or a pure kernel-law comparison.

Per-image and mean CSVs: figures/literal_projected_x{2,3,4}_{per_image,means}.csv.
All previous experiments, including the failed extension, are preserved.

## Verification and local backup

-150 new plus300 reused control records; none missing.
-450 unique model/prediction pairs checked for finite values and expected shapes.
-150 matching crop-schedule comparisons; no reused ours one-step model.
-All saved PSNR values recompute within7.11e-15 dB.
-Independent audit matched all30 source-result hashes and all five-image means.
-540 new-run files, including150 checkpoints and150 floating predictions, are
  local and checksum-identical to the server copy.
-See verification_literal_projected_complete.json and
  verification_literal_projected_local_copy.json.

No manuscript changes, hyperparameter tuning, extra seeds, or result selection
were performed. The detached runner completed normally; no further training
is required for this rerun.
