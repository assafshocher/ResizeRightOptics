# Matched-anchor ZSSR: completed results

Completed 19 September 2026, 02:52 Israel. All six panels contain all five Set5 images, one seed, total factors ×2/×3/×4 and learned step counts 1–10. No runs were discarded or parameters adjusted based on scores.

900 scored records correspond to 870 unique model/prediction pairs: 585 newly trained and 285 reused from the completed band-limited ours/AA runs. All saved checkpoints were finite; recomputed PSNR matched the recorded scores within 1.1e-14 dB. Completed artifacts are retained locally and on desk2.

## Main finding

The band-limited comparison supports the benefit of the scale law for multi-step learned SR. The literal-bicubic **regularized extension does not**: it loses to matched AA at every tested multi-step setting in the Set5 mean. Its failure must not be presented as a positive robustness result, or as a test of an exact alias-free theorem. The cause is not established by these runs.

The PyTorch-grid arm changes ONLY the downsampling coordinates/output size while keeping the matched AA filter, RR normalization and reflection. The shared upsampler remains RR-centered. Its large degradation therefore involves a mismatch of coordinate conventions; it is neither an isolated AA-law effect nor a benchmark of unmodified PyTorch end-to-end resizing.

## Ours minus matched AA at ten steps (dB)

| Input mode | ×2 | ×3 | ×4 |
|---|---:|---:|---:|
| Band-limited bicubic | 4.54 | 4.32 | 3.29 |
| Literal bicubic; ours regularized | -4.92 | -4.69 | -6.41 |

## Protocol qualifications

- Literal-mode ours uses a fixed AA-centered Tikhonov extension, eta=0.001, chosen before any literal-mode SR scores. It preserves the literal cubic anchor exactly and retains out-of-band content. It is not the exact band-limited family; see [protocol](MATCHED_PROTOCOL.md).
- Ours and AA share the one-step network in each mode. The grid arm shares the continuous kernel but can have different sample positions.
- Same model, seed, initialization, crop schedule and 10,000-update budget across arms. Training pairs are generated after each fresh crop. No back-projection or GT-dependent stopping/selection.
- Metric: clipped float predictions, BT.601 Y, shave S HR pixels; arithmetic mean of five per-image PSNR values. One seed/five images does not establish statistical significance. Original image dimensions and RR grids are retained; these are not asserted to reproduce published leaderboard protocols.
- The academic-plotting workflow supplied consistent method colors/markers, readable labels and vector PDF plus 300-DPI PNG exports. All favorable and unfavorable points remain visible.

## Figures and machine-readable data

[Six-panel PDF](figures/fig_set5_zssr_matched_six_panel.pdf) · [PNG](figures/fig_set5_zssr_matched_six_panel.png) · [Verification](verification_matched_complete.json).

Each panel also has an individual PDF/PNG and full-precision means/per-image CSVs in figures/. The table values below are rounded to two decimals only for readability.

## Band-limited bicubic ×2

Bicubic interpolation only: **33.18 dB**.

| Steps | Ours | AA | AA (PyTorch grid) |
|---:|---:|---:|---:|
| 1 | 35.43 | 35.43 | 35.46 |
| 2 | 35.74 | 35.44 | 33.16 |
| 3 | 35.62 | 34.75 | 34.52 |
| 4 | 35.45 | 33.98 | 25.44 |
| 5 | 35.30 | 33.02 | 19.58 |
| 6 | 34.99 | 31.96 | 24.96 |
| 7 | 34.84 | 31.13 | 20.38 |
| 8 | 34.45 | 29.56 | 17.26 |
| 9 | 34.29 | 29.79 | 16.97 |
| 10 | 33.81 | 29.28 | 18.96 |

[Per-image CSV](figures/matched_bandlimited_x2_per_image.csv) · [Means CSV](figures/matched_bandlimited_x2_means.csv).

## Band-limited bicubic ×3

Bicubic interpolation only: **29.96 dB**.

| Steps | Ours | AA | AA (PyTorch grid) |
|---:|---:|---:|---:|
| 1 | 30.37 | 30.37 | 29.17 |
| 2 | 31.23 | 31.08 | 21.64 |
| 3 | 31.49 | 31.05 | 22.21 |
| 4 | 31.61 | 30.45 | 24.75 |
| 5 | 31.61 | 29.94 | 22.61 |
| 6 | 31.55 | 29.38 | 17.96 |
| 7 | 31.47 | 28.75 | 26.06 |
| 8 | 31.31 | 28.26 | 17.16 |
| 9 | 31.00 | 27.34 | 18.93 |
| 10 | 30.93 | 26.61 | 18.01 |

[Per-image CSV](figures/matched_bandlimited_x3_per_image.csv) · [Means CSV](figures/matched_bandlimited_x3_means.csv).

## Band-limited bicubic ×4

Bicubic interpolation only: **28.01 dB**.

| Steps | Ours | AA | AA (PyTorch grid) |
|---:|---:|---:|---:|
| 1 | 26.36 | 26.36 | 25.95 |
| 2 | 28.46 | 28.47 | 27.33 |
| 3 | 28.82 | 28.62 | 23.94 |
| 4 | 28.74 | 28.05 | 22.38 |
| 5 | 28.64 | 27.57 | 22.55 |
| 6 | 28.79 | 27.20 | 18.71 |
| 7 | 28.83 | 26.69 | 17.39 |
| 8 | 28.67 | 26.12 | 17.58 |
| 9 | 28.70 | 25.77 | 17.50 |
| 10 | 28.50 | 25.21 | 18.84 |

[Per-image CSV](figures/matched_bandlimited_x4_per_image.csv) · [Means CSV](figures/matched_bandlimited_x4_means.csv).

## Literal bicubic (ours: regularized) ×2

Bicubic interpolation only: **33.69 dB**.

| Steps | Ours | AA | AA (PyTorch grid) |
|---:|---:|---:|---:|
| 1 | 36.41 | 36.41 | 36.42 |
| 2 | 34.32 | 36.26 | 33.54 |
| 3 | 32.00 | 35.39 | 35.24 |
| 4 | 32.02 | 34.56 | 25.50 |
| 5 | 31.21 | 33.80 | 19.54 |
| 6 | 30.10 | 32.62 | 25.35 |
| 7 | 29.12 | 31.69 | 20.53 |
| 8 | 27.95 | 30.39 | 17.25 |
| 9 | 26.32 | 30.30 | 16.99 |
| 10 | 25.04 | 29.96 | 19.29 |

[Per-image CSV](figures/matched_literal_x2_per_image.csv) · [Means CSV](figures/matched_literal_x2_means.csv).

## Literal bicubic (ours: regularized) ×3

Bicubic interpolation only: **30.42 dB**.

| Steps | Ours | AA | AA (PyTorch grid) |
|---:|---:|---:|---:|
| 1 | 31.37 | 31.37 | 30.02 |
| 2 | 29.55 | 32.16 | 21.63 |
| 3 | 28.30 | 31.95 | 22.22 |
| 4 | 25.02 | 31.38 | 24.82 |
| 5 | 23.25 | 30.68 | 22.61 |
| 6 | 22.88 | 29.96 | 17.94 |
| 7 | 22.99 | 29.30 | 26.73 |
| 8 | 22.97 | 29.03 | 17.14 |
| 9 | 22.91 | 27.69 | 18.95 |
| 10 | 22.55 | 27.24 | 18.05 |

[Per-image CSV](figures/matched_literal_x3_per_image.csv) · [Means CSV](figures/matched_literal_x3_means.csv).

## Literal bicubic (ours: regularized) ×4

Bicubic interpolation only: **28.43 dB**.

| Steps | Ours | AA | AA (PyTorch grid) |
|---:|---:|---:|---:|
| 1 | 27.45 | 27.45 | 26.83 |
| 2 | 19.52 | 29.33 | 27.96 |
| 3 | 24.90 | 29.39 | 24.20 |
| 4 | 24.84 | 29.18 | 22.66 |
| 5 | 22.46 | 28.69 | 22.82 |
| 6 | 19.80 | 28.08 | 18.75 |
| 7 | 20.84 | 27.49 | 17.43 |
| 8 | 18.78 | 26.89 | 17.71 |
| 9 | 19.89 | 26.58 | 17.71 |
| 10 | 19.97 | 26.38 | 18.98 |

[Per-image CSV](figures/matched_literal_x4_per_image.csv) · [Means CSV](figures/matched_literal_x4_means.csv).

