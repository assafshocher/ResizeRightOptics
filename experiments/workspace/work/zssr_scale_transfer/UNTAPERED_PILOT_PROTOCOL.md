# Direct untapered bicubic pilot — 19 September 2026

User asked to test the direct construction after discovering that the prior
ZSSR family included an extra smooth spectral taper. This pilot removes that
taper and adds no replacement window, ridge, or full-spectrum extension.

For anchor reduction S, H(w)=product_j=0..17 U(w/S^j), where U is the Fourier
transform of the actual RR Keys cubic. The elementary multiplier is H(rw)/H(w).
Evaluate on |w|<=pi/r, the output Nyquist band of the earlier Fourier realization.
This last hard frequency boundary remains explicit: "untapered" does not mean
an unrestricted full-spectrum quotient. Literal spatial RR bicubic contains
replicas; exact all-phase equality to it is not asserted or manufactured.

## Fixed pilot, selected before scores

- baby image, ordinary bicubic-AA input at total scale2, m=1..10.
- New independent networks, including m1; reuse no scores as new training.
- Same old8-layer width64 CNN, shared bicubic residual, seed0,10000 updates,
  Adam lr1e-4, same warmup/decay, batch1, crop64/crop-margin16, radius64,
  density512/quadrature4097, RR weights onCPU, same online crop-first pairs.
- Existing ordinary-AA, tapered-ours, and interpolation results are controls.
- No back-projection, HR tuning, model selection, or score-based retuning.
- Output: runs/untapered_baby_x2_v1/baby_seed0; prior files remain untouched.

## Numerical checks before learning

Check the telescope and anchor in the Fourier domain, phase-dependent sampled
response, near-identity behavior, finite values, and support-doubling behavior.
Measure native-AA anchor mismatch on observed LR only. No HR pixels or SR scores
are used for these checks. The hard cutoff has longer spatial tails, so report
finite-radius errors directly rather than implying FFT-machine-precision
composition for the truncated spatial implementation. Do not add a taper to
make the diagnostic errors smaller.

Report all10 pilot scores (one image, one seed), recompute saved predictions'
PSNR, verify finite checkpoints and identical crop-coordinate hashes. This is
not a Set5 average and does not authorize silently replacing the completed
150-run comparison. The pilot runner is detached and saves status/provenance.
