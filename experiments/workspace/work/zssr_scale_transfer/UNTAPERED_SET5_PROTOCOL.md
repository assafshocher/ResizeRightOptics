# Untapered Set5 extension — 19 September 2026

User requests all five images before drawing conclusions from the baby pilot.
Scope: Set5, total reduction x2, m=1..10, seed0. Reuse all ten completed baby
models in place; train forty new models for bird, butterfly, head and woman.
No x3/x4 expansion and no kernel/support/training/selection changes.

The construction and complete scientific settings remain those of
UNTAPERED_PILOT_PROTOCOL.md. In particular radius64, density512, quadrature4097,
18-term untapered camera product, hard output-Nyquist integration, shared RR
upsampling/bicubic residual, online crop-first pairs, 10000 Adam updates, lr1e-4,
seed0, crop64/margin16. Ordinary bicubic-AA inputs are unchanged. The sampled
anchor match remains approximate. No smooth taper, ridge, back-projection or
ground-truth hyperparameter selection is introduced.

Store new models in runs/untapered_set5_x2_v1. Leave every prior run untouched.
Reuse ordinary AA, AA with PyTorch-style downsampling grid, and previous tapered
ours as controls. The grid control uses RR filters and RR upsampling, not stock
PyTorch end to end. Compute arithmetic means of all five per-image PSNR scores,
never incomplete Set5 means. Report all ten m values regardless of outcome.

Detached runner saves status and input/source hashes; explicit resume only.
Verify finite checkpoints/predictions, full training budgets, pixel-identical
bicubic baselines and identical crop schedules; recompute all comparison scores.
Verify new method source hashes match the pilot. Numerical-convergence caveats
from the pilot remain; the requested extension is an image-generalization check,
not a numerical-convergence test. One seed, five images; no significance claim.
