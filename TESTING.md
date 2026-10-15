# Test plan

The central distinction is between the camera law and its finite implementation.
Exact identities are tested on alias-free Fourier modes with compatible periodic
grids. Finite spatial kernels are tested for convergence, not declared exact.

1. Independent grid oracles: centered coordinates, impulses, even/odd sizes,
   explicit scale versus inferred scale, singleton axes, crops and nonintegers.
2. Implementation equivalence: NumPy/PyTorch, ordinary/polyphase convolution,
   separable/full N-D, all supported padding modes and permuted tensor axes.
3. Algebra: monotone composition, commutativity when bands permit it, up-down,
   down-up equals the retained-band projection, irrational and rational scales.
4. Cameras: Gaussian, Airy, cubic anchor, sampled/custom PSFs, asymmetric PSFs,
   anisotropy, interior zeros, regularization and finite-product error.
5. Gradients: image, PSF samples, sigma, covariance, scale on a fixed output grid;
   finite-difference gradcheck and repeated optimizer steps (no stale graphs).
6. Safety: invalid arguments, allocation limits, dtype/device, optional Torch,
   read-only/noncontiguous arrays and tiny inputs with large padding.
7. Report actual accuracy and timings. GPU claims require GPU runs. Rounded
   finite crops and boundary extension are not covered by the exact semigroup.
