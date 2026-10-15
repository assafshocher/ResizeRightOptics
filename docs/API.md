# Resize-Right Optics

**One camera. A different, derived kernel at each scale. The ResizeRight grid.**

A research implementation building on [Assaf Shocher's ResizeRight](https://github.com/assafshocher/ResizeRight).
Like RR, the code follows the resampling steps with their explanation alongside
them. Unlike RR, an interpolation kernel is not stretched by an AA heuristic.
The camera defines a whole family:

\[
\widehat{k_s}(\omega)=H(\omega/s)H(\omega)^\dagger.
\]

There are efficient separable and genuine nonseparable N-D paths, NumPy and
PyTorch support, continuous kernel evaluation, and rational polyphase
convolutions. PyTorch gradients pass through the image and camera parameters.

**Status:** tested research code, not yet a production release. See
[VALIDATION.md](../VALIDATION.md) for measured errors, timings and tested devices.
In particular, short spatial kernels are **not** exactly consistent, and an
ill-conditioned camera inverse requires care. No data or results from the
paper experiments are modified by this project.

## Install

```bash
pip install .             # NumPy
pip install '.[torch]'    # optionally install PyTorch
pip install '.[test]'     # test dependencies
python -m pytest -q
```

## Start with a camera

```python
import numpy as np
from resize_right_optics import resize, Gaussian, Airy, Cubic

image = np.random.default_rng(0).random((96, 128, 3))  # HWC, float

# A separable Gaussian PSF, sigma in pixels.
small = resize(image, 0.6, camera=Gaussian(sigma=0.7))

# The same family at another scale: no antialiasing option is needed.
large = resize(small, 1 / 0.6, camera=Gaussian(sigma=0.7))

# If no camera is given, use the family inferred from Keys cubic at scale 1/2.
# This is NOT bicubic with its support stretched to every requested scale.
small = resize(image, 0.6)  # same as camera=Cubic(reference_scale=0.5)

# A circular-pupil camera: genuinely 2-D, not a product of two 1-D kernels.
# The Fourier path is particularly useful for its long-tailed inverse.
small = resize(image, 0.5, camera=Airy(cutoff=2.7), method="fourier")
```

These are different camera models, not quality presets. Choose the model that
describes the acquisition. In particular, a cubic-derived PSF is not certified
nonnegative/physical, and a Gaussian's spectrum is not optically band-limited.
The resampler enforces the digital Nyquist band in either case; it cannot repair
aliasing that was already present in the input image.

## PSF versus reference kernel

Keep these concepts separate in the API rather than giving `scale=0` a hidden
meaning. A PSF is accepted by `from_psf`; a resize kernel is accepted by
`from_kernel`, with its reference scale stated explicitly.

```python
from resize_right_optics import (from_psf, from_function, from_kernel,
                                from_kernel_spectrum, Camera, Separable)

# Sample values of a continuous PSF on a fine, centered grid.
psf = np.exp(-np.arange(-24, 25)**2 * 0.125**2 / (2 * 0.7**2))
camera = from_psf(psf, spacing=0.125)
small = resize(image, 0.6, camera=camera)

# Or supply the continuous PSF function. This constructor uses quadrature.
camera = from_function(lambda x: np.exp(-x*x / (2*0.7**2)),
                       support=8, spacing=0.125)

# A measured 2-D PSF is automatically treated as nonseparable.
u, v = np.meshgrid(np.linspace(-3, 3, 25), np.linspace(-3, 3, 25), indexing="ij")
psf2 = np.exp(-(u*u + v*v + 0.6*u*v))
camera2 = from_psf(psf2, spacing=0.25)

# Separate, different cameras along the specified dimensions.
camera_sep = Separable(Gaussian(0.6), Gaussian(0.9))
small = resize(image, [0.5, 0.75], dims=[0, 1], camera=camera_sep)

# If U is a known relative kernel spectrum at r=1/2:
U = lambda w: np.exp(-0.5 * 0.7**2 * (1 / 0.5**2 - 1) * w*w)
camera = from_kernel_spectrum(U, scale=0.5, terms=32)
```

`from_kernel(samples, scale=..., spacing=...)` uses the same coordinates as
`from_psf`, but interprets the values as a *relative resize kernel*. Its width
must already correspond to that reference scale, in **input-pixel units**.
Do not stretch it a second time. Both downscale and upscale anchors are
supported; scale 1 contains no information about the camera.

The camera inference is a finite approximation to an infinite product.
Increase `terms` and check convergence, especially near reference scale 1.
Not every arbitrary kernel defines a physical or well-conditioned camera.
Reference-kernel DC must be one (sampled kernels are normalized for you).

For a custom OTF, use `Camera(otf, ndim=d)`. A 1-D callable receives an array of
scalar frequencies; a d-D callable receives `(..., d)` coordinates. Return
an array in the same framework/device. A real PSF requires Hermitian symmetry,
`H(-w) = conj(H(w))`. No symmetry or nonnegativity is silently imposed.

## The grid is part of the API

For input length `n`, output length `m`, and requested scale `s`, output index
`j` reads the input coordinate

\[
x_j=(j-(m-1)/2)/s+(n-1)/2.
\]

- Scale only: output size is `ceil(n*s)` (allowing for floating-point roundoff).
- Output size only: scale is `m/n`.
- Both: **honor both**. Never replace the requested scale with `m/n`.
- A scale-1 resize with a different output size is a centered crop/extension.
- A scalar scale defaults to the first two axes in NumPy, last two in PyTorch.
  A scale list addresses that many first/last axes. For tensors, explicit
  `dims` is usually clearer. Axes may be negative or in any order.
- `out_shape` supplies the sizes for the selected axes, not necessarily the
  full tensor shape. Unselected batch/channel axes are untouched.

```python
small = resize(image, 0.6, out_shape=[58, 77], dims=[0, 1])

import torch
x = torch.rand(2, 3, 96, 128)  # NCHW
y = resize(x, [0.5, 0.75], dims=[2, 3], camera=Gaussian(0.7))
```

## Two numerical realizations

### Spatial: finite, local, RR-style

`method="spatial"` is the default. The implementation tabulates the inverse
Fourier transform of the camera ratio and evaluates that kernel at continuous
offsets on RR's grid. Its finite approximation has explicit controls:

| Parameter | Default | Meaning |
|---|---|---|
| `support_sz` | 32 separable / 12 full N-D | Full kernel width, in input pixels; scalar or per-axis list |
| `density` | 8 separable / 4 full N-D | Spatial kernel table samples per pixel |
| `period_factor` | 4 | Fourier quadrature period relative to support |
| `table_interp` | `"cubic"` | Interpolation of the kernel table; `"linear"` is cheaper |
| `window` | `"hann"` | Smooth finite-support window; also `"lanczos"` or `None` |
| `normalize` | `True` | Normalize phase weights to preserve constants |
| `pad_mode` | `"reflect"` | Also `"symmetric"`, `"edge"`, `"wrap"`, `"constant"` (zero) |

These parameters control approximation, **not the optical scale law**. Increasing
support alone does not guarantee monotonically smaller error. Test convergence
of support, density and Fourier quadrature. Abrupt truncation (`window=None`)
is especially problematic for inverse filters. Constant padding, naturally,
does not preserve a constant image outside its boundary.

Use `by_convs=True` for rational scales `p/q`. There are `p` phase filters, each
at input stride `q`, with outputs interleaved. PyTorch's separable path uses
native `conv1d`; NumPy uses strided windows. The nonseparable path uses N-D
polyphase gather/contraction, including beyond three dimensions. It does not
pretend that a nonseparable kernel can be applied axis by axis.

The scale is **never snapped** to a nearby rational. If the numerator cannot
be kept below `max_numerator` (default 16) at roundoff precision, execution
falls back to the ordinary path with a warning. The two paths share the exact
same grid, camera, finite kernel and boundary rule.

Full N-D support grows exponentially. Work is chunked, with allocation guards
`max_kernel_points` and `max_work_elements`. In high dimensions, choose smaller
support/density or use the Fourier path; the code will not silently reduce
accuracy to fit memory.

### Fourier: periodic, no finite-kernel truncation

`method="fourier"` filters the input Fourier coefficients and evaluates them on
the **same centered grid**. When `m=s*n`, a phase-corrected FFT crop/pad makes
this fast. Otherwise, a direct Fourier-series evaluation honors the requested
scale rather than substituting a different one.

This path assumes periodic boundaries; spatial padding/window/normalization
controls do not apply. It is useful as a reference and for full N-D cameras
whose spatial inverse kernels are prohibitively long. For non-compatible
extents, it may require a dense Fourier basis; memory guards apply.

## What "consistent" means here

The exact operator identities concern alias-free signals on compatible domains:

- Down-down and up-up chains compose.
- Up-down restores a valid input on the retained camera band.
- Down-up restores **the intermediate band's projection**, not lost detail.
- Mixed/anisotropic chains retain the intersection of their transported bands.

The tests check these statements to double-precision roundoff with the Fourier
path, including circular/anisotropic bands. They do **not** claim exact
identities after arbitrary finite crops, rounded periodic extents, padding,
regularization, phase normalization or kernel truncation.

Exact Nyquist-edge modes are excluded in the periodic reference. Spatial
quadrature treats the continuous band edge by its limiting integral; agreement
on that exceptional finite-array mode is not promised. Use captures strictly
inside the retained band when checking identities.

## Refinement and regularization

Canonical refinement is not a learned super-resolution prior. It can strongly
amplify noise or numerical error where the camera OTF is small—particularly
near an Airy cutoff. Use explicit regularization when appropriate:

```python
y = resize(x, 2, camera=Gaussian(1.0), regularization=0.01)
```

The denominator then becomes `abs(H)**2 + eta**2`, with `eta=regularization`.
The default is zero: no hidden ridge is inserted. `rcond=1e-12` excludes
near-null OTF frequencies relative to DC in both source and target.
Regularization trades the exact identities for stability. Neither the window
nor automatic warnings substitute for choosing a suitable regularization level.
For a camera with a hard optical cutoff, the unregularized inverse spectrum
can be singular enough that an ordinary spatial inverse-transform integral
does not converge. Sampling that spectrum more densely is not a cure; use a
regularized realization or the appropriate range-restricted Fourier operator.

For a separable camera, regularization and `rcond` apply **per axis**. A joint
N-D Tikhonov denominator generally destroys separability. If you need that
version, pass `Separable(...).as_full()` through the full path. Without these
numerical modifications, the product and separable camera laws agree.

## Differentiability

```python
sigma_raw = torch.nn.Parameter(torch.tensor(-0.4))
x = torch.rand(32, 32, requires_grad=True)
# Recreate derived parameters in each forward pass, as in any PyTorch model.
y = resize(x, 0.75, camera=Gaussian(sigma_raw.exp()))
y.square().mean().backward()
assert sigma_raw.grad is not None and x.grad is not None

# A learned sampled PSF stays differentiable too.
logits = torch.nn.Parameter(torch.zeros(9, 9))
psf = logits.flatten().softmax(0).reshape(9, 9)
y = resize(x, 0.75, camera=from_psf(psf, spacing=0.25),
           support_sz=6, density=2)
y.square().mean().backward()
```

No input or camera tensor is converted to NumPy, detached for numerical
evaluation, or cached with a stale graph. The small scalar reads used to
choose shapes/branches are not used in the differentiable arithmetic.
For scale gradients, fix `out_shape` and leave `by_convs=False`. Output size,
support selection, rational phase count and hard spectral band membership
are discrete choices—not globally differentiable operations.

Camera containers are not `nn.Module`s: register learned parameters in your
own module/optimizer. A `Camera` callback can call such a module directly.
Half/bfloat16 inputs compute in float32 then return their original dtype;
float64 stays float64. NumPy inputs reject Torch camera parameters rather
than silently detaching them.

## Code map

- `resize_right.py`: centered grid → fields of view → weights → application;
  separable, full N-D, polyphase and periodic Fourier realizations.
- `cameras.py`: camera definitions and inference from a reference kernel.
- `kernels.py`: spectrum ratio → continuously evaluable spatial kernel.
- `_backend.py`: small NumPy/PyTorch API differences.
- `tests/`: deterministic alignment, algebra, gradients, devices and safety.
- `examples/`: small runnable examples and the reproducible validation report.

The original RR baseline inspected for this implementation was commit
`510d4d5b67dccf4efdee9f311ed42609a71f17c5`. Its files are not modified or required
at runtime. This is a new API, not a drop-in replacement accepting RR's old
`interp_method`/`antialiasing` arguments.
