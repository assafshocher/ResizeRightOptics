"""Resize on ResizeRight's centered grid, with kernels derived from a camera.

The steps deliberately follow the original ResizeRight: project the grid,
find each pixel's field of view, calculate weights, and apply them. The new
step is deriving those weights from the *same camera* at every scale.
"""
import math
import numbers
import warnings
from fractions import Fraction
from itertools import product
import numpy as np
from ._backend import (array, arange, framework, moveaxis, stack, mesh, take,
                       cat, fftfreq, is_torch, torch)
from ._backend import scalar
from .cameras import Camera, Cubic, Separable
from .kernels import Kernel, transfer


def projected_grid(in_size, out_size, scale, like):
    """Map output pixel centers to input pixel coordinates, without rounding s."""
    return (arange(out_size, like) - (out_size - 1) / 2) / scale + (in_size - 1) / 2


def _sequence(value):
    if value is None:
        return None
    if np.isscalar(value) or (is_torch(value) and value.ndim == 0):
        return [value]
    return list(value)


def _geometry(input, scale_factors, out_shape, dims):
    scales, sizes = _sequence(scale_factors), _sequence(out_shape)
    if scales is None and sizes is None:
        raise ValueError("supply scale_factors or out_shape")
    if dims is None:
        length = max(len(scales or []), len(sizes or []))
        length = min(2, input.ndim) if length == 1 and sizes is None else length
        dims = tuple(range(input.ndim - length, input.ndim) if is_torch(input) else range(length))
    elif isinstance(dims, numbers.Integral):
        dims = (dims,)
    else:
        dims = tuple(dims)
    if not dims or any(not isinstance(d, numbers.Integral) or isinstance(d, bool)
                       or not -input.ndim <= d < input.ndim for d in dims):
        raise ValueError("dims must be nonempty valid tensor axes")
    dims = tuple(int(d) % input.ndim for d in dims)
    if len(set(dims)) != len(dims):
        raise ValueError("dims must not repeat")
    if scales is not None and len(scales) == 1:
        scales *= len(dims)
    if (scales is not None and len(scales) != len(dims)) or (sizes is not None and len(sizes) != len(dims)):
        raise ValueError("scales and out_shape must match dims (or use one shared scale)")
    if scales is not None and any(isinstance(s, bool) or not math.isfinite(scalar(s)) or scalar(s) <= 0 for s in scales):
        raise ValueError("scales must be finite and positive")
    if sizes is not None and any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in sizes):
        raise ValueError("output sizes must be positive integers")
    if scales is None:
        scales = [m / input.shape[d] for d, m in zip(dims, sizes)]
    if sizes is None:
        # Guard only floating point roundoff at an integer, not a material
        # discrepancy. The scale itself is never replaced by out/in.
        lengths = [input.shape[d] * scalar(s) for d, s in zip(dims, scales)]
        sizes = [max(1, math.ceil(x - 8 * math.ulp(x))) for x in lengths]
    return dims, scales, sizes


def _pad_indices(index, size, mode):
    fw = framework(index)
    valid = (index >= 0) & (index < size)
    if mode == "constant" or mode in ("edge", "replicate"):
        mapped = index.clamp(0, size - 1) if is_torch(index) else index.clip(0, size - 1)
    elif mode in ("wrap", "circular"):
        mapped = index % size
    elif mode == "reflect":
        t = index % max(1, 2 * (size - 1))
        mapped = fw.minimum(t, 2 * (size - 1) - t) if size > 1 else t * 0
    elif mode == "symmetric":
        t = index % (2 * size)
        mapped = fw.minimum(t, 2 * size - 1 - t)
    else:
        raise ValueError(f"unknown pad_mode {mode!r}")
    return mapped, valid if mode == "constant" else fw.ones_like(valid)


def _field(grid, support):
    fw = framework(grid)
    # An extra endpoint is harmless (the kernel is zero outside support)
    # and keeps all phases identical at integer/half-integer boundaries.
    left = fw.floor(grid - support / 2)
    left = left.to(fw.int64) if is_torch(left) else left.astype("int64")
    return left[..., None] + arange(math.ceil(support) + 2, grid, integer=True)


def _normalize(weights, enabled):
    if not enabled:
        return weights
    fw = framework(weights)
    total = weights.sum(-1, keepdim=True) if is_torch(weights) else weights.sum(-1, keepdims=True)
    if bool((abs(total) < 32 * fw.finfo(weights.dtype).eps).any()):
        raise ValueError("Kernel has a nearly zero phase sum; enlarge support or disable normalize")
    return weights / total


def _rational(scale, max_numerator):
    if is_torch(scale) and scale.requires_grad:
        raise ValueError("by_convs fixes rational phases; use by_convs=False for scale gradients")
    s = float(scale)
    candidate = 1 / Fraction(1 / s).limit_denominator(max_numerator)
    # Strictly roundoff-sized tolerance: this optimizes a scale, never snaps
    # an approximately rational requested scale onto a different grid.
    if abs(float(candidate) - s) > 16 * math.ulp(s):
        warnings.warn("Scale has no small rational numerator; using ordinary resampling without changing the grid.",
                      RuntimeWarning, stacklevel=3)
        return None
    return candidate.numerator, candidate.denominator


def _spatial_1d(input, axis, out_size, scale, kernel, pad_mode, normalize, chunk_size):
    image = moveaxis(input, axis, 0)
    grid = projected_grid(len(image), out_size, scale, input)
    output = []
    for start in range(0, out_size, chunk_size):
        g = grid[start:start + chunk_size]
        neighbors = _field(g, kernel.support[0])
        weights = _normalize(kernel(g[:, None] - neighbors), normalize)
        indices, valid = _pad_indices(neighbors, len(image), pad_mode)
        values = take(image, indices)
        weights = (weights * valid).reshape(weights.shape + (1,) * (image.ndim - 1))
        output.append((values * weights).sum(1))
    return moveaxis(cat(output), 0, axis)


def _convolution_1d(input, axis, out_size, scale, kernel, pad_mode, normalize, rational):
    """p convolutions, each at stride q, interleaved for a p/q resize.

    Torch uses native conv1d. NumPy uses strided windows and contraction.
    Both are correlations: the weights are already in neighbor order.
    """
    p, q = rational
    fw = framework(input)
    image = moveaxis(input, axis, 0)
    grid = projected_grid(len(image), out_size, scale, input)
    pieces, ordering = [], []
    for phase in range(min(p, out_size)):
        g = grid[phase:phase + 1]
        neighbors = _field(g, kernel.support[0])[0]
        weights = _normalize(kernel(g[0] - neighbors), normalize)
        count = (out_size - 1 - phase) // p + 1
        indices = neighbors[0] + arange((count - 1) * q + len(neighbors), input, integer=True)
        mapped, valid = _pad_indices(indices, len(image), pad_mode)
        values = take(image, mapped) * valid.reshape((-1,) + (1,) * (image.ndim - 1))
        columns = values.reshape(len(values), -1).T
        if is_torch(input):
            out = fw.nn.functional.conv1d(columns[:, None, :], weights[None, None, :], stride=q)[:, 0, :].T
        else:
            windows = np.lib.stride_tricks.sliding_window_view(columns, len(weights), axis=-1)[:, ::q]
            out = (windows @ weights).T
        pieces.append(out.reshape((count,) + image.shape[1:]))
        ordering.extend(range(phase, out_size, p))
    permutation = array(np.argsort(ordering), input, integer=True)
    return moveaxis(take(cat(pieces), permutation), 0, axis)


def _spatial_nd(input, dims, sizes, scales, kernel, pad_mode, normalize,
                chunk_size, max_work_elements, rationals=None):
    """Full N-D fields of view; never factor a nonseparable kernel.

    In polyphase mode each Cartesian phase uses one shared kernel and its
    input windows advance by the denominator strides. This is an N-D
    strided convolution expressed as gather/contraction, including d>3.
    """
    fw = framework(input)
    ndim = len(dims)
    image = moveaxis(input, dims, tuple(range(ndim)))
    shape, passive = image.shape[:ndim], image.shape[ndim:]
    flat = image.reshape(math.prod(shape), math.prod(passive))
    strides = [math.prod(shape[d + 1:]) for d in range(ndim)]
    counts = [math.ceil(w) + 2 for w in kernel.support]
    neighbors = math.prod(counts)
    per_output = neighbors * max(1, flat.shape[1])
    if per_output > max_work_elements:
        raise ValueError(f"One N-D field needs {per_output:,} elements; reduce support or raise max_work_elements")
    chunk_size = min(chunk_size, max(1, max_work_elements // per_output))
    offsets = stack(mesh([arange(n, input, integer=True) for n in counts])).reshape(-1, ndim)
    grids = [projected_grid(n, m, s, input) for n, m, s in zip(shape, sizes, scales)]
    output, positions = [], []
    # With no phase decomposition one group covers the entire output grid.
    ps = [r[0] for r in rationals] if rationals else [1] * ndim
    for phase in product(*[range(min(p, m)) for p, m in zip(ps, sizes)]):
        axes = [arange((m - 1 - a) // p + 1, input, integer=True) * p + a
                for a, p, m in zip(phase, ps, sizes)]
        index = stack(mesh(axes)).reshape(-1, ndim)
        shared = None
        if rationals:
            anchor = stack([grids[d][phase[d]] for d in range(ndim)], axis=0)
            left = fw.floor(anchor - array(kernel.support, input) / 2)
            shared = _normalize(kernel(anchor - (left + offsets)), normalize)
        for start in range(0, len(index), chunk_size):
            ix = index[start:start + chunk_size]
            g = stack([grids[d][ix[:, d]] for d in range(ndim)])
            if rationals:
                # Derive integer window shifts, not floating floor(g).
                # This avoids phase-boundary drift at rational coordinates.
                shifts = stack([(ix[:, d] - phase[d]) // ps[d] * rationals[d][1]
                                for d in range(ndim)])
                lefts = left + shifts
                weights = shared[None, :]
            else:
                lefts = fw.floor(g - array(kernel.support, input) / 2)
            raw = lefts[:, None, :] + offsets[None, :, :]
            raw = raw.to(fw.int64) if is_torch(raw) else raw.astype("int64")
            if not rationals:
                weights = _normalize(kernel(g[:, None, :] - raw), normalize)
            linear = fw.zeros_like(raw[..., 0])
            valid = fw.ones_like(raw[..., 0], dtype=fw.bool if is_torch(input) else bool)
            for d in range(ndim):
                mapped, ok = _pad_indices(raw[..., d], shape[d], pad_mode)
                linear = linear + mapped * strides[d]
                valid = valid & ok
            values = take(flat, linear)
            output.append((values * (weights * valid)[..., None]).sum(1))
            positions.append(sum(ix[:, d] * math.prod(sizes[d + 1:]) for d in range(ndim)))
    pos = cat(positions)
    order = fw.argsort(pos)
    result = take(cat(output), order).reshape(tuple(sizes) + passive)
    return moveaxis(result, tuple(range(ndim)), dims)


def _fourier(input, dims, sizes, scales, camera, regularization, rcond, max_work_elements):
    """Periodic reference: filter Fourier coefficients, evaluate on the RR grid.

    Evaluation is a Fourier sum, not a size-ratio FFT resize. Thus explicit
    scales and rounded sizes still use exactly the requested coordinates.
    Compatible periodic extents (m=s*n) are needed for exact chain identities.
    """
    fw = framework(input)
    image = moveaxis(input, dims, tuple(range(len(dims))))
    shape = image.shape[:len(dims)]
    frequencies = [2 * math.pi * fftfreq(n, 1, input) for n in shape]
    spectrum = fw.fft.fftn(image, dim=tuple(range(len(dims)))) if is_torch(input) else fw.fft.fftn(image, axes=tuple(range(len(dims))))
    grid = stack(mesh(frequencies))
    multiplier = transfer(camera, grid[..., 0] if camera.ndim == 1 else grid,
                          scales, regularization, rcond)
    result = spectrum * multiplier.reshape(tuple(shape) + (1,) * (image.ndim - len(dims)))
    compatible = all(abs(n * scalar(s) - m) <= 16 * math.ulp(float(m))
                     and not (is_torch(s) and s.requires_grad)
                     for n, m, s in zip(shape, sizes, scales))
    if compatible:
        # When periods agree, evaluating the Fourier series is just an IFFT.
        # The phase below retains RR's center alignment, including even/odd
        # changes. A plain FFT crop/pad would have the wrong grid origin.
        for axis, (n, m, s) in enumerate(zip(shape, sizes, scales)):
            k = fw.round(fftfreq(m, 1, input) * m)
            index = (k.to(fw.int64) if is_torch(k) else k.astype('int64')) % n
            inside = (k >= -(n // 2)) & (k <= (n - 1) // 2)
            offset = (n - 1) / 2 - (m - 1) / (2 * s)
            phase = (m / n) * fw.exp(2j * math.pi * k * offset / n) * inside
            result = take(result, index, axis=axis)
            broadcast = [1] * result.ndim
            broadcast[axis] = m
            result = result * phase.reshape(broadcast)
        axes = tuple(range(len(dims)))
        result = fw.fft.ifftn(result, dim=axes) if is_torch(input) else fw.fft.ifftn(result, axes=axes)
        return moveaxis(result.real, axes, dims)
    for axis, (n, m, s, w) in enumerate(zip(shape, sizes, scales, frequencies)):
        if n * m > max_work_elements:
            raise ValueError("Non-period-compatible Fourier evaluation needs a dense n-by-m basis. "
                             "Use spatial resampling, compatible extents, or raise max_work_elements.")
        coordinates = projected_grid(n, m, s, input)
        basis = fw.exp(1j * coordinates[:, None] * w[None, :]) / n
        along = moveaxis(result, axis, 0)
        result = moveaxis((basis @ along.reshape(n, -1)).reshape((m,) + along.shape[1:]), 0, axis)
    return moveaxis(result.real, tuple(range(len(dims))), dims)


def resize(input, scale_factors=None, out_shape=None, *, camera=None, dims=None,
           by_convs=False, method="spatial", support_sz=None, density=None,
           period_factor=4, pad_mode="reflect", normalize=True, window="hann",
           table_interp="cubic",
           regularization=0.0, rcond=1e-12, max_numerator=16,
           chunk_size=256, max_kernel_points=1048576, max_work_elements=8388608):
    """Resize a real floating NumPy array or PyTorch tensor.

    Default axes follow RR: NumPy's first two, Torch's last two for a shared
    scale; a scale list addresses that many axes. Explicit ``dims`` removes
    this convention. ``out_shape`` lists sizes only for those axes.

    ``method='spatial'`` uses a finite kernel; ``by_convs=True`` selects the
    rational polyphase implementation. ``method='fourier'`` is a periodic
    reference (no boundary padding or finite-kernel approximation).

    The default camera is inferred from Keys cubic at scale 1/2. This is a
    cubic-derived family, not a claim of a nonnegative physical PSF.
    No antialiasing switch: the camera ratio includes the scale law. Inputs
    are assumed to be alias-free captures; this function cannot undo aliasing.
    """
    if not isinstance(input, np.ndarray) and not is_torch(input):
        raise TypeError("input must be a NumPy array or PyTorch tensor")
    if input.ndim == 0 or any(n == 0 for n in input.shape):
        raise ValueError("input must have nonempty dimensions")
    if (is_torch(input) and (not input.is_floating_point())) or (not is_torch(input) and input.dtype.kind != "f"):
        raise TypeError("input must be real floating point; convert integer images explicitly")
    if method not in ("spatial", "fourier"):
        raise ValueError("method must be 'spatial' or 'fourier'")
    if scalar(regularization) < 0 or rcond < 0 or not math.isfinite(scalar(regularization)) or not math.isfinite(rcond):
        raise ValueError("regularization and rcond must be nonnegative")
    if any(not isinstance(n, int) or n < 1 for n in (chunk_size, max_numerator, max_kernel_points, max_work_elements)):
        raise ValueError("chunk and allocation limits must be positive integers")
    dims, scales, sizes = _geometry(input, scale_factors, out_shape, dims)
    # Work in at least float32 (FFT support and conditioning), then restore
    # the input dtype. Float64 is preserved throughout, including gradients.
    original_dtype = input.dtype
    if is_torch(input):
        input = input.float() if input.dtype in (torch.float16, torch.bfloat16) else input
        scales = [array(s, input) if is_torch(s) else s for s in scales]
    elif input.dtype.itemsize < 4:
        input = input.astype(np.float32)
    camera = Cubic() if camera is None else camera
    if not isinstance(camera, (Camera, Separable)):
        raise TypeError("camera must be a Camera or Separable; use from_psf/from_kernel for arrays")
    separable = isinstance(camera, Separable) or camera.ndim == 1
    cameras = list(camera.cameras) if isinstance(camera, Separable) else [camera] * len(dims)
    if (separable and len(cameras) != len(dims)) or (not separable and camera.ndim != len(dims)):
        raise ValueError("camera dimensionality must match the resized axes")
    flags = [by_convs] * len(dims) if isinstance(by_convs, bool) else list(by_convs)
    if len(flags) != len(dims) or any(not isinstance(f, bool) for f in flags):
        raise ValueError("by_convs must be a bool or one bool per resized axis")
    if method == "fourier" and any(flags):
        raise ValueError("by_convs is a spatial implementation, not a Fourier option")
    if not separable and any(flags) and not all(flags):
        raise ValueError("a nonseparable kernel needs all-or-none by_convs")
    widths = ([support_sz] * len(dims) if support_sz is None or np.isscalar(support_sz) else list(support_sz))
    if len(widths) != len(dims):
        raise ValueError("support_sz must match dims")
    widths = [w if w is not None else (32 if separable else 12) for w in widths]
    density = (8 if separable else 4) if density is None else density
    options = dict(density=density, period_factor=period_factor, regularization=regularization,
                   rcond=rcond, max_kernel_points=max_kernel_points, window=window, table_interp=table_interp)
    result = input
    if separable:
        # Reductions first saves work; cameras/scales remain attached to their
        # original axes even when the execution order changes.
        for j in sorted(range(len(dims)), key=lambda j: scalar(scales[j])):
            d, s, m, cam = dims[j], scales[j], sizes[j], cameras[j]
            if method == "fourier":
                result = _fourier(result, (d,), (m,), (s,), cam, regularization, rcond, max_work_elements)
            else:
                kernel = Kernel(cam, (s,), result, support=widths[j], **options)
                rational = _rational(s, max_numerator) if flags[j] else None
                field_elements = (math.ceil(widths[j]) + 2) * (result.numel() // result.shape[d]
                                  if is_torch(result) else result.size // result.shape[d])
                if field_elements > max_work_elements:
                    raise ValueError(f"One field needs {field_elements:,} elements; reduce support/batch size "
                                     "or raise max_work_elements")
                step = min(chunk_size, max(1, max_work_elements // field_elements))
                result = (_convolution_1d(result, d, m, s, kernel, pad_mode, normalize, rational)
                          if rational else _spatial_1d(result, d, m, s, kernel, pad_mode, normalize, step))
    elif method == "fourier":
        result = _fourier(result, dims, sizes, scales, camera, regularization, rcond, max_work_elements)
    else:
        kernel = Kernel(camera, scales, result, support=widths, **options)
        rationals = [_rational(s, max_numerator) for s in scales] if all(flags) else None
        if rationals and any(r is None for r in rationals):
            rationals = None
        result = _spatial_nd(result, dims, sizes, scales, kernel, pad_mode, normalize,
                             chunk_size, max_work_elements, rationals)
    return result.to(original_dtype) if is_torch(result) else result.astype(original_dtype, copy=False)
