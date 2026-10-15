"""Spatial Gaussian kernels; their finite support is the only truncation.

Widths are in the current input's pixels. Disable ResizeRight antialiasing:
the prescribed width already contains the scale law, and RR normalizes each
sampling phase. Only the untruncated continuous family composes exactly.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
except ImportError:
    torch = None

METHODS = ("gaussian_consistent", "gaussian_aa")


def _positive(value, name, lower=0):
    value = float(value)
    if not math.isfinite(value) or value <= lower:
        raise ValueError(f"{name} must be finite and greater than {lower}")
    return value


class TruncatedGaussian:
    """Gaussian density inside +/- truncate*sigma, zero outside.

    No minimum width, support floor, Fourier cutoff, taper, or ridge is used.
    Consequently, support below one pixel can miss some sampling phases.
    """

    def __init__(self, sigma, truncate=4.0):
        self.sigma = _positive(sigma, "sigma")
        self.truncate = _positive(truncate, "truncate")
        self.radius = self.truncate * self.sigma
        self.support_sz = 2 * self.radius
        self.normalizer = self.sigma * math.sqrt(2 * math.pi)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.support_sz, self.normalizer, 1 / self.normalizer)):
            raise ValueError("sigma and truncate exceed numerical range")

    def __call__(self, x):
        if torch is not None and isinstance(x, torch.Tensor):
            density = torch.exp(-0.5 * (x / self.sigma).square()) / self.normalizer
            return torch.where(x.abs() <= self.radius, density, torch.zeros_like(density))
        x = np.asarray(x)
        density = np.exp(-0.5 * (x / self.sigma) ** 2) / self.normalizer
        return np.where(np.abs(x) <= self.radius, density, 0.0)


def gaussian_kernel(method, factor, anchor=2, base_sigma=0.5):
    """Return the kernel callable; use it with RR antialiasing=False.

    The common anchor width is base_sigma*anchor. Factoring r^2-1 reduces
    cancellation near one. Both methods take the identical anchor branch.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown Gaussian method: {method}")
    factor = _positive(factor, "factor", 1)
    anchor = _positive(anchor, "anchor", 1)
    base_sigma = _positive(base_sigma, "base_sigma")
    if factor == anchor:
        sigma = base_sigma * anchor
    elif method == "gaussian_consistent":
        sigma = base_sigma * anchor * math.sqrt(
            ((factor - 1) * (factor + 1)) / ((anchor - 1) * (anchor + 1)))
    else:
        sigma = base_sigma * factor
    return TruncatedGaussian(sigma)
