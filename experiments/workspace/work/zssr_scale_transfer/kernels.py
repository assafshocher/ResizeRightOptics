"""Continuous, separable kernels for the ResizeRight interpolation interface.

The canonical family below is calibrated to the *retained-band* continuous
bicubic-AA anchor symbol, not silently equated to literal library bicubic-AA.
The latter has out-of-band leakage and sampling replicas. Audit both.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "ResizeRight"))
import interp_methods
import resize_right as rr


@lru_cache(maxsize=1)
def cubic_spectrum_table():
    # Integrate the actual ResizeRight Keys function, not a library substitute.
    x = np.linspace(0, 2, 4097)
    values = interp_methods.cubic(x)
    omega = np.linspace(0, np.pi, 8193)
    transfer = np.empty_like(omega)
    for start in range(0, len(omega), 128):
        block = omega[start:start + 128]
        transfer[start:start + len(block)] = 2 * np.trapezoid(
            np.cos(block[:, None] * x) * values, x, axis=1)
    transfer /= transfer[0]
    return omega, transfer


def cubic_spectrum(omega):
    grid, values = cubic_spectrum_table()
    if np.max(np.abs(omega)) > np.pi + 1e-9:
        raise ValueError("The camera is defined only inside its digital band")
    return np.interp(np.abs(omega), grid, values)


def band_window(omega):
    """Fixed C4 roll-off over the top 20% of the band (not fitted to images).

    It is identically one below 0.8*pi and zero at/beyond pi. Since it is
    one throughout the half band, the x2 telescope has the simple anchor
    U(2*omega)*W(2*omega). Smoothness controls spatial-tail truncation error.
    """
    t = np.clip((np.abs(omega)/np.pi - 0.8) / 0.2, 0, 1)
    # The smoothstep complement obeys W(t) = 1-W(1-t). Evaluate its
    # smaller tail directly: subtracting the expanded polynomial from one
    # rounds a positive window to zero near the cutoff (e.g. pi/1.0001).
    u = np.minimum(t, 1-t)
    tail = u**5 * (126 + u*(-420 + u*(540 + u*(-315 + 70*u))))
    window = np.where(t <= .5, 1-tail, tail)
    return np.where(np.abs(omega) >= np.pi, 0., window)


def camera(omega, anchor=2):
    if anchor < 2:
        raise ValueError("The supported anchor reduction is >= 2")
    result = np.ones_like(omega, dtype=np.float64)
    for j in range(18):
        result *= cubic_spectrum(omega / anchor**j)
    return result * band_window(omega)


class ContinuousKernel:
    """An even continuous scalar function, densely tabulated in input pixels.

    Inverse cosine quadrature evaluates the camera-ratio multiplier. Linear
    interpolation supplies arbitrary fractional offsets to ResizeRight.
    Finite support is an explicit numerical approximation, not a theorem.
    """
    def __init__(self, factor, radius=64, density=512, quadrature=4097, law="consistent", anchor=2):
        if factor < 1:
            raise ValueError("factor is a reduction factor >= 1")
        self.factor = float(factor)
        self.radius = radius
        self.support_sz = 2 * radius
        self.density = density
        self.law = law
        self.anchor = float(anchor)
        x = np.arange(radius * density + 1, dtype=np.float64) / density
        omega = np.linspace(0, np.pi / factor, quadrature)
        if law == "aa_bandlimited" or abs(factor - anchor) < 1e-12:
            # Reuse the common anchor exactly; also test the telescope identity.
            spectrum = cubic_spectrum(factor * omega) * band_window(factor * omega)
        elif law == "consistent":
            spectrum = camera(factor * omega, anchor) / camera(omega, anchor)
        else:
            raise ValueError(law)
        self.values = np.empty_like(x)
        for start in range(0, len(x), 128):
            block = x[start:start + 128]
            self.values[start:start + len(block)] = np.trapezoid(
                np.cos(block[:, None] * omega) * spectrum, omega, axis=1) / np.pi
        self._tensors = {}

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            absolute = np.abs(x)
            return np.interp(absolute, np.arange(len(self.values)) / self.density,
                             self.values, right=0) * (absolute < self.radius)
        key = (x.device, x.dtype)
        if key not in self._tensors:
            self._tensors[key] = torch.as_tensor(self.values, device=x.device, dtype=x.dtype)
        table = self._tensors[key]
        position = x.abs() * self.density
        index = position.long().clamp(0, len(table) - 2)
        fraction = (position - index).clamp(0, 1)
        return (table[index] * (1 - fraction) + table[index + 1] * fraction) * (x.abs() < self.radius)


class CachedResize:
    """Cache ResizeRight's linear weights, not any resized training images.

    Applying RR to an identity basis gives its exact separable linear map,
    including boundary handling, normalization, scale and center-grid rules.
    Reusing that map avoids reconstructing indices/weights at every SGD step.
    Unit tests compare these applications against unmodified rr.resize.
    """
    def __init__(self, shape, scale, out_shape=None, method=interp_methods.cubic,
                 antialiasing=True, device="cuda", pad_mode="reflect", weight_device="cpu"):
        self.shape = tuple(shape)
        self.out_shape = tuple(out_shape or [math.ceil(n * scale) for n in shape])
        self.scale = scale
        self.method = method
        self.antialiasing = antialiasing
        self.pad_mode = pad_mode
        self.weight_device = weight_device
        self.axes = []
        for n, target in zip(self.shape, self.out_shape):
            basis = torch.eye(n, device=weight_device)[None, None]
            weights = rr.resize(basis, scale_factors=[1, scale], out_shape=[n, target],
                                interp_method=method, antialiasing=antialiasing,
                                pad_mode=pad_mode)[0, 0].T.contiguous()
            self.axes.append(weights.to(device))

    def __call__(self, image):
        if tuple(image.shape[-2:]) != self.shape:
            raise ValueError((image.shape, self.shape))
        return torch.matmul(torch.matmul(self.axes[0], image), self.axes[1].T)

    def direct(self, image):
        return rr.resize(image, scale_factors=self.scale, out_shape=self.out_shape,
                         interp_method=self.method, antialiasing=self.antialiasing,
                         pad_mode=self.pad_mode)
