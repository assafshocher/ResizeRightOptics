"""Matched-kernel comparisons: change ResizeRight's grid, not its filter.

The native-grid arm is NOT unmodified torch.interpolate: reflection and
phase-normalized filter evaluation remain ResizeRight in every arm.
"""
from contextlib import contextmanager
import math
import numpy as np

from kernels import CachedResize, ContinuousKernel, interp_methods, rr
from literal_spectrum import anchored_ridge_spectrum


class LiteralAnchorKernel(ContinuousKernel):
    """Literal bicubic plus a sampled, regularized camera-law correction.

The bicubic term is evaluated analytically, preserving its tails in Fourier
space rather than projecting them away. Only the correction is tabulated.
All scales use eta=1e-3, fixed independently of images/scores. The transform
period and frequency extent are numerical integration settings, not a
Nyquist band restriction. See MATCHED_PROTOCOL.md for the changed guarantee.
    """
    ETA = .001

    def __init__(self, factor, radius, anchor, density=512, length=1 << 20,
                 omega_max=32*np.pi):
        self.factor = float(factor)
        self.radius = radius
        self.support_sz = 2*radius
        self.density = density
        self.law = "literal_anchor_ridge"
        self.anchor = float(anchor)
        omega = 2*np.pi*np.fft.rfftfreq(length, d=1/density)
        active = omega <= omega_max
        _, correction = anchored_ridge_spectrum(omega[active], factor, anchor, self.ETA)
        spectrum = np.zeros_like(omega)
        spectrum[active] = correction
        spatial = np.fft.irfft(spectrum, n=length)*density
        self.values = spatial[:radius*density+1].copy()
        self._tensors = {}
        self.diagnostics = {
            "eta": self.ETA, "fft_length": length, "density": density,
            "omega_max": float(omega_max), "anchor": anchor, "factor": factor,
            "spatial_correction_l1_tail": float(2*np.sum(np.abs(spatial[radius*density:length//2]))/density),
            "correction_at_frequency_boundary": float(np.max(np.abs(correction[-100:]))),
        }

    def __call__(self, x):
        base = interp_methods.cubic(x / self.factor) / self.factor
        return base + super().__call__(x)


def half_pixel_grid(in_sz, out_sz, scale_factor, fw, by_convs, device=None):
    if by_convs:
        raise ValueError("The controlled grid comparison does not use by_convs")
    coordinates = rr.fw_arange(out_sz, fw, device)
    return (coordinates + .5) / float(scale_factor) - .5


@contextmanager
def grid_convention(convention):
    """Local synchronous RR weight construction; never alter vendor sources."""
    original = rr.get_projected_grid
    if convention == "torch_half_pixel":
        rr.get_projected_grid = half_pixel_grid
    elif convention != "rr_centered":
        raise ValueError(convention)
    try:
        yield
    finally:
        rr.get_projected_grid = original


class MatchedTorchGridResize(CachedResize):
    def __init__(self, shape, scale, **kwargs):
        out_shape = tuple(math.floor(n * scale) for n in shape)
        if min(out_shape) < 1:
            raise ValueError((shape, scale, out_shape))
        with grid_convention("torch_half_pixel"):
            super().__init__(shape, scale, out_shape=out_shape, **kwargs)

    def direct(self, image):
        with grid_convention("torch_half_pixel"):
            return super().direct(image)
