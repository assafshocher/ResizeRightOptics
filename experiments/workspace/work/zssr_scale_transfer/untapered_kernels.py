"""Direct bicubic telescope on the retained Fourier band, with no taper/ridge.

The integration boundary is the output Nyquist frequency, as in the earlier
Fourier realization. It is not an unrestricted full-spectrum extension or an
exact emulation of literal spatial bicubic-AA at every sampling phase.
"""
import numpy as np

from kernels import ContinuousKernel, cubic_spectrum


def untapered_camera(omega, anchor=2, terms=18):
    if anchor < 2:
        raise ValueError("The supported anchor reduction is >= 2")
    omega = np.asarray(omega, dtype=np.float64)
    value = np.ones_like(omega)
    for j in range(terms):
        value *= cubic_spectrum(omega / anchor**j)
    return value


def untapered_multiplier(omega, factor, anchor=2):
    omega = np.asarray(omega, dtype=np.float64)
    if factor < 1 or np.any(np.abs(factor*omega) > np.pi + 1e-9):
        raise ValueError("Evaluate only on the output's retained band")
    return untapered_camera(factor*omega, anchor) / untapered_camera(omega, anchor)


class UntaperedKernel(ContinuousKernel):
    """Same RR LUT and quadrature machinery; only the smooth taper is absent."""
    def __init__(self, factor, radius=64, density=512, quadrature=4097, anchor=2):
        if factor < 1:
            raise ValueError("factor is a reduction factor >= 1")
        self.factor = float(factor)
        self.radius = radius
        self.support_sz = 2*radius
        self.density = density
        self.law = "consistent_untapered"
        self.anchor = float(anchor)
        x = np.arange(radius*density+1, dtype=np.float64)/density
        omega = np.linspace(0, np.pi/factor, quadrature)
        spectrum = untapered_multiplier(omega, factor, anchor)
        if not np.isfinite(spectrum).all():
            raise ArithmeticError("Nonfinite untapered multiplier")
        self.values = np.empty_like(x)
        for start in range(0, len(x), 128):
            block = x[start:start+128]
            self.values[start:start+len(block)] = np.trapezoid(
                np.cos(block[:, None]*omega)*spectrum, omega, axis=1)/np.pi
        self._tensors = {}
