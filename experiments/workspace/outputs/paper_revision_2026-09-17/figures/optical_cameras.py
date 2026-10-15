"""Physically valid non-Gaussian camera OTFs and their relative kernels.

The lens OTF is the normalized autocorrelation of a circular pupil. The
available pupils are clear, 50%-radius centrally obscured (annular), and
even astigmatic with phase of 0.25 waves. Each includes the Fourier
transform of unit-width square pixel integration.
The cutoff is 0.9 pi radians per reference-camera pixel.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


CUTOFF = 0.9 * np.pi
PUPIL_SAMPLES = 401
ASTIGMATISM_WAVES = 0.25
ANNULAR_INNER_RADIUS = 0.5


@lru_cache(maxsize=3)
def pupil_autocorrelation(camera: str) -> tuple[np.ndarray, float]:
    if camera not in {"clear pupil", "annular pupil", "astigmatic pupil"}:
        raise ValueError(camera)
    axis = np.linspace(-1.0, 1.0, PUPIL_SAMPLES)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    support = xx * xx + yy * yy <= 1.0
    if camera == "annular pupil":
        support &= xx * xx + yy * yy >= ANNULAR_INNER_RADIUS**2
    if camera != "astigmatic pupil":
        phase = np.zeros_like(xx)
    else:
        phase = 2.0 * np.pi * ASTIGMATISM_WAVES * (xx * xx - yy * yy)
    pupil = np.where(support, np.exp(1j * phase), 0.0)
    autocorrelation = fftconvolve(pupil, np.conj(pupil[::-1, ::-1]), mode="full")
    center = PUPIL_SAMPLES - 1
    autocorrelation /= autocorrelation[center, center]
    return autocorrelation, float(axis[1] - axis[0])


def camera_transfer(camera: str, omega_x: np.ndarray, omega_y: np.ndarray) -> np.ndarray:
    """Evaluate the zoom-invariant camera symbol at pixel-frequency coordinates."""
    correlation, pupil_step = pupil_autocorrelation(camera)
    shift_x = 2.0 * np.asarray(omega_x, dtype=np.float64) / CUTOFF
    shift_y = 2.0 * np.asarray(omega_y, dtype=np.float64) / CUTOFF
    center = PUPIL_SAMPLES - 1
    coordinates = [center + shift_y / pupil_step, center + shift_x / pupil_step]
    real = map_coordinates(correlation.real, coordinates, order=1, mode="constant", cval=0.0)
    imag = map_coordinates(correlation.imag, coordinates, order=1, mode="constant", cval=0.0)
    lens = real + 1j * imag
    lens = np.where(shift_x * shift_x + shift_y * shift_y < 4.0, lens, 0.0)
    pixel_box = np.sinc(omega_x / (2.0 * np.pi)) * np.sinc(omega_y / (2.0 * np.pi))
    return lens * pixel_box


@lru_cache(maxsize=50)
def spectral_relative_kernel(input_size: int, output_size: int, camera: str) -> np.ndarray:
    """Normalized Fourier-basis multiplier for one coarsening step.

    A common center/FOV and odd sample counts ensure the same physical
    frequency modes are retained at every step. The ratio is evaluated on
    those modes only; pseudoinverse-null frequencies contribute zero.
    """
    if input_size % 2 != 1 or output_size % 2 != 1 or output_size > input_size:
        raise ValueError("odd square coarsening sizes are required")
    modes = np.arange(-(output_size // 2), output_size // 2 + 1, dtype=np.float64)
    horizontal, vertical = np.meshgrid(modes, modes)
    source_omega_x = 2.0 * np.pi * horizontal / input_size
    source_omega_y = 2.0 * np.pi * vertical / input_size
    target_omega_x = 2.0 * np.pi * horizontal / output_size
    target_omega_y = 2.0 * np.pi * vertical / output_size
    source_symbol = camera_transfer(camera, source_omega_x, source_omega_y)
    target_symbol = camera_transfer(camera, target_omega_x, target_omega_y)
    safe = np.abs(source_symbol) > 1e-10
    relative = np.zeros_like(target_symbol)
    relative[safe] = target_symbol[safe] / source_symbol[safe]
    # The centered grids differ by a scale-correct half-pixel phase.
    input_offset = -(input_size - 1) / 2.0
    output_offset = -(input_size / output_size) * (output_size - 1) / 2.0
    phase = np.exp(
        1j * (source_omega_x + source_omega_y) * (output_offset - input_offset)
    )
    relative *= phase * (output_size / input_size) ** 2
    return relative


def resize_optical(image: np.ndarray, output_size: int, camera: str) -> np.ndarray:
    input_size = image.shape[0]
    if image.shape[1] != input_size:
        raise ValueError("square input expected")
    start = (input_size - output_size) // 2
    stop = start + output_size
    multiplier = spectral_relative_kernel(input_size, output_size, camera)
    result = np.empty((output_size, output_size, image.shape[2]), dtype=np.float32)
    for channel in range(image.shape[2]):
        spectrum = np.fft.fftshift(np.fft.fft2(image[..., channel]))
        output_spectrum = spectrum[start:stop, start:stop] * multiplier
        result[..., channel] = np.fft.ifft2(np.fft.ifftshift(output_spectrum)).real.astype(np.float32)
    return result


def spatial_kernel_slice(camera: str, scale: float = 0.5, samples: int = 1025) -> np.ndarray:
    """Sample the canonical continuous K_s on an input-pixel spatial grid.

    Frequency grid spans [-pi,pi]. The displayed signed 2D kernel is the
    inverse Fourier transform of H(omega/s) H(omega)^dagger, including
    the optical and sampling cutoff; it is not the camera PSF itself.
    """
    if samples % 2 != 1:
        raise ValueError("odd frequency-grid size required")
    frequencies = 2.0 * np.pi * np.fft.fftfreq(samples)
    horizontal, vertical = np.meshgrid(frequencies, frequencies)
    source = camera_transfer(camera, horizontal, vertical)
    target = camera_transfer(camera, horizontal / scale, vertical / scale)
    safe = np.abs(source) > 1e-10
    symbol = np.zeros_like(source)
    symbol[safe] = target[safe] / source[safe]
    kernel = np.fft.fftshift(np.fft.ifft2(symbol).real)
    return kernel
