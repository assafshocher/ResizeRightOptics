"""Simulate two captures from one continuous Fourier scene at prescribed zoom.

Unlike a size-only resizer, `target_capture(scene, scale, output_size, ...)`
keeps the physical scale and output count independent. The scene is the
periodic Shannon reconstruction of an RGB sample array; camera transfer
is a computed pupil OTF, not a fitted PSF or a claim about the JPG camera.
The target is synthesized at the centered projected grid directly from
retained Fourier modes using separable chirp-z transforms.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import czt

from optical_cameras import CUTOFF, camera_transfer


def source_capture(scene: np.ndarray, camera: str) -> np.ndarray:
    source_size = scene.shape[0]
    if scene.shape[1] != source_size or source_size % 2 != 1:
        raise ValueError("odd square scene expected")
    frequencies = 2.0 * np.pi * np.fft.fftfreq(source_size)
    horizontal, vertical = np.meshgrid(frequencies, frequencies)
    transfer = camera_transfer(camera, horizontal, vertical)
    result = np.empty_like(scene, dtype=np.float32)
    for channel in range(scene.shape[2]):
        spectrum = np.fft.fft2(scene[..., channel]) * transfer
        result[..., channel] = np.fft.ifft2(spectrum).real.astype(np.float32)
    return result


def target_capture(
    scene: np.ndarray,
    physical_scale: float,
    output_size: int,
    camera: str,
    grid_scale: float | None = None,
) -> np.ndarray:
    """Compute K_0 M_s(scene) at G_{r,N}; r defaults to physical s.

    Passing `grid_scale=N/N0` changes only the sampling coordinates,
    leaving the optical target transfer function at physical scale s.
    """
    source_size = scene.shape[0]
    if scene.shape[1] != source_size or source_size % 2 != 1:
        raise ValueError("odd square scene expected")
    if physical_scale <= 0 or physical_scale > 1:
        raise ValueError("this simulation covers coarsening only")
    if output_size % 2 != 1:
        raise ValueError("odd output size expected")
    grid_scale = physical_scale if grid_scale is None else grid_scale
    if grid_scale <= 0:
        raise ValueError("grid scale must be positive")
    # Frequencies outside this square cannot pass the radial pupil cutoff.
    maximum_mode = int(np.ceil(CUTOFF * physical_scale * source_size / (2.0 * np.pi)))
    modes = np.arange(-maximum_mode, maximum_mode + 1, dtype=np.float64)
    horizontal, vertical = np.meshgrid(modes, modes)
    omega_x = 2.0 * np.pi * horizontal / source_size
    omega_y = 2.0 * np.pi * vertical / source_size
    transfer = camera_transfer(camera, omega_x / physical_scale, omega_y / physical_scale)
    center = source_size // 2
    start, stop = center - maximum_mode, center + maximum_mode + 1
    spectrum = np.fft.fftshift(np.fft.fft2(scene, axes=(0, 1)), axes=(0, 1))
    retained = spectrum[start:stop, start:stop, :] * transfer[..., None]

    indices = np.arange(output_size, dtype=np.float64)
    projected = (
        (indices - (output_size - 1) / 2.0) / grid_scale
        + (source_size - 1) / 2.0
    )
    first_coordinate = float(projected[0])
    ratio = np.exp(2j * np.pi / (grid_scale * source_size))
    anchor = np.exp(-2j * np.pi * first_coordinate / source_size)
    # `czt` treats the retained modes as 0,...,2K. Undo the mode origin -K.
    x_synthesized = czt(retained, m=output_size, w=ratio, a=anchor, axis=1)
    x_synthesized *= np.exp(-2j * np.pi * maximum_mode * projected / source_size)[None, :, None]
    image = czt(x_synthesized, m=output_size, w=ratio, a=anchor, axis=0)
    image *= np.exp(-2j * np.pi * maximum_mode * projected / source_size)[:, None, None]
    return (image.real / source_size**2).astype(np.float32)
