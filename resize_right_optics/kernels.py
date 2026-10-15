"""From one camera spectrum to a continuous kernel at a requested scale."""
import math
import warnings
import numbers
from itertools import product
from ._backend import array, arange, framework, fftfreq, mesh, stack, is_torch
from .cameras import keys


def transfer(camera, omega, scales, regularization=0.0, rcond=1e-12):
    """H(w/s) H(w)^dagger, restricted to source and target Nyquist bands.

    ``regularization`` is eta in conj(H)/(abs(H)**2 + eta**2).
    A nonzero eta is useful with noise but changes the exact identities.
    ``rcond`` removes near-zero source OTF values relative to DC; it defines
    one fixed source band, rather than clipping a different ratio per scale.
    """
    fw = framework(omega)
    scales = stack([array(s, omega) for s in scales], axis=0)
    scaled = omega / (scales[0] if camera.ndim == 1 else scales)
    # Values outside either digital band never enter the result. Query DC
    # there, so an extrapolated/learned spectrum need not evaluate far-away
    # frequencies merely because the spatial lookup table is oversampled.
    source_band, target_band = abs(omega) < math.pi, abs(scaled) < math.pi
    if camera.ndim > 1:
        source_band, target_band = source_band.all(-1), target_band.all(-1)
    a = source_band[..., None] if camera.ndim > 1 else source_band
    b = target_band[..., None] if camera.ndim > 1 else target_band
    source = camera(fw.where(a, omega, fw.zeros_like(omega)))
    target = camera(fw.where(b, scaled, fw.zeros_like(scaled)))
    dc = abs(camera(fw.zeros_like(omega.reshape(-1)[:camera.ndim])
                    if camera.ndim == 1 else fw.zeros_like(omega.reshape(-1, camera.ndim)[:1]))).max()
    band = source_band & target_band
    # Removing the same near-null frequencies at either scale is necessary
    # for a consistent retained-band interpretation.
    band = band & (abs(source) > rcond * dc) & (abs(target) > rcond * dc)
    safe = fw.where(band, source, fw.ones_like(source))
    eta = array(regularization, omega)
    inverse = fw.conj(safe) / (abs(safe) ** 2 + eta ** 2)
    return fw.where(band, target * inverse, fw.zeros_like(target))


class Kernel:
    """A continuously evaluable, differentiable, finite-support kernel table.

    Fourier quadrature, table interpolation and spatial truncation
    are controlled independently. ``support`` is a full width per axis.
    No table is persisted across resize calls, so learned PSFs remain live.
    """
    def __init__(self, camera, scales, like, support=32, density=8,
                 period_factor=4, regularization=0.0, rcond=1e-12,
                 max_kernel_points=1048576, window="hann", table_interp="cubic"):
        self.ndim = camera.ndim
        self.support = ((float(support),) * self.ndim if isinstance(support, numbers.Real)
                        else tuple(float(s) for s in support))
        if len(self.support) != self.ndim or min(self.support) <= 0 or not all(math.isfinite(s) for s in self.support):
            raise ValueError("support must give a positive width per kernel dimension")
        if density < 1 or period_factor < 2 or not math.isfinite(density) or not math.isfinite(period_factor):
            raise ValueError("density must be >=1 and period_factor >=2")
        self.density = density
        if window not in (None, "hann", "lanczos"):
            raise ValueError("window must be None, 'hann', or 'lanczos'")
        self.window = window
        if table_interp not in ("linear", "cubic"):
            raise ValueError("table_interp must be 'linear' or 'cubic'")
        self.table_interp = table_interp
        self.shape = tuple(2 * math.ceil(max(16, period_factor * s) * density / 2)
                           for s in self.support)
        points = math.prod(self.shape)
        if points > max_kernel_points:
            raise ValueError(f"Kernel table needs {points:,} points; limit is {max_kernel_points:,}. "
                             "Reduce support/density, use a separable camera, or explicitly raise max_kernel_points.")
        fw = framework(like)
        # Midpoint quadrature avoids giving the Nyquist discontinuity an
        # entire Fourier bin. In particular, the identity kernel samples
        # to delta at integer offsets instead of acquiring a 1/period bias.
        frequency = stack(mesh([2 * math.pi * (fftfreq(n, 1 / density, like) + density / (2 * n))
                                for n in self.shape]))
        multiplier = transfer(camera, frequency[..., 0] if self.ndim == 1 else frequency,
                              scales, regularization, rcond)
        if bool((abs(multiplier) > 1000).any()):
            warnings.warn("Camera inverse gain exceeds 1000. Finite kernels and noise may be unstable; "
                          "consider regularization and check against method='fourier'.",
                          RuntimeWarning, stacklevel=3)
        # An inverse DFT is a quadrature of the continuous inverse transform.
        # Its 1/N normalization becomes d(omega)/(2*pi) after this factor.
        spatial = fw.fft.fftshift(fw.fft.ifftn(multiplier))
        phase = sum(mesh([math.pi * (arange(n, like) - n // 2) / n for n in self.shape]))
        self.table = (spatial * fw.exp(1j * phase)).real * density ** self.ndim

    def __call__(self, offsets):
        fw = framework(offsets)
        coords = offsets[..., None] if self.ndim == 1 else offsets
        t = coords * self.density + array([n // 2 for n in self.shape], coords)
        base = fw.floor(t)
        fraction = t - base
        base = base.to(dtype=fw.int64) if is_torch(base) else base.astype("int64")
        result = fw.zeros_like(coords[..., 0])
        # Interpolate the *kernel table*, not the image. Cubic interpolation
        # reduces lookup error without multiplying the Fourier grid size.
        # Both choices support gradients through table values and positions.
        stencil = (0, 1) if self.table_interp == "linear" else (-1, 0, 1, 2)
        for corner in product(stencil, repeat=self.ndim):
            index, weight = [], fw.ones_like(result)
            for d, bit in enumerate(corner):
                idx = base[..., d] + bit
                index.append(idx.clamp(0, self.shape[d] - 1) if is_torch(idx)
                             else idx.clip(0, self.shape[d] - 1))
                weight = weight * (keys(fraction[..., d] - bit) if self.table_interp == "cubic"
                                   else fraction[..., d] if bit else 1 - fraction[..., d])
            result = result + self.table[tuple(index)] * weight
        inside = (abs(coords) <= array(self.support, coords) / 2).all(-1)
        # A window controls ringing from *finite spatial support*. It does
        # not change the camera or reintroduce the old AA-stretch rule.
        u = 2 * coords / array(self.support, coords)
        if self.window == "hann":
            result = result * ((1 + fw.cos(math.pi * u)) / 2).prod(-1)
        elif self.window == "lanczos":
            result = result * fw.sinc(u).prod(-1)
        return fw.where(inside, result, fw.zeros_like(result))


def kernel_at(camera, scale_factors, like, **options):
    """Expose the continuous offset function for custom use or inspection.

    Example: k = kernel_at(Gaussian(1), [0.6], numpy.zeros(1)); k(offsets).
    The generated kernel already accounts for the scale. Do not AA-stretch it.
    """
    if isinstance(scale_factors, (int, float)):
        scale_factors = [scale_factors] * camera.ndim
    if len(scale_factors) != camera.ndim or any(float(s) <= 0 for s in scale_factors):
        raise ValueError("one positive scale per camera dimension is required")
    return Kernel(camera, scale_factors, like, **options)
