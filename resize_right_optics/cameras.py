"""A camera is specified once. Its resizing kernels are derived at every scale.

Frequencies are radians per input pixel. A 1-D camera is applied separably;
a d-dimensional camera takes coordinates with shape (..., d). Parameters
may be PyTorch tensors: no detached spectrum or kernel is cached.
"""
import math
import numpy as np
from ._backend import array, arange, framework, mesh, stack, cat, is_torch, scalar


class Camera:
    """User-defined OTF. The callable must use the backend of its coordinates.

    A real PSF requires H(-w) = conj(H(w)). Its DC gain should be one.
    ``ndim=1`` means a separable camera; use Separable for different axes.
    """
    def __init__(self, otf, ndim=1, name="custom"):
        if not isinstance(ndim, int) or ndim < 1:
            raise ValueError("ndim must be a positive integer")
        self.otf, self.ndim, self.name = otf, ndim, name

    def __call__(self, omega):
        return self.otf(omega)


class Separable:
    """One 1-D camera per resized dimension, in the order of ``dims``."""
    def __init__(self, *cameras):
        if not cameras or any(c.ndim != 1 for c in cameras):
            raise ValueError("Separable needs one or more 1-D cameras")
        self.cameras = cameras
        self.ndim = len(cameras)

    def as_full(self):
        """Use the same OTF through the full N-D path.

        Useful for checking equivalence, or applying *joint* Tikhonov
        regularization instead of regularizing each 1-D factor separately.
        """
        def spectrum(omega):
            value = self.cameras[0](omega if self.ndim == 1 else omega[..., 0])
            for d, camera in enumerate(self.cameras[1:], 1):
                value = value * camera(omega[..., d])
            return value
        return Camera(spectrum, self.ndim, name="full product camera")


class Gaussian(Camera):
    """Gaussian PSF, sigma in pixels (possibly a differentiable tensor).

    The PSF itself has no spectral cutoff. The resampler separately enforces
    the digital Nyquist band: it cannot undo aliasing already in the input.
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        super().__init__(self.spectrum, name="Gaussian")

    def spectrum(self, omega):
        if not math.isfinite(scalar(self.sigma)) or scalar(self.sigma) <= 0:
            raise ValueError("Gaussian sigma must be finite and positive")
        return framework(omega).exp(-0.5 * (omega * array(self.sigma, omega)) ** 2)


class GaussianCovariance(Camera):
    """A genuinely nonseparable Gaussian, given a positive-definite covariance.

    Covariance validity is the caller's responsibility for learned tensors;
    parameterizing it as L @ L.T keeps it positive definite during training.
    """
    def __init__(self, covariance):
        shape = np.shape(covariance) if not is_torch(covariance) else covariance.shape
        if len(shape) != 2 or shape[0] != shape[1] or not shape[0]:
            raise ValueError("covariance must be a square matrix")
        self.covariance = covariance
        super().__init__(self.spectrum, ndim=shape[0], name="Gaussian covariance")

    def spectrum(self, omega):
        cov = array(self.covariance, omega)
        if self.ndim == 1:
            return framework(omega).exp(-0.5 * omega ** 2 * cov[0, 0])
        return framework(omega).exp(-0.5 * ((omega @ cov) * omega).sum(-1))


class Airy(Camera):
    """Incoherent circular-pupil Airy PSF, represented by its exact 2-D OTF.

    ``cutoff`` is radial, in radians/pixel. A cutoff <= pi fits within the
    Nyquist square. It is not a separable approximation to the Airy pattern.
    """
    def __init__(self, cutoff=math.pi):
        self.cutoff = cutoff
        super().__init__(self.spectrum, ndim=2, name="Airy")

    def spectrum(self, omega):
        if not math.isfinite(scalar(self.cutoff)) or scalar(self.cutoff) <= 0:
            raise ValueError("Airy cutoff must be finite and positive")
        fw = framework(omega)
        # A tiny floor avoids the undefined derivative of norm at DC. DC is
        # assigned its exact value below, and is independent of the cutoff.
        radius2 = (omega ** 2).sum(-1)
        eps = fw.finfo(omega.dtype).eps
        q = fw.sqrt(fw.maximum(radius2, array(eps ** 2, omega))) / array(self.cutoff, omega)
        u = fw.minimum(q, array(1 - eps, omega))
        value = (2 / math.pi) * (fw.arccos(u) - u * fw.sqrt(1 - u ** 2))
        value = fw.where(q < 1, value, fw.zeros_like(value))
        return fw.where(radius2 == 0, fw.ones_like(value), value)


class IdealLowpass(Camera):
    """A useful reference camera, not a nonnegative physical PSF."""
    def __init__(self, cutoff=math.pi):
        self.cutoff = cutoff
        super().__init__(self.spectrum, name="ideal lowpass")

    def spectrum(self, omega):
        return (abs(omega) < array(self.cutoff, omega)) * framework(omega).ones_like(omega)


def from_psf(samples, spacing=1.0, center=None):
    """A sampled PSF on a Cartesian grid, with optional subpixel center.

    This is quadrature of a continuous PSF, not a resize kernel. ``spacing``
    is the PSF sample spacing in pixels; use a fine grid for a continuous
    model. Sum-normalization preserves DC and gradients through PSF values.
    Even-sized arrays default to their geometric center, not an FFT corner.
    """
    shape = tuple(samples.shape)
    if not shape or any(n == 0 for n in shape):
        raise ValueError("PSF samples must be a nonempty array")
    ndim = len(shape)
    if (is_torch(samples) and samples.is_complex()) or (not is_torch(samples) and np.iscomplexobj(samples)):
        raise TypeError("PSF samples must be real (signed values are allowed)")
    spacings = (spacing,) * ndim if np.isscalar(spacing) or (is_torch(spacing) and spacing.ndim == 0) else tuple(spacing)
    centers = (tuple((n - 1) / 2 for n in shape) if center is None else
               ((center,) if np.isscalar(center) or (is_torch(center) and center.ndim == 0) else tuple(center)))
    if len(spacings) != ndim or len(centers) != ndim or any(s <= 0 for s in spacings):
        raise ValueError("spacing and center must match the PSF dimensions; spacing must be positive")

    def spectrum(omega):
        fw = framework(omega)
        values = array(samples, omega).reshape(-1)
        total = values.sum()
        if bool(total == 0):
            raise ValueError("PSF has zero DC; cannot normalize it")
        positions = stack(mesh([(arange(n, omega) - c) * dx
                                for n, c, dx in zip(shape, centers, spacings)])).reshape(-1, ndim)
        query = omega.reshape(-1, ndim) if ndim > 1 else omega.reshape(-1, 1)
        pieces = []
        # Bound temporary storage when a dense OTF grid meets a large PSF.
        chunk = max(1, 262144 // len(values))
        for start in range(0, len(query), chunk):
            phase = query[start:start + chunk] @ positions.T
            pieces.append((fw.exp(-1j * phase) * (values / total)).sum(-1))
        return cat(pieces).reshape(omega.shape[:-1] if ndim > 1 else omega.shape)

    return Camera(spectrum, ndim, name="sampled PSF")


def from_function(psf, support, spacing=0.125, ndim=1):
    """Quadrature of a continuous PSF callable on a centered Cartesian grid.

    The callable receives a scalar coordinate array (1-D), or (..., ndim).
    ``support`` is the full width per axis, not the radius. Truncation of
    the supplied PSF is an explicit model approximation.
    """
    widths = (support,) * ndim if np.isscalar(support) else tuple(support)
    if len(widths) != ndim or min(widths) <= 0 or spacing <= 0:
        raise ValueError("positive support/spacing required, with one width per dimension")

    def spectrum(omega):
        vectors = [(arange(2 * math.ceil(w / (2 * spacing)) + 1, omega)
                    - math.ceil(w / (2 * spacing))) * spacing for w in widths]
        grid = stack(mesh(vectors))
        values = psf(grid[..., 0] if ndim == 1 else grid)
        return from_psf(values, spacing)(omega)

    return Camera(spectrum, ndim, name="continuous PSF quadrature")


def from_kernel_spectrum(spectrum, scale, ndim=1, terms=24):
    """Infer a camera from a relative kernel U at reference scale r != 1.

        H(w) = product_{j=1..infinity} U(r**j * w),       0 < r < 1,
        H(w) = product_{j=0..infinity} 1/U(w / r**j),          r > 1.

    U uses *input-pixel coordinates*, includes the reference kernel's
    width, and must have U(0)=1. This finite product is an approximation;
    it need not describe a nonnegative optical PSF. Digital band projection
    still applies, so an aliased software resize is not reproduced exactly.
    """
    if not math.isfinite(scale) or scale <= 0 or scale == 1 or not isinstance(terms, int) or terms < 1:
        raise ValueError("reference scale must be positive and not 1, with positive terms")

    def otf(omega):
        fw = framework(omega)
        zero = fw.zeros_like(omega.reshape(-1)[:1] if ndim == 1 else omega.reshape(-1, ndim)[:1])
        if bool((abs(spectrum(zero) - 1) > 1e-6).any()):
            raise ValueError("A reference kernel must have unit DC, U(0)=1")
        result = fw.ones_like(omega[..., 0] if ndim > 1 else omega)
        for j in range(terms):
            value = spectrum(omega * scale ** (j + 1) if scale < 1 else omega / scale ** j)
            if scale > 1:
                # Zeros carry no recoverable anchor information. Keep them
                # outside the inferred band instead of dividing by zero.
                nonzero = abs(value) > 0
                safe = fw.where(nonzero, value, fw.ones_like(value))
                value = fw.where(nonzero, 1 / safe, fw.zeros_like(value))
            result = result * value
        return result

    return Camera(otf, ndim, name=f"kernel anchor at {scale:g}")


def from_kernel(samples, scale, spacing=1.0, center=None, terms=24):
    """Sampled relative kernel -> camera. For a PSF, use from_psf instead."""
    kernel = from_psf(samples, spacing, center)
    return from_kernel_spectrum(kernel, scale, kernel.ndim, terms)


def keys(x):
    """Keys' cubic kernel, a=-1/2, as in the original ResizeRight."""
    x = abs(x)
    return ((1.5 * x ** 3 - 2.5 * x ** 2 + 1) * (x <= 1)
            + (-0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2) * ((x > 1) & (x < 2)))


def Cubic(reference_scale=0.5, terms=24):
    """Camera inferred from a stretched Keys cubic reference kernel.

    This is a *cubic-derived family*, not a claim that cubic is a physical
    PSF. Only the reference scale uses the familiar cubic spectrum; kernels
    at other scales follow the camera ratio, not AA stretching.
    """
    if not 0 < reference_scale < 1:
        raise ValueError("Cubic's stretched reference is a downscale in (0, 1)")
    # Integrate each polynomial piece separately. Fixed quadrature nodes
    # are constants; the evaluation itself stays in the input's framework.
    nodes, weights = np.polynomial.legendre.leggauss(24)
    nodes = np.concatenate(((nodes + 1) / 2, (nodes + 3) / 2))
    weights = np.tile(weights / 2, 2) * 2 * keys(nodes)

    def spectrum(omega):
        fw = framework(omega)
        return (fw.cos(omega[..., None] * array(nodes, omega) / reference_scale)
                * array(weights, omega)).sum(-1)

    return from_kernel_spectrum(spectrum, reference_scale, terms=terms)
