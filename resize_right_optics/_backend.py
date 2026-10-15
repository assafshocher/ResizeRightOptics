"""Small differences between NumPy and PyTorch, kept out of the mathematics."""
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def is_torch(x):
    return torch is not None and isinstance(x, torch.Tensor)


def framework(x):
    return torch if is_torch(x) else np


def array(value, like, integer=False):
    if is_torch(like):
        dtype = torch.int64 if integer else like.real.dtype
        return torch.as_tensor(value, dtype=dtype, device=like.device)
    if is_torch(value):
        raise TypeError("A PyTorch camera parameter needs a PyTorch input; it is not detached to NumPy.")
    return np.asarray(value, dtype=np.int64 if integer else like.real.dtype)


def arange(n, like, integer=False):
    if is_torch(like):
        return torch.arange(n, dtype=torch.int64 if integer else like.real.dtype,
                            device=like.device)
    return np.arange(n, dtype=np.int64 if integer else like.real.dtype)


def moveaxis(x, source, destination):
    return x.movedim(source, destination) if is_torch(x) else np.moveaxis(x, source, destination)


def mesh(vectors):
    return framework(vectors[0]).meshgrid(*vectors, indexing="ij")


def stack(values, axis=-1):
    return torch.stack(values, dim=axis) if is_torch(values[0]) else np.stack(values, axis=axis)


def cat(values, axis=0):
    return torch.cat(values, dim=axis) if is_torch(values[0]) else np.concatenate(values, axis=axis)


def fftfreq(n, spacing, like):
    if is_torch(like):
        return torch.fft.fftfreq(n, d=spacing, dtype=like.real.dtype, device=like.device)
    return np.fft.fftfreq(n, d=spacing).astype(like.real.dtype)


def take(x, indices, axis=0):
    if is_torch(x):
        return torch.index_select(x, axis, indices.reshape(-1)).reshape(
            x.shape[:axis] + indices.shape + x.shape[axis + 1:])
    return np.take(x, indices, axis=axis)


def scalar(value):
    """Read a shape/validation decision, never an arithmetic camera value."""
    return value.detach().item() if is_torch(value) else float(value)
