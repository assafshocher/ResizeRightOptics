"""Run from the project root: python examples/quickstart.py (after pip install .)."""
import numpy as np
import torch
from resize_right_optics import resize, Gaussian, Airy, Separable, from_psf

# Explicit axes make NumPy HWC and Torch CHW conventions unambiguous.
image = np.random.default_rng(0).random((48, 64, 3))
small = resize(image, [0.5, 0.75], dims=[0, 1], camera=Gaussian(0.7), by_convs=True)
print("NumPy:", image.shape, "->", small.shape)

# A camera need not be separable: this is the circular-pupil OTF in two dims.
optical = resize(image, 0.5, dims=[0, 1], camera=Airy(2.7), method="fourier")
print("Circular pupil:", optical.shape)

# Every forward pass builds its kernel from the current PSF parameters.
x = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float64)
logits = torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64))
psf = logits.flatten().softmax(0).reshape(5, 5)
y = resize(x, 0.75, camera=from_psf(psf, spacing=0.25), support_sz=6, density=2)
y.square().mean().backward()
print("Learned PSF gradient norm:", float(logits.grad.norm()))

# Both executors use the same camera and grid, not merely the same out_shape.
cam = Separable(Gaussian(0.5), Gaussian(0.8))
a = resize(image, [2/3, 0.75], dims=[0, 1], camera=cam)
b = resize(image, [2/3, 0.75], dims=[0, 1], camera=cam, by_convs=True)
print("Ordinary/polyphase max error:", float(np.max(abs(a - b))))

