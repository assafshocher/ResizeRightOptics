#!/usr/bin/env python3
"""Regenerate the endpoint figure with the exact hard-thresholded family."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
N = 16384
DX = 0.25
W = np.pi / 3
ETA = 0.05
x = (np.arange(N) - N // 2) * DX
w = 2 * np.pi * np.fft.fftfreq(N, DX)
x_mask = np.abs(x) <= 11
colors = plt.cm.viridis


def no_spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def camera_symbol(omega):
    return np.clip(1 - np.abs(omega) / W, 0, None)


def thresholded_symbol(omega):
    value = camera_symbol(omega)
    return np.where(value >= ETA, value, 0.0)


def pseudo_inverse(value):
    output = np.zeros_like(value)
    np.divide(1.0, value, out=output, where=np.abs(value) > 0)
    return output


def spatial(multiplier):
    return np.fft.fftshift(np.fft.ifft(multiplier).real) / DX


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), dpi=200)

# In output-pixel coordinates, K_s(s omega) = H(omega)/H(s omega).
ax = axes[0]
for index, scale in enumerate((1 / 2, 1 / 4, 1 / 8, 1 / 16)):
    symbol = thresholded_symbol(w) * pseudo_inverse(thresholded_symbol(scale * w))
    ax.plot(x[x_mask], spatial(symbol)[x_mask], color=colors(index / 4), lw=0.9)
ax.plot(x[x_mask], spatial(thresholded_symbol(w))[x_mask], "k--", lw=1.2)
ax.set_yticks([])
ax.set_xlabel(r"$x$")
ax.set_title(r"$s\to0$: $\to K_0$", fontsize=8.5)
no_spines(ax)

# In input coordinates, K_s tends to the camera pseudoinverse.
ax = axes[1]
for index, scale in enumerate((2, 4, 8, 16)):
    symbol = thresholded_symbol(w / scale) * pseudo_inverse(thresholded_symbol(w))
    ax.plot(x[x_mask], spatial(symbol)[x_mask], color=colors(index / 4), lw=0.9)
ax.plot(x[x_mask], spatial(pseudo_inverse(thresholded_symbol(w)))[x_mask], "k--", lw=1.2)
ax.set_yticks([])
ax.set_xlabel(r"$x$")
ax.set_title(r"$s\to\infty$: $\to K_\infty$", fontsize=8.5)
no_spines(ax)

# The formal Gaussian-kernel variance is analytic, including on the backward-heat side.
ax = axes[2]
scales = np.exp(np.linspace(np.log(1 / 16), np.log(16), 401))
normalized_variance = scales ** -2 - 1
ax.semilogx(scales, normalized_variance, color="k", lw=1.2)
ax.axhline(0, color="0.85", lw=0.4)
ax.axhline(-1, color="0.6", ls="--", lw=0.7)
ax.axvline(1, color="0.85", lw=0.4)
ax.set_yscale("symlog", linthresh=0.1)
ax.set_yticks((-1, 0, 1, 10, 100))
ax.set_yticklabels((r"$-1$", r"$0$", r"$1$", r"$10$", r"$100$"))
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$t(s)/\kappa_2$")
ax.set_title(r"formal variance of $K_s$", fontsize=8.5)
no_spines(ax)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_endpoints.pdf"))
fig.savefig(os.path.join(OUT, "fig_endpoints.png"), dpi=300)
plt.close(fig)
