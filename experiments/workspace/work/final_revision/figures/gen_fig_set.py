#!/usr/bin/env python3
"""Regenerate the solution-set schematic from an exact bounded camera symbol."""

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
BAND = 1.0
SCALE = 2.0
omega = np.linspace(0, 2.6 * BAND, 2400)


def camera_symbol(value):
    inside = np.abs(value) <= BAND
    return np.where(inside, 1 - 0.6 * (value / BAND) ** 2, 0.0)


source = camera_symbol(omega)
target = camera_symbol(omega / SCALE)
canonical = np.zeros_like(omega)
np.divide(target, source, out=canonical, where=source != 0)

fig, ax = plt.subplots(figsize=(3.6, 2.5), dpi=200)
ax.plot(omega / BAND, canonical, color="k", lw=1.4)
ax.axvspan(1, SCALE, color="crimson", alpha=0.12)

free_1 = 0.18 * np.exp(-((omega / BAND - 1.35) / 0.18) ** 2)
free_2 = 0.16 * np.exp(-((omega / BAND - 1.70) / 0.20) ** 2)
ax.plot(omega / BAND, canonical + free_1, color="#D55E00", lw=0.9)
ax.plot(omega / BAND, canonical + free_2, color="#7570B3", lw=0.9)

ax.set_xticks((0, 1, 2))
ax.set_xticklabels((r"$0$", r"$B$", r"$sB$"))
ax.set_yticks([])
ax.set_xlabel(r"$\omega$")
ax.text(1.5, 0.86 * ax.get_ylim()[1], "free", color="crimson", ha="center")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_set.pdf"))
fig.savefig(os.path.join(OUT, "fig_set.png"), dpi=300)
plt.close(fig)
