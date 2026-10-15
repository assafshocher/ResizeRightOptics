"""Publication figure and companion animation for the projected-grid appendix."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)

BLACK = "#202124"
GREY = "#7a7f87"
LIGHT = "#eef1f4"
GREEN = "#2e8b57"
GREEN_LIGHT = "#eaf4ee"
ORANGE = "#e67e22"
RED = "#c75450"
BLUE = "#3f6fa0"


def projected_grid(n0, n, s):
    q = (n0 - 1) / 2 - (n - 1) / (2 * s)
    return np.arange(n) / s + q


def draw_cells(ax, n, y, height, face, edge, centers=None, width=1.0, lw=1.0):
    if centers is None:
        centers = np.arange(n)
    for x in centers:
        ax.add_patch(
            Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                facecolor=face,
                edgecolor=edge,
                lw=lw,
                zorder=1,
            )
        )
    ax.scatter(centers, np.full(n, y), s=22, c=edge, zorder=3)


def rounded_box(ax, xy, w, h, text, fc=LIGHT, ec="none", color=BLACK, fs=10):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=color, fontsize=fs)


def make_static():
    n0, n, s = 6, 3, 0.47
    g = projected_grid(n0, n, s)
    center = (n0 - 1) / 2
    q = g[0]

    fig = plt.figure(figsize=(12.2, 4.25), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.08, 1.02], wspace=0.25)

    # (a) The construction.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("a   Space the samples, then center them", loc="left", fontweight="bold")
    draw_cells(ax, n0, 1.05, 0.56, "white", GREY)
    draw_cells(ax, n, -0.05, 0.56, GREEN_LIGHT, GREEN, centers=g, width=1 / s, lw=1.2)
    for x in g:
        ax.plot([x, x], [0.23, 0.78], color=GREEN, lw=0.8, ls=(0, (2, 2)), alpha=0.8)
        ax.scatter([x], [1.05], marker="x", s=42, lw=1.6, c=GREEN, zorder=4)
    ax.plot(
        [center, center],
        [-0.35, 1.34],
        color=BLACK,
        lw=1,
        ls=(0, (3, 2)),
        alpha=0.65,
    )
    ax.text(center, 1.48, "shared center", ha="center", va="bottom", color=BLACK)
    ax.annotate(
        "",
        xy=(g[1], -0.51),
        xytext=(g[0], -0.51),
        arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.2),
    )
    ax.text((g[0] + g[1]) / 2, -0.68, r"spacing $1/s$", ha="center", color=GREEN)
    # The true offset is short in this example. A capped dimension line stays
    # legible where opposing arrowheads would visually collapse into a star.
    ax.plot([0, q], [0.52, 0.52], color=BLUE, lw=1.2)
    ax.plot([0, 0], [0.47, 0.57], color=BLUE, lw=1.2)
    ax.plot([q, q], [0.47, 0.57], color=BLUE, lw=1.2)
    ax.text(q / 2, 0.60, r"offset $q$", ha="center", color=BLUE)
    ax.text(-0.82, 1.05, "input", ha="right", va="center", color=GREY)
    ax.text(-0.82, -0.05, "output\nprojected", ha="right", va="center", color=GREEN)
    ax.text(
        2.5,
        -1.08,
        r"$g_n=\dfrac{n}{s}+q,\qquad q=\dfrac{N_0-1}{2}-\dfrac{N-1}{2s}$",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.text(2.5, -1.39, rf"example: $N_0={n0}$, $N={n}$, $s={s}$", ha="center", color=GREY)
    ax.set_xlim(-1.25, 5.85)
    ax.set_ylim(-1.58, 1.78)
    ax.axis("off")

    # (b) Failure modes.
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("b   Two shortcuts move the optical axis", loc="left", fontweight="bold")
    ax.plot(
        [center, center],
        [-0.72, 1.30],
        color=BLACK,
        lw=1,
        ls=(0, (3, 2)),
        alpha=0.7,
    )
    ax.text(center, 1.52, "input center", ha="center", color=BLACK)
    rows = [
        ("centered", g, GREEN, 1.08),
        ("half-pixel", np.arange(n) / s + 0.5 * (1 / s - 1), ORANGE, 0.30),
        (r"$n/s$", np.arange(n) / s, RED, -0.48),
    ]
    for label, points, color, y in rows:
        ax.plot([points[0], points[-1]], [y, y], color=color, lw=1.4, alpha=0.7)
        ax.scatter(points, np.full(n, y), s=45, c=color, edgecolors="white", linewidths=0.7, zorder=3)
        row_center = (points[0] + points[-1]) / 2
        ax.plot([row_center, row_center], [y - 0.18, y + 0.18], color=color, lw=2)
        delta = row_center - center
        suffix = "" if abs(delta) < 1e-10 else "\n" + rf"$\Delta c={delta:+.2f}$ px"
        ax.text(-0.26, y, label + suffix, ha="right", va="center", color=color, fontsize=8.8)
    ax.text(
        2.5,
        -1.15,
        "half-pixel is exact only when " + r"$N=sN_0$" + "\n" + r"$n/s$ is exact only when $q=0$",
        ha="center",
        va="center",
        fontsize=8.7,
    )
    ax.set_xlim(-1.50, 5.75)
    ax.set_ylim(-1.58, 1.78)
    ax.axis("off")

    # (c) Scale and size are distinct.
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("c   A round trip needs scale and size", loc="left", fontweight="bold")
    rounded_box(ax, (0.02, 0.90), 0.23, 0.32, "9 px", fc=LIGHT, fs=10)
    rounded_box(ax, (0.39, 0.90), 0.23, 0.32, "5 px", fc=GREEN_LIGHT, color=GREEN, fs=10)
    rounded_box(ax, (0.76, 0.90), 0.23, 0.32, "9 px", fc=GREEN_LIGHT, color=GREEN, fs=10)
    ax.annotate("", xy=(0.39, 1.06), xytext=(0.25, 1.06), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax.annotate("", xy=(0.76, 1.06), xytext=(0.62, 1.06), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax.text(0.32, 1.40, r"$s=\frac{1}{2},\ N=5$", ha="center", color=GREEN, fontsize=8.6)
    ax.text(0.69, 1.40, r"$s=2,\ N=9$", ha="center", color=GREEN, fontsize=8.6)

    rounded_box(ax, (0.39, 0.17), 0.23, 0.32, "5 px", fc="#fff4e8", color=ORANGE, fs=10)
    rounded_box(ax, (0.76, 0.17), 0.23, 0.32, "10 px", fc="#fff4e8", color=ORANGE, fs=10)
    ax.annotate("", xy=(0.76, 0.33), xytext=(0.62, 0.33), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    ax.text(0.69, 0.66, r"scale only: $s=2$", ha="center", color=ORANGE, fontsize=8.6)

    rounded_box(ax, (0.39, -0.56), 0.23, 0.32, "5 px", fc="#fdeceb", color=RED, fs=10)
    rounded_box(ax, (0.76, -0.56), 0.23, 0.32, "9 px", fc="#fdeceb", color=RED, fs=10)
    ax.annotate("", xy=(0.76, -0.40), xytext=(0.62, -0.40), arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(0.69, -0.07, r"size only: infers $s=9/5$", ha="center", color=RED, fontsize=8.6)

    ax.text(0.5, -1.00, "Scale fixes spacing; size fixes extent.", ha="center", color=BLACK, fontweight="bold", fontsize=9.2)
    ax.text(0.5, -1.30, "Pass both to preserve the intended\nzoom and center.", ha="center", color=GREY, fontsize=8.5)
    ax.set_xlim(-0.03, 1.04)
    ax.set_ylim(-1.58, 1.78)
    ax.axis("off")

    fig.savefig(OUT / "fig_grid.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig_grid.png", dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_animation():
    n0 = 9
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.set_xlim(-1.0, n0)
    ax.set_ylim(-0.95, 1.35)
    ax.axis("off")
    ax.axvline((n0 - 1) / 2, color=BLACK, lw=1, ls=(0, (3, 2)), alpha=0.65)
    draw_cells(ax, n0, 0.9, 0.42, "white", GREY)
    ax.text(-0.65, 0.9, "input", ha="right", va="center", color=GREY)
    correct_line, = ax.plot([], [], color=GREEN, lw=1.5)
    correct_pts = ax.scatter([], [], s=42, color=GREEN, edgecolors="white", linewidths=0.6)
    shortcut_line, = ax.plot([], [], color=ORANGE, lw=1.5)
    shortcut_pts = ax.scatter([], [], s=42, color=ORANGE, edgecolors="white", linewidths=0.6)
    ax.text(-0.65, 0.20, "centered", ha="right", va="center", color=GREEN)
    ax.text(-0.65, -0.48, "half-pixel", ha="right", va="center", color=ORANGE)
    title = ax.text(4.0, 1.25, "", ha="center", va="center", fontsize=11, fontweight="bold")
    note = ax.text(4.0, -0.85, "", ha="center", va="center", color=GREY)

    scales = np.concatenate([np.linspace(0.35, 1.35, 45), np.linspace(1.35, 0.35, 45)])

    def update(s):
        n = int(np.ceil(s * n0))
        centered = projected_grid(n0, n, s)
        half = np.arange(n) / s + 0.5 * (1 / s - 1)
        correct_line.set_data([centered[0], centered[-1]], [0.20, 0.20])
        correct_pts.set_offsets(np.c_[centered, np.full(n, 0.20)])
        shortcut_line.set_data([half[0], half[-1]], [-0.48, -0.48])
        shortcut_pts.set_offsets(np.c_[half, np.full(n, -0.48)])
        drift = (half[0] + half[-1]) / 2 - (n0 - 1) / 2
        title.set_text(rf"$s={s:.2f},\quad N=\lceil sN_0\rceil={n}$")
        note.set_text(rf"half-pixel center drift: {drift:+.2f} px")
        return correct_line, correct_pts, shortcut_line, shortcut_pts, title, note

    ani = animation.FuncAnimation(fig, update, frames=scales, interval=70, blit=False)
    ani.save(OUT / "fig_grid_animation.gif", writer=animation.PillowWriter(fps=14), dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    make_static()
    make_animation()
