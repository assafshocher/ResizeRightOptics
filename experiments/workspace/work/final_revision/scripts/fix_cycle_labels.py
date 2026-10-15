"""Correct physical/model terminology in the existing cycle figure."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "figs" / "fig_cycle_original.png"
dst = ROOT / "figs" / "fig_cycle.png"
im = Image.open(src).convert("RGB")
draw = ImageDraw.Draw(im)

font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
font = ImageFont.truetype(str(font_path), 31)
small = ImageFont.truetype(str(font_path), 27)


def centered_text(box, text, text_font):
    draw.rectangle(box, fill="white")
    x0, y0, x1, y1 = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(
        ((x0 + x1 - w) / 2, (y0 + y1 - h) / 2 - bounds[1]),
        text,
        fill=(25, 25, 25),
        font=text_font,
    )


centered_text((250, 0, 835, 72), "bicubic kernel model", font)
centered_text((1260, 0, 1920, 72), "Gaussian idealization", font)
centered_text((2260, 0, 3055, 72), "defocus-disk camera", font)
centered_text((28, 472, 235, 520), "endpoint", small)

im.save(dst, dpi=(300, 300), optimize=True)
