"""Relabel formal endpoint models in two supplemental cycle figures."""

from pathlib import Path
import fitz


ROOT = Path(__file__).resolve().parents[1] / "figs"

for stem in ("fig_cycle_bilinear", "fig_cycle_lanczos3"):
    source = ROOT / f"{stem}_original.pdf"
    target = ROOT / f"{stem}.pdf"
    doc = fitz.open(source)
    page = doc[0]

    endpoint_box = fitz.Rect(44, 96, 82, 113)
    suffix_box = fitz.Rect(108, 97, 179, 114)
    page.add_redact_annot(endpoint_box, fill=(1, 1, 1))
    page.add_redact_annot(suffix_box, fill=(1, 1, 1))
    page.apply_redactions()
    page.insert_textbox(
        endpoint_box,
        "endpoint K0",
        fontsize=6.2,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER,
    )
    page.insert_textbox(
        suffix_box,
        "up: classical",
        fontsize=7.4,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER,
    )

    tmp = ROOT / f"{stem}_relabeled.pdf"
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    tmp.replace(target)
