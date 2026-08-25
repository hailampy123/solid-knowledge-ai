"""Generate the sample conservation PDF (dev-only; output is committed).

Run: uv run --group dev python scripts/make_sample_pdf.py
"""
from pathlib import Path

from fpdf import FPDF

PARAS = [
    "Orca Conservation",
    "Orcas face several human-caused threats. Chemical pollutants such as PCBs "
    "accumulate in their blubber and impair reproduction and immune function.",
    "Prey depletion is a major threat to fish-eating resident orcas. Declines in "
    "Chinook salmon reduce the food available to Southern Resident killer whales.",
    "Underwater noise from ships interferes with echolocation and communication, "
    "making it harder for orcas to hunt and coordinate.",
    "The Southern Resident population of the northeastern Pacific is listed as "
    "endangered, with only around seventy individuals remaining.",
]


def main() -> None:
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for para in PARAS:
        pdf.multi_cell(0, 8, para)
        pdf.ln(4)
    out = Path(__file__).resolve().parent.parent / "data" / "docs" / "orca-conservation.pdf"
    pdf.output(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
