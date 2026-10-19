"""PDF version of the report, for the "Download report (PDF)" button.

    build_pdf(result, summary, warnings, generated_at=None) -> bytes

It carries the same content as the plain-text report in report.py (score,
verdict, the two raw numbers, summary, skill lists, warnings and the two
closing notes), laid out on one or two A4 pages. The note texts are imported
from report.py, so they are written in one place only.

Uses the fpdf2 package and its built-in Helvetica font. That font covers only
Latin-1 characters, so any other character (for example in a file name that
appears in a warning) is replaced by a close ASCII form or by "?" instead of
crashing the download.
"""

from __future__ import annotations

from datetime import datetime

from fpdf import FPDF

from resume_matcher.schemas import AnalysisResult
from report import GUIDE_NOT_VERDICT, HOW_TO_READ
from summary import MUST_HAVE, NICE_TO_HAVE, UNKNOWN, group_missing
from ui_helpers import (
    GROUP_TITLES,
    coverage_line,
    format_score,
    meaning_match_line,
)

FONT = "Helvetica"
INK = (31, 36, 48)
MUTED = (107, 114, 128)
ACCENT = (79, 70, 229)
LINE = (229, 231, 235)

# Verdict colours (text on a light tint), matching the page badges in spirit.
VERDICT_COLOURS = {
    "Fit": (22, 101, 52),
    "Partial Fit": (146, 64, 14),
    "No-Fit": (153, 27, 27),
}

# Common characters outside Latin-1, mapped to a plain equivalent.
_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "•": "-",
    " ": " ", "→": "->",
}


def pdf_safe(text: str) -> str:
    """Make text printable in the built-in Latin-1 font (never raises)."""
    for char, plain in _REPLACEMENTS.items():
        text = text.replace(char, plain)
    return text.encode("latin-1", "replace").decode("latin-1")


class _ReportPdf(FPDF):
    """FPDF with a small footer: the page number."""

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font(FONT, size=8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")


def _heading(pdf: FPDF, text: str) -> None:
    pdf.ln(4)
    pdf.set_font(FONT, "B", 11)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 7, pdf_safe(text.upper()), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(*INK)


def _paragraph(pdf: FPDF, text: str, *, size: int = 10, muted: bool = False) -> None:
    pdf.set_font(FONT, size=size)
    pdf.set_text_color(*(MUTED if muted else INK))
    pdf.multi_cell(0, 5.5, pdf_safe(text), new_x="LMARGIN", new_y="NEXT")


def _bullets(pdf: FPDF, names: list[str]) -> None:
    pdf.set_font(FONT, size=10)
    pdf.set_text_color(*INK)
    for name in names:
        pdf.set_x(pdf.l_margin + 4)
        pdf.multi_cell(0, 5.5, pdf_safe(f"- {name}"), new_x="LMARGIN", new_y="NEXT")


def _stat(pdf: FPDF, label: str, value: str) -> None:
    pdf.set_font(FONT, size=10)
    pdf.set_text_color(*MUTED)
    pdf.cell(42, 6, pdf_safe(label))
    pdf.set_font(FONT, "B", 10)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6, pdf_safe(value), new_x="LMARGIN", new_y="NEXT")


def _skill_group(pdf: FPDF, title: str, note: str, names: list[str]) -> None:
    _heading(pdf, f"{title} ({len(names)})")
    _paragraph(pdf, note, size=9, muted=True)
    pdf.ln(1)
    if names:
        _bullets(pdf, names)
    else:
        _paragraph(pdf, "None.")


def build_pdf(
    result: AnalysisResult,
    summary: str,
    warnings: list[str],
    generated_at: datetime | None = None,
) -> bytes:
    """Return the whole report as the bytes of a PDF file."""
    pdf = _ReportPdf(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 16, 18)
    pdf.set_title("Resume Match Report")
    pdf.set_creator("Resume-to-Job Description Matcher")
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title block.
    pdf.set_font(FONT, "B", 20)
    pdf.set_text_color(*INK)
    pdf.cell(0, 10, "Resume Match Report", new_x="LMARGIN", new_y="NEXT")
    if generated_at is not None:
        pdf.set_font(FONT, size=9)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, f"Generated: {generated_at:%Y-%m-%d %H:%M}",
                 new_x="LMARGIN", new_y="NEXT")

    # Overall block.
    _heading(pdf, "Overall")
    pdf.set_font(FONT, "B", 30)
    pdf.set_text_color(*INK)
    pdf.cell(24, 14, format_score(result.overall_score))
    pdf.set_font(FONT, size=12)
    pdf.set_text_color(*MUTED)
    pdf.cell(20, 14, "/ 100")
    pdf.set_font(FONT, "B", 13)
    pdf.set_text_color(*VERDICT_COLOURS.get(result.verdict, INK))
    pdf.cell(0, 14, pdf_safe(result.verdict), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    _stat(pdf, "Meaning match", meaning_match_line(result))
    _stat(pdf, "Skill coverage", coverage_line(result))

    _heading(pdf, "Summary")
    _paragraph(pdf, summary)

    # Missing skills, grouped, most important first.
    _heading(pdf, "Skills to add (most important first)")
    grouped = group_missing(result.missing_skills)
    listed_any = False
    for key in (MUST_HAVE, NICE_TO_HAVE, UNKNOWN):
        names = grouped[key]
        if not names:
            continue
        listed_any = True
        pdf.set_font(FONT, "B", 10)
        pdf.set_text_color(*INK)
        pdf.cell(0, 6.5, pdf_safe(f"{GROUP_TITLES[key]} ({len(names)})"),
                 new_x="LMARGIN", new_y="NEXT")
        _bullets(pdf, names)
        pdf.ln(1)
    if not listed_any:
        _paragraph(pdf, "No required skills are missing.")

    _skill_group(
        pdf,
        "Skills you already have",
        "Asked for by the job description and found on the resume.",
        list(dict.fromkeys(result.matched_skills)),
    )
    _skill_group(
        pdf,
        "Extra skills",
        "On the resume, but not asked for in this job description.",
        list(dict.fromkeys(result.extra_skills)),
    )

    if warnings:
        _heading(pdf, "Things to check")
        _bullets(pdf, warnings)

    _heading(pdf, "About this report")
    _paragraph(pdf, HOW_TO_READ, size=9, muted=True)
    pdf.ln(2)
    _paragraph(pdf, GUIDE_NOT_VERDICT, size=9, muted=True)

    return bytes(pdf.output())
