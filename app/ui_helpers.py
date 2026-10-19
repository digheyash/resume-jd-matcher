"""Plain helper functions for the Streamlit dashboard.

Nothing in this module imports Streamlit, so every function can be tested with
pytest and no browser. The page script (streamlit_app.py) only calls these
functions and draws what they return.

Three groups:
    1. Reading input   read_uploaded_file, decode_text_bytes, input_problems
    2. Messages        friendly_error, combine_warnings
    3. Display         format_*, chips, score ring, verdict badge, cards

All HTML is built as ONE LINE (no newlines, no indentation). Streamlit passes
it through a Markdown step first, and a blank line or an indented line would
make Markdown treat the HTML as code. Every piece of text that comes from the
user's documents (skill names, warnings, messages) goes through html.escape.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Callable

from resume_matcher.analyzer.analyzer import AnalysisError
from resume_matcher.parsing.loader import (
    DocumentLoadError,
    LoadedDocument,
    from_text,
    load_pdf,
)
from resume_matcher.schemas import AnalysisResult, SkillGap
from summary import MUST_HAVE, NICE_TO_HAVE, UNKNOWN, group_missing, semantic_not_used

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
# Largest upload accepted. Keep equal to server.maxUploadSize in
# .streamlit/config.toml (that setting stops bigger files in the browser; this
# one protects the helper if it is called from somewhere else).
MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

PDF_SUFFIXES = (".pdf",)
TEXT_SUFFIXES = (".txt", ".md")

# Colours (same palette as the dashboard design).
INK = "#1F2430"
MUTED = "#5B6170"

# Verdict -> badge background, badge text, ring colour.
VERDICT_STYLES = {
    "Fit": {"bg": "#E6F4EA", "fg": "#1E6B3A", "ring": "#2E9D5B"},
    "Partial Fit": {"bg": "#FEF3DC", "fg": "#8A5300", "ring": "#D97706"},
    "No-Fit": {"bg": "#FDECEC", "fg": "#9F1D1D", "ring": "#D14343"},
}
NEUTRAL_STYLE = {"bg": "#EEEDEA", "fg": "#4A4F5C", "ring": "#8A8F9C"}

# Chip kind -> (background, text colour).
CHIP_STYLES = {
    "matched": ("#E6F4EA", "#1E6B3A"),
    MUST_HAVE: ("#FDECEC", "#9F1D1D"),
    NICE_TO_HAVE: ("#FEF3DC", "#8A5300"),
    UNKNOWN: ("#EEEDEA", "#4A4F5C"),
    "extra": ("#EEEDEA", "#4A4F5C"),
}

# Heading text and colour for each group of missing skills.
GROUP_TITLES = {
    MUST_HAVE: "Must-have",
    NICE_TO_HAVE: "Nice-to-have",
    UNKNOWN: "Importance not stated",
}
GROUP_COLORS = {MUST_HAVE: "#9F1D1D", NICE_TO_HAVE: "#8A5300", UNKNOWN: "#4A4F5C"}

CARD_STYLE = (
    "background:#FFFFFF;border:1px solid #ECE8E1;border-radius:16px;"
    "box-shadow:0 1px 2px rgba(31,36,48,0.04),0 8px 24px rgba(31,36,48,0.04);"
    "padding:28px;box-sizing:border-box;margin-bottom:16px;"
)

RING_RADIUS = 58
RING_CIRCUMFERENCE = 2 * math.pi * RING_RADIUS


def _esc(text: object) -> str:
    """Escape text for safe use inside HTML."""
    return html.escape(str(text), quote=True)


# ----------------------------------------------------------------------
# 1. Reading input
# ----------------------------------------------------------------------
def decode_text_bytes(data: bytes) -> str:
    """Decode an uploaded text file, trying the same encodings as the loader.

    utf-8-sig copes with a Notepad BOM, cp1252 is the usual Windows fallback,
    and latin-1 can decode any byte, so this never fails.
    """
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def read_uploaded_file(
    name: str,
    data: bytes,
    pdf_loader: Callable[[bytes], LoadedDocument] = load_pdf,
    text_loader: Callable[[str], LoadedDocument] = from_text,
) -> LoadedDocument:
    """Turn an uploaded file (its name and raw bytes) into a LoadedDocument.

    The two loaders are parameters so tests can pass fakes.

    Raises:
        DocumentLoadError: empty file, file over the size limit, unsupported
            type, or a PDF the loader cannot open.
    """
    if not data:
        raise DocumentLoadError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentLoadError(
            f"The file is larger than {MAX_UPLOAD_MB} MB. "
            "Please use a smaller file or paste the text."
        )
    suffix = Path(name).suffix.lower()
    if suffix in PDF_SUFFIXES:
        return pdf_loader(data)
    if suffix in TEXT_SUFFIXES:
        return text_loader(decode_text_bytes(data))
    raise DocumentLoadError(
        f"Unsupported file type '{suffix or name}'. "
        "Please upload a PDF or a .txt file."
    )


def input_problems(resume_text: str, jd_text: str) -> list[str]:
    """Return friendly messages for missing input (empty list = good to go)."""
    problems = []
    if not (resume_text or "").strip():
        problems.append("Add your resume: upload a file or paste the text.")
    if not (jd_text or "").strip():
        problems.append("Paste the job description.")
    return problems


# ----------------------------------------------------------------------
# 2. Messages
# ----------------------------------------------------------------------
def friendly_error(exc: Exception) -> tuple[str, str]:
    """Return (title, message) for an error, never a stack trace."""
    if isinstance(exc, DocumentLoadError):
        return "We could not read that file", str(exc)
    if isinstance(exc, AnalysisError):
        return "We could not analyse these texts", str(exc)
    return (
        "Something went wrong",
        "An unexpected problem stopped the analysis. Please try again, "
        "or paste the text instead of uploading a file.",
    )


def combine_warnings(
    file_warnings: list[str], result_warnings: list[str], prefix: str = "Resume: "
) -> list[str]:
    """Merge warnings from reading the resume file with those from analyze().

    File warnings come first and get the prefix. A warning is not listed twice
    when analyze() already reported the same text (with or without the prefix).
    """
    combined: list[str] = []
    for warning in file_warnings:
        tagged = f"{prefix}{warning}"
        if tagged not in result_warnings and warning not in result_warnings:
            combined.append(tagged)
    for warning in result_warnings:
        if warning not in combined:
            combined.append(warning)
    return combined


# ----------------------------------------------------------------------
# 3. Display: numbers
# ----------------------------------------------------------------------
def format_score(score: float) -> str:
    """Whole-number score, kept between 0 and 100."""
    return str(max(0, min(100, round(score))))


NO_SKILLS_TEXT = "no skills recognised"


def format_cosine(value: float) -> str:
    """Raw meaning-match value with two decimals, kept between 0 and 1."""
    return f"{max(0.0, min(1.0, value)):.2f}"


def format_percent(value: float) -> str:
    """Whole-number percentage, e.g. 0.357 -> '36%'."""
    return f"{round(100 * value)}%"


def skill_count_text(result: AnalysisResult) -> str:
    """'5 of 14 skills', counted from the matched and missing lists."""
    matched = len(dict.fromkeys(result.matched_skills))
    missing = sum(len(names) for names in group_missing(result.missing_skills).values())
    required = matched + missing
    if required == 0:
        return NO_SKILLS_TEXT
    word = "skill" if required == 1 else "skills"
    return f"{matched} of {required} {word}"


NOT_USED = "Not used"


def meaning_match_text(result: AnalysisResult) -> str:
    """The meaning match for display: '0.42', or 'Not used' when it was skipped.

    analyze() returns a placeholder 0.0 when the job description has no
    requirement lines; showing that as "0.00" would look like a measured result.
    """
    if semantic_not_used(result):
        return NOT_USED
    return format_cosine(result.semantic_score)


def meaning_match_detail(result: AnalysisResult) -> str:
    """Small second line under the meaning match (empty when it was measured)."""
    return "no requirement lines found" if semantic_not_used(result) else ""


def coverage_text(result: AnalysisResult) -> str:
    """Skill coverage for display: '36%', or 'Not used' when no skills were found."""
    if skill_count_text(result) == NO_SKILLS_TEXT:
        return NOT_USED
    return format_percent(result.skill_coverage)


def coverage_detail(result: AnalysisResult) -> str:
    """Small second line under the coverage: '5 of 14 skills'."""
    return skill_count_text(result)


def _with_detail(text: str, detail: str) -> str:
    return f"{text} ({detail})" if detail else text


def meaning_match_line(result: AnalysisResult) -> str:
    """One-line form for the text and PDF reports."""
    return _with_detail(meaning_match_text(result), meaning_match_detail(result))


def coverage_line(result: AnalysisResult) -> str:
    """One-line form for the text and PDF reports, e.g. '36% (5 of 14 skills)'."""
    return _with_detail(coverage_text(result), coverage_detail(result))


# ----------------------------------------------------------------------
# 3. Display: HTML pieces
# ----------------------------------------------------------------------
def verdict_style(verdict: str) -> dict[str, str]:
    """Colours for a verdict; an unknown verdict gets a neutral grey."""
    return dict(VERDICT_STYLES.get(verdict, NEUTRAL_STYLE))


def verdict_badge_html(verdict: str) -> str:
    """The rounded verdict pill (Fit / Partial Fit / No-Fit)."""
    style = verdict_style(verdict)
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'background:{style["bg"]};color:{style["fg"]};border-radius:999px;'
        f'padding:8px 16px;font-size:14px;font-weight:650">'
        f'<span style="width:8px;height:8px;border-radius:50%;'
        f'background:{style["ring"]};display:inline-block"></span>'
        f"{_esc(verdict)}</div>"
    )


def score_ring_html(score: float, verdict: str) -> str:
    """A ring gauge with the whole-number score in the middle."""
    value = max(0.0, min(100.0, float(score)))
    offset = RING_CIRCUMFERENCE * (1 - value / 100)
    colour = verdict_style(verdict)["ring"]
    label = f"Match score {format_score(score)} out of 100, {verdict}"
    svg = (
        f'<svg width="176" height="176" viewBox="0 0 140 140" role="img" '
        f'aria-label="{_esc(label)}">'
        f'<circle cx="70" cy="70" r="{RING_RADIUS}" fill="none" '
        f'stroke="#F0EDE6" stroke-width="12"></circle>'
        f'<circle cx="70" cy="70" r="{RING_RADIUS}" fill="none" '
        f'stroke="{colour}" stroke-width="12" stroke-linecap="round" '
        f'stroke-dasharray="{RING_CIRCUMFERENCE:.1f}" '
        f'stroke-dashoffset="{offset:.1f}" transform="rotate(-90 70 70)">'
        f"</circle></svg>"
    )
    centre = (
        '<div style="position:absolute;inset:0;display:flex;flex-direction:column;'
        'align-items:center;justify-content:center">'
        f'<div style="font-size:46px;font-weight:700;letter-spacing:-0.03em;'
        f'line-height:1;color:{INK}">{format_score(score)}</div>'
        f'<div style="font-size:13px;color:{MUTED};margin-top:4px">out of 100</div>'
        "</div>"
    )
    return (
        '<div style="position:relative;width:176px;height:176px;margin:0 auto">'
        f"{svg}{centre}</div>"
    )


def stat_html(label: str, value: str, detail: str = "") -> str:
    """A small tile with a label, a value and an optional detail line."""
    detail_html = (
        f'<div style="font-size:12px;color:{MUTED}">{_esc(detail)}</div>'
        if detail
        else ""
    )
    return (
        '<div style="flex:1;min-width:0;background:#FAF9F6;border-radius:12px;'
        'padding:14px">'
        f'<div style="font-size:12px;color:{MUTED};margin-bottom:4px">{_esc(label)}</div>'
        f'<div style="font-size:20px;font-weight:650;color:{INK}">{_esc(value)}</div>'
        f"{detail_html}</div>"
    )


def chip_html(label: str, kind: str) -> str:
    """One rounded skill chip. An unknown kind is drawn in neutral grey."""
    background, colour = CHIP_STYLES.get(kind, CHIP_STYLES["extra"])
    return (
        f'<span style="background:{background};color:{colour};border-radius:999px;'
        f'padding:8px 14px;font-size:14px;font-weight:550;display:inline-block">'
        f"{_esc(label)}</span>"
    )


def chips_html(labels: list[str], kind: str, empty_text: str = "None.") -> str:
    """A wrapped row of chips, or a quiet line of text when there are none."""
    if not labels:
        return f'<div style="font-size:14px;color:{MUTED}">{_esc(empty_text)}</div>'
    chips = "".join(chip_html(label, kind) for label in labels)
    return f'<div style="display:flex;flex-wrap:wrap;gap:10px">{chips}</div>'


def missing_section_html(missing: list[SkillGap]) -> str:
    """Missing skills grouped by importance, most important group first."""
    grouped = group_missing(missing)
    parts = []
    for key in (MUST_HAVE, NICE_TO_HAVE, UNKNOWN):
        names = grouped[key]
        if not names:
            continue
        heading = (
            f'<div style="font-size:12px;font-weight:650;letter-spacing:0.06em;'
            f"text-transform:uppercase;color:{GROUP_COLORS[key]};margin-bottom:12px"
            f'">{_esc(GROUP_TITLES[key])} &middot; {len(names)}</div>'
        )
        parts.append(f'<div style="margin-bottom:22px">{heading}{chips_html(names, key)}</div>')
    if not parts:
        return chips_html([], UNKNOWN, empty_text="No required skills are missing.")
    return "".join(parts)


def card_html(title: str, body_html: str, subtitle: str = "") -> str:
    """A white card with a title, an optional subtitle and trusted HTML inside.

    body_html must come from the helpers in this module (already escaped).
    """
    subtitle_html = (
        f'<div style="font-size:14px;color:{MUTED};margin-top:4px">{_esc(subtitle)}</div>'
        if subtitle
        else ""
    )
    heading = (
        f'<div style="margin-bottom:18px"><div style="font-size:18px;font-weight:650;'
        f'letter-spacing:-0.01em;color:{INK}">{_esc(title)}</div>{subtitle_html}</div>'
    )
    return f'<div style="{CARD_STYLE}">{heading}{body_html}</div>'


def warnings_box_html(warnings: list[str]) -> str:
    """The amber 'Things to check' box; an empty string when there are none."""
    if not warnings:
        return ""
    lines = "".join(
        f'<div style="font-size:14px;line-height:1.55;color:#6B4200;margin-top:4px">'
        f"{_esc(w)}</div>"
        for w in warnings
    )
    return (
        '<div style="background:#FFF8E8;border:1px solid #F3DFB0;border-radius:14px;'
        'padding:18px 22px;margin-bottom:16px">'
        '<div style="font-size:14px;font-weight:650;color:#6B4200">Things to check</div>'
        f"{lines}</div>"
    )


def error_box_html(title: str, message: str) -> str:
    """A calm rose-tinted box for an error message."""
    return (
        '<div style="background:#FDECEC;border:1px solid #F3C6C6;border-radius:14px;'
        'padding:18px 22px;margin-bottom:16px">'
        f'<div style="font-size:15px;font-weight:650;color:#7A1616">{_esc(title)}</div>'
        f'<div style="font-size:14px;line-height:1.55;color:#7A1616;margin-top:4px">'
        f"{_esc(message)}</div></div>"
    )
