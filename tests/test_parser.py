"""Tests for resume_matcher.parsing.parser (the load -> clean -> sections chain)."""

import pymupdf
import pytest

from resume_matcher.config import SAMPLES_DIR
from resume_matcher.parsing.loader import DocumentLoadError
from resume_matcher.parsing.parser import parse_document, parse_text


def _pdf_bytes(lines):
    """Build a small PDF in memory, one line of text per entry."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 20
    data = doc.tobytes()
    doc.close()
    return data


def _section_warnings(parsed):
    """Warnings produced by section detection (they all mention 'section')."""
    return [w for w in parsed.warnings if "section" in w.lower()]


# -------------------------------------------------------------- sample files
def test_parse_sample_resume_file():
    parsed = parse_document(SAMPLES_DIR / "sample_resume.txt", kind="resume")

    assert parsed.kind == "resume"
    assert parsed.source == "text"
    assert "skills" in parsed.sections
    assert "@" not in parsed.text
    assert parsed.removed["emails"] == ["priya.sharma@example.com"]
    assert _section_warnings(parsed) == []


def test_parse_sample_jd_file():
    parsed = parse_document(SAMPLES_DIR / "sample_jd.txt", kind="jd")

    assert parsed.kind == "jd"
    assert "requirements" in parsed.sections
    assert "nice_to_have" in parsed.sections
    assert _section_warnings(parsed) == []


# ------------------------------------------------------------------ PDF input
def test_parse_pdf_bytes():
    data = _pdf_bytes(
        [
            "PRIYA SHARMA",
            "SKILLS",
            "Python, SQL, Pandas",
            "PROJECTS",
            "Churn predictor using scikit-learn",
            "EDUCATION",
            "B.Tech in Computer Science",
        ]
    )
    parsed = parse_document(data, kind="resume")

    assert parsed.page_count == 1
    assert parsed.source in ("pdfplumber", "pymupdf")
    for name in ("skills", "projects", "education"):
        assert name in parsed.sections
    assert "Python" in parsed.sections["skills"]


def test_blank_pdf_is_reported_not_crashed():
    parsed = parse_document(_pdf_bytes([]), kind="resume")

    assert parsed.text == ""
    assert parsed.sections == {}
    assert any("scanned" in w for w in parsed.warnings)


# ------------------------------------------------------------ pasted text
def test_parse_text_jd_sections():
    parsed = parse_text(
        "Junior ML Engineer\nRequirements\nPython\nNice to have\nDocker", kind="jd"
    )
    assert parsed.sections["requirements"] == "Python"
    assert parsed.sections["nice_to_have"] == "Docker"


def test_raw_text_is_kept_and_clean_text_is_cleaned():
    raw = "SKILLS\r\nPython a@b.com"
    parsed = parse_text(raw, kind="resume")

    assert parsed.raw_text == raw
    assert "@" not in parsed.text
    assert "\r" not in parsed.text


def test_normalized_text_is_computed_on_demand():
    parsed = parse_text("SKILLS\nI built models and studied results", kind="resume")
    tokens = parsed.normalized_text().split()

    assert "build" in tokens
    assert "model" in tokens
    assert "and" not in tokens


# ----------------------------------------------------------------- warnings
def test_resume_without_skills_section_warns():
    parsed = parse_text("PRIYA\nEDUCATION\nB.Tech in CS", kind="resume")
    assert any("No Skills section" in w for w in parsed.warnings)


def test_text_without_headings_warns():
    parsed = parse_text("just some plain words here", kind="resume")

    assert list(parsed.sections) == ["full_text"]
    assert any("No section headings" in w for w in parsed.warnings)


def test_jd_without_requirements_warns():
    parsed = parse_text("ABOUT THE ROLE\nWe build things", kind="jd")
    assert any("Requirements" in w for w in parsed.warnings)


def test_short_text_keeps_the_loader_warning():
    parsed = parse_text("Python", kind="resume")
    assert any("Very little text" in w for w in parsed.warnings)


def test_empty_text_does_not_crash():
    parsed = parse_text("", kind="resume")

    assert parsed.text == ""
    assert parsed.sections == {}
    assert any("No text" in w for w in parsed.warnings)


# ------------------------------------------------------------------- errors
def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        parse_text("anything", kind="cv")
    with pytest.raises(ValueError):
        parse_document("whatever.pdf", kind="cv")


def test_missing_file_raises_loader_error(tmp_path):
    with pytest.raises(DocumentLoadError):
        parse_document(tmp_path / "nope.pdf", kind="resume")
