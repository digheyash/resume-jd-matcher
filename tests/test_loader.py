"""Tests for resume_matcher.parsing.loader.

PDFs are generated on the fly with PyMuPDF inside pytest's temporary
folder, so no binary test files need to be stored in the repo.
"""

import pymupdf
import pytest

from resume_matcher.config import MIN_TEXT_CHARS, SAMPLES_DIR
from resume_matcher.parsing.loader import (
    DocumentLoadError,
    from_text,
    load_document,
    load_pdf,
    load_text_file,
)

LONG_LINES = [
    "SKILLS: Python, SQL, Machine Learning, scikit-learn, pandas",
    "PROJECTS: Built a spam classifier using TF-IDF and logistic regression",
    "EDUCATION: B.Tech in Computer Science, graduated with distinction",
    "EXPERIENCE: Intern at a data analytics company, built dashboards",
    "More text so that the document is comfortably above the minimum length",
]


def _make_pdf(path, lines):
    """Create a PDF with one line of text per entry (none if lines is empty)."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 20
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------- text input
def test_from_text_empty_gives_warning():
    result = from_text("   \n  ")
    assert any("No text" in w for w in result.warnings)


def test_from_text_short_gives_warning():
    result = from_text("Python developer")
    assert any("Very little text" in w for w in result.warnings)


def test_from_text_long_has_no_warnings():
    result = from_text("x" * (MIN_TEXT_CHARS + 50))
    assert result.warnings == []
    assert result.source == "text"


def test_load_text_file_sample_resume():
    result = load_text_file(SAMPLES_DIR / "sample_resume.txt")
    assert result.source == "text"
    # The sample file writes the name in capitals, so compare ignoring case.
    assert "priya sharma" in result.text.lower()


# ----------------------------------------------------------------- PDF input
def test_load_pdf_from_path(tmp_path):
    pdf_path = _make_pdf(tmp_path / "resume.pdf", LONG_LINES)
    result = load_pdf(pdf_path)
    assert "Python" in result.text
    assert result.page_count == 1
    assert result.source in ("pdfplumber", "pymupdf")
    assert result.warnings == []


def test_load_pdf_from_bytes(tmp_path):
    pdf_path = _make_pdf(tmp_path / "resume.pdf", LONG_LINES)
    result = load_pdf(pdf_path.read_bytes())
    assert "Machine Learning" in result.text


def test_blank_pdf_warns_about_scanned_image(tmp_path):
    pdf_path = _make_pdf(tmp_path / "blank.pdf", [])
    result = load_pdf(pdf_path)
    assert result.text.strip() == ""
    assert any("scanned" in w for w in result.warnings)


def test_corrupted_pdf_raises():
    with pytest.raises(DocumentLoadError):
        load_pdf(b"this is definitely not a pdf")


def test_missing_pdf_raises(tmp_path):
    with pytest.raises(DocumentLoadError):
        load_pdf(tmp_path / "does_not_exist.pdf")


# ------------------------------------------------------------- load_document
def test_load_document_picks_pdf_reader(tmp_path):
    pdf_path = _make_pdf(tmp_path / "resume.pdf", LONG_LINES)
    result = load_document(pdf_path)
    assert "Python" in result.text


def test_load_document_rejects_unknown_type(tmp_path):
    odd_file = tmp_path / "resume.xyz"
    odd_file.write_text("hello")
    with pytest.raises(DocumentLoadError):
        load_document(odd_file)
