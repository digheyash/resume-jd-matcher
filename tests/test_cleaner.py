"""Tests for resume_matcher.parsing.cleaner.

Special characters are written as \\u escapes so this file stays plain ASCII.
"""

from resume_matcher.config import SAMPLES_DIR
from resume_matcher.parsing.cleaner import clean_text
from resume_matcher.parsing.loader import load_text_file


# ------------------------------------------------------------ contact details
def test_removes_email_and_records_it():
    result = clean_text("Contact: priya.sharma@example.com for details")
    assert "@" not in result.text
    assert result.text == "Contact: for details"
    assert result.removed["emails"] == ["priya.sharma@example.com"]


def test_removes_phone_with_country_code():
    result = clean_text("Call +91 90000 00000 now")
    assert result.text == "Call now"
    assert result.removed["phones"] == ["+91 90000 00000"]


def test_keeps_year_range_that_looks_like_a_number():
    result = clean_text("B.Tech in CS (2022-2026)")
    assert "2022-2026" in result.text
    assert result.removed["phones"] == []


def test_keeps_skill_names_that_contain_dots_and_symbols():
    result = clean_text("Skills: Node.js, ASP.NET, Vue.js, C++, C#, .NET")
    for skill in ("Node.js", "ASP.NET", "Vue.js", "C++", "C#", ".NET"):
        assert skill in result.text
    assert result.removed["urls"] == []


def test_removes_urls_and_keeps_sentence_punctuation():
    result = clean_text("See https://example.com/page. Also visit www.example.org today")
    assert "https" not in result.text
    assert "www" not in result.text
    assert result.removed["urls"] == ["https://example.com/page", "www.example.org"]
    assert "today" in result.text


def test_removes_profile_links_without_http():
    result = clean_text("Code at github.com/example-priya")
    assert "github" not in result.text
    assert result.removed["urls"] == ["github.com/example-priya"]


# -------------------------------------------------------------------- bullets
def test_removes_bullet_symbols_but_keeps_dash_lines():
    raw = "\u2022 Built a model\n\u25cf Wrote tests\n- Kept dash\n* Star item"
    result = clean_text(raw)
    assert result.text.split("\n") == [
        "Built a model",
        "Wrote tests",
        "- Kept dash",
        "Star item",
    ]
    assert result.removed["bullets"] == ["\u2022", "\u25cf", "*"]


def test_removes_private_use_bullet_from_pdf_fonts():
    result = clean_text("\uf0b7 Python\n\uf0b7 SQL")
    assert result.text == "Python\nSQL"


# --------------------------------------------------------------- whitespace
def test_normalizes_line_endings():
    result = clean_text("Line one\r\nLine two\rLine three")
    assert "\r" not in result.text
    assert result.text.split("\n") == ["Line one", "Line two", "Line three"]


def test_collapses_extra_spaces_tabs_and_blank_lines():
    result = clean_text("Python    SQL\t\tPandas\n\n\n\n\nNext")
    assert result.text == "Python SQL Pandas\n\nNext"


def test_fixes_ligatures_and_removes_zero_width_characters():
    assert "office" in clean_text("of\ufb01ce").text
    assert clean_text("data\u200bscience").text == "datascience"


def test_contact_line_disappears_completely():
    raw = "A@b.com | +91 90000 00000 | github.com/x\nSKILLS\nPython"
    result = clean_text(raw)
    assert result.text == "SKILLS\nPython"


# -------------------------------------------------------- whole sample file
def test_sample_resume_is_cleaned():
    raw = load_text_file(SAMPLES_DIR / "sample_resume.txt").text
    result = clean_text(raw)

    assert result.removed["emails"] == ["priya.sharma@example.com"]
    assert len(result.removed["phones"]) == 1
    assert result.removed["urls"] == ["github.com/example-priya"]

    assert "@" not in result.text
    assert "\r" not in result.text
    assert "example-priya" not in result.text
    assert "SKILLS" in result.text
    assert "Python" in result.text
    assert "2022-2026" in result.text  # the year range must survive


def test_cleaning_twice_changes_nothing():
    raw = load_text_file(SAMPLES_DIR / "sample_resume.txt").text
    once = clean_text(raw).text
    twice = clean_text(once).text
    assert once == twice


# ---------------------------------------------------------------- edge cases
def test_empty_input():
    result = clean_text("")
    assert result.text == ""
    assert result.summary() == "nothing removed"


def test_summary_describes_removals():
    result = clean_text("a@b.com")
    assert result.text == ""
    assert "1 emails" in result.summary()
