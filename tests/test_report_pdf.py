"""Tests for app/report_pdf.py (the PDF download).

The PDF is built in memory and checked in two ways:
  * its bytes (it is a PDF, it has pages, it never raises), and
  * its text, read back with a PDF text reader (pypdf or pdfplumber, whichever
    is installed; those tests are skipped if neither is).

All numbers and names below are made-up test data.
"""

import io
from datetime import datetime

import pytest

from report import GUIDE_NOT_VERDICT, HOW_TO_READ
from report_pdf import build_pdf, pdf_safe
from resume_matcher.schemas import AnalysisResult, SkillGap

WHEN = datetime(2026, 10, 19, 9, 5)


def make_result(**changes) -> AnalysisResult:
    values = dict(
        overall_score=52.8,
        verdict="Partial Fit",
        semantic_score=0.424,
        skill_coverage=0.5,
        matched_skills=["Python", "Pandas"],
        missing_skills=[
            SkillGap("Docker", "must-have"),
            SkillGap("PyTorch", "must-have"),
            SkillGap("AWS", "nice-to-have"),
            SkillGap("Kubernetes", "unknown"),
        ],
        extra_skills=["Git"],
        warnings=[],
    )
    values.update(changes)
    return AnalysisResult(**values)


def make_pdf(result=None, summary="This resume is a partial fit.", warnings=(),
             generated_at=WHEN) -> bytes:
    return build_pdf(result or make_result(), summary, list(warnings), generated_at)


def read_text(data: bytes) -> str:
    """Extract the text of a PDF, or skip the test if no reader is installed."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        pytest.skip("neither pypdf nor pdfplumber is installed")


def flat(data: bytes) -> str:
    """Extracted text with all white space collapsed to single spaces."""
    return " ".join(read_text(data).split())


# ----------------------------------------------------------------------
# pdf_safe
# ----------------------------------------------------------------------
def test_pdf_safe_keeps_plain_text():
    assert pdf_safe("Deep Learning, 53/100") == "Deep Learning, 53/100"


def test_pdf_safe_keeps_latin1_letters():
    assert pdf_safe("résumé") == "résumé"


def test_pdf_safe_replaces_curly_quotes_and_dashes():
    assert pdf_safe("“a” ‘b’ c–d e—f") == '"a" \'b\' c-d e-f'


def test_pdf_safe_replaces_ellipsis_and_bullet():
    assert pdf_safe("wait… • item") == "wait... - item"


def test_pdf_safe_replaces_unknown_characters_with_question_marks():
    assert pdf_safe("日本") == "??"


def test_pdf_safe_handles_emoji_without_raising():
    assert pdf_safe("ok 😀") == "ok ?"


def test_pdf_safe_empty_string():
    assert pdf_safe("") == ""


def test_pdf_safe_result_always_fits_latin1():
    pdf_safe("mixed “quotes” 日本 😀 café").encode("latin-1")


# ----------------------------------------------------------------------
# The file itself
# ----------------------------------------------------------------------
def test_build_pdf_returns_bytes():
    assert isinstance(make_pdf(), bytes)


def test_output_starts_like_a_pdf():
    assert make_pdf().startswith(b"%PDF-")


def test_output_ends_like_a_pdf():
    assert make_pdf().rstrip().endswith(b"%%EOF")


def test_output_is_not_trivially_small():
    assert len(make_pdf()) > 1000


def test_output_has_at_least_one_page():
    assert read_text(make_pdf()).strip() != ""


def test_generated_at_is_optional():
    assert make_pdf(generated_at=None).startswith(b"%PDF-")


def test_generated_time_appears_when_given():
    assert "Generated: 2026-10-19 09:05" in flat(make_pdf())


def test_no_generated_line_when_time_is_not_given():
    assert "Generated:" not in flat(make_pdf(generated_at=None))


# ----------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------
# Section headings are printed in capitals (for example "EXTRA SKILLS (1)");
# the two group labels under "Skills to add" keep normal case.
def test_title_is_present():
    assert "Resume Match Report" in flat(make_pdf())


def test_score_is_the_rounded_whole_number():
    text = flat(make_pdf())
    assert "53" in text
    assert "52.8" not in text


def test_score_out_of_100_is_present():
    assert "/ 100" in flat(make_pdf())


def test_verdict_is_present():
    assert "Partial Fit" in flat(make_pdf())


def test_each_verdict_prints():
    for verdict in ("Fit", "Partial Fit", "No-Fit"):
        assert verdict in flat(make_pdf(make_result(verdict=verdict)))


def test_unknown_verdict_does_not_crash():
    assert "Maybe" in flat(make_pdf(make_result(verdict="Maybe")))


def test_meaning_match_is_shown_with_two_decimals():
    text = flat(make_pdf())
    assert "Meaning match" in text
    assert "0.42" in text


def test_skill_coverage_is_shown_as_percent_and_count():
    text = flat(make_pdf())
    assert "Skill coverage" in text
    assert "50%" in text


def test_summary_text_is_present():
    text = flat(make_pdf(summary="UNIQUE SUMMARY SENTENCE here."))
    assert "UNIQUE SUMMARY SENTENCE here." in text


def test_missing_skills_are_listed_by_group():
    text = flat(make_pdf())
    assert "Must-have (2)" in text
    assert "Nice-to-have (1)" in text
    for name in ("Docker", "PyTorch", "AWS", "Kubernetes"):
        assert name in text


def test_must_have_comes_before_nice_to_have():
    text = flat(make_pdf())
    assert text.index("Must-have (2)") < text.index("Nice-to-have (1)")


def test_unknown_importance_group_is_listed():
    text = flat(make_pdf())
    assert "Kubernetes" in text


def test_no_missing_skills_says_so():
    assert "No required skills are missing." in flat(
        make_pdf(make_result(missing_skills=[]))
    )


def test_matched_and_extra_skills_are_listed():
    text = flat(make_pdf())
    assert "SKILLS YOU ALREADY HAVE (2)" in text
    assert "Pandas" in text
    assert "EXTRA SKILLS (1)" in text
    assert "Git" in text


def test_empty_skill_lists_say_none():
    text = flat(make_pdf(make_result(matched_skills=[], extra_skills=[])))
    assert "SKILLS YOU ALREADY HAVE (0)" in text
    assert "EXTRA SKILLS (0)" in text
    assert "None." in text


def test_duplicate_skill_names_are_listed_once():
    text = flat(make_pdf(make_result(matched_skills=["Python", "Python"])))
    assert "SKILLS YOU ALREADY HAVE (1)" in text


def test_both_closing_notes_are_present_word_for_word():
    text = flat(make_pdf())
    assert " ".join(HOW_TO_READ.split()) in text
    assert " ".join(GUIDE_NOT_VERDICT.split()) in text


def test_duplicate_extra_skills_are_listed_once():
    text = flat(make_pdf(make_result(extra_skills=["Git", "Git"])))
    assert "EXTRA SKILLS (1)" in text


# ----------------------------------------------------------------------
# Warnings
# ----------------------------------------------------------------------
def test_warnings_section_appears_when_there_are_warnings():
    text = flat(make_pdf(warnings=["The job description is very short."]))
    assert "THINGS TO CHECK" in text
    assert "The job description is very short." in text


def test_no_warnings_section_without_warnings():
    assert "THINGS TO CHECK" not in flat(make_pdf())


def test_awkward_characters_in_warnings_do_not_crash():
    data = make_pdf(warnings=["file “résumé—final.pdf” 日本語 😀 looked short"])
    text = flat(data)
    assert "looked short" in text
    assert "résumé-final.pdf" in text


def test_awkward_characters_in_skill_names_do_not_crash():
    data = make_pdf(make_result(matched_skills=["C++ ✓", "Café"]))
    assert "Café" in flat(data)


def test_awkward_characters_in_the_summary_do_not_crash():
    assert make_pdf(summary="Summary with 日本語 and “quotes”.").startswith(b"%PDF-")


# ----------------------------------------------------------------------
# Size and layout
# ----------------------------------------------------------------------
def test_a_long_report_runs_onto_more_pages_without_crashing():
    many = [SkillGap(f"Skill {i}", "must-have") for i in range(120)]
    data = make_pdf(make_result(missing_skills=many))
    assert "Page 2" in flat(data)
    assert "Skill 119" in flat(data)


def test_page_numbers_are_printed():
    assert "Page 1 of" in flat(make_pdf())


def test_a_very_long_unbroken_word_does_not_crash():
    data = make_pdf(summary="x" * 400)
    assert data.startswith(b"%PDF-")


def test_does_not_change_its_inputs():
    result = make_result()
    warnings = ["one"]
    build_pdf(result, "s", warnings, WHEN)
    assert warnings == ["one"]
    assert [g.skill for g in result.missing_skills][0] == "Docker"


# ----------------------------------------------------------------------
# "Not used" values
# ----------------------------------------------------------------------
def test_meaning_match_is_not_used_when_the_job_has_no_requirement_lines():
    warning = ("No requirement lines were found in the job description; the "
               "score uses skill coverage only.")
    result = make_result(semantic_score=0.0, warnings=[warning])
    text = flat(make_pdf(result, warnings=[warning]))
    assert "Meaning match Not used (no requirement lines found)" in text
    assert "0.00" not in text


def test_coverage_is_not_used_when_no_skills_were_recognised():
    result = make_result(matched_skills=[], missing_skills=[], skill_coverage=0.0)
    assert "Skill coverage Not used (no skills recognised)" in flat(make_pdf(result))
