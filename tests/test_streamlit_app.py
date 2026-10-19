"""Tests for app/streamlit_app.py, the Streamlit page.

The page is run with Streamlit's own headless test tool (AppTest): it executes
the script, lets the test "click" buttons and "type" into boxes, and returns
the elements that were drawn. No browser and no web server are needed.

pipeline.analyze() is replaced by a fake in every test, so these tests do not
load the Sentence-BERT model and run in a few seconds. They check how the PAGE
behaves (what it shows for each situation). They do not check the quality of
the real scores; that is covered by the Phase 3-5 tests.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from resume_matcher import pipeline
from resume_matcher.analyzer.analyzer import AnalysisError
from resume_matcher.schemas import AnalysisResult, SkillGap

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
SAMPLE_RESUME = ROOT / "data" / "samples" / "sample_resume.txt"
SAMPLE_JD = ROOT / "data" / "samples" / "sample_jd.txt"

RESUME = "Python developer. Built models with Pandas and scikit-learn."
JD = "Must have: Python, Docker and PyTorch."


def make_result(**changes) -> AnalysisResult:
    """A made-up result (these numbers are test data, not real output)."""
    values = dict(
        overall_score=52.8,
        verdict="Partial Fit",
        semantic_score=0.424,
        skill_coverage=0.5,
        matched_skills=["Python", "Pandas"],
        missing_skills=[SkillGap("Docker", "must-have"), SkillGap("AWS", "nice-to-have")],
        extra_skills=["Git"],
        warnings=[],
    )
    values.update(changes)
    return AnalysisResult(**values)


class FakeAnalyze:
    """Stands in for pipeline.analyze and remembers how it was called."""

    def __init__(self, result=None, error=None):
        self.result = result if result is not None else make_result()
        self.error = error
        self.calls = []

    def __call__(self, resume_text, jd_text, *args, **kwargs):
        self.calls.append((resume_text, jd_text))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def fake(monkeypatch):
    """Install a fake analyze() and return it."""
    fake_analyze = FakeAnalyze()
    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    return fake_analyze


def new_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30).run()


def button(at: AppTest, key: str):
    return next(b for b in at.button if b.key == key)


def page_text(at: AppTest) -> str:
    """All the markdown the page drew, as one string."""
    return " ".join(element.value for element in at.markdown)


def paste_and_analyze(at: AppTest, resume: str = RESUME, jd: str = JD) -> AppTest:
    at.radio(key="resume_mode").set_value("Paste text").run()
    at.text_area(key="resume_text").set_value(resume)
    at.text_area(key="jd_text").set_value(jd)
    return button(at, "analyze").click().run()


# ----------------------------------------------------------------------
# First view
# ----------------------------------------------------------------------
def test_first_view_has_no_exception(fake):
    at = new_app()
    assert not at.exception


def test_first_view_shows_the_inputs_and_buttons(fake):
    at = new_app()
    assert at.radio(key="resume_mode").value == "Upload file"
    assert at.text_area(key="jd_text").value == ""
    assert {b.key for b in at.button} >= {"load_sample", "analyze"}


def test_first_view_shows_the_heading(fake):
    at = new_app()
    assert "How well does your resume fit the job?" in page_text(at)


def test_first_view_has_no_results_and_no_download(fake):
    at = new_app()
    assert "Overall match" not in page_text(at)
    assert len(at.get("download_button")) == 0
    assert fake.calls == []


def test_uploading_is_the_default_so_no_resume_text_box(fake):
    at = new_app()
    assert [t.key for t in at.text_area] == ["jd_text"]


# ----------------------------------------------------------------------
# Missing input
# ----------------------------------------------------------------------
def test_empty_click_shows_both_problems_and_does_not_analyze(fake):
    at = new_app()
    at = button(at, "analyze").click().run()
    text = page_text(at)
    assert "Something is missing" in text
    assert "Add your resume" in text
    assert "Paste the job description" in text
    assert fake.calls == []
    assert not at.exception


def test_only_resume_pasted_asks_for_the_job_description(fake):
    at = new_app()
    at = paste_and_analyze(at, resume=RESUME, jd="   ")
    text = page_text(at)
    assert "Paste the job description" in text
    assert "Add your resume" not in text
    assert fake.calls == []


def test_only_jd_pasted_asks_for_the_resume(fake):
    at = new_app()
    at = paste_and_analyze(at, resume="", jd=JD)
    text = page_text(at)
    assert "Add your resume" in text
    assert "Paste the job description" not in text
    assert fake.calls == []


def test_problems_disappear_after_a_good_run(fake):
    at = new_app()
    at = button(at, "analyze").click().run()
    assert "Something is missing" in page_text(at)
    at.radio(key="resume_mode").set_value("Paste text").run()
    at.text_area(key="resume_text").set_value(RESUME)
    at.text_area(key="jd_text").set_value(JD)
    at = button(at, "analyze").click().run()
    assert "Something is missing" not in page_text(at)
    assert "Overall match" in page_text(at)


# ----------------------------------------------------------------------
# A normal analysis
# ----------------------------------------------------------------------
def test_pasted_texts_are_passed_to_analyze_unchanged(fake):
    at = paste_and_analyze(new_app())
    assert fake.calls == [(RESUME, JD)]
    assert not at.exception


def test_results_show_every_section(fake):
    text = page_text(paste_and_analyze(new_app()))
    for heading in (
        "Summary", "Overall match", "Skills to add", "Skills you already have",
        "Extra skills", "How to read this", "A guide, not a verdict",
    ):
        assert heading in text, heading


def test_results_show_the_numbers_from_the_result(fake):
    text = page_text(paste_and_analyze(new_app()))
    assert ">53<" in text.replace(" ", "")          # the whole-number score
    assert "Partial Fit" in text
    assert "0.42" in text                            # meaning match
    assert "50%" in text                             # skill coverage


def test_meaning_match_tile_says_not_used_when_requirements_were_missing(fake):
    fake.result = make_result(
        semantic_score=0.0,
        warnings=["No requirement lines were found in the job description; the "
                  "score uses skill coverage only."],
    )
    text = page_text(paste_and_analyze(new_app()))
    assert "Not used" in text
    assert "no requirement lines found" in text
    assert "0.00" not in text


def test_results_show_the_skill_names(fake):
    text = page_text(paste_and_analyze(new_app()))
    for name in ("Docker", "AWS", "Python", "Pandas", "Git"):
        assert name in text, name


def test_summary_paragraph_uses_the_same_score(fake):
    text = page_text(paste_and_analyze(new_app()))
    assert "a partial fit for the role, with a score of 53 out of 100" in text


def test_both_download_buttons_appear_after_analysis(fake):
    at = paste_and_analyze(new_app())
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["Download report (.txt)", "Download report (PDF)"]


def test_the_pdf_failing_does_not_break_the_page(fake, monkeypatch):
    import report_pdf

    def broken(*args, **kwargs):
        raise RuntimeError("pdf engine exploded")

    monkeypatch.setattr(report_pdf, "build_pdf", broken)
    at = paste_and_analyze(new_app())
    assert not at.exception
    text = page_text(at) + " ".join(c.value for c in at.caption)
    assert "The PDF report could not be created" in text
    assert "pdf engine exploded" not in text
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["Download report (.txt)"]
    assert "Overall match" in page_text(at)


def test_a_second_analysis_replaces_the_first(fake):
    at = paste_and_analyze(new_app())
    fake.result = make_result(overall_score=81.0, verdict="Fit", skill_coverage=0.9)
    at = button(at, "analyze").click().run()
    text = page_text(at)
    assert "Fit" in text
    assert "a good fit" in text
    assert "Partial Fit" not in text
    assert len(fake.calls) == 2


def test_result_stays_when_the_user_types_again(fake):
    at = paste_and_analyze(new_app())
    at.text_area(key="jd_text").set_value(JD + " More text.").run()
    assert "Overall match" in page_text(at)
    assert len(fake.calls) == 1


# ----------------------------------------------------------------------
# Warnings
# ----------------------------------------------------------------------
def test_warnings_from_analyze_are_shown(fake):
    fake.result = make_result(warnings=["The job description is very short."])
    text = page_text(paste_and_analyze(new_app()))
    assert "The job description is very short." in text


def test_no_warning_box_when_there_are_no_warnings(fake):
    text = page_text(paste_and_analyze(new_app()))
    assert "The job description is very short." not in text


def test_html_in_a_skill_or_warning_is_escaped(fake):
    fake.result = make_result(
        matched_skills=["<script>alert(1)</script>"],
        warnings=["<b>bold</b>"],
    )
    text = page_text(paste_and_analyze(new_app()))
    assert "<script>" not in text
    assert "<b>bold</b>" not in text
    assert "&lt;script&gt;" in text


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
def test_analysis_error_shows_a_friendly_box(fake):
    fake.error = AnalysisError("Both texts must contain real words.")
    at = paste_and_analyze(new_app())
    text = page_text(at)
    assert "We could not analyse these texts" in text
    assert "Both texts must contain real words." in text
    assert "Overall match" not in text
    assert not at.exception


def test_unexpected_error_shows_a_friendly_box_without_a_traceback(fake):
    fake.error = RuntimeError("secret internal detail")
    at = paste_and_analyze(new_app())
    text = page_text(at)
    assert "Something went wrong" in text
    assert "secret internal detail" not in text
    assert "Traceback" not in text
    assert not at.exception


def test_an_old_result_is_cleared_when_the_next_run_fails(fake):
    at = paste_and_analyze(new_app())
    assert "Overall match" in page_text(at)
    fake.error = AnalysisError("Nope.")
    at = button(at, "analyze").click().run()
    text = page_text(at)
    assert "Overall match" not in text
    assert "Nope." in text


# ----------------------------------------------------------------------
# Sample pair
# ----------------------------------------------------------------------
needs_samples = pytest.mark.skipif(
    not (SAMPLE_RESUME.exists() and SAMPLE_JD.exists()),
    reason="sample files are missing from data/samples",
)


@needs_samples
def test_sample_button_fills_both_boxes(fake):
    at = button(new_app(), "load_sample").click().run()
    assert at.radio(key="resume_mode").value == "Paste text"
    assert at.text_area(key="resume_text").value.strip() != ""
    assert at.text_area(key="jd_text").value.strip() != ""
    assert not at.exception


@needs_samples
def test_sample_button_does_not_run_the_analysis(fake):
    button(new_app(), "load_sample").click().run()
    assert fake.calls == []


@needs_samples
def test_sample_then_analyze_sends_the_sample_texts(fake):
    at = button(new_app(), "load_sample").click().run()
    at = button(at, "analyze").click().run()
    assert len(fake.calls) == 1
    resume_sent, jd_sent = fake.calls[0]
    assert resume_sent == at.text_area(key="resume_text").value
    assert jd_sent == at.text_area(key="jd_text").value
    assert "Overall match" in page_text(at)


@needs_samples
def test_sample_button_clears_an_earlier_result(fake):
    at = paste_and_analyze(new_app())
    at = button(at, "load_sample").click().run()
    assert "Overall match" not in page_text(at)


def test_html_in_the_summary_paragraph_is_escaped(fake):
    # The summary names the most important missing skills, so a skill name
    # with markup in it ends up inside the summary paragraph.
    fake.result = make_result(missing_skills=[SkillGap("<i>Docker</i>", "must-have")])
    text = page_text(paste_and_analyze(new_app()))
    assert "<i>Docker</i>" not in text
    assert "&lt;i&gt;Docker&lt;/i&gt;" in text
