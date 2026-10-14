"""Tests for resume_matcher.parsing.sections."""

import pytest

from resume_matcher.config import SAMPLES_DIR
from resume_matcher.parsing.cleaner import clean_text
from resume_matcher.parsing.loader import load_text_file
from resume_matcher.parsing.sections import (
    FULL_TEXT_KEY,
    HEADER_KEY,
    detect_sections,
)


# ----------------------------------------------------------- basic splitting
def test_plain_headings_split_a_resume():
    text = (
        "PRIYA SHARMA\n\nSUMMARY\nStudent.\n\nSKILLS\nPython, SQL\n\n"
        "PROJECTS\nChurn predictor\n\nEDUCATION\nB.Tech"
    )
    sections = detect_sections(text)
    assert list(sections) == [HEADER_KEY, "summary", "skills", "projects", "education"]
    assert sections[HEADER_KEY] == "PRIYA SHARMA"
    assert sections["skills"] == "Python, SQL"


def test_heading_case_and_trailing_colon_do_not_matter():
    sections = detect_sections("Work Experience:\nIntern at X\nTechnical Skills\nPython")
    assert sections == {"experience": "Intern at X", "skills": "Python"}


def test_inline_heading_with_content_on_same_line():
    sections = detect_sections("Skills: Python, SQL, Pandas\nEducation: B.Tech in CS")
    assert sections == {"skills": "Python, SQL, Pandas", "education": "B.Tech in CS"}


def test_sentence_is_not_mistaken_for_a_heading():
    sections = detect_sections("SKILLS\nI have project management experience\nPython")
    assert sections == {"skills": "I have project management experience\nPython"}


def test_spaced_out_letters_are_recognised():
    assert detect_sections("S K I L L S\nPython") == {"skills": "Python"}


def test_repeated_headings_are_merged():
    sections = detect_sections("PROJECTS\nOne\nSKILLS\nPython\nPROJECTS\nTwo")
    assert sections["projects"] == "One\nTwo"
    assert sections["skills"] == "Python"


def test_empty_sections_are_left_out():
    assert detect_sections("SKILLS\n\nPROJECTS\nChurn") == {"projects": "Churn"}


def test_languages_and_hobbies_do_not_pollute_skills():
    text = "SKILLS\nPython\nLanguages: English, Hindi\nHobbies\nChess"
    sections = detect_sections(text)
    assert sections["skills"] == "Python"
    assert sections["other"] == "English, Hindi\nChess"


# ---------------------------------------------------------------- edge cases
def test_text_without_headings_returns_full_text():
    text = "Just some plain text\nwithout headings"
    assert detect_sections(text) == {FULL_TEXT_KEY: text}


def test_empty_input_returns_empty_dict():
    assert detect_sections("") == {}


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        detect_sections("anything", kind="cv")


# --------------------------------------------------------- job descriptions
def test_jd_mode_splits_requirements_and_nice_to_have():
    text = (
        "Junior ML Engineer\nAbout the role\nWe build things.\n"
        "Requirements\nPython\nSQL\nNice to have\nDocker"
    )
    sections = detect_sections(text, kind="jd")
    assert sections[HEADER_KEY] == "Junior ML Engineer"
    assert sections["about"] == "We build things."
    assert sections["requirements"] == "Python\nSQL"
    assert sections["nice_to_have"] == "Docker"


def test_resume_mode_does_not_know_jd_headings():
    text = "Requirements\nPython"
    assert detect_sections(text, kind="resume") == {FULL_TEXT_KEY: text}
    assert detect_sections(text, kind="jd") == {"requirements": "Python"}


# -------------------------------------------------------- the sample files
def test_sample_resume_sections():
    raw = load_text_file(SAMPLES_DIR / "sample_resume.txt").text
    sections = detect_sections(clean_text(raw).text, kind="resume")

    assert sections[HEADER_KEY] == "PRIYA SHARMA"
    for name in ("summary", "skills", "projects", "education"):
        assert name in sections
    assert "Python" in sections["skills"]
    assert "Sales Dashboard" in sections["projects"]
    assert "B.Tech" in sections["education"]


def test_sample_jd_sections():
    raw = load_text_file(SAMPLES_DIR / "sample_jd.txt").text
    sections = detect_sections(clean_text(raw).text, kind="jd")

    assert "requirements" in sections
    assert "nice_to_have" in sections
    assert "python" in sections["requirements"].lower()
