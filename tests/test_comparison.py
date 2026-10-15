"""Tests for resume_matcher.skills.comparison (Phase 3)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from resume_matcher.skills.comparison import (
    SkillComparison,
    compare_documents,
    compare_skills,
)
from resume_matcher.skills.extractor import SkillExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "samples"


@pytest.fixture(scope="module")
def ext():
    return SkillExtractor()


def compare_text(ext, resume_text, jd_text):
    """Compare two plain skill lists (both treated as a 'skills' section)."""
    return compare_skills(
        ext.extract_from_text(resume_text, "skills"),
        ext.extract_from_text(jd_text, "skills"),
    )


# ------------------------------------------------------------ set logic


def test_tiny_example_from_the_docstring(ext):
    result = compare_text(ext, "Python, SQL, Git", "Python, SQL, Docker")
    assert result.matched == ("Python", "SQL")
    assert result.missing == ("Docker",)
    assert result.extra == ("Git",)
    assert result.jd_total == 3
    assert result.coverage == pytest.approx(2 / 3)
    assert result.coverage_percent == pytest.approx(200 / 3)
    assert result.has_jd_skills


def test_identical_skill_sets_give_full_coverage(ext):
    result = compare_text(ext, "Python, SQL", "Python, SQL")
    assert result.missing == () and result.extra == ()
    assert result.coverage == 1.0


def test_disjoint_skill_sets_give_zero_coverage(ext):
    result = compare_text(ext, "Java, Git", "Python, SQL")
    assert result.matched == ()
    assert result.missing == ("Python", "SQL")
    assert result.extra == ("Git", "Java")
    assert result.coverage == 0.0


def test_empty_resume_misses_every_jd_skill(ext):
    result = compare_text(ext, "", "Python, SQL")
    assert result.missing == ("Python", "SQL")
    assert result.coverage == 0.0
    assert result.has_jd_skills


def test_jd_without_skills_is_flagged_not_hidden(ext):
    result = compare_text(ext, "Python", "We are a friendly team.")
    assert result.jd_total == 0
    assert not result.has_jd_skills
    assert result.coverage == 0.0
    assert result.coverage_percent == 0.0
    assert result.extra == ("Python",)


def test_aliases_are_compared_by_canonical_name(ext):
    result = compare_text(ext, "sklearn, golang", "Scikit-learn, Go")
    assert result.matched == ("Go", "Scikit-learn")
    assert result.missing == ()


def test_outputs_are_sorted_ignoring_case(ext):
    result = compare_text(ext, "", "NumPy, AWS, Docker, Pandas")
    assert result.missing == ("AWS", "Docker", "NumPy", "Pandas")


# --------------------------------------------------------------- sources


def test_sources_record_where_skills_came_from(ext):
    resume = ext.extract_from_sections(
        {"skills": "Python, Git", "experience": "Used Python and SQL"}
    )
    jd = ext.extract_from_sections(
        {"requirements": "Python and Docker", "nice_to_have": "Experience with SQL"}
    )
    result = compare_skills(resume, jd)
    assert result.jd_sources == {
        "Docker": ("requirements",),
        "Python": ("requirements",),
        "SQL": ("nice_to_have",),
    }
    assert result.resume_sources == {
        "Git": ("skills",),
        "Python": ("skills", "experience"),
        "SQL": ("experience",),
    }


def test_missing_from_filters_by_jd_section(ext):
    jd = ext.extract_from_sections(
        {"requirements": "Python and Docker", "nice_to_have": "Experience with AWS"}
    )
    result = compare_skills(ext.extract_from_text("Python", "skills"), jd)
    assert result.missing_from("requirements") == ("Docker",)
    assert result.missing_from("nice_to_have") == ("AWS",)
    assert result.missing_from("header") == ()


def test_summary_line(ext):
    result = compare_text(ext, "Python, SQL, Git", "Python, SQL, Docker")
    assert result.summary() == (
        "matched 2, missing 1, extra 1, coverage 67% of 3 JD skills"
    )


# ------------------------------------------------------ document input


def test_compare_documents_uses_parsed_sections(ext):
    resume = SimpleNamespace(sections={"skills": "Python, Git"}, text="")
    jd = SimpleNamespace(sections={"requirements": "Python and Docker"}, text="")
    result = compare_documents(resume, jd, ext)
    assert isinstance(result, SkillComparison)
    assert result.matched == ("Python",)
    assert result.missing == ("Docker",)


def test_compare_documents_works_with_the_default_extractor():
    resume = SimpleNamespace(sections={}, text="Python and Git")
    jd = SimpleNamespace(sections={}, text="Python and Docker")
    assert compare_documents(resume, jd).missing == ("Docker",)


# ------------------------------------------------ literal sample sections


def test_literal_sample_pair(ext):
    resume = ext.extract_from_sections(
        {
            "header": "PRIYA SHARMA",
            "skills": "Python, SQL, Pandas, NumPy, Scikit-learn, Git, Machine Learning",
            "projects": "Built a classifier with Scikit-learn and Pandas. "
            "Measured precision, recall, F1. Queried data with SQL.",
        }
    )
    jd = ext.extract_from_sections(
        {
            "header": "Junior Machine Learning Engineer",
            "requirements": "- Strong Python programming\n"
            "- Experience with Pandas and NumPy\n"
            "- Hands-on with Scikit-learn and model evaluation\n"
            "- Knowledge of Deep Learning using TensorFlow or PyTorch\n"
            "- Familiarity with Docker and REST APIs",
            "nice_to_have": "- Experience with NLP and Hugging Face Transformers\n"
            "- Basic knowledge of AWS",
        }
    )
    result = compare_skills(resume, jd)
    assert result.matched == (
        "Machine Learning",
        "NumPy",
        "Pandas",
        "Python",
        "Scikit-learn",
    )
    assert result.extra == ("Git", "SQL")
    assert set(result.missing) == {
        "AWS",
        "Deep Learning",
        "Docker",
        "Hugging Face Transformers",
        "Model Evaluation",
        "Natural Language Processing",
        "PyTorch",
        "REST API",
        "TensorFlow",
    }
    assert result.jd_total == 14
    assert result.jd_sources["Machine Learning"] == ("header",)


# ----------------------------------------- real sample files (Phase 2)


def test_sample_files_end_to_end(ext):
    parser = pytest.importorskip("resume_matcher.parsing.parser")
    resume = parser.parse_document(str(SAMPLES / "sample_resume.txt"), "resume")
    jd = parser.parse_document(str(SAMPLES / "sample_jd.txt"), "jd")
    result = compare_documents(resume, jd, ext)

    assert {"Python", "Pandas", "NumPy", "Scikit-learn"} <= set(result.matched)
    for gap in ["TensorFlow", "PyTorch", "Docker", "REST API", "AWS"]:
        assert gap in result.missing, gap
    assert {"SQL", "Git"} <= set(result.extra)
    assert 0.0 < result.coverage < 1.0
