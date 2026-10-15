"""Tests for resume_matcher.skills.extractor (Phase 3)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from resume_matcher.skills.extractor import (
    FULL_TEXT_KEY,
    SKIP_SECTIONS,
    SkillExtractor,
    get_default_extractor,
    strip_dash_bullets,
)
from resume_matcher.skills.ontology import parse_ontology

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "samples"


@pytest.fixture(scope="module")
def ext():
    return SkillExtractor()


def found(extractor, text, section=FULL_TEXT_KEY):
    """Set of canonical names found in one block of text."""
    return extractor.extract_from_text(text, section).skills()


# ------------------------------------------------------- boundary rules


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Java and JavaScript developer", {"Java", "JavaScript"}),
        ("JavaScript only", {"JavaScript"}),
        ("C++ developer", {"C++"}),
        ("C# and .NET", {"C#", ".NET"}),
        ("Built it with ASP.NET", set()),
        ("Node.js and Scikit-learn", {"Node.js", "Scikit-learn"}),
        ("Git and GitHub", {"Git", "GitHub"}),
        ("GitHub only", {"GitHub"}),
        ("NoSQL and MySQL", {"MySQL"}),
        ("Sentence ends with Python.", {"Python"}),
        ("(Docker)", {"Docker"}),
    ],
)
def test_boundaries(ext, text, expected):
    assert found(ext, text) == expected


def test_matching_is_case_insensitive_for_normal_skills(ext):
    assert found(ext, "PYTHON, pandas and NumPy") == {"Python", "Pandas", "NumPy"}


def test_whitespace_inside_a_name_is_flexible(ext):
    assert found(ext, "Machine\nLearning and scikit   learn") == {
        "Machine Learning",
        "Scikit-learn",
    }


def test_leading_dash_bullets_are_ignored(ext):
    assert found(ext, "- Python\n- Docker") == {"Python", "Docker"}


def test_strip_dash_bullets_only_touches_line_starts():
    assert strip_dash_bullets("- a\n  - b\nx - y\nScikit-learn") == (
        "a\nb\nx - y\nScikit-learn"
    )


# ----------------------------------------------------------- aliases


def test_aliases_map_to_canonical_names(ext):
    result = ext.extract_from_text("sklearn, k8s, NLP, golang", "skills")
    assert result.skills() == {
        "Scikit-learn",
        "Kubernetes",
        "Natural Language Processing",
        "Go",
    }


def test_longest_spelling_wins(ext):
    result = ext.extract_from_text("Hugging Face Transformers and Apache Spark")
    assert result.skills() == {"Hugging Face Transformers", "Apache Spark"}
    assert result.hits["Hugging Face Transformers"].count == 1


def test_r_programming_alias_needs_no_extra_context(ext):
    result = ext.extract_from_text("I know R programming well", "experience")
    assert result.skills() == {"R"}
    assert result.hits["R"].surfaces == ("R programming",)


def test_surfaces_and_count_are_recorded(ext):
    hit = ext.extract_from_text("Scikit-learn, sklearn, scikit-learn").hits[
        "Scikit-learn"
    ]
    assert hit.count == 3
    assert hit.surfaces == ("Scikit-learn", "sklearn", "scikit-learn")


# -------------------------------------------------- ambiguous spellings


@pytest.mark.parametrize(
    "text, section",
    [
        ("R&D team", "skills"),
        ("Go to market strategy", "requirements"),
        ("Ability to excel in a fast-paced team", "requirements"),
        ("Excel in a fast-paced team", "requirements"),
        ("Spring 2024 semester", "experience"),
        ("I like to react quickly", "experience"),
        ("experience with go and react", "experience"),
        ("Python, R, Go", "experience"),
        ("Strong C level communication", "requirements"),
    ],
)
def test_ambiguous_false_friends_are_rejected(ext, text, section):
    result = found(ext, text, section)
    assert not (result & {"R", "Go", "Excel", "Spring Boot", "React", "C"}), result


def test_ambiguous_list_items_count_in_a_skills_section(ext):
    assert found(ext, "Python, R, Go, Excel", "skills") == {
        "Python",
        "R",
        "Go",
        "Excel",
    }


def test_ambiguous_list_items_count_in_requirements(ext):
    assert found(ext, "Python, R", "requirements") == {"Python", "R"}


def test_ambiguous_list_outside_skill_sections_is_rejected(ext):
    assert found(ext, "Python, R, Go", "experience") == {"Python"}


def test_hint_word_accepts_ambiguous_names_anywhere(ext):
    assert found(ext, "Experience with Go", "experience") == {"Go"}
    assert found(ext, "Built dashboards using React", "projects") == {"React"}
    assert found(ext, "Knowledge of Spark", "requirements") == {"Apache Spark"}


def test_ambiguous_names_need_exact_capitalisation(ext):
    assert found(ext, "experience with go", "experience") == set()


def test_hint_words_do_not_cross_lines(ext):
    assert found(ext, "Experience with\nGo", "experience") == set()


def test_prose_mention_without_signal_is_missed_on_purpose(ext):
    # Known trade-off: strict rules prefer a missed C over a false R or Go.
    result = found(ext, "Also C, Java and JavaScript", "skills")
    assert result == {"Java", "JavaScript"}


# ------------------------------------------------- sections and results


def test_extract_from_text_defaults_to_full_text_section(ext):
    result = ext.extract_from_text("Python")
    assert result.sections_of("Python") == (FULL_TEXT_KEY,)


def test_sections_are_tracked_per_skill(ext):
    result = ext.extract_from_sections(
        {
            "header": "Jane Doe",
            "skills": "Python, R",
            "experience": "Used Python and Pandas",
        }
    )
    assert result.hits["Python"].sections == ("skills", "experience")
    assert result.hits["Python"].count == 2
    assert result.hits["R"].sections == ("skills",)
    assert result.hits["Pandas"].sections == ("experience",)
    assert result.hits["Pandas"].in_section("experience")
    assert not result.hits["Pandas"].in_section("skills")


def test_skipped_sections_are_ignored(ext):
    assert "other" in SKIP_SECTIONS and "benefits" in SKIP_SECTIONS
    result = ext.extract_from_sections({"other": "Python", "benefits": "Docker"})
    assert len(result) == 0


def test_result_helpers(ext):
    result = ext.extract_from_text("pandas, Python, Docker")
    assert result.names() == ["Docker", "Pandas", "Python"]
    assert "Python" in result
    assert "Java" not in result
    assert result.hits["Python"].category == "programming_language"
    assert result.sections_of("Java") == ()


def test_empty_text_gives_empty_result(ext):
    assert len(ext.extract_from_text("")) == 0
    assert len(ext.extract_from_sections({})) == 0


def test_find_matches_reports_exact_spans(ext):
    text = "We use Python and Docker."
    for match in ext.find_matches(text):
        assert text[match.start : match.end] == match.text


# ------------------------------------------------- parsed-document input


def test_extract_from_parsed_uses_sections(ext):
    doc = SimpleNamespace(sections={"skills": "Python, SQL"}, text="ignored Docker")
    assert ext.extract_from_parsed(doc).skills() == {"Python", "SQL"}


def test_extract_from_parsed_falls_back_to_full_text(ext):
    doc = SimpleNamespace(sections={}, text="Python and Docker")
    assert ext.extract_from_parsed(doc).skills() == {"Python", "Docker"}


def test_full_text_key_matches_phase_2_constant():
    sections = pytest.importorskip("resume_matcher.parsing.sections")
    assert FULL_TEXT_KEY == sections.FULL_TEXT_KEY


# ----------------------------------------------- literal sample sections

RESUME_SECTIONS = {
    "header": "PRIYA SHARMA",
    "skills": "Python, SQL, Pandas, NumPy, Scikit-learn, Git, Machine Learning",
    "projects": "Built a classifier with Scikit-learn and Pandas. Measured "
    "precision, recall, F1. Queried data with SQL.",
}

JD_SECTIONS = {
    "header": "Junior Machine Learning Engineer",
    "requirements": "- Strong Python programming\n"
    "- Experience with Pandas and NumPy\n"
    "- Hands-on with Scikit-learn and model evaluation\n"
    "- Knowledge of Deep Learning using TensorFlow or PyTorch\n"
    "- Familiarity with Docker and REST APIs",
    "nice_to_have": "- Experience with NLP and Hugging Face Transformers\n"
    "- Basic knowledge of AWS",
}


def test_literal_resume_sections(ext):
    result = ext.extract_from_sections(RESUME_SECTIONS)
    assert result.skills() == {
        "Python",
        "SQL",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Git",
        "Machine Learning",
    }


def test_literal_jd_sections_record_where_each_skill_came_from(ext):
    result = ext.extract_from_sections(JD_SECTIONS)
    must = {
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Model Evaluation",
        "Deep Learning",
        "TensorFlow",
        "PyTorch",
        "Docker",
        "REST API",
    }
    nice = {"Natural Language Processing", "Hugging Face Transformers", "AWS"}
    for skill in must:
        assert result.hits[skill].sections == ("requirements",), skill
    for skill in nice:
        assert result.hits[skill].sections == ("nice_to_have",), skill


# ------------------------------------------------------- custom ontology


def test_extractor_accepts_a_custom_ontology():
    onto = parse_ontology(
        {"skills": [{"canonical": "Foo", "category": "x", "aliases": ["Bar"]}]}
    )
    custom = SkillExtractor(onto)
    assert custom.extract_from_text("bar and FOO and Python").skills() == {"Foo"}


def test_default_extractor_is_cached():
    assert get_default_extractor() is get_default_extractor()


# ------------------------------------------------ real sample files (Phase 2)


def _parse_sample(name, kind):
    parser = pytest.importorskip("resume_matcher.parsing.parser")
    path = SAMPLES / name
    assert path.exists(), f"missing sample file: {path}"
    return parser.parse_document(str(path), kind)


def test_sample_resume_file(ext):
    result = ext.extract_from_parsed(_parse_sample("sample_resume.txt", "resume"))
    expected = {
        "Python",
        "SQL",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Git",
        "Machine Learning",
    }
    assert expected <= result.skills()
    assert result.hits["Python"].in_section("skills")
    for absent in ["TensorFlow", "PyTorch", "Docker", "AWS"]:
        assert absent not in result


def test_sample_jd_file(ext):
    result = ext.extract_from_parsed(_parse_sample("sample_jd.txt", "jd"))
    for skill in ["Python", "Pandas", "TensorFlow", "PyTorch", "Docker", "REST API"]:
        assert result.hits[skill].in_section("requirements"), skill
    for skill in ["Natural Language Processing", "Hugging Face Transformers", "AWS"]:
        assert result.hits[skill].in_section("nice_to_have"), skill
