"""Tests for app/summary.py (the one-paragraph summary of an analysis result).

No model, no PDF and no Streamlit is needed: build_summary() reads only the
fields of an AnalysisResult. Expected sentences are written out by hand from
the rules in summary.py, so a wording change in the module shows up here.
"""

import copy
from pathlib import Path

import pytest

from resume_matcher.schemas import AnalysisResult, SkillGap
from summary import (
    _coverage_sentence,
    _gap_sentences,
    _join_names,
    _opening_sentence,
    _skill_word,
    build_summary,
    group_missing,
    NO_REQUIREMENTS_MARK,
    semantic_not_used,
)

ANALYZER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src" / "resume_matcher" / "analyzer" / "analyzer.py"
)


# ----------------------------------------------------------------------
# Small builders
# ----------------------------------------------------------------------
def gap(skill, importance="must-have"):
    return SkillGap(skill, importance)


def make_result(*, score=50.0, verdict="Partial Fit", matched=(), missing=(),
                warnings=()):
    """An AnalysisResult with only the fields the summary reads filled in."""
    return AnalysisResult(
        overall_score=score,
        verdict=verdict,
        semantic_score=0.4,
        skill_coverage=0.5,
        matched_skills=list(matched),
        missing_skills=list(missing),
        extra_skills=[],
        warnings=list(warnings),
    )


def groups(must=(), nice=(), unknown=()):
    """The dict shape that group_missing() returns."""
    return {
        "must-have": list(must),
        "nice-to-have": list(nice),
        "unknown": list(unknown),
    }


# ----------------------------------------------------------------------
# _join_names and _skill_word
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "names, expected",
    [
        ([], ""),
        (["A"], "A"),
        (["A", "B"], "A and B"),
        (["A", "B", "C"], "A, B and C"),
        (["A", "B", "C", "D"], "A, B, C and D"),
        # Five names: four are shown, one is hidden.
        (["A", "B", "C", "D", "E"], "A, B, C, D and 1 more"),
        (["A", "B", "C", "D", "E", "F"], "A, B, C, D and 2 more"),
    ],
)
def test_join_names(names, expected):
    assert _join_names(names) == expected


def test_join_names_respects_a_custom_limit():
    # Limit 2 shows A and B and hides one name.
    assert _join_names(["A", "B", "C"], limit=2) == "A, B and 1 more"


@pytest.mark.parametrize("count, word", [(0, "skills"), (1, "skill"), (2, "skills")])
def test_skill_word(count, word):
    assert _skill_word(count) == word


# ----------------------------------------------------------------------
# group_missing
# ----------------------------------------------------------------------
def test_group_missing_splits_by_importance_and_keeps_order():
    missing = [
        gap("Docker", "must-have"),
        gap("AWS", "nice-to-have"),
        gap("PyTorch", "must-have"),
        gap("Git", "unknown"),
    ]
    assert group_missing(missing) == {
        "must-have": ["Docker", "PyTorch"],
        "nice-to-have": ["AWS"],
        "unknown": ["Git"],
    }


def test_group_missing_lists_a_repeated_skill_once():
    missing = [gap("Docker"), gap("Docker")]
    assert group_missing(missing)["must-have"] == ["Docker"]


def test_group_missing_treats_an_unexpected_label_as_unknown():
    missing = [gap("Docker", "critical")]
    assert group_missing(missing) == groups(unknown=["Docker"])


def test_group_missing_of_nothing_has_three_empty_groups():
    assert group_missing([]) == groups()


# ----------------------------------------------------------------------
# Sentence 1: verdict and score
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "verdict, score, expected",
    [
        ("Fit", 81.4, "This resume is a good fit for the role, with a score of 81 out of 100."),
        ("Partial Fit", 52.8, "This resume is a partial fit for the role, with a score of 53 out of 100."),
        ("No-Fit", 12.0, "This resume is not a fit for the role, with a score of 12 out of 100."),
    ],
)
def test_opening_sentence_for_each_verdict(verdict, score, expected):
    assert _opening_sentence(make_result(score=score, verdict=verdict)) == expected


def test_opening_sentence_clamps_the_score_to_0_100():
    high = _opening_sentence(make_result(score=130.0, verdict="Fit"))
    low = _opening_sentence(make_result(score=-5.0, verdict="Fit"))
    assert "a score of 100 out of 100" in high
    assert "a score of 0 out of 100" in low


def test_opening_sentence_survives_an_unknown_verdict():
    text = _opening_sentence(make_result(score=40.2, verdict="Maybe"))
    assert text == "This resume scored 40 out of 100 (verdict: Maybe)."


# ----------------------------------------------------------------------
# Sentence 2: how many JD skills the resume shows
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "matched, required, expected",
    [
        (0, 0, "No skills from the job description were recognised, so the score reflects the wording match only."),
        (0, 1, "It does not show the one skill the job description asks for."),
        (0, 3, "It shows none of the 3 skills the job description asks for."),
        (1, 1, "It shows the one skill the job description asks for."),
        (2, 2, "It shows both skills the job description asks for."),
        (5, 5, "It shows all 5 skills the job description asks for."),
        (5, 14, "It shows 5 of the 14 skills the job description asks for."),
    ],
)
def test_coverage_sentence(matched, required, expected):
    assert _coverage_sentence(matched, required) == expected


# ----------------------------------------------------------------------
# Sentence 3: the gaps
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "grp, expected",
    [
        (groups(), "No required skills are missing."),
        (groups(must=["Docker", "PyTorch"]),
         "The biggest gaps are the must-have skills Docker and PyTorch."),
        (groups(must=["Docker"]),
         "The biggest gap is the must-have skill Docker."),
        (groups(must=["Docker", "PyTorch"], nice=["AWS"]),
         "The biggest gaps are the must-have skills Docker and PyTorch, "
         "while AWS is only nice-to-have."),
        (groups(must=["Docker"], nice=["AWS", "Git"]),
         "The biggest gap is the must-have skill Docker, "
         "while AWS and Git are only nice-to-have."),
        (groups(must=["Docker"], unknown=["Git"]),
         "The biggest gap is the must-have skill Docker. "
         "1 other missing skill could not be rated for importance."),
        (groups(must=["Docker"], unknown=["Git", "SQL"]),
         "The biggest gap is the must-have skill Docker. "
         "2 other missing skills could not be rated for importance."),
        (groups(nice=["AWS", "Git"]),
         "The missing skills are all nice-to-have: AWS and Git."),
        (groups(nice=["AWS"]),
         "The only missing skill, AWS, is nice-to-have."),
        (groups(nice=["AWS"], unknown=["Git"]),
         "The missing nice-to-have skill is AWS. "
         "1 other missing skill could not be rated for importance."),
        (groups(nice=["AWS", "Git"], unknown=["SQL"]),
         "The missing nice-to-have skills are AWS and Git. "
         "1 other missing skill could not be rated for importance."),
        (groups(unknown=["Docker"]),
         "The missing skill is Docker; the job description "
         "does not say how important it is."),
        (groups(unknown=["Docker", "Git"]),
         "The missing skills are Docker and Git; the job description "
         "does not say how important they are."),
        (groups(must=["A", "B", "C", "D", "E", "F"]),
         "The biggest gaps are the must-have skills A, B, C, D and 2 more."),
    ],
)
def test_gap_sentences(grp, expected):
    assert _gap_sentences(grp) == expected


# ----------------------------------------------------------------------
# build_summary: the whole paragraph
# ----------------------------------------------------------------------
SAMPLE_PARAGRAPH = (
    "This resume is a partial fit for the role, with a score of 53 out of 100. "
    "It shows 5 of the 14 skills the job description asks for. "
    "The biggest gaps are the must-have skills Deep Learning, Docker, "
    "Model Evaluation, PyTorch and 2 more, while AWS, Hugging Face Transformers "
    "and Natural Language Processing are only nice-to-have. "
    "If you really have any of the must-have skills, make sure they appear "
    "clearly on your resume; they count for more than the nice-to-have ones."
)


def sample_like_result():
    """Shaped like the Phase 5 demo: 5 matched, 6 must-have and 3 nice-to-have missing.

    The five matched names are stand-ins; only their count matters here.
    """
    must = ["Deep Learning", "Docker", "Model Evaluation", "PyTorch",
            "REST API", "TensorFlow"]
    nice = ["AWS", "Hugging Face Transformers", "Natural Language Processing"]
    return make_result(
        score=52.8,
        verdict="Partial Fit",
        matched=["Python", "Machine Learning", "scikit-learn", "Pandas", "NumPy"],
        missing=[gap(s, "must-have") for s in must]
        + [gap(s, "nice-to-have") for s in nice],
    )


def test_sample_like_result_gives_the_expected_paragraph():
    assert build_summary(sample_like_result()) == SAMPLE_PARAGRAPH


def test_summary_is_a_single_paragraph():
    assert "\n" not in build_summary(sample_like_result())


def test_fit_with_nothing_missing():
    result = make_result(score=81.4, verdict="Fit", matched=["Python", "SQL"])
    assert build_summary(result) == (
        "This resume is a good fit for the role, with a score of 81 out of 100. "
        "It shows both skills the job description asks for. "
        "No required skills are missing."
    )


def test_no_recognised_jd_skills_skips_the_gap_sentence():
    # Nothing matched and nothing missing: the coverage sentence says so, and
    # "No required skills are missing" would contradict it, so it is left out.
    result = make_result(score=40.2, verdict="No-Fit",
                         warnings=["Job description: no skills recognised"])
    assert build_summary(result) == (
        "This resume is not a fit for the role, with a score of 40 out of 100. "
        "No skills from the job description were recognised, so the score "
        "reflects the wording match only. "
        "Please read the notes under the results, as they may affect how "
        "reliable this is."
    )


def test_advice_sentence_appears_only_when_a_must_have_skill_is_missing():
    with_must = build_summary(make_result(matched=["Python"], missing=[gap("Docker")]))
    nice_only = build_summary(
        make_result(matched=["Python"], missing=[gap("AWS", "nice-to-have")])
    )
    assert "must-have skills, make sure they appear" in with_must
    assert "make sure they appear" not in nice_only


def test_warning_sentence_appears_only_when_there_are_warnings():
    without = build_summary(make_result(matched=["Python"]))
    with_warning = build_summary(make_result(matched=["Python"], warnings=["x"]))
    assert "Please read the notes" not in without
    assert with_warning.endswith("as they may affect how reliable this is.")


def test_repeated_matched_skill_is_counted_once():
    # Matched: Python, SQL (Python repeated). Missing: Docker. Required = 3.
    result = make_result(matched=["Python", "Python", "SQL"],
                         missing=[gap("Docker")])
    assert "It shows 2 of the 3 skills" in build_summary(result)


def test_unexpected_importance_label_is_reported_as_unrated():
    result = make_result(matched=["Python"], missing=[gap("Docker", "critical")])
    assert build_summary(result).endswith(
        "The missing skill is Docker; the job description "
        "does not say how important it is."
    )


def test_sentences_come_in_the_planned_order():
    text = build_summary(sample_like_result())
    opening = text.index("This resume is")
    coverage = text.index("It shows")
    gaps = text.index("The biggest gaps")
    advice = text.index("If you really have")
    assert opening < coverage < gaps < advice


def test_build_summary_does_not_change_its_input():
    result = sample_like_result()
    before = copy.deepcopy(result)
    build_summary(result)
    assert result == before


# ----------------------------------------------------------------------
# semantic_not_used: the job description had no requirement lines
# ----------------------------------------------------------------------
NO_REQUIREMENTS_WARNING = (
    "No requirement lines were found in the job description; the score uses "
    "skill coverage only."
)
NO_REQUIREMENTS_SENTENCE = (
    "The job description has no clear list of requirements, so this score "
    "rests on the skill match alone."
)


def test_marker_text_is_still_what_the_analyzer_writes():
    # If analyzer.py changes its warning, the dashboard would silently stop
    # noticing it. This test reads the source file to catch that.
    assert NO_REQUIREMENTS_MARK in ANALYZER_SOURCE.read_text(encoding="utf-8")


def test_semantic_not_used_when_the_analyzer_warning_is_present():
    result = make_result(warnings=[NO_REQUIREMENTS_WARNING])
    assert semantic_not_used(result) is True


def test_semantic_not_used_when_the_warning_has_a_prefix_or_other_warnings():
    result = make_result(warnings=["Resume is short.", "Note: " + NO_REQUIREMENTS_WARNING])
    assert semantic_not_used(result) is True


def test_semantic_is_used_without_warnings():
    assert semantic_not_used(make_result()) is False


def test_semantic_is_used_with_unrelated_warnings():
    result = make_result(warnings=["Resume: the text is very short."])
    assert semantic_not_used(result) is False


def test_summary_explains_a_score_based_on_skills_alone():
    result = make_result(
        score=67.0, verdict="Fit", matched=["Java", "C++"],
        missing=[gap("Python", "unknown")], warnings=[NO_REQUIREMENTS_WARNING],
    )
    assert NO_REQUIREMENTS_SENTENCE in build_summary(result)


def test_summary_has_no_such_sentence_for_a_normal_result():
    assert NO_REQUIREMENTS_SENTENCE not in build_summary(sample_like_result())


def test_requirements_sentence_comes_right_after_the_coverage_sentence():
    result = make_result(
        score=67.0, verdict="Fit", matched=["Java", "C++"],
        missing=[gap("Python", "unknown")], warnings=[NO_REQUIREMENTS_WARNING],
    )
    text = build_summary(result)
    coverage = text.index("It shows 2 of the 3 skills")
    note = text.index(NO_REQUIREMENTS_SENTENCE)
    gaps = text.index("The missing skill is Python")
    pointer = text.index("Please read the notes")
    assert coverage < note < gaps < pointer
