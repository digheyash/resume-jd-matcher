"""Tests for app/report.py (the plain-text report behind "Download report").

The summary paragraph is passed in as a fixed string, so these tests do not
depend on the summary rules. Expected lines are written out by hand.
"""

import copy
import dataclasses
from datetime import datetime

import pytest

from resume_matcher.schemas import AnalysisResult, SkillGap
from report import GUIDE_NOT_VERDICT, HOW_TO_READ, WIDTH, build_report

FIXED_TIME = datetime(2026, 10, 18, 22, 40)

# A long paragraph, so the report has to wrap it. "nice-to-have" falls right
# at the line end, which is where a plain wrap would split it at a hyphen.
SUMMARY_TEXT = (
    "This resume is a partial fit for the role, with a score of 53 out of 100. "
    "It shows 5 of the 14 skills the job description asks for. "
    "The biggest gaps are the must-have skills Deep Learning, Docker, "
    "Model Evaluation, PyTorch and 2 more, while AWS, Hugging Face Transformers "
    "and Natural Language Processing are only nice-to-have. "
    "If you really have any of the must-have skills, make sure they appear "
    "clearly on your resume; they count for more than the nice-to-have ones."
)


def gap(skill, importance="must-have"):
    return SkillGap(skill, importance)


def make_result(*, score=52.8, verdict="Partial Fit", semantic=0.424,
                coverage=5 / 14, matched=(), missing=(), extra=()):
    return AnalysisResult(
        overall_score=score,
        verdict=verdict,
        semantic_score=semantic,
        skill_coverage=coverage,
        matched_skills=list(matched),
        missing_skills=list(missing),
        extra_skills=list(extra),
        warnings=[],
    )


def sample_result():
    """5 matched, 6 must-have and 3 nice-to-have missing, 3 extra."""
    must = ["Deep Learning", "Docker", "Model Evaluation", "PyTorch",
            "REST API", "TensorFlow"]
    nice = ["AWS", "Hugging Face Transformers", "Natural Language Processing"]
    return make_result(
        matched=["Python", "Machine Learning", "scikit-learn", "Pandas", "NumPy"],
        missing=[gap(s) for s in must] + [gap(s, "nice-to-have") for s in nice],
        extra=["Data Analysis", "Git", "SQL"],
    )


def sample_report(warnings=(), generated_at=FIXED_TIME):
    return build_report(sample_result(), SUMMARY_TEXT, list(warnings), generated_at)


def lines_of(text):
    return text.splitlines()


def block(text, start, end):
    """The lines between two headings (the start heading is included)."""
    lines = lines_of(text)
    first = next(i for i, line in enumerate(lines) if line.startswith(start))
    last = next(i for i, line in enumerate(lines) if line.startswith(end))
    return lines[first:last]


# ----------------------------------------------------------------------
# Header and overall block
# ----------------------------------------------------------------------
def test_report_starts_with_its_title():
    assert lines_of(sample_report())[0] == "RESUME MATCH REPORT"


def test_generated_line_appears_when_a_time_is_given():
    assert lines_of(sample_report())[1] == "Generated: 2026-10-18 22:40"


def test_generated_line_is_left_out_without_a_time():
    text = sample_report(generated_at=None)
    assert "Generated:" not in text
    assert lines_of(text)[1] == ""


def test_overall_block_lines():
    overall = block(sample_report(), "OVERALL", "SUMMARY")
    assert overall == [
        "OVERALL",
        "  Score:           53 / 100",
        "  Verdict:         Partial Fit",
        "  Meaning match:   0.42",
        "  Skill coverage:  36% (5 of 14 skills)",
        "",
    ]


def test_an_unknown_verdict_is_printed_as_it_is():
    result = make_result(verdict="Maybe", matched=["Python"])
    text = build_report(result, "s", [])
    assert "  Verdict:         Maybe" in lines_of(text)


# ----------------------------------------------------------------------
# Summary block
# ----------------------------------------------------------------------
def test_summary_is_wrapped_indented_and_complete():
    summary_lines = block(sample_report(), "SUMMARY", "SKILLS TO ADD")[1:-1]
    assert len(summary_lines) > 1
    # Indented by exactly two spaces (not zero, not four).
    assert all(line.startswith("  ") and not line.startswith("   ")
               for line in summary_lines)
    assert " ".join(line.strip() for line in summary_lines) == SUMMARY_TEXT


def test_no_line_is_longer_than_the_report_width():
    text = sample_report(warnings=["w " * 60])
    assert max(len(line) for line in lines_of(text)) <= WIDTH


def test_hyphenated_words_are_not_split_across_lines():
    assert "nice-\n" not in sample_report()


# ----------------------------------------------------------------------
# Skills to add
# ----------------------------------------------------------------------
def test_missing_skills_are_listed_by_importance_with_counts():
    section = block(sample_report(), "SKILLS TO ADD", "SKILLS YOU ALREADY HAVE")
    assert section == [
        "SKILLS TO ADD (most important first)",
        "  Must-have (6)",
        "    - Deep Learning",
        "    - Docker",
        "    - Model Evaluation",
        "    - PyTorch",
        "    - REST API",
        "    - TensorFlow",
        "  Nice-to-have (3)",
        "    - AWS",
        "    - Hugging Face Transformers",
        "    - Natural Language Processing",
        "",
    ]


def test_groups_come_in_order_and_empty_groups_are_left_out():
    result = make_result(
        matched=["Python"],
        missing=[gap("Git", "unknown"), gap("AWS", "nice-to-have"), gap("Docker")],
    )
    text = build_report(result, "s", [])
    must = text.index("  Must-have (1)")
    nice = text.index("  Nice-to-have (1)")
    unknown = text.index("  Importance not stated (1)")
    assert must < nice < unknown

    only_nice = build_report(
        make_result(missing=[gap("AWS", "nice-to-have")]), "s", []
    )
    # Look for the heading line: the closing note also says "Must-have skills".
    assert "  Must-have (" not in only_nice
    assert "  Importance not stated (" not in only_nice
    assert "  Nice-to-have (1)" in only_nice


def test_nothing_missing_says_so():
    text = build_report(make_result(matched=["Python"], missing=[]), "s", [])
    section = block(text, "SKILLS TO ADD", "SKILLS YOU ALREADY HAVE")
    assert "  No required skills are missing." in section


# ----------------------------------------------------------------------
# Matched and extra skills
# ----------------------------------------------------------------------
def test_matched_skills_block():
    section = block(sample_report(), "SKILLS YOU ALREADY HAVE", "EXTRA SKILLS")
    assert section == [
        "SKILLS YOU ALREADY HAVE (5)",
        "  Asked for by the job description and found on the resume.",
        "    - Python",
        "    - Machine Learning",
        "    - scikit-learn",
        "    - Pandas",
        "    - NumPy",
        "",
    ]


def test_extra_skills_block():
    section = block(sample_report(), "EXTRA SKILLS", "ABOUT THIS REPORT")
    assert section == [
        "EXTRA SKILLS (3)",
        "  On the resume, but not asked for in this job description.",
        "    - Data Analysis",
        "    - Git",
        "    - SQL",
        "",
    ]


def test_empty_skill_lists_say_none():
    text = build_report(make_result(missing=[gap("Docker")]), "s", [])
    assert "SKILLS YOU ALREADY HAVE (0)" in text
    assert "EXTRA SKILLS (0)" in text
    assert lines_of(text).count("    None.") == 2


def test_repeated_names_are_listed_once():
    result = make_result(matched=["Python", "Python"], extra=["Git", "Git"])
    text = build_report(result, "s", [])
    assert "SKILLS YOU ALREADY HAVE (1)" in text
    assert "EXTRA SKILLS (1)" in text
    assert lines_of(text).count("    - Python") == 1
    assert lines_of(text).count("    - Git") == 1


def test_non_ascii_skill_names_survive_utf8_encoding():
    text = build_report(make_result(matched=["Café"]), "s", [])
    assert "    - Café" in lines_of(text)
    assert text.encode("utf-8").decode("utf-8") == text


# ----------------------------------------------------------------------
# Warnings
# ----------------------------------------------------------------------
def test_no_warnings_means_no_warnings_section():
    assert "THINGS TO CHECK" not in sample_report()


def test_warnings_are_bulleted():
    text = sample_report(warnings=["Resume: first", "Job description: second"])
    section = block(text, "THINGS TO CHECK", "ABOUT THIS REPORT")
    assert section == [
        "THINGS TO CHECK",
        "  - Resume: first",
        "  - Job description: second",
        "",
    ]


def test_a_long_warning_wraps_with_a_hanging_indent_and_keeps_hyphenated_words():
    # 68 letters, then a hyphenated word: a plain wrap would split "nice-to-have".
    warning = "a" * 68 + " nice-to-have words follow"
    section = block(sample_report(warnings=[warning]), "THINGS TO CHECK", "ABOUT THIS REPORT")
    assert section[1] == "  - " + "a" * 68
    assert section[2] == "    nice-to-have words follow"
    assert max(len(line) for line in section) <= WIDTH


# ----------------------------------------------------------------------
# Closing notes and overall shape
# ----------------------------------------------------------------------
def test_both_notes_are_printed_in_full():
    flat = " ".join(sample_report().split())
    assert HOW_TO_READ in flat
    assert GUIDE_NOT_VERDICT in flat


def test_the_honesty_note_keeps_its_key_statements():
    # The report must not be mistaken for more than it is.
    assert "24 rated" in GUIDE_NOT_VERDICT
    assert "synthetic" in GUIDE_NOT_VERDICT
    assert "rough guide" in GUIDE_NOT_VERDICT
    assert "dictionary" in GUIDE_NOT_VERDICT


def test_sections_come_in_the_planned_order():
    text = sample_report(warnings=["w"])
    headings = [
        "OVERALL", "SUMMARY", "SKILLS TO ADD", "SKILLS YOU ALREADY HAVE",
        "EXTRA SKILLS", "THINGS TO CHECK", "ABOUT THIS REPORT",
    ]
    positions = [lines_of(text).index(next(l for l in lines_of(text) if l.startswith(h)))
                 for h in headings]
    assert positions == sorted(positions)


def test_report_ends_with_exactly_one_newline():
    text = sample_report()
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_build_report_does_not_change_its_inputs():
    result, warnings = sample_result(), ["Resume: a"]
    result_before, warnings_before = copy.deepcopy(result), list(warnings)
    build_report(result, SUMMARY_TEXT, warnings, FIXED_TIME)
    assert result == result_before
    assert warnings == warnings_before


@pytest.mark.parametrize("score, shown", [(130.0, "100 / 100"), (-4.0, "0 / 100")])
def test_score_is_held_between_0_and_100(score, shown):
    text = build_report(make_result(score=score), "s", [])
    assert f"  Score:           {shown}" in lines_of(text)


def test_overall_block_says_not_used_when_the_job_has_no_requirement_lines():
    result = dataclasses.replace(
        make_result(matched=["Java", "C++"], missing=[gap("Python", "unknown")],
                    score=66.7, verdict="Fit", semantic=0.0, coverage=2 / 3),
        warnings=["No requirement lines were found in the job description; "
                  "the score uses skill coverage only."],
    )
    text = build_report(result, "Summary.", result.warnings, FIXED_TIME)
    overall = block(text, "OVERALL", "SUMMARY")
    assert "  Meaning match:   Not used (no requirement lines found)" in overall
    assert "  Skill coverage:  67% (2 of 3 skills)" in overall
    assert "0.00" not in text


def test_overall_block_says_not_used_when_no_skills_were_recognised():
    result = make_result(semantic=0.5, coverage=0.0)
    text = build_report(result, "Summary.", [], FIXED_TIME)
    overall = block(text, "OVERALL", "SUMMARY")
    assert "  Meaning match:   0.50" in overall
    assert "  Skill coverage:  Not used (no skills recognised)" in overall
