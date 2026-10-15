"""Tests for resume_matcher.skills.evaluation (Phase 3)."""

import json

import pytest

from resume_matcher.skills.evaluation import (
    CaseResult,
    EvaluationReport,
    LabelError,
    LabelledCase,
    Scores,
    default_labels_path,
    evaluate,
    format_report,
    load_cases,
    main,
    parse_cases,
)
from resume_matcher.skills.extractor import SkillExtractor
from resume_matcher.skills.ontology import get_default_ontology


@pytest.fixture(scope="module")
def ext():
    return SkillExtractor()


def case(case_id, text, skills, kind="resume", section="skills", note=""):
    return LabelledCase(
        id=case_id,
        kind=kind,
        section=section,
        text=text,
        skills=frozenset(skills),
        note=note,
    )


def raw_case(**overrides):
    base = {
        "id": "c1",
        "kind": "resume",
        "section": "skills",
        "text": "Python",
        "skills": ["Python"],
    }
    base.update(overrides)
    return base


# ----------------------------------------------------------------- Scores


def test_scores_for_the_tiny_example():
    scores = Scores(tp=3, fp=1, fn=1)
    assert scores.precision == pytest.approx(0.75)
    assert scores.recall == pytest.approx(0.75)
    assert scores.f1 == pytest.approx(0.75)


def test_f1_is_between_precision_and_recall():
    scores = Scores(tp=2, fp=0, fn=2)  # precision 1.0, recall 0.5
    assert scores.precision == 1.0
    assert scores.recall == 0.5
    assert scores.f1 == pytest.approx(2 / 3)


def test_zero_denominators_give_zero_not_an_error():
    empty = Scores()
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)
    assert Scores(fp=2).precision == 0.0
    assert Scores(fn=2).recall == 0.0


def test_scores_add_up():
    assert Scores(1, 2, 3) + Scores(4, 5, 6) == Scores(5, 7, 9)


# ------------------------------------------------------------- CaseResult


def test_case_result_sets():
    result = CaseResult(
        case=case("x", "", {"Python", "SQL", "Docker"}),
        predicted=frozenset({"Python", "SQL", "Git"}),
    )
    assert result.true_positives == {"Python", "SQL"}
    assert result.false_positives == {"Git"}
    assert result.false_negatives == {"Docker"}
    assert not result.is_exact
    assert result.scores == Scores(tp=2, fp=1, fn=1)


def test_exact_result():
    result = CaseResult(case=case("x", "", {"Python"}), predicted=frozenset({"Python"}))
    assert result.is_exact


# --------------------------------------------------------------- evaluate


def test_evaluate_counts_and_groups_by_kind(ext):
    cases = [
        case("a", "Python, SQL", {"Python", "SQL"}, kind="resume"),
        case("b", "Python, SQL, Git", {"Python", "SQL", "Docker"}, kind="jd"),
        case("c", "We are a friendly team.", set(), kind="jd"),
    ]
    report = evaluate(cases, ext)
    assert isinstance(report, EvaluationReport)
    assert report.overall == Scores(tp=4, fp=1, fn=1)
    assert report.by_kind["resume"] == Scores(tp=2, fp=0, fn=0)
    assert report.by_kind["jd"] == Scores(tp=2, fp=1, fn=1)
    assert report.exact_matches == 2
    assert [r.case.id for r in report.mistakes] == ["b"]


def test_evaluate_uses_the_case_section(ext):
    in_list = case("l", "Python, R", {"Python", "R"}, section="skills")
    in_prose = case("p", "Python, R", {"Python"}, section="experience")
    report = evaluate([in_list, in_prose], ext)
    assert report.exact_matches == 2


def test_evaluate_works_with_the_default_extractor():
    report = evaluate([case("a", "Python", {"Python"})])
    assert report.overall == Scores(tp=1)


# ----------------------------------------------------------------- report


def test_format_report_lists_mistakes(ext):
    report = evaluate(
        [case("bad", "Python, SQL, Git", {"Python", "Docker"}, note="my note")], ext
    )
    text = format_report(report)
    assert "OVERALL" in text
    assert "bad [resume/skills] my note" in text
    assert "false positive" in text and "Git, SQL" in text
    assert "false negative" in text and "Docker" in text


def test_format_report_without_mistakes(ext):
    text = format_report(evaluate([case("ok", "Python", {"Python"})], ext))
    assert "No mistakes." in text
    assert "MISTAKES" not in text


# ------------------------------------------------------------ label files


def test_parse_cases_builds_cases():
    parsed = parse_cases({"cases": [raw_case(note="hi")]})
    assert parsed == [case("c1", "Python", {"Python"}, note="hi")]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {},
        {"cases": []},
        {"cases": "x"},
        {"cases": ["x"]},
        {"cases": [{"id": "c1"}]},
        {"cases": [raw_case(id="")]},
        {"cases": [raw_case(kind="email")]},
        {"cases": [raw_case(text=5)]},
        {"cases": [raw_case(skills="Python")]},
        {"cases": [raw_case(skills=[1])]},
        {"cases": [raw_case(skills=["Python", "Python"])]},
        {"cases": [raw_case(), raw_case()]},
    ],
)
def test_malformed_labels_are_rejected(bad):
    with pytest.raises(LabelError):
        parse_cases(bad)


def test_load_cases_from_file(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"cases": [raw_case()]}), encoding="utf-8")
    assert len(load_cases(path)) == 1


def test_missing_labels_file(tmp_path):
    with pytest.raises(LabelError, match="not found"):
        load_cases(tmp_path / "nope.json")


def test_invalid_json_labels_file(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(LabelError, match="not valid JSON"):
        load_cases(path)


# ------------------------------------------------- the real labels file


def test_real_labels_file_loads():
    assert default_labels_path().exists()
    cases = load_cases()
    assert len(cases) > 0
    assert {c.kind for c in cases} == {"resume", "jd"}
    assert any(not c.skills for c in cases), "keep some no-skill cases"


def test_every_real_label_is_a_known_ontology_skill():
    known = set(get_default_ontology().canonical_names())
    for c in load_cases():
        assert c.skills <= known, (c.id, sorted(c.skills - known))


def test_real_evaluation_produces_valid_scores():
    report = evaluate(load_cases())
    scores = report.overall
    assert scores.tp + scores.fn > 0
    for value in (scores.precision, scores.recall, scores.f1):
        assert 0.0 <= value <= 1.0


def test_main_prints_a_report(capsys):
    main()
    out = capsys.readouterr().out
    assert "OVERALL" in out
    assert "precision" in out and "recall" in out and "F1" in out
