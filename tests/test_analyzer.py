"""
Tests for resume_matcher.analyzer.analyzer (Phase 5).

The parser, the skill extractor and the semantic matcher are replaced by tiny
fakes, so these tests need no spaCy model, no Sentence-BERT and no data files
(except the one smoke test at the end, which uses the real parser and skips
itself if it is unavailable).

THE BASE CASE, worked out by hand
    resume skills : Python, Docker, Extra
    JD skills     : Python (requirements), SQL (requirements),
                    Docker (nice_to_have)
    matched       : Docker, Python      missing: SQL      extra: Extra
    plain coverage    = 2 / 3                       = 0.6667
    weighted coverage = (1.0 + 0.5) / (1.0 + 0.5 + 1.0) = 0.6
    params            : anchors 0.2..0.6, weight 0.5, Fit >= 0.7, Partial >= 0.4
    raw semantic 0.4  -> (0.4 - 0.2) / (0.6 - 0.2) = 0.5
    final score       = 0.5 * 0.5 + 0.5 * 0.6 = 0.55  -> 55.0 -> Partial Fit
"""
import pytest

from resume_matcher.analyzer.analyzer import (
    REQUIREMENTS_SECTION,
    AnalysisError,
    Analyzer,
)
from resume_matcher.analyzer.scoring import VERDICTS, ScoringParams
from resume_matcher.schemas import AnalysisResult, SkillGap


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class FakeDoc:
    def __init__(self, text, kind, warnings=()):
        self.text = text
        self.kind = kind
        self.warnings = list(warnings)


class FakeParse:
    """parse_fn(text, kind) -> FakeDoc; remembers every call."""

    DOC_WARNINGS = {
        "resume_warn": ["text is very short"],
        "jd_warn": ["no section headings found"],
    }

    def __init__(self):
        self.calls = []

    def __call__(self, text, kind):
        self.calls.append((text, kind))
        return FakeDoc(text, kind, self.DOC_WARNINGS.get(text, []))


class FakeExtraction:
    def __init__(self, table):
        self.table = table

    def skills(self):
        return set(self.table)

    def sections_of(self, name):
        return tuple(self.table[name])


class FakeExtractor:
    """Looks up the skills of a document by its text."""

    def __init__(self, table):
        self.table = table

    def extract_from_parsed(self, doc):
        return FakeExtraction(self.table[doc.text])


class FakeSectionResult:
    def __init__(self, value, asked):
        self.value = value
        self.asked = asked

    def mean_score(self, section):
        self.asked.append(section)
        return self.value


class FakeMatcher:
    """match_documents(...) -> an object whose mean_score returns `value`."""

    def __init__(self, value):
        self.value = value
        self.asked = []
        self.calls = 0

    def match_documents(self, resume_doc, jd_doc):
        self.calls += 1
        return FakeSectionResult(self.value, self.asked)


TABLE = {
    "resume": {
        "Python": ("skills",),
        "Docker": ("skills",),
        "Extra": ("skills",),
    },
    "resume_full": {
        "Python": ("skills",),
        "SQL": ("skills",),
        "Docker": ("skills",),
    },
    "resume_none": {},
    "resume_warn": {"Python": ("skills",)},
    "jd": {
        "Python": ("requirements",),
        "SQL": ("requirements",),
        "Docker": ("nice_to_have",),
    },
    "jd_none": {},
    "jd_warn": {"Python": ("requirements",)},
    "jd_many": {
        "SQL": ("requirements",),
        "Airflow": ("requirements",),
        "Docker": ("nice_to_have",),
        "Git": ("header",),
    },
}

PARAMS = ScoringParams(
    anchor_low=0.2,
    anchor_high=0.6,
    weight_semantic=0.5,
    fit_threshold=0.7,
    partial_threshold=0.4,
)


def make(value, params=PARAMS):
    """An Analyzer wired to fakes, plus the fakes (for inspection)."""
    matcher = FakeMatcher(value)
    parse = FakeParse()
    analyzer = Analyzer(matcher, FakeExtractor(TABLE), params, parse)
    return analyzer, matcher, parse


# --------------------------------------------------------------------------
# The base case
# --------------------------------------------------------------------------
class TestBaseCase:
    def test_returns_an_analysis_result(self):
        analyzer, _, _ = make(0.4)
        assert isinstance(analyzer.analyze("resume", "jd"), AnalysisResult)

    def test_overall_score_and_verdict(self):
        analyzer, _, _ = make(0.4)
        result = analyzer.analyze("resume", "jd")
        assert result.overall_score == pytest.approx(55.0)
        assert result.verdict == "Partial Fit"

    def test_reports_raw_cosine_and_plain_coverage(self):
        analyzer, _, _ = make(0.4)
        result = analyzer.analyze("resume", "jd")
        assert result.semantic_score == pytest.approx(0.4)
        assert result.skill_coverage == pytest.approx(2 / 3)

    def test_skill_lists(self):
        analyzer, _, _ = make(0.4)
        result = analyzer.analyze("resume", "jd")
        assert result.matched_skills == ["Docker", "Python"]
        assert result.missing_skills == [SkillGap(skill="SQL", importance="must-have")]
        assert result.extra_skills == ["Extra"]

    def test_no_warnings_for_a_clean_pair(self):
        analyzer, _, _ = make(0.4)
        assert analyzer.analyze("resume", "jd").warnings == []

    def test_asks_the_matcher_for_the_requirements_section(self):
        analyzer, matcher, _ = make(0.4)
        analyzer.analyze("resume", "jd")
        assert matcher.asked == [REQUIREMENTS_SECTION] == ["requirements"]

    def test_parses_resume_and_jd_with_the_right_kinds(self):
        analyzer, _, parse = make(0.4)
        analyzer.analyze("resume", "jd")
        assert parse.calls == [("resume", "resume"), ("jd", "jd")]

    def test_same_input_gives_same_output(self):
        analyzer, _, _ = make(0.4)
        first = analyzer.analyze("resume", "jd")
        second = analyzer.analyze("resume", "jd")
        assert first == second


# --------------------------------------------------------------------------
# Verdict range
# --------------------------------------------------------------------------
class TestVerdicts:
    def test_strong_pair_is_fit(self):
        # raw 0.6 -> 1.0 ; resume has every JD skill -> weighted coverage 1.0
        # score = 0.5 * 1.0 + 0.5 * 1.0 = 1.0 -> 100.0
        analyzer, _, _ = make(0.6)
        result = analyzer.analyze("resume_full", "jd")
        assert result.overall_score == pytest.approx(100.0)
        assert result.verdict == "Fit"
        assert result.missing_skills == []

    def test_weak_pair_is_no_fit(self):
        # raw 0.1 -> clipped to 0.0 ; resume has no skills -> coverage 0.0
        analyzer, _, _ = make(0.1)
        result = analyzer.analyze("resume_none", "jd")
        assert result.overall_score == pytest.approx(0.0)
        assert result.verdict == "No-Fit"
        assert result.matched_skills == []

    def test_verdict_is_always_a_known_label(self):
        for value in (0.0, 0.2, 0.4, 0.6, 0.9):
            analyzer, _, _ = make(value)
            assert analyzer.analyze("resume", "jd").verdict in VERDICTS

    def test_score_stays_between_zero_and_hundred(self):
        for value in (-0.5, 0.0, 0.4, 1.0, 1.7):
            analyzer, _, _ = make(value)
            score = analyzer.analyze("resume", "jd").overall_score
            assert 0.0 <= score <= 100.0


# --------------------------------------------------------------------------
# Parameters really drive the result
# --------------------------------------------------------------------------
class TestParams:
    def test_weight_one_uses_only_the_semantic_part(self):
        # rescaled semantic 0.5 -> 50.0
        params = ScoringParams(0.2, 0.6, 1.0, 0.7, 0.4)
        analyzer, _, _ = make(0.4, params)
        assert analyzer.analyze("resume", "jd").overall_score == pytest.approx(50.0)

    def test_weight_zero_uses_only_weighted_coverage(self):
        # weighted coverage 0.6 -> 60.0 (plain coverage 0.6667 would give 66.67)
        params = ScoringParams(0.2, 0.6, 0.0, 0.7, 0.4)
        analyzer, _, _ = make(0.4, params)
        assert analyzer.analyze("resume", "jd").overall_score == pytest.approx(60.0)

    def test_thresholds_decide_the_verdict(self):
        params = ScoringParams(0.2, 0.6, 0.5, 0.5, 0.3)  # 55.0 is now a Fit
        analyzer, _, _ = make(0.4, params)
        assert analyzer.analyze("resume", "jd").verdict == "Fit"

    def test_default_params_are_used_when_none_given(self):
        analyzer = Analyzer(FakeMatcher(0.4), FakeExtractor(TABLE), parse_fn=FakeParse())
        assert analyzer.params == ScoringParams()


# --------------------------------------------------------------------------
# Missing-skill ranking inside the result
# --------------------------------------------------------------------------
def test_missing_skills_are_ranked_by_importance():
    # resume has nothing; JD asks for SQL and Airflow (must-have), Docker
    # (nice-to-have) and Git (header -> unknown).
    analyzer, _, _ = make(0.4)
    result = analyzer.analyze("resume_none", "jd_many")
    assert [(g.skill, g.importance) for g in result.missing_skills] == [
        ("Airflow", "must-have"),
        ("SQL", "must-have"),
        ("Docker", "nice-to-have"),
        ("Git", "unknown"),
    ]


# --------------------------------------------------------------------------
# Missing signals
# --------------------------------------------------------------------------
class TestMissingSignals:
    def test_no_requirement_lines_falls_back_to_weighted_coverage(self):
        analyzer, _, _ = make(None)
        result = analyzer.analyze("resume", "jd")
        assert result.overall_score == pytest.approx(60.0)  # weighted coverage 0.6
        assert result.semantic_score == 0.0
        assert result.verdict == "Partial Fit"
        assert any("requirement lines" in w for w in result.warnings)

    def test_no_jd_skills_falls_back_to_the_semantic_score(self):
        # raw 0.6 -> rescaled 1.0, weight forced to 1.0 -> 100.0
        analyzer, _, _ = make(0.6)
        result = analyzer.analyze("resume", "jd_none")
        assert result.overall_score == pytest.approx(100.0)
        assert result.verdict == "Fit"
        assert result.skill_coverage == 0.0
        assert result.matched_skills == []
        assert result.missing_skills == []
        assert result.extra_skills == ["Docker", "Extra", "Python"]
        assert any("recognised skills" in w for w in result.warnings)

    def test_no_jd_skills_does_not_change_the_analyzers_own_params(self):
        analyzer, _, _ = make(0.6)
        analyzer.analyze("resume", "jd_none")
        assert analyzer.params.weight_semantic == 0.5

    def test_nothing_to_score_raises(self):
        analyzer, _, _ = make(None)
        with pytest.raises(AnalysisError):
            analyzer.analyze("resume", "jd_none")


# --------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------
class TestInputChecks:
    @pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
    def test_empty_resume_raises(self, bad):
        analyzer, matcher, parse = make(0.4)
        with pytest.raises(AnalysisError):
            analyzer.analyze(bad, "jd")
        assert parse.calls == [] and matcher.calls == 0

    @pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
    def test_empty_jd_raises(self, bad):
        analyzer, matcher, parse = make(0.4)
        with pytest.raises(AnalysisError):
            analyzer.analyze("resume", bad)
        assert parse.calls == [] and matcher.calls == 0

    def test_none_text_raises(self):
        analyzer, _, _ = make(0.4)
        with pytest.raises(AnalysisError):
            analyzer.analyze(None, "jd")
        with pytest.raises(AnalysisError):
            analyzer.analyze("resume", None)

    def test_analysis_error_is_a_value_error(self):
        assert issubclass(AnalysisError, ValueError)

    def test_error_message_names_the_empty_input(self):
        analyzer, _, _ = make(0.4)
        with pytest.raises(AnalysisError, match="resume"):
            analyzer.analyze("", "jd")
        with pytest.raises(AnalysisError, match="job description"):
            analyzer.analyze("resume", "")


# --------------------------------------------------------------------------
# Warnings and clipping
# --------------------------------------------------------------------------
class TestWarningsAndClipping:
    def test_parser_warnings_are_passed_on_with_a_prefix(self):
        analyzer, _, _ = make(0.4)
        result = analyzer.analyze("resume_warn", "jd_warn")
        assert "Resume: text is very short" in result.warnings
        assert "Job description: no section headings found" in result.warnings

    def test_document_without_a_warnings_attribute_is_fine(self):
        class Bare:
            pass

        def parse(text, kind):
            return Bare()

        class Ext:
            def extract_from_parsed(self, doc):
                return FakeExtraction({"Python": ("requirements",)})

        analyzer = Analyzer(FakeMatcher(0.4), Ext(), PARAMS, parse)
        assert analyzer.analyze("a", "b").warnings == []

    def test_reported_semantic_score_is_clipped_to_zero_one(self):
        high, _, _ = make(1.3)
        low, _, _ = make(-0.2)
        assert high.analyze("resume", "jd").semantic_score == 1.0
        assert low.analyze("resume", "jd").semantic_score == 0.0


# --------------------------------------------------------------------------
# Smoke test with the REAL parser and the REAL default extractor
# (only the semantic matcher is a fake). It checks the wiring, not the numbers.
# --------------------------------------------------------------------------
def test_wiring_with_the_real_parser_and_extractor():
    pytest.importorskip("resume_matcher.parsing.parser")
    pytest.importorskip("resume_matcher.skills.extractor")

    resume = (
        "Asha Rao\n"
        "asha.rao@example.com | +91 90000 00000\n"
        "SKILLS\n"
        "Python, SQL, pandas, scikit-learn\n"
        "EXPERIENCE\n"
        "Data Analyst Intern, Example Analytics, 2025\n"
        "- Built a churn model with scikit-learn and cleaned customer data with pandas.\n"
        "- Wrote SQL queries to prepare weekly reports for the sales team.\n"
        "EDUCATION\n"
        "B.Tech, Computer Science, 2026\n"
    )
    jd = (
        "Data Analyst\n"
        "Example Analytics is hiring a data analyst to work with the sales team.\n"
        "REQUIREMENTS\n"
        "- Strong Python and SQL for analysing customer data\n"
        "- Experience with pandas for cleaning and reshaping tables\n"
        "- Ability to explain results to non-technical colleagues\n"
        "NICE TO HAVE\n"
        "- Docker\n"
        "- Exposure to scikit-learn\n"
    )
    assert len(resume) > 200 and len(jd) > 200
    result = Analyzer(FakeMatcher(0.5)).analyze(resume, jd)

    assert isinstance(result, AnalysisResult)
    assert result.verdict in VERDICTS
    assert 0.0 <= result.overall_score <= 100.0
    assert 0.0 <= result.skill_coverage <= 1.0
    assert isinstance(result.matched_skills, list)
    assert isinstance(result.missing_skills, list)
    assert isinstance(result.warnings, list)
