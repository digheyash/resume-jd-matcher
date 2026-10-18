"""Phase 1 smoke test: proves the skeleton imports, loads data and is wired.

(Phase 5: the old "analyze() is still a stub" test was replaced by two tests of
the real entry point. The full behaviour is covered in test_pipeline.py and
test_analyzer.py; these two only check that the real wiring rejects bad input.)
"""
import pytest

from resume_matcher import config
from resume_matcher.analyzer.analyzer import AnalysisError
from resume_matcher.pipeline import analyze
from resume_matcher.schemas import AnalysisResult, SkillGap


def test_sample_files_exist_and_are_not_empty():
    for name in ("sample_resume.txt", "sample_jd.txt"):
        text = (config.SAMPLES_DIR / name).read_text(encoding="utf-8")
        assert len(text) > 100


def test_weights_sum_to_one():
    assert config.WEIGHT_SEMANTIC + config.WEIGHT_SKILL_COVERAGE == pytest.approx(1.0)


def test_result_schema_builds():
    r = AnalysisResult(
        overall_score=0.0, verdict="No-Fit", semantic_score=0.0, skill_coverage=0.0,
        missing_skills=[SkillGap("Docker", "must-have")],
    )
    assert r.missing_skills[0].skill == "Docker"


def test_analyze_rejects_empty_text():
    with pytest.raises(AnalysisError):
        analyze("", "jd")
    with pytest.raises(AnalysisError):
        analyze("resume", "   ")


def test_analyze_says_when_there_is_nothing_to_score():
    # Texts with no requirement lines and no recognised skills cannot be scored.
    # (Observed in the Phase 5 run of the full suite with the real parser.)
    with pytest.raises(AnalysisError):
        analyze("resume", "jd")
