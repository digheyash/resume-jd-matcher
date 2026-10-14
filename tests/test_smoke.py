"""Phase 1 smoke test: proves the skeleton imports, loads data and is wired."""
import pytest

from resume_matcher import config
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


def test_analyze_is_still_a_stub():
    with pytest.raises(NotImplementedError):
        analyze("resume", "jd")