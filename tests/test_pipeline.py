"""
Tests for resume_matcher.pipeline (Phase 5).

The analyzer is built from fakes, so these tests need no model download and no
spaCy. The one test that builds the REAL default analyzer only checks how it is
wired (cached, tuned numbers); Sentence-BERT's weights load lazily and are
never touched here.
"""
import pytest

from resume_matcher import config, pipeline
from resume_matcher.analyzer.analyzer import AnalysisError, Analyzer
from resume_matcher.analyzer.scoring import ScoringParams
from resume_matcher.schemas import AnalysisResult, SkillGap


# --------------------------------------------------------------------------
# Fakes (same idea as test_analyzer.py)
# --------------------------------------------------------------------------
class FakeDoc:
    def __init__(self, text, kind):
        self.text = text
        self.kind = kind
        self.warnings = []


class FakeExtraction:
    def __init__(self, table):
        self.table = table

    def skills(self):
        return set(self.table)

    def sections_of(self, name):
        return tuple(self.table[name])


class FakeExtractor:
    TABLE = {
        "resume": {"Python": ("skills",), "Docker": ("skills",), "Extra": ("skills",)},
        "jd": {
            "Python": ("requirements",),
            "SQL": ("requirements",),
            "Docker": ("nice_to_have",),
        },
    }

    def extract_from_parsed(self, doc):
        return FakeExtraction(self.TABLE[doc.text])


class FakeSectionResult:
    def mean_score(self, section):
        return 0.4


class FakeMatcher:
    def match_documents(self, resume_doc, jd_doc):
        return FakeSectionResult()


PARAMS = ScoringParams(0.2, 0.6, 0.5, 0.7, 0.4)


def fake_analyzer():
    """Base case from test_analyzer.py: score 55.0, Partial Fit, SQL missing."""
    return Analyzer(FakeMatcher(), FakeExtractor(), PARAMS, lambda text, kind: FakeDoc(text, kind))


# --------------------------------------------------------------------------
# tuned_params
# --------------------------------------------------------------------------
class TestTunedParams:
    def test_values_come_from_config(self):
        p = pipeline.tuned_params()
        assert isinstance(p, ScoringParams)
        assert p.anchor_low == config.SEMANTIC_ANCHOR_LOW
        assert p.anchor_high == config.SEMANTIC_ANCHOR_HIGH
        assert p.weight_semantic == config.WEIGHT_SEMANTIC
        assert p.fit_threshold == config.FIT_THRESHOLD
        assert p.partial_threshold == config.PARTIAL_THRESHOLD

    def test_the_two_weights_in_config_agree(self):
        # The score uses 1 - WEIGHT_SEMANTIC for coverage, so the config constant
        # for the coverage weight must not drift away from it.
        total = config.WEIGHT_SEMANTIC + config.WEIGHT_SKILL_COVERAGE
        assert total == pytest.approx(1.0)

    def test_thresholds_are_in_order(self):
        assert 0.0 <= config.PARTIAL_THRESHOLD < config.FIT_THRESHOLD <= 1.0

    def test_anchors_are_in_order(self):
        assert config.SEMANTIC_ANCHOR_LOW < config.SEMANTIC_ANCHOR_HIGH


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------
class TestAnalyze:
    def test_uses_the_analyzer_it_is_given(self):
        result = pipeline.analyze("resume", "jd", fake_analyzer())
        assert isinstance(result, AnalysisResult)
        assert result.overall_score == pytest.approx(55.0)
        assert result.verdict == "Partial Fit"
        assert result.missing_skills == [SkillGap(skill="SQL", importance="must-have")]

    def test_passes_the_texts_through_unchanged(self):
        seen = []

        class Recorder:
            def analyze(self, resume_text, jd_text):
                seen.append((resume_text, jd_text))
                return "sentinel"

        assert pipeline.analyze("R text", "J text", Recorder()) == "sentinel"
        assert seen == [("R text", "J text")]

    def test_falls_back_to_the_default_analyzer(self, monkeypatch):
        calls = []

        class Recorder:
            def analyze(self, resume_text, jd_text):
                calls.append((resume_text, jd_text))
                return "from default"

        monkeypatch.setattr(pipeline, "get_default_analyzer", lambda: Recorder())
        assert pipeline.analyze("a", "b") == "from default"
        assert calls == [("a", "b")]

    def test_analysis_error_reaches_the_caller(self):
        with pytest.raises(AnalysisError):
            pipeline.analyze("", "jd", fake_analyzer())


# --------------------------------------------------------------------------
# get_default_analyzer (wiring only; the model is never loaded)
# --------------------------------------------------------------------------
class TestDefaultAnalyzer:
    def test_is_cached_and_uses_the_tuned_numbers(self):
        pytest.importorskip("resume_matcher.matching.sbert")
        pytest.importorskip("resume_matcher.matching.section_match")
        pytest.importorskip("resume_matcher.parsing.parser")
        pipeline.get_default_analyzer.cache_clear()
        first = pipeline.get_default_analyzer()
        assert isinstance(first, Analyzer)
        assert pipeline.get_default_analyzer() is first
        assert first.params == pipeline.tuned_params()


# --------------------------------------------------------------------------
# format_result
# --------------------------------------------------------------------------
def sample_result(**changes):
    values = dict(
        overall_score=72.46,
        verdict="Fit",
        semantic_score=0.5123,
        skill_coverage=0.6667,
        matched_skills=["Python", "SQL"],
        missing_skills=[
            SkillGap("Airflow", "must-have"),
            SkillGap("Docker", "nice-to-have"),
        ],
        extra_skills=["Git"],
        warnings=[],
    )
    values.update(changes)
    return AnalysisResult(**values)


class TestFormatResult:
    def test_headline_numbers(self):
        text = pipeline.format_result(sample_result())
        assert "Overall score : 72.5 / 100" in text
        assert "Verdict       : Fit" in text
        assert "0.512" in text
        assert "67%" in text

    def test_skill_lists(self):
        text = pipeline.format_result(sample_result())
        assert "Matched skills (2): Python, SQL" in text
        assert "Extra skills (1), in the resume but not asked for: Git" in text

    def test_missing_skills_are_numbered_in_the_given_order(self):
        text = pipeline.format_result(sample_result())
        assert "Missing skills (2), most important first:" in text
        assert text.index("  1. Airflow (must-have)") < text.index("  2. Docker (nice-to-have)")

    def test_empty_lists_say_none(self):
        text = pipeline.format_result(
            sample_result(matched_skills=[], missing_skills=[], extra_skills=[])
        )
        assert "Matched skills (0): none" in text
        assert "Missing skills (0), most important first:\n  none" in text
        assert "but not asked for: none" in text

    def test_warnings_block_only_when_there_are_warnings(self):
        assert "Warnings:" not in pipeline.format_result(sample_result())
        text = pipeline.format_result(sample_result(warnings=["Resume: very short"]))
        assert "Warnings:\n  - Resume: very short" in text


# --------------------------------------------------------------------------
# main (command line)
# --------------------------------------------------------------------------
class TestMain:
    def test_prints_the_result_and_the_note(self, tmp_path, capsys):
        resume = tmp_path / "r.txt"
        jd = tmp_path / "j.txt"
        resume.write_text("resume", encoding="utf-8")
        jd.write_text("jd", encoding="utf-8")
        code = pipeline.main(["--resume", str(resume), "--jd", str(jd)], fake_analyzer())
        out = capsys.readouterr().out
        assert code == 0
        assert "Overall score : 55.0 / 100" in out
        assert "Verdict       : Partial Fit" in out
        assert "1. SQL (must-have)" in out
        assert "24 synthetic rated pairs" in out

    def test_missing_file_returns_2(self, tmp_path, capsys):
        jd = tmp_path / "j.txt"
        jd.write_text("jd", encoding="utf-8")
        code = pipeline.main(
            ["--resume", str(tmp_path / "nope.txt"), "--jd", str(jd)], fake_analyzer()
        )
        assert code == 2
        assert "Cannot read file" in capsys.readouterr().err

    def test_empty_text_returns_1(self, tmp_path, capsys):
        resume = tmp_path / "r.txt"
        jd = tmp_path / "j.txt"
        resume.write_text("   ", encoding="utf-8")
        jd.write_text("jd", encoding="utf-8")
        code = pipeline.main(["--resume", str(resume), "--jd", str(jd)], fake_analyzer())
        assert code == 1
        assert "Cannot analyse" in capsys.readouterr().err

    def test_help_mentions_the_sample_defaults(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            pipeline.main(["--help"])
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "--resume" in out and "--jd" in out
        assert "sample" in out
