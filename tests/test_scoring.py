"""
Tests for resume_matcher.analyzer.scoring (Phase 5).

All tests use small hand-made inputs; no models, no data files. Expected
values are worked out by hand in the comments.
"""
import dataclasses

import pytest

from resume_matcher import config
from resume_matcher.analyzer.scoring import (
    IMPORTANCE_WEIGHTS,
    VERDICTS,
    ScoringParams,
    combine,
    importance_of,
    rescale,
    score_pair,
    verdict_from_score,
    weighted_coverage,
)


# --------------------------------------------------------------------------
# importance_of
# --------------------------------------------------------------------------
class TestImportanceOf:
    def test_requirements_is_must_have(self):
        assert importance_of(("requirements",)) == "must-have"

    def test_nice_to_have_section(self):
        assert importance_of(("nice_to_have",)) == "nice-to-have"

    def test_both_sections_counts_as_must_have(self):
        assert importance_of(("nice_to_have", "requirements")) == "must-have"

    def test_other_section_is_unknown(self):
        assert importance_of(("header",)) == "unknown"
        assert importance_of(("responsibilities",)) == "unknown"

    def test_no_sources_is_unknown(self):
        assert importance_of(()) == "unknown"

    def test_accepts_a_list(self):
        assert importance_of(["requirements"]) == "must-have"


# --------------------------------------------------------------------------
# weighted_coverage
# --------------------------------------------------------------------------
class TestWeightedCoverage:
    def test_docstring_example(self):
        # Python must-have (1.0) matched; Docker nice-to-have (0.5) missing.
        # 1.0 / (1.0 + 0.5) = 0.6667
        src = {"Python": ("requirements",), "Docker": ("nice_to_have",)}
        assert weighted_coverage(["Python"], ["Docker"], src) == pytest.approx(2 / 3)

    def test_everything_matched_is_one(self):
        src = {"A": ("requirements",), "B": ("nice_to_have",)}
        assert weighted_coverage(["A", "B"], [], src) == pytest.approx(1.0)

    def test_nothing_matched_is_zero(self):
        src = {"A": ("requirements",), "B": ("nice_to_have",)}
        assert weighted_coverage([], ["A", "B"], src) == 0.0

    def test_empty_jd_is_zero(self):
        assert weighted_coverage([], [], {}) == 0.0

    def test_no_sections_equals_plain_coverage(self):
        # All four skills are "unknown" (equal weights): 1 of 4 = 0.25.
        src = {s: ("header",) for s in ("A", "B", "C", "D")}
        assert weighted_coverage(["A"], ["B", "C", "D"], src) == pytest.approx(0.25)

    def test_missing_must_have_hurts_more_than_missing_nice_to_have(self):
        src = {"A": ("requirements",), "B": ("nice_to_have",)}
        # have A (1.0), lack B (0.5): 1.0 / 1.5 = 0.6667
        missing_nice = weighted_coverage(["A"], ["B"], src)
        # have B (0.5), lack A (1.0): 0.5 / 1.5 = 0.3333
        missing_must = weighted_coverage(["B"], ["A"], src)
        assert missing_nice == pytest.approx(2 / 3)
        assert missing_must == pytest.approx(1 / 3)
        assert missing_must < missing_nice

    def test_skill_absent_from_sources_is_unknown(self):
        # A must-have (1.0) matched, B not in sources -> unknown (0.5) missing.
        # 1.0 / 1.5 = 0.6667
        assert weighted_coverage(["A"], ["B"], {"A": ("requirements",)}) == pytest.approx(
            2 / 3
        )

    def test_result_stays_between_zero_and_one(self):
        src = {"A": ("requirements",), "B": ("nice_to_have",), "C": ()}
        value = weighted_coverage(["A", "C"], ["B"], src)
        assert 0.0 <= value <= 1.0

    def test_importance_weights_are_as_documented(self):
        assert IMPORTANCE_WEIGHTS == {
            "must-have": 1.0,
            "nice-to-have": 0.5,
            "unknown": 0.5,
        }


# --------------------------------------------------------------------------
# rescale
# --------------------------------------------------------------------------
class TestRescale:
    def test_midpoint(self):
        # (0.435 - 0.28) / (0.59 - 0.28) = 0.155 / 0.31 = 0.5
        assert rescale(0.435, 0.28, 0.59) == pytest.approx(0.5)

    def test_anchors_map_to_zero_and_one(self):
        assert rescale(0.28, 0.28, 0.59) == pytest.approx(0.0)
        assert rescale(0.59, 0.28, 0.59) == pytest.approx(1.0)

    def test_below_low_is_clipped_to_zero(self):
        assert rescale(0.1, 0.28, 0.59) == 0.0

    def test_above_high_is_clipped_to_one(self):
        assert rescale(0.9, 0.28, 0.59) == 1.0

    def test_identity_anchors_leave_value_unchanged(self):
        assert rescale(0.37, 0.0, 1.0) == pytest.approx(0.37)

    def test_equal_anchors_raise(self):
        with pytest.raises(ValueError):
            rescale(0.5, 0.4, 0.4)

    def test_reversed_anchors_raise(self):
        with pytest.raises(ValueError):
            rescale(0.5, 0.6, 0.4)


# --------------------------------------------------------------------------
# ScoringParams
# --------------------------------------------------------------------------
class TestScoringParams:
    def test_defaults_come_from_config(self):
        p = ScoringParams()
        assert p.weight_semantic == config.WEIGHT_SEMANTIC
        assert p.fit_threshold == config.FIT_THRESHOLD
        assert p.partial_threshold == config.PARTIAL_THRESHOLD

    def test_default_anchors_mean_no_rescaling(self):
        p = ScoringParams()
        assert (p.anchor_low, p.anchor_high) == (0.0, 1.0)

    def test_custom_values_are_stored(self):
        p = ScoringParams(0.2, 0.6, 0.3, 0.8, 0.5)
        assert (p.anchor_low, p.anchor_high) == (0.2, 0.6)
        assert p.weight_semantic == 0.3
        assert (p.fit_threshold, p.partial_threshold) == (0.8, 0.5)

    def test_weight_extremes_are_allowed(self):
        ScoringParams(weight_semantic=0.0)
        ScoringParams(weight_semantic=1.0)

    @pytest.mark.parametrize("weight", [-0.1, 1.1])
    def test_weight_out_of_range_raises(self, weight):
        with pytest.raises(ValueError):
            ScoringParams(weight_semantic=weight)

    def test_equal_anchors_raise(self):
        with pytest.raises(ValueError):
            ScoringParams(anchor_low=0.5, anchor_high=0.5)

    def test_reversed_anchors_raise(self):
        with pytest.raises(ValueError):
            ScoringParams(anchor_low=0.6, anchor_high=0.2)

    def test_partial_must_be_below_fit(self):
        with pytest.raises(ValueError):
            ScoringParams(fit_threshold=0.5, partial_threshold=0.5)
        with pytest.raises(ValueError):
            ScoringParams(fit_threshold=0.4, partial_threshold=0.6)

    def test_thresholds_must_be_within_zero_and_one(self):
        with pytest.raises(ValueError):
            ScoringParams(fit_threshold=1.2, partial_threshold=0.5)
        with pytest.raises(ValueError):
            ScoringParams(fit_threshold=0.7, partial_threshold=-0.1)

    def test_params_are_frozen(self):
        p = ScoringParams()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.weight_semantic = 0.9  # type: ignore[misc]


# --------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------
class TestCombine:
    def test_default_equal_weights(self):
        # 0.5 * 0.8 + 0.5 * 0.4 = 0.6
        p = ScoringParams(weight_semantic=0.5)
        assert combine(0.8, 0.4, p) == pytest.approx(0.6)

    def test_custom_weight(self):
        # 0.25 * 0.8 + 0.75 * 0.4 = 0.2 + 0.3 = 0.5
        p = ScoringParams(weight_semantic=0.25)
        assert combine(0.8, 0.4, p) == pytest.approx(0.5)

    def test_weight_one_uses_only_semantic(self):
        p = ScoringParams(weight_semantic=1.0)
        assert combine(0.8, 0.1, p) == pytest.approx(0.8)

    def test_weight_zero_uses_only_coverage(self):
        p = ScoringParams(weight_semantic=0.0)
        assert combine(0.8, 0.1, p) == pytest.approx(0.1)

    def test_missing_semantic_falls_back_to_coverage(self):
        assert combine(None, 0.4, ScoringParams()) == pytest.approx(0.4)

    def test_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            combine(0.5, 1.2, ScoringParams())
        with pytest.raises(ValueError):
            combine(None, -0.1, ScoringParams())

    def test_semantic_out_of_range_raises(self):
        with pytest.raises(ValueError):
            combine(1.2, 0.5, ScoringParams())
        with pytest.raises(ValueError):
            combine(-0.1, 0.5, ScoringParams())

    def test_result_stays_between_zero_and_one(self):
        p = ScoringParams(weight_semantic=0.7)
        for sem in (0.0, 0.5, 1.0):
            for cov in (0.0, 0.5, 1.0):
                assert 0.0 <= combine(sem, cov, p) <= 1.0


# --------------------------------------------------------------------------
# score_pair
# --------------------------------------------------------------------------
class TestScorePair:
    def test_rescales_then_combines(self):
        # anchors 0.2..0.6, raw 0.4 -> (0.4-0.2)/(0.6-0.2) = 0.5
        # weight 0.5, coverage 1.0 -> 0.5*0.5 + 0.5*1.0 = 0.75
        p = ScoringParams(anchor_low=0.2, anchor_high=0.6, weight_semantic=0.5)
        assert score_pair(0.4, 1.0, p) == pytest.approx(0.75)

    def test_raw_above_high_is_clipped_before_combining(self):
        # raw 0.9 -> clipped to 1.0; weight 0.5, coverage 0.0 -> 0.5
        p = ScoringParams(anchor_low=0.2, anchor_high=0.6, weight_semantic=0.5)
        assert score_pair(0.9, 0.0, p) == pytest.approx(0.5)

    def test_raw_below_low_is_clipped_before_combining(self):
        # raw 0.0 -> clipped to 0.0; weight 0.5, coverage 1.0 -> 0.5
        p = ScoringParams(anchor_low=0.2, anchor_high=0.6, weight_semantic=0.5)
        assert score_pair(0.0, 1.0, p) == pytest.approx(0.5)

    def test_no_semantic_score_uses_coverage_only(self):
        p = ScoringParams(anchor_low=0.2, anchor_high=0.6)
        assert score_pair(None, 0.3, p) == pytest.approx(0.3)

    def test_default_params_are_identity_rescale(self):
        # weight 0.5, raw 0.8, coverage 0.4 -> 0.5*0.8 + 0.5*0.4 = 0.6
        p = ScoringParams(weight_semantic=0.5)
        assert score_pair(0.8, 0.4, p) == pytest.approx(0.6)


# --------------------------------------------------------------------------
# verdict_from_score
# --------------------------------------------------------------------------
class TestVerdict:
    P = ScoringParams(fit_threshold=0.70, partial_threshold=0.45)

    @pytest.mark.parametrize(
        "score, expected",
        [
            (1.0, "Fit"),
            (0.85, "Fit"),
            (0.70, "Fit"),  # exactly on the Fit threshold goes UP
            (0.699, "Partial Fit"),
            (0.45, "Partial Fit"),  # exactly on the Partial threshold goes UP
            (0.449, "No-Fit"),
            (0.0, "No-Fit"),
        ],
    )
    def test_boundaries(self, score, expected):
        assert verdict_from_score(score, self.P) == expected

    def test_uses_the_given_thresholds(self):
        p = ScoringParams(fit_threshold=0.9, partial_threshold=0.2)
        assert verdict_from_score(0.8, p) == "Partial Fit"
        assert verdict_from_score(0.1, p) == "No-Fit"
        assert verdict_from_score(0.95, p) == "Fit"

    def test_every_verdict_is_a_known_label(self):
        for s in (0.0, 0.3, 0.5, 0.8, 1.0):
            assert verdict_from_score(s, self.P) in VERDICTS


def test_verdicts_constant():
    assert VERDICTS == ("Fit", "Partial Fit", "No-Fit")


def test_verdicts_match_the_pairs_file_labels():
    # Skipped where the matching package is unavailable; in the real project
    # this checks that the labels in the ratings file and the analyzer agree.
    pairs = pytest.importorskip("resume_matcher.matching.pairs")
    assert tuple(pairs.VALID_VERDICTS) == VERDICTS
