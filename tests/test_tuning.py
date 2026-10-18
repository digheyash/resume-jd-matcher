"""
Tests for resume_matcher.analyzer.tuning (Phase 5).

Everything here uses small hand-made data sets and fakes: no Sentence-BERT, no
spaCy, no data files. The three data sets are built so the right answer can be
checked by eye:

    SEMANTIC_LED   the raw semantic score follows the ratings, the coverage is
                   noisy      -> the best weight is the HIGH limit (0.8)
    COVERAGE_LED   the raw semantic score runs AGAINST the ratings, coverage
                   follows them -> the best weight is the LOW limit (0.2)
    SEPARABLE      both signals rise with the ratings and the three verdict
                   groups are cleanly apart -> thresholds can be perfect
"""
import dataclasses
import json
import math

import pytest

import resume_matcher.analyzer.tuning as tuning
from resume_matcher.analyzer.analyzer import Analyzer
from resume_matcher.analyzer.scoring import (
    VERDICTS,
    ScoringParams,
    score_pair,
    verdict_from_score,
)
from resume_matcher.analyzer.tuning import (
    WEIGHT_MAX,
    WEIGHT_MIN,
    PairFeatures,
    build_features,
    confusion_matrix,
    default_output_path,
    evaluate,
    fit_anchors,
    fit_params,
    fit_thresholds,
    fit_weight,
    format_report,
    leave_one_out,
    run_tuning,
    score_features,
    to_json_dict,
)


def feats(rows):
    """rows: (pair_id, human_score, human_verdict, raw, weighted_coverage)."""
    return [PairFeatures(i, float(h), v, raw, cov, cov) for i, h, v, raw, cov in rows]


SEMANTIC_LED = feats(
    [
        (1, 0, "No-Fit", 0.20, 1.0),
        (2, 10, "No-Fit", 0.25, 0.6),
        (3, 20, "No-Fit", 0.30, 0.2),
        (4, 30, "No-Fit", 0.35, 0.0),
        (5, 40, "No-Fit", 0.40, 0.6),
        (6, 50, "No-Fit", 0.55, 0.0),
        (7, 60, "No-Fit", 0.60, 0.6),
    ]
)

COVERAGE_LED = feats(
    [
        (1, 0, "No-Fit", 0.60, 0.0),
        (2, 10, "No-Fit", 0.55, 0.2),
        (3, 20, "No-Fit", 0.40, 0.3),
        (4, 30, "No-Fit", 0.35, 0.5),
        (5, 40, "No-Fit", 0.30, 0.7),
        (6, 50, "No-Fit", 0.25, 0.8),
        (7, 60, "No-Fit", 0.20, 1.0),
    ]
)

SEPARABLE = feats(
    [
        (1, 0, "No-Fit", 0.20, 0.0),
        (2, 10, "No-Fit", 0.22, 0.1),
        (3, 20, "No-Fit", 0.25, 0.2),
        (4, 45, "Partial Fit", 0.35, 0.4),
        (5, 50, "Partial Fit", 0.38, 0.5),
        (6, 60, "Partial Fit", 0.40, 0.5),
        (7, 85, "Fit", 0.52, 0.8),
        (8, 95, "Fit", 0.55, 0.9),
        (9, 100, "Fit", 0.60, 1.0),
    ]
)


# --------------------------------------------------------------------------
# Correlation helper, confusion matrix, evaluate
# --------------------------------------------------------------------------
class TestStatistics:
    def test_perfect_correlation(self):
        assert tuning._correlation("pearson", [1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
        assert tuning._correlation("spearman", [1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        assert tuning._correlation("spearman", [1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)

    def test_constant_input_gives_nan(self):
        assert math.isnan(tuning._correlation("pearson", [1, 1, 1], [1, 2, 3]))
        assert math.isnan(tuning._correlation("spearman", [1, 2, 3], [5, 5, 5]))

    def test_too_few_points_gives_nan(self):
        assert math.isnan(tuning._correlation("pearson", [1, 2], [1, 2]))

    def test_confusion_matrix_by_hand(self):
        true = ["Fit", "No-Fit", "Partial Fit", "No-Fit"]
        pred = ["Fit", "Fit", "No-Fit", "No-Fit"]
        # rows/cols in VERDICTS order: Fit, Partial Fit, No-Fit
        assert confusion_matrix(true, pred) == (
            (1, 0, 0),  # Fit      -> predicted Fit
            (0, 0, 1),  # Partial  -> predicted No-Fit
            (1, 0, 1),  # No-Fit   -> one Fit, one No-Fit
        )

    def test_evaluate_perfect_agreement(self):
        data = feats([(1, 0, "No-Fit", 0.2, 0.0), (2, 50, "Partial Fit", 0.4, 0.5), (3, 100, "Fit", 0.6, 1.0)])
        result = evaluate([0.0, 0.5, 1.0], ["No-Fit", "Partial Fit", "Fit"], data)
        assert result.n == 3
        assert result.accuracy == pytest.approx(1.0)
        assert result.pearson == pytest.approx(1.0)
        assert result.spearman == pytest.approx(1.0)
        assert result.confusion == ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def test_evaluate_one_wrong_verdict(self):
        data = feats([(1, 0, "No-Fit", 0.2, 0.0), (2, 50, "Partial Fit", 0.4, 0.5), (3, 100, "Fit", 0.6, 1.0)])
        result = evaluate([0.0, 0.5, 1.0], ["No-Fit", "Fit", "Fit"], data)
        assert result.accuracy == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# fit_anchors
# --------------------------------------------------------------------------
class TestFitAnchors:
    def test_lowest_and_highest(self):
        assert fit_anchors([0.4, 0.2, 0.6, 0.3]) == (0.2, 0.6)

    def test_accepts_a_generator(self):
        assert fit_anchors(x for x in (0.5, 0.1)) == (0.1, 0.5)

    def test_all_equal_raises(self):
        with pytest.raises(ValueError):
            fit_anchors([0.3, 0.3, 0.3])

    def test_single_value_raises(self):
        with pytest.raises(ValueError):
            fit_anchors([0.3])


# --------------------------------------------------------------------------
# fit_weight
# --------------------------------------------------------------------------
def anchors_of(data):
    return fit_anchors(f.raw_semantic for f in data)


class TestFitWeight:
    def test_semantic_led_data_picks_the_high_limit(self):
        # Spearman is 1.0 at weight 0.8 but only about 0.86 at 0.75.
        assert fit_weight(SEMANTIC_LED, *anchors_of(SEMANTIC_LED)) == pytest.approx(WEIGHT_MAX)

    def test_coverage_led_data_picks_the_low_limit(self):
        assert fit_weight(COVERAGE_LED, *anchors_of(COVERAGE_LED)) == pytest.approx(WEIGHT_MIN)

    def test_identical_signals_tie_and_go_to_one_half(self):
        # raw = 0.2 + 0.4 * coverage, anchors 0.2..0.6 -> rescaled raw == coverage,
        # so every weight gives the same scores and the tie-break picks 0.5.
        data = feats(
            [
                (1, 0, "No-Fit", 0.2, 0.0),
                (2, 25, "No-Fit", 0.3, 0.25),
                (3, 50, "Partial Fit", 0.4, 0.5),
                (4, 75, "Fit", 0.5, 0.75),
                (5, 100, "Fit", 0.6, 1.0),
            ]
        )
        assert fit_weight(data, 0.2, 0.6) == pytest.approx(0.5)

    def test_custom_range_is_respected(self):
        low, high = anchors_of(SEMANTIC_LED)
        assert fit_weight(SEMANTIC_LED, low, high, weight_min=0.0, weight_max=0.6) <= 0.6

    def test_range_without_a_grid_value_raises(self):
        low, high = anchors_of(SEMANTIC_LED)
        with pytest.raises(ValueError):
            fit_weight(SEMANTIC_LED, low, high, weight_min=0.51, weight_max=0.52)

    def test_result_is_always_inside_the_default_range(self):
        for data in (SEMANTIC_LED, COVERAGE_LED, SEPARABLE):
            w = fit_weight(data, *anchors_of(data))
            assert WEIGHT_MIN <= w <= WEIGHT_MAX


# --------------------------------------------------------------------------
# fit_thresholds
# --------------------------------------------------------------------------
class TestFitThresholds:
    def test_separable_scores_get_perfect_cut_offs(self):
        scores = [0.1, 0.2, 0.5, 0.6, 0.9, 0.95]
        verdicts = ["No-Fit", "No-Fit", "Partial Fit", "Partial Fit", "Fit", "Fit"]
        # partial cut halfway between 0.2 and 0.5, fit cut halfway between 0.6 and 0.9
        partial, fit = fit_thresholds(scores, verdicts)
        assert partial == pytest.approx(0.35)
        assert fit == pytest.approx(0.75)

    def test_cut_offs_classify_the_training_scores_correctly(self):
        scores = [0.1, 0.2, 0.5, 0.6, 0.9, 0.95]
        verdicts = ["No-Fit", "No-Fit", "Partial Fit", "Partial Fit", "Fit", "Fit"]
        partial, fit = fit_thresholds(scores, verdicts)
        params = ScoringParams(partial_threshold=partial, fit_threshold=fit)
        assert [verdict_from_score(s, params) for s in scores] == verdicts

    def test_equal_accuracy_prefers_the_widest_gaps(self):
        # Two cut-off pairs give 3 of 4 right: (0.15, 0.35) and (0.35, 0.70).
        # The second keeps a bigger distance from every score, so it wins.
        partial, fit = fit_thresholds(
            [0.1, 0.2, 0.5, 0.9], ["No-Fit", "No-Fit", "Fit", "Fit"]
        )
        assert partial == pytest.approx(0.35)
        assert fit == pytest.approx(0.70)

    def test_partial_is_always_below_fit(self):
        partial, fit = fit_thresholds(
            [0.9, 0.1, 0.5, 0.3, 0.7],
            ["Fit", "No-Fit", "Partial Fit", "No-Fit", "Fit"],
        )
        assert partial < fit

    def test_fewer_than_three_distinct_scores_raises(self):
        with pytest.raises(ValueError):
            fit_thresholds([0.2, 0.2, 0.8, 0.8], ["No-Fit", "No-Fit", "Fit", "Fit"])


# --------------------------------------------------------------------------
# fit_params, score_features
# --------------------------------------------------------------------------
class TestFitParams:
    def test_returns_valid_scoring_params(self):
        p = fit_params(SEPARABLE)
        assert isinstance(p, ScoringParams)
        assert p.partial_threshold < p.fit_threshold
        assert WEIGHT_MIN <= p.weight_semantic <= WEIGHT_MAX

    def test_anchors_are_the_rounded_lowest_and_highest_raw_score(self):
        p = fit_params(SEPARABLE)
        assert p.anchor_low == pytest.approx(0.20)
        assert p.anchor_high == pytest.approx(0.60)

    def test_anchors_are_rounded_to_four_decimals(self):
        data = feats(
            [
                (1, 0, "No-Fit", 0.123456, 0.0),
                (2, 50, "Partial Fit", 0.3, 0.5),
                (3, 60, "Partial Fit", 0.4, 0.6),
                (4, 100, "Fit", 0.654321, 1.0),
            ]
        )
        p = fit_params(data)
        assert p.anchor_low == 0.1235
        assert p.anchor_high == 0.6543

    def test_separable_data_is_classified_perfectly_in_sample(self):
        p = fit_params(SEPARABLE)
        _, verdicts = score_features(SEPARABLE, p)
        assert verdicts == [f.human_verdict for f in SEPARABLE]

    def test_fewer_than_three_pairs_raises(self):
        with pytest.raises(ValueError):
            fit_params(SEPARABLE[:2])

    def test_score_features_matches_score_pair(self):
        p = fit_params(SEPARABLE)
        scores, verdicts = score_features(SEPARABLE, p)
        for f, s, v in zip(SEPARABLE, scores, verdicts):
            assert s == pytest.approx(score_pair(f.raw_semantic, f.weighted_coverage, p))
            assert v == verdict_from_score(s, p)
            assert v in VERDICTS


# --------------------------------------------------------------------------
# leave_one_out
# --------------------------------------------------------------------------
class TestLeaveOneOut:
    def test_one_result_per_pair(self):
        loo = leave_one_out(SEPARABLE)
        assert len(loo.scores) == len(loo.verdicts) == len(loo.fold_params) == 9

    def test_each_fold_is_fitted_without_its_own_pair(self):
        loo = leave_one_out(SEPARABLE)
        # Pair 1 has the lowest raw score (0.20); without it the lowest is 0.22.
        assert loo.fold_params[0].anchor_low == pytest.approx(0.22)
        # Pair 9 has the highest raw score (0.60); without it the highest is 0.55.
        assert loo.fold_params[-1].anchor_high == pytest.approx(0.55)
        # Leaving out a middle pair does not move the anchors.
        assert loo.fold_params[4].anchor_low == pytest.approx(0.20)
        assert loo.fold_params[4].anchor_high == pytest.approx(0.60)

    def test_held_out_score_uses_that_folds_params(self):
        loo = leave_one_out(SEPARABLE)
        for f, s, v, p in zip(SEPARABLE, loo.scores, loo.verdicts, loo.fold_params):
            assert s == pytest.approx(score_pair(f.raw_semantic, f.weighted_coverage, p))
            assert v == verdict_from_score(s, p)

    def test_the_input_list_is_not_changed(self):
        before = list(SEPARABLE)
        leave_one_out(SEPARABLE)
        assert SEPARABLE == before


# --------------------------------------------------------------------------
# run_tuning
# --------------------------------------------------------------------------
class TestRunTuning:
    def test_collects_everything(self):
        result = run_tuning(SEPARABLE, notes=["a note"])
        assert result.params == fit_params(SEPARABLE)
        assert result.notes == ("a note",)
        assert result.in_sample.n == result.held_out.n == 9
        assert len(result.in_sample_scores) == len(result.loo.scores) == 9

    def test_in_sample_matches_a_direct_evaluation(self):
        result = run_tuning(SEPARABLE)
        scores, verdicts = score_features(SEPARABLE, result.params)
        direct = evaluate(scores, verdicts, SEPARABLE)
        assert result.in_sample == direct
        assert result.in_sample.accuracy == pytest.approx(1.0)

    def test_majority_baseline(self):
        # All 7 pairs of SEMANTIC_LED are rated No-Fit, so always answering
        # "No-Fit" is right 7 times out of 7 -> baseline accuracy 1.0.
        result = run_tuning(SEMANTIC_LED)
        assert result.baseline_label == "No-Fit"
        assert result.baseline_accuracy == pytest.approx(1.0)

    def test_three_single_signals_are_reported(self):
        result = run_tuning(SEPARABLE)
        names = [name for name, _, _ in result.single_signals]
        assert names == [
            "raw requirement-line cosine",
            "plain skill coverage",
            "weighted skill coverage",
        ]


# --------------------------------------------------------------------------
# build_features, with fakes
# --------------------------------------------------------------------------
class FakeDoc:
    def __init__(self, text, kind):
        self.text = text
        self.kind = kind
        self.warnings = []


def fake_parse(text, kind):
    return FakeDoc(text, kind)


class FakeExtraction:
    def __init__(self, table):
        self.table = table

    def skills(self):
        return set(self.table)

    def sections_of(self, name):
        return tuple(self.table[name])


class FakeExtractor:
    def __init__(self, table):
        self.table = table

    def extract_from_parsed(self, doc):
        return FakeExtraction(self.table[doc.text])


class FakeSectionResult:
    def __init__(self, value):
        self.value = value

    def mean_score(self, section):
        assert section == "requirements"
        return self.value


class FakeMatcher:
    """Returns the value stored for (resume text, jd text)."""

    def __init__(self, values):
        self.values = values

    def match_documents(self, resume_doc, jd_doc):
        return FakeSectionResult(self.values[(resume_doc.text, jd_doc.text)])


class FakePair:
    def __init__(self, pair_id, resume, jd, score=80, verdict="Fit", rated=True):
        self.pair_id = pair_id
        self._resume = resume
        self._jd = jd
        self.human_score = score if rated else None
        self.human_verdict = verdict if rated else None

    @property
    def is_rated(self):
        return self.human_score is not None

    def read_resume(self):
        return self._resume

    def read_jd(self):
        return self._jd


TABLE = {
    "resume": {"Python": ("skills",), "Docker": ("skills",), "Extra": ("skills",)},
    "jd": {
        "Python": ("requirements",),
        "SQL": ("requirements",),
        "Docker": ("nice_to_have",),
    },
    "jd_none": {},
}


class TestBuildFeatures:
    def test_signals_for_a_normal_pair(self):
        # matched Docker (0.5) + Python (1.0) = 1.5 of 2.5 -> 0.6 ; plain 2/3
        matcher = FakeMatcher({("resume", "jd"): 0.42})
        found, notes = build_features(
            [FakePair(7, "resume", "jd", score=65, verdict="Partial Fit")],
            matcher,
            FakeExtractor(TABLE),
            fake_parse,
        )
        assert notes == []
        assert found == [
            PairFeatures(
                pair_id=7,
                human_score=65.0,
                human_verdict="Partial Fit",
                raw_semantic=0.42,
                weighted_coverage=pytest.approx(0.6),
                plain_coverage=pytest.approx(2 / 3),
            )
        ]

    def test_pair_without_requirement_lines_is_left_out_with_a_note(self):
        matcher = FakeMatcher({("resume", "jd"): None})
        found, notes = build_features(
            [FakePair(3, "resume", "jd")], matcher, FakeExtractor(TABLE), fake_parse
        )
        assert found == []
        assert notes == ["pair 3: left out (no requirement lines)"]

    def test_pair_without_jd_skills_is_left_out_with_a_note(self):
        matcher = FakeMatcher({("resume", "jd_none"): 0.5})
        found, notes = build_features(
            [FakePair(4, "resume", "jd_none")], matcher, FakeExtractor(TABLE), fake_parse
        )
        assert found == []
        assert notes == ["pair 4: left out (no recognised JD skills)"]

    def test_unrated_pair_raises(self):
        matcher = FakeMatcher({("resume", "jd"): 0.5})
        with pytest.raises(ValueError, match="no rating"):
            build_features(
                [FakePair(9, "resume", "jd", rated=False)],
                matcher,
                FakeExtractor(TABLE),
                fake_parse,
            )

    def test_features_give_the_same_score_as_the_analyzer(self):
        # The numbers tuned here must be the numbers analyze() uses.
        params = ScoringParams(0.2, 0.6, 0.5, 0.7, 0.4)
        matcher = FakeMatcher({("resume", "jd"): 0.4})
        extractor = FakeExtractor(TABLE)
        found, _ = build_features(
            [FakePair(1, "resume", "jd")], matcher, extractor, fake_parse
        )
        from_features = score_pair(found[0].raw_semantic, found[0].weighted_coverage, params)
        result = Analyzer(matcher, extractor, params, fake_parse).analyze("resume", "jd")
        assert result.overall_score == pytest.approx(from_features * 100.0)
        assert result.overall_score == pytest.approx(55.0)


# --------------------------------------------------------------------------
# Report and JSON
# --------------------------------------------------------------------------
class TestReport:
    def test_report_contains_every_section(self):
        text = format_report(run_tuning(SEPARABLE, notes=["pair 99: left out (test)"]))
        for expected in (
            "PHASE 5",
            "Pairs used: 9",
            "pair 99: left out (test)",
            "Each signal on its own",
            "Tuned numbers",
            "In-sample results",
            "Leave-one-out results",
            "Baseline: always answering",
            "Per pair",
            "Lines for config.py",
            "SEMANTIC_ANCHOR_LOW",
            "SEMANTIC_ANCHOR_HIGH",
            "WEIGHT_SEMANTIC",
            "WEIGHT_SKILL_COVERAGE",
            "FIT_THRESHOLD",
            "PARTIAL_THRESHOLD",
            "Caveats",
            "SYNTHETIC",
            "AFTER looking",
        ):
            assert expected in text, expected

    def test_report_lists_every_pair(self):
        text = format_report(run_tuning(SEPARABLE))
        per_pair = text.split("6. Per pair")[1].split("7. Lines")[0]
        assert len([l for l in per_pair.splitlines() if l.strip().startswith(tuple("123456789"))]) == 9

    def test_config_lines_use_the_tuned_values(self):
        result = run_tuning(SEPARABLE)
        text = format_report(result)
        p = result.params
        assert f"SEMANTIC_ANCHOR_LOW = {p.anchor_low}" in text
        assert f"FIT_THRESHOLD = {p.fit_threshold}" in text
        assert f"PARTIAL_THRESHOLD = {p.partial_threshold}" in text
        assert f"WEIGHT_SKILL_COVERAGE = {round(1.0 - p.weight_semantic, 10)}" in text

    def test_limit_note_appears_only_when_the_weight_is_on_a_limit(self):
        result = run_tuning(SEPARABLE)
        on_limit = dataclasses.replace(
            result, params=dataclasses.replace(result.params, weight_semantic=WEIGHT_MAX)
        )
        inside = dataclasses.replace(
            result, params=dataclasses.replace(result.params, weight_semantic=0.5)
        )
        assert "sits on a limit" in format_report(on_limit)
        assert "sits on a limit" not in format_report(inside)

    def test_json_summary_is_valid_and_matches_the_result(self):
        result = run_tuning(SEPARABLE)
        data = json.loads(json.dumps(to_json_dict(result)))
        p = result.params
        assert data["params"] == {
            "anchor_low": p.anchor_low,
            "anchor_high": p.anchor_high,
            "weight_semantic": p.weight_semantic,
            "partial_threshold": p.partial_threshold,
            "fit_threshold": p.fit_threshold,
        }
        assert data["pairs_used"] == 9
        assert data["verdict_order"] == list(VERDICTS)
        assert data["held_out"]["n"] == 9

    def test_default_output_path(self):
        assert default_output_path().name == "tuned_scoring.json"
        assert default_output_path().parent.name == "processed"
