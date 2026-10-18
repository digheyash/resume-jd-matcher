"""
Tuning and evaluation of the final score (Phase 5).

WHAT THIS MODULE DOES
    Finds the five numbers that turn the two signals into a score and a verdict,
    using the rated pairs, and measures how well that works.

        anchor_low, anchor_high   map the raw requirement-line cosine onto 0..1
        weight_semantic           share of the semantic part in the final score
        partial_threshold,
        fit_threshold             score cut-offs for the three verdicts

HOW EACH NUMBER IS CHOSEN
    anchors     the lowest and highest raw semantic score among the training
                pairs. This does NOT use the ratings (label-free).
    weight      tried at 0.00, 0.05 ... 1.00; the one whose final scores rank
                the pairs most like the human ratings (Spearman). Ties go to
                the value nearest 0.5, then to the smaller one.
    thresholds  every cut-off halfway between two neighbouring scores is a
                candidate; the pair of cut-offs with the best 3-class accuracy
                wins. Ties go to the pair furthest from any real score.

HOW IT IS EVALUATED HONESTLY
    In-sample    numbers use the same pairs the tuning saw (optimistic).
    Leave-one-out refits ALL five numbers on N-1 pairs and scores the pair that
                was left out, once per pair. These held-out numbers are the
                better estimate, but with 24 synthetic pairs they are noisy.

Run (loads Sentence-BERT, about 20 s the first time):
    PYTHONPATH=src python -m resume_matcher.analyzer.tuning
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from scipy import stats

from .. import config
from ..skills.comparison import compare_documents
from .analyzer import REQUIREMENTS_SECTION
from .scoring import (
    VERDICTS,
    ScoringParams,
    score_pair,
    verdict_from_score,
    weighted_coverage,
)

ROUND_DIGITS = 4  # tuned numbers are rounded so config.py reproduces them exactly

# DESIGN CHOICE (not tuned): the weight search only tries values in this range,
# so BOTH signals always count. A score that ignored skill coverage could say
# "Fit" while the gap report lists missing must-haves. If the tuned weight lands
# on a limit, the report says so.
WEIGHT_MIN = 0.2
WEIGHT_MAX = 0.8


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PairFeatures:
    """Everything the score needs for one rated pair."""

    pair_id: int
    human_score: float          # 0-100, the project author's rating
    human_verdict: str          # "Fit" | "Partial Fit" | "No-Fit"
    raw_semantic: float         # mean best-line cosine over the requirements
    weighted_coverage: float    # skill coverage, must-haves counting more
    plain_coverage: float       # matched / required (only used in the report)


def build_features(
    pairs: Iterable[Any],
    section_matcher: Any,
    extractor: Any = None,
    parse_fn: Callable[[str, str], Any] | None = None,
) -> tuple[list[PairFeatures], list[str]]:
    """Compute the signals for every rated pair, exactly as Analyzer does.

    Returns (features, notes). A pair that lacks one of the two signals (no
    requirement lines, or no recognised JD skills) cannot be tuned on, so it is
    left out and a note says why.
    """
    if parse_fn is None:
        from ..parsing.parser import parse_text

        parse_fn = parse_text

    features: list[PairFeatures] = []
    notes: list[str] = []
    for pair in pairs:
        if not pair.is_rated:
            raise ValueError(f"Pair {pair.pair_id} has no rating.")
        resume_doc = parse_fn(pair.read_resume(), "resume")
        jd_doc = parse_fn(pair.read_jd(), "jd")
        comparison = compare_documents(resume_doc, jd_doc, extractor)
        raw = section_matcher.match_documents(resume_doc, jd_doc).mean_score(
            REQUIREMENTS_SECTION
        )
        if raw is None:
            notes.append(f"pair {pair.pair_id}: left out (no requirement lines)")
            continue
        if not comparison.has_jd_skills:
            notes.append(f"pair {pair.pair_id}: left out (no recognised JD skills)")
            continue
        features.append(
            PairFeatures(
                pair_id=pair.pair_id,
                human_score=float(pair.human_score),
                human_verdict=pair.human_verdict,
                raw_semantic=float(raw),
                weighted_coverage=weighted_coverage(
                    comparison.matched, comparison.missing, comparison.jd_sources
                ),
                plain_coverage=comparison.coverage,
            )
        )
    return features, notes


# --------------------------------------------------------------------------
# Small statistics helpers
# --------------------------------------------------------------------------
def _correlation(kind: str, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson or Spearman correlation; nan when it is undefined."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    function = stats.pearsonr if kind == "pearson" else stats.spearmanr
    return float(function(list(xs), list(ys))[0])


def confusion_matrix(
    true: Sequence[str], predicted: Sequence[str]
) -> tuple[tuple[int, ...], ...]:
    """3x3 counts. Rows = human verdict, columns = predicted, order = VERDICTS."""
    index = {name: i for i, name in enumerate(VERDICTS)}
    grid = [[0] * len(VERDICTS) for _ in VERDICTS]
    for t, p in zip(true, predicted):
        grid[index[t]][index[p]] += 1
    return tuple(tuple(row) for row in grid)


@dataclass(frozen=True)
class Evaluation:
    """How well scores and verdicts agree with the human ratings."""

    n: int
    pearson: float
    spearman: float
    accuracy: float
    confusion: tuple[tuple[int, ...], ...]


def evaluate(
    scores: Sequence[float],
    predicted: Sequence[str],
    features: Sequence[PairFeatures],
) -> Evaluation:
    """Compare final scores / verdicts with the human ratings of `features`."""
    human = [f.human_score for f in features]
    true = [f.human_verdict for f in features]
    correct = sum(1 for t, p in zip(true, predicted) if t == p)
    return Evaluation(
        n=len(features),
        pearson=_correlation("pearson", scores, human),
        spearman=_correlation("spearman", scores, human),
        accuracy=correct / len(features) if features else float("nan"),
        confusion=confusion_matrix(true, predicted),
    )


# --------------------------------------------------------------------------
# Fitting the five numbers
# --------------------------------------------------------------------------
def fit_anchors(raw_scores: Iterable[float]) -> tuple[float, float]:
    """Lowest and highest raw score (label-free)."""
    values = [float(v) for v in raw_scores]
    if len(set(values)) < 2:
        raise ValueError("Need at least two different semantic scores to set anchors.")
    return min(values), max(values)


def fit_weight(
    features: Sequence[PairFeatures],
    anchor_low: float,
    anchor_high: float,
    step: float = 0.05,
    weight_min: float = WEIGHT_MIN,
    weight_max: float = WEIGHT_MAX,
) -> float:
    """Weight of the semantic part that best ranks the pairs like the humans.

    Only weights between weight_min and weight_max are tried.
    """
    human = [f.human_score for f in features]
    best_weight: float | None = None
    best_key: tuple[float, float, float] | None = None
    for i in range(int(round(1.0 / step)) + 1):
        weight = round(i * step, 10)
        if weight < weight_min - 1e-9 or weight > weight_max + 1e-9:
            continue
        params = ScoringParams(
            anchor_low=anchor_low, anchor_high=anchor_high, weight_semantic=weight
        )
        scores = [
            score_pair(f.raw_semantic, f.weighted_coverage, params) for f in features
        ]
        rho = _correlation("spearman", scores, human)
        if math.isnan(rho):
            rho = -2.0
        key = (round(rho, 9), -abs(weight - 0.5), -weight)
        if best_key is None or key > best_key:
            best_key, best_weight = key, weight
    if best_weight is None:
        raise ValueError(
            f"No weight between {weight_min} and {weight_max} is on the search grid."
        )
    return best_weight


def fit_thresholds(
    scores: Sequence[float], true_verdicts: Sequence[str]
) -> tuple[float, float]:
    """Return (partial_threshold, fit_threshold) with the best 3-class accuracy.

    Candidates sit halfway between neighbouring distinct scores. Among equally
    accurate pairs, the one with the widest gap to the nearest scores wins.
    """
    distinct = sorted(set(scores))
    if len(distinct) < 3:
        raise ValueError("Need at least three different scores to set two thresholds.")
    cuts = [((a + b) / 2.0, (b - a) / 2.0) for a, b in zip(distinct, distinct[1:])]

    best: tuple[float, float] = (0.0, 1.0)
    best_key: tuple[float, float] | None = None
    for i, (partial, half_p) in enumerate(cuts):
        for fit, half_f in cuts[i + 1:]:
            params = ScoringParams(partial_threshold=partial, fit_threshold=fit)
            correct = sum(
                1
                for s, t in zip(scores, true_verdicts)
                if verdict_from_score(s, params) == t
            )
            key = (round(correct / len(scores), 9), min(half_p, half_f))
            if best_key is None or key > best_key:
                best_key, best = key, (partial, fit)
    return best


def fit_params(features: Sequence[PairFeatures]) -> ScoringParams:
    """Fit all five numbers on `features` (rounded to ROUND_DIGITS)."""
    if len(features) < 3:
        raise ValueError("Need at least three pairs to tune the score.")
    low, high = fit_anchors(f.raw_semantic for f in features)
    low, high = round(low, ROUND_DIGITS), round(high, ROUND_DIGITS)
    weight = fit_weight(features, low, high)
    base = ScoringParams(anchor_low=low, anchor_high=high, weight_semantic=weight)
    scores = [score_pair(f.raw_semantic, f.weighted_coverage, base) for f in features]
    partial, fit = fit_thresholds(scores, [f.human_verdict for f in features])
    return ScoringParams(
        anchor_low=low,
        anchor_high=high,
        weight_semantic=weight,
        fit_threshold=round(fit, ROUND_DIGITS),
        partial_threshold=round(partial, ROUND_DIGITS),
    )


# --------------------------------------------------------------------------
# Scoring with fitted parameters, and leave-one-out
# --------------------------------------------------------------------------
def score_features(
    features: Sequence[PairFeatures], params: ScoringParams
) -> tuple[list[float], list[str]]:
    """Final scores (0..1) and verdicts for every pair under `params`."""
    scores = [score_pair(f.raw_semantic, f.weighted_coverage, params) for f in features]
    return scores, [verdict_from_score(s, params) for s in scores]


@dataclass(frozen=True)
class LooResult:
    """Held-out scores and verdicts, plus the parameters of every fold."""

    scores: tuple[float, ...]
    verdicts: tuple[str, ...]
    fold_params: tuple[ScoringParams, ...]


def leave_one_out(features: Sequence[PairFeatures]) -> LooResult:
    """Refit on N-1 pairs, score the pair left out; repeat for every pair."""
    scores: list[float] = []
    verdicts: list[str] = []
    folds: list[ScoringParams] = []
    for i, held in enumerate(features):
        training = [f for j, f in enumerate(features) if j != i]
        params = fit_params(training)
        score = score_pair(held.raw_semantic, held.weighted_coverage, params)
        scores.append(score)
        verdicts.append(verdict_from_score(score, params))
        folds.append(params)
    return LooResult(tuple(scores), tuple(verdicts), tuple(folds))


@dataclass(frozen=True)
class TuningResult:
    """Everything the report prints."""

    features: tuple[PairFeatures, ...]
    notes: tuple[str, ...]
    params: ScoringParams
    in_sample_scores: tuple[float, ...]
    in_sample_verdicts: tuple[str, ...]
    in_sample: Evaluation
    loo: LooResult
    held_out: Evaluation
    baseline_label: str
    baseline_accuracy: float
    single_signals: tuple[tuple[str, float, float], ...]  # (name, pearson, spearman)


def run_tuning(
    features: Sequence[PairFeatures], notes: Sequence[str] = ()
) -> TuningResult:
    """Fit on all pairs, run leave-one-out, and collect every number to report."""
    params = fit_params(features)
    scores, verdicts = score_features(features, params)
    loo = leave_one_out(features)

    counts = {name: 0 for name in VERDICTS}
    for f in features:
        counts[f.human_verdict] += 1
    label = max(VERDICTS, key=lambda name: counts[name])

    human = [f.human_score for f in features]
    signals = []
    for name, values in (
        ("raw requirement-line cosine", [f.raw_semantic for f in features]),
        ("plain skill coverage", [f.plain_coverage for f in features]),
        ("weighted skill coverage", [f.weighted_coverage for f in features]),
    ):
        signals.append(
            (
                name,
                _correlation("pearson", values, human),
                _correlation("spearman", values, human),
            )
        )

    return TuningResult(
        features=tuple(features),
        notes=tuple(notes),
        params=params,
        in_sample_scores=tuple(scores),
        in_sample_verdicts=tuple(verdicts),
        in_sample=evaluate(scores, verdicts, features),
        loo=loo,
        held_out=evaluate(loo.scores, loo.verdicts, features),
        baseline_label=label,
        baseline_accuracy=counts[label] / len(features),
        single_signals=tuple(signals),
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
CAVEATS = (
    "The pairs are few (see n above) and the texts are SYNTHETIC (AI-written); "
    "the ratings are the project author's. Only a handful of pairs are Partial Fit, "
    "so the Partial Fit threshold rests on very little evidence.",
    "The requirement-line method and the must-have weights were chosen AFTER looking "
    "at these same pairs. Leave-one-out does not account for that, so even the "
    "held-out numbers are somewhat optimistic.",
    "In-sample numbers are optimistic by construction. Held-out numbers are the "
    "better estimate, but each comes from single-pair tests and is noisy; a different "
    "data set would give different numbers.",
    "The anchors are the lowest and highest raw score seen; unseen pairs outside "
    "that range are clipped to 0 or 1.",
    f"The semantic weight was only searched between {WEIGHT_MIN} and {WEIGHT_MAX} "
    "(a design choice, so that both signals always count), not over 0 to 1.",
)

_PARAM_NAMES = (
    "anchor_low",
    "anchor_high",
    "weight_semantic",
    "partial_threshold",
    "fit_threshold",
)


def _num(value: float, digits: int = 3) -> str:
    return "nan" if math.isnan(value) else f"{value:.{digits}f}"


def _confusion_text(evaluation: Evaluation) -> list[str]:
    short = ("Fit", "Partial", "No-Fit")
    lines = ["    human \\ predicted   " + "  ".join(f"{s:>7}" for s in short)]
    for name, row in zip(VERDICTS, evaluation.confusion):
        lines.append(f"    {name:<18}  " + "  ".join(f"{c:>7}" for c in row))
    return lines


def format_report(result: TuningResult) -> str:
    """Plain-text report of the tuning run."""
    p = result.params
    lines: list[str] = []
    add = lines.append

    add("=" * 70)
    add("PHASE 5 - SCORE TUNING AND EVALUATION")
    add("=" * 70)
    add(f"Pairs used: {len(result.features)}")
    for note in result.notes:
        add(f"  note: {note}")

    add("")
    add("1. Each signal on its own vs the human rating (no tuning involved)")
    add(f"   {'signal':<30} {'Pearson':>8} {'Spearman':>9}")
    for name, pearson, spearman in result.single_signals:
        add(f"   {name:<30} {_num(pearson):>8} {_num(spearman):>9}")

    add("")
    add(f"2. Tuned numbers (fitted on all {len(result.features)} pairs)")
    for name in _PARAM_NAMES:
        add(f"   {name:<18} {getattr(p, name)}")
    if p.weight_semantic in (WEIGHT_MIN, WEIGHT_MAX):
        add(
            f"   note: weight_semantic sits on a limit of the allowed range "
            f"({WEIGHT_MIN} to {WEIGHT_MAX}); the data would prefer "
            f"{'more' if p.weight_semantic == WEIGHT_MAX else 'less'} of the semantic signal."
        )

    add("")
    add("3. In-sample results (OPTIMISTIC: same pairs the tuning saw)")
    add(
        f"   Pearson {_num(result.in_sample.pearson)}   "
        f"Spearman {_num(result.in_sample.spearman)}   "
        f"verdict accuracy {_num(result.in_sample.accuracy)}"
    )
    lines.extend(_confusion_text(result.in_sample))

    add("")
    add("4. Leave-one-out results (HELD-OUT: each pair scored by a fit that never saw it)")
    add(
        f"   Pearson {_num(result.held_out.pearson)}   "
        f"Spearman {_num(result.held_out.spearman)}   "
        f"verdict accuracy {_num(result.held_out.accuracy)}"
    )
    lines.extend(_confusion_text(result.held_out))
    add(
        f"   Baseline: always answering '{result.baseline_label}' gives accuracy "
        f"{_num(result.baseline_accuracy)}"
    )

    add("")
    add("5. How much the tuned numbers moved between leave-one-out folds")
    add(f"   {'number':<18} {'all pairs':>10} {'fold min':>10} {'fold max':>10}")
    for name in _PARAM_NAMES:
        values = [getattr(fp, name) for fp in result.loo.fold_params]
        add(
            f"   {name:<18} {getattr(p, name):>10} "
            f"{min(values):>10} {max(values):>10}"
        )

    add("")
    add("6. Per pair (scores are 0-100)")
    add(
        f"   {'id':>2} {'human':>5} {'verdict':<11} "
        f"{'in-sample':>9} {'pred':<11} {'held-out':>8} {'pred':<11}"
    )
    for f, s, v, hs, hv in zip(
        result.features,
        result.in_sample_scores,
        result.in_sample_verdicts,
        result.loo.scores,
        result.loo.verdicts,
    ):
        add(
            f"   {f.pair_id:>2} {f.human_score:>5.0f} {f.human_verdict:<11} "
            f"{s * 100:>9.1f} {v:<11} {hs * 100:>8.1f} {hv:<11}"
        )

    add("")
    add("7. Lines for config.py (only because this run produced them)")
    add(f"   SEMANTIC_ANCHOR_LOW = {p.anchor_low}")
    add(f"   SEMANTIC_ANCHOR_HIGH = {p.anchor_high}")
    add(f"   WEIGHT_SEMANTIC = {p.weight_semantic}")
    add(f"   WEIGHT_SKILL_COVERAGE = {round(1.0 - p.weight_semantic, 10)}")
    add(f"   FIT_THRESHOLD = {p.fit_threshold}")
    add(f"   PARTIAL_THRESHOLD = {p.partial_threshold}")

    add("")
    add("Caveats")
    for caveat in CAVEATS:
        add(f"  - {caveat}")
    return "\n".join(lines)


def to_json_dict(result: TuningResult) -> dict:
    """The tuned numbers and headline metrics, for saving next to the data."""
    p = result.params

    def metrics(e: Evaluation) -> dict:
        return {
            "n": e.n,
            "pearson": e.pearson,
            "spearman": e.spearman,
            "accuracy": e.accuracy,
            "confusion_rows_human_cols_predicted": [list(r) for r in e.confusion],
        }

    return {
        "params": {name: getattr(p, name) for name in _PARAM_NAMES},
        "pairs_used": len(result.features),
        "in_sample": metrics(result.in_sample),
        "held_out": metrics(result.held_out),
        "baseline": {
            "label": result.baseline_label,
            "accuracy": result.baseline_accuracy,
        },
        "verdict_order": list(VERDICTS),
    }


def default_output_path() -> Path:
    return config.PROCESSED_DIR / "tuned_scoring.json"


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune and evaluate the Phase 5 score.")
    parser.add_argument("--pairs", help="pairs CSV (default: data/labels/pairs.csv)")
    parser.add_argument("--output", help="where to save the JSON summary")
    parser.add_argument("--no-save", action="store_true", help="do not write the JSON file")
    args = parser.parse_args(argv)

    # Heavy imports only when running for real.
    from ..matching.pairs import load_pairs
    from ..matching.sbert import SbertEmbedder
    from ..matching.section_match import SectionMatcher

    pairs = load_pairs(args.pairs) if args.pairs else load_pairs()
    print("Loading Sentence-BERT and computing the signals (about 30 s)...", flush=True)
    matcher = SectionMatcher(SbertEmbedder())
    features, notes = build_features(pairs, matcher)
    if len(features) < 3:
        print("Not enough usable pairs to tune the score.", file=sys.stderr)
        return 1

    result = run_tuning(features, notes)
    print(format_report(result))

    if not args.no_save:
        path = Path(args.output) if args.output else default_output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json_dict(result), indent=2), encoding="utf-8")
        print(f"\nSaved summary to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
