"""
Compare the three similarity methods against the human ratings.

RUN (after ALL rows of data/labels/pairs.csv are rated):
    PYTHONPATH=src python -m resume_matcher.matching.compare_methods

THE THREE METHODS
    tfidf     TF-IDF cosine on the lemmatised text (Phase 2 normalized_text).
              The vocabulary and IDF are fitted on ALL texts in the pairs file
              (no ratings are used, so no labels leak into the fit).
    sbert     Sentence-BERT cosine of two whole-document vectors.
    sections  Sentence-BERT: for each JD requirement line, the best resume
              line; the score is the mean of those best scores.

WHAT IS REPORTED
    * Pearson r and Spearman rho between each method's score and the human
      score, with p-values and n. Pearson looks for a straight-line relation;
      Spearman only for the same ORDER, which is what matters for ranking.
    * The average score per human verdict (Fit / Partial Fit / No-Fit): a
      good method gives clearly higher averages for better verdicts.
    * Seconds per pair on this machine (CPU). TF-IDF includes the spaCy
      normalisation it needs. One-time costs (TF-IDF fit, model load) are
      listed separately, and the model gets one warm-up call first.
      Parsing and cleaning are not counted: every method needs them.
    * A per-pair table, to see exactly where methods and ratings disagree.

HOW TO READ THE RESULT (honest limits, also printed with the report)
    * Cosine values are NOT probabilities. A correlation measures whether the
      ORDER of the scores agrees with the ratings, not whether the numbers are
      calibrated. Calibration (thresholds, verdicts) is Phase 5.
    * There are few pairs, and they are synthetic (written by the assistant);
      the ratings are the user's. Treat the numbers as indicative only. A
      p-value from 24 pairs is a rough guide, not proof.
    * The results are saved to data/processed/comparison_results.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from resume_matcher import config
from resume_matcher.matching.pairs import (
    VALID_VERDICTS,
    LabelledPair,
    PairsError,
    load_pairs,
)
from resume_matcher.matching.sbert import SbertEmbedder
from resume_matcher.matching.section_match import MUST_HAVE_SECTION, SectionMatcher
from resume_matcher.matching.tfidf import TfidfMatcher

# (key used in PairResult, name shown in the report)
METHODS = (
    ("tfidf", "TF-IDF"),
    ("sbert", "SBERT, whole document"),
    ("sections", "SBERT, requirement lines"),
)

CAVEATS = (
    "Cosine scores are not probabilities; a correlation says whether the ORDER agrees "
    "with the ratings, not whether the numbers are calibrated (that is Phase 5).",
    "Few pairs, all synthetic (written by the assistant); the ratings are the user's. "
    "Treat these numbers as indicative only.",
    "The TF-IDF vocabulary is fitted on the same texts it scores (no ratings used).",
    "Seconds per pair exclude parsing and one-time costs; they come from this machine's CPU.",
)


# ---------------------------------------------------------------- statistics
@dataclass(frozen=True)
class Correlation:
    """A correlation coefficient with its p-value and the number of pairs.

    r and p are None when the correlation is undefined: fewer than 3 pairs, or
    one of the two score lists is constant.
    """

    r: float | None
    p: float | None
    n: int


def correlate(human: Sequence[float], scores: Sequence[float], method: str = "pearson") -> Correlation:
    """Pearson or Spearman correlation between human scores and method scores."""
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    x = np.asarray(human, dtype=float)
    y = np.asarray(scores, dtype=float)
    if x.shape != y.shape:
        raise ValueError("human and scores must have the same length")
    n = int(x.size)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return Correlation(None, None, n)
    function = stats.pearsonr if method == "pearson" else stats.spearmanr
    result = function(x, y)
    r, p = float(result.statistic), float(result.pvalue)
    if math.isnan(r):
        return Correlation(None, None, n)
    return Correlation(r, p, n)


def mean_by_verdict(verdicts: Sequence[str], scores: Sequence[float]) -> dict[str, float | None]:
    """Average score for each verdict; None for a verdict with no pairs."""
    result: dict[str, float | None] = {}
    for verdict in VALID_VERDICTS:
        chosen = [s for v, s in zip(verdicts, scores) if v == verdict]
        result[verdict] = float(np.mean(chosen)) if chosen else None
    return result


# ------------------------------------------------------------------- results
@dataclass(frozen=True)
class PairResult:
    """The three method scores for one rated pair (sections is None if the JD
    has no requirement lines)."""

    pair_id: int
    human_score: float
    human_verdict: str
    tfidf: float
    sbert: float
    sections: float | None


@dataclass(frozen=True)
class MethodSummary:
    key: str
    name: str
    pearson: Correlation
    spearman: Correlation
    seconds_per_pair: float
    mean_by_verdict: dict


@dataclass(frozen=True)
class ComparisonResult:
    pair_results: tuple[PairResult, ...]
    summaries: tuple[MethodSummary, ...]
    corpus_size: int
    tfidf_vocabulary_size: int
    tfidf_fit_seconds: float
    sbert_load_seconds: float


# ------------------------------------------------------------------ the run
def _default_parse(text: str, kind: str):
    """The real Phase 2 parser (imported late so tests need no spaCy)."""
    from resume_matcher.parsing.parser import parse_text

    return parse_text(text, kind=kind)


def run_comparison(
    pairs: Sequence[LabelledPair],
    embedder: SbertEmbedder | None = None,
    tfidf_matcher: TfidfMatcher | None = None,
    parse_fn: Callable[[str, str], object] | None = None,
) -> ComparisonResult:
    """Score every rated pair with the three methods and summarise.

    Args:
        pairs: rated pairs (see pairs.load_pairs).
        embedder: a SbertEmbedder; a real one is created if not given.
        tfidf_matcher: a TfidfMatcher; a new one if not given.
        parse_fn: parse_fn(text, kind) -> parsed document with `.text`,
            `.sections` and `.normalized_text()`. Defaults to Phase 2 parse_text.

    Raises:
        ValueError: if there are no pairs or a pair is not rated.
    """
    if not pairs:
        raise ValueError("no pairs to compare")
    unrated = [p.pair_id for p in pairs if not p.is_rated]
    if unrated:
        raise ValueError(f"pairs not rated yet: {unrated}")

    parse = parse_fn or _default_parse
    embedder = embedder or SbertEmbedder()
    tfidf = tfidf_matcher or TfidfMatcher()
    section_matcher = SectionMatcher(embedder)

    # Parse every distinct text once.
    resumes: dict = {}
    jds: dict = {}
    for pair in pairs:
        if pair.resume_file not in resumes:
            resumes[pair.resume_file] = parse(pair.read_resume(), "resume")
        if pair.jd_file not in jds:
            jds[pair.jd_file] = parse(pair.read_jd(), "jd")

    # One-time: fit TF-IDF on the reference corpus (no ratings involved).
    corpus = [doc.normalized_text() for doc in [*resumes.values(), *jds.values()]]
    start = time.perf_counter()
    tfidf.fit(corpus)
    fit_seconds = time.perf_counter() - start

    # One-time: load the model, then warm it up so the first pair is not slower.
    start = time.perf_counter()
    embedder.model
    load_seconds = time.perf_counter() - start
    warm_text = next(iter(resumes.values())).text
    if warm_text.strip():
        embedder.embed_document(warm_text)

    seconds = {"tfidf": 0.0, "sbert": 0.0, "sections": 0.0}
    results: list[PairResult] = []
    for pair in pairs:
        resume = resumes[pair.resume_file]
        jd = jds[pair.jd_file]

        start = time.perf_counter()
        tfidf_score = tfidf.similarity(resume.normalized_text(), jd.normalized_text())
        seconds["tfidf"] += time.perf_counter() - start

        start = time.perf_counter()
        sbert_score = embedder.similarity(resume.text, jd.text)
        seconds["sbert"] += time.perf_counter() - start

        start = time.perf_counter()
        section_result = section_matcher.match_documents(resume, jd)
        seconds["sections"] += time.perf_counter() - start

        results.append(
            PairResult(
                pair_id=pair.pair_id,
                human_score=pair.human_score,
                human_verdict=pair.human_verdict,
                tfidf=tfidf_score,
                sbert=sbert_score,
                sections=section_result.mean_score(MUST_HAVE_SECTION),
            )
        )

    summaries = []
    for key, name in METHODS:
        usable = [r for r in results if getattr(r, key) is not None]
        human = [r.human_score for r in usable]
        scores = [getattr(r, key) for r in usable]
        summaries.append(
            MethodSummary(
                key=key,
                name=name,
                pearson=correlate(human, scores, "pearson"),
                spearman=correlate(human, scores, "spearman"),
                seconds_per_pair=seconds[key] / len(results),
                mean_by_verdict=mean_by_verdict([r.human_verdict for r in usable], scores),
            )
        )

    return ComparisonResult(
        pair_results=tuple(results),
        summaries=tuple(summaries),
        corpus_size=len(corpus),
        tfidf_vocabulary_size=tfidf.vocabulary_size,
        tfidf_fit_seconds=fit_seconds,
        sbert_load_seconds=load_seconds,
    )


# ----------------------------------------------------------------- reporting
def _fmt_correlation(c: Correlation) -> str:
    if c.r is None:
        return "n/a"
    return f"{c.r:+.3f} (p={c.p:.3f}, n={c.n})"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_report(result: ComparisonResult) -> str:
    """The comparison as plain text (what the command prints)."""
    n_pairs = len(result.pair_results)
    name_width = max(len(name) for _, name in METHODS)
    lines = [
        f"Compared {n_pairs} rated pairs.",
        f"TF-IDF corpus: {result.corpus_size} texts, vocabulary {result.tfidf_vocabulary_size} words "
        f"(fit {result.tfidf_fit_seconds:.3f} s, one-time).",
        f"SBERT model load: {result.sbert_load_seconds:.2f} s (one-time).",
        "",
        "Agreement with the human score",
        f"{'Method':<{name_width}}  {'Pearson r':<26}{'Spearman rho':<26}{'s/pair':>8}",
    ]
    for s in result.summaries:
        lines.append(
            f"{s.name:<{name_width}}  {_fmt_correlation(s.pearson):<26}"
            f"{_fmt_correlation(s.spearman):<26}{s.seconds_per_pair:>8.4f}"
        )

    lines += ["", "Average method score per human verdict"]
    lines.append(f"{'Method':<{name_width}}  " + "".join(f"{v:>13}" for v in VALID_VERDICTS))
    for s in result.summaries:
        lines.append(
            f"{s.name:<{name_width}}  "
            + "".join(f"{_fmt(s.mean_by_verdict[v]):>13}" for v in VALID_VERDICTS)
        )

    lines += ["", "Per pair"]
    lines.append(
        f"{'id':>3} {'human':>6} {'verdict':<12}{'TF-IDF':>8}{'SBERT':>8}{'sections':>10}"
    )
    for r in result.pair_results:
        lines.append(
            f"{r.pair_id:>3} {r.human_score:>6g} {r.human_verdict:<12}"
            f"{r.tfidf:>8.3f}{r.sbert:>8.3f}{_fmt(r.sections):>10}"
        )

    lines += ["", "Caveats"]
    lines += [f"- {text}" for text in CAVEATS]
    return "\n".join(lines)


def save_csv(result: ComparisonResult, path: str | Path) -> Path:
    """Write the per-pair scores to a CSV file; returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["pair_id", "human_score", "human_verdict", "tfidf", "sbert", "sbert_sections"]
        )
        for r in result.pair_results:
            writer.writerow(
                [r.pair_id, f"{r.human_score:g}", r.human_verdict,
                 f"{r.tfidf:.6f}", f"{r.sbert:.6f}",
                 "" if r.sections is None else f"{r.sections:.6f}"]
            )
    return target


def default_output_path() -> Path:
    return config.PROCESSED_DIR / "comparison_results.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare TF-IDF and Sentence-BERT with the human ratings.")
    parser.add_argument("--pairs", type=Path, default=None, help="pairs CSV (default data/labels/pairs.csv)")
    parser.add_argument("--output", type=Path, default=None, help="where to save the per-pair CSV")
    parser.add_argument("--no-save", action="store_true", help="do not write the CSV file")
    args = parser.parse_args(argv)

    try:
        pairs = load_pairs(args.pairs)
    except FileNotFoundError as error:
        print(f"File not found: {error.filename}")
        return 1
    except PairsError as error:
        print(f"Cannot compare yet: {len(error.problems)} problem(s) in the pairs file:")
        for problem in error.problems:
            print("  -", problem)
        print("Fix them (or finish rating) and run again. "
              "Status: PYTHONPATH=src python -m resume_matcher.matching.pairs")
        return 1

    result = run_comparison(pairs)
    print(format_report(result))
    if not args.no_save:
        saved = save_csv(result, args.output or default_output_path())
        print(f"\nSaved per-pair scores to {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
