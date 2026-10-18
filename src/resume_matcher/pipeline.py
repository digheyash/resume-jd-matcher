"""
Single entry point of the system: analyze(resume_text, jd_text).

Each phase filled in one stage:
    Phase 2 -> parse + clean      Phase 3 -> skills + set logic
    Phase 4 -> similarity         Phase 5 -> score + verdict + gap report
The UI (Phase 6) calls ONLY analyze().

THE TUNED NUMBERS
    The scoring numbers (anchors, weight, thresholds) come from config.py, where
    they were stored after analyzer/tuning.py produced them on the 24 rated,
    SYNTHETIC pairs. Treat the score as indicative, not as a precise measurement.

THE MODEL
    Sentence-BERT is built once and cached. Its weights load on the first real
    use (about 20 s on a laptop CPU), so the first analyze() call is slow and
    later calls are fast.

Command line (text files only; defaults to the sample resume and sample JD):
    PYTHONPATH=src python -m resume_matcher.pipeline
    PYTHONPATH=src python -m resume_matcher.pipeline --resume my_resume.txt --jd job.txt
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from . import config
from .analyzer.analyzer import AnalysisError, Analyzer
from .analyzer.scoring import ScoringParams
from .schemas import AnalysisResult


def tuned_params() -> ScoringParams:
    """The scoring numbers stored in config.py (tuned in Phase 5)."""
    return ScoringParams(
        anchor_low=config.SEMANTIC_ANCHOR_LOW,
        anchor_high=config.SEMANTIC_ANCHOR_HIGH,
        weight_semantic=config.WEIGHT_SEMANTIC,
        fit_threshold=config.FIT_THRESHOLD,
        partial_threshold=config.PARTIAL_THRESHOLD,
    )


@lru_cache(maxsize=1)
def get_default_analyzer() -> Analyzer:
    """The real analyzer (Sentence-BERT matcher + tuned numbers), built once."""
    from .matching.sbert import SbertEmbedder
    from .matching.section_match import SectionMatcher

    return Analyzer(SectionMatcher(SbertEmbedder()), params=tuned_params())


def analyze(
    resume_text: str, jd_text: str, analyzer: Analyzer | None = None
) -> AnalysisResult:
    """Compare one resume against one job description.

    `analyzer` is for tests and special setups; normally leave it out and the
    cached default analyzer is used. Raises AnalysisError when the pair cannot
    be analysed (empty text, or nothing at all to score).
    """
    active = analyzer if analyzer is not None else get_default_analyzer()
    return active.analyze(resume_text, jd_text)


def format_result(result: AnalysisResult) -> str:
    """A readable plain-text view of one result (used by the command line)."""

    def listing(names: Sequence[str]) -> str:
        return ", ".join(names) if names else "none"

    lines = [
        f"Overall score : {result.overall_score:.1f} / 100",
        f"Verdict       : {result.verdict}",
        f"Semantic match (raw cosine)          : {result.semantic_score:.3f}",
        f"Skill coverage (matched / required)  : {result.skill_coverage * 100:.0f}%",
        "",
        f"Matched skills ({len(result.matched_skills)}): {listing(result.matched_skills)}",
        f"Missing skills ({len(result.missing_skills)}), most important first:",
    ]
    if result.missing_skills:
        for rank, gap in enumerate(result.missing_skills, start=1):
            lines.append(f"  {rank}. {gap.skill} ({gap.importance})")
    else:
        lines.append("  none")
    lines.append(
        f"Extra skills ({len(result.extra_skills)}), in the resume but not asked for: "
        f"{listing(result.extra_skills)}"
    )
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, analyzer: Analyzer | None = None) -> int:
    """Analyse two text files and print the result. Returns an exit code."""
    parser = argparse.ArgumentParser(description="Analyse one resume against one job description.")
    parser.add_argument(
        "--resume",
        default=str(config.SAMPLES_DIR / "sample_resume.txt"),
        help="resume as a text file (default: the sample resume)",
    )
    parser.add_argument(
        "--jd",
        default=str(config.SAMPLES_DIR / "sample_jd.txt"),
        help="job description as a text file (default: the sample JD)",
    )
    args = parser.parse_args(argv)

    try:
        resume_text = Path(args.resume).read_text(encoding="utf-8")
        jd_text = Path(args.jd).read_text(encoding="utf-8")
    except OSError as error:
        print(f"Cannot read file: {error}", file=sys.stderr)
        return 2

    try:
        result = analyze(resume_text, jd_text, analyzer)
    except AnalysisError as error:
        print(f"Cannot analyse: {error}", file=sys.stderr)
        return 1

    print(format_result(result))
    print()
    print(
        "Note: the scoring numbers were tuned on 24 synthetic rated pairs; "
        "treat this score as indicative."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
