"""
Score formula and verdict rules (Phase 5).

This module contains only small, pure functions, so every step of the final
score can be explained (and tested) on its own:

    importance_of(sources)        which JD section a skill came from -> importance
    weighted_coverage(...)        share of JD skills the resume covers, must-haves
                                  counting more than nice-to-haves
    rescale(raw, low, high)       map a raw cosine score onto 0..1
    combine(semantic, cov, p)     w * semantic + (1 - w) * coverage
    verdict_from_score(s, p)      Fit / Partial Fit / No-Fit from two thresholds

NOTE ON NUMBERS
  * The importance weights below are DESIGN CHOICES (not tuned on data).
  * ScoringParams defaults for the anchors, the weight and the thresholds are
    PLACEHOLDERS until the Phase 5 tuning script has produced values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .. import config

VERDICTS = ("Fit", "Partial Fit", "No-Fit")

# JD section names (from Phase 2 section detection) that decide importance.
MUST_HAVE_SECTIONS = ("requirements",)
NICE_TO_HAVE_SECTIONS = ("nice_to_have",)

# DESIGN CHOICE, not tuned: a skill asked for under "requirements" counts fully;
# anything else counts half. If a JD has no recognised sections, every skill
# is "unknown", all weights are equal, and the result equals plain coverage.
IMPORTANCE_WEIGHTS = {
    "must-have": 1.0,
    "nice-to-have": 0.5,
    "unknown": 0.5,
}


def importance_of(sources: Iterable[str]) -> str:
    """Return 'must-have', 'nice-to-have' or 'unknown' for one JD skill.

    `sources` is the tuple of JD section names where the skill was found
    (SkillComparison.jd_sources[skill]). A skill found in both a requirements
    and a nice-to-have section counts as must-have.
    """
    names = set(sources)
    if names & set(MUST_HAVE_SECTIONS):
        return "must-have"
    if names & set(NICE_TO_HAVE_SECTIONS):
        return "nice-to-have"
    return "unknown"


def weighted_coverage(
    matched: Sequence[str],
    missing: Sequence[str],
    jd_sources: Mapping[str, Sequence[str]],
) -> float:
    """Weighted share of the JD's skills that the resume covers (0.0 to 1.0).

    Example: JD asks for Python (must-have) and Docker (nice-to-have); the
    resume has only Python. Weights 1.0 and 0.5 -> 1.0 / 1.5 = 0.667,
    whereas plain coverage would be 0.5.

    Returns 0.0 when the JD has no recognised skills (nothing to cover).
    """
    def weight(skill: str) -> float:
        return IMPORTANCE_WEIGHTS[importance_of(jd_sources.get(skill, ()))]

    got = sum(weight(s) for s in matched)
    total = got + sum(weight(s) for s in missing)
    if total <= 0:
        return 0.0
    return got / total


def rescale(raw: float, low: float, high: float) -> float:
    """Linearly map `raw` so that `low` -> 0.0 and `high` -> 1.0, clipped.

    Example: low=0.28, high=0.59, raw=0.435 -> 0.5. Values outside the anchors
    are clipped, so the result is always between 0.0 and 1.0.
    """
    if high <= low:
        raise ValueError(f"high ({high}) must be greater than low ({low}).")
    value = (raw - low) / (high - low)
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class ScoringParams:
    """All tunable numbers of the score in one place.

    anchor_low / anchor_high : raw requirement-line cosine mapped to 0 / 1.
                               PLACEHOLDER (0.0, 1.0 = no rescaling) until tuned.
    weight_semantic          : share of the semantic part, 0..1 (PLACEHOLDER).
    fit_threshold            : score >= this -> Fit (PLACEHOLDER).
    partial_threshold        : score >= this -> Partial Fit (PLACEHOLDER).
    """

    anchor_low: float = 0.0
    anchor_high: float = 1.0
    weight_semantic: float = config.WEIGHT_SEMANTIC
    fit_threshold: float = config.FIT_THRESHOLD
    partial_threshold: float = config.PARTIAL_THRESHOLD

    def __post_init__(self) -> None:
        if self.anchor_high <= self.anchor_low:
            raise ValueError("anchor_high must be greater than anchor_low.")
        if not 0.0 <= self.weight_semantic <= 1.0:
            raise ValueError("weight_semantic must be between 0 and 1.")
        if not 0.0 <= self.partial_threshold < self.fit_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= partial_threshold < fit_threshold <= 1."
            )


def combine(
    semantic: float | None, coverage: float, params: ScoringParams
) -> float:
    """Final score on 0..1 from an already-rescaled semantic score and coverage.

    If `semantic` is None (the JD had no requirement lines to compare), the
    score is the skill coverage alone.
    """
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"coverage must be between 0 and 1, got {coverage}.")
    if semantic is None:
        return coverage
    if not 0.0 <= semantic <= 1.0:
        raise ValueError(f"semantic must be between 0 and 1, got {semantic}.")
    w = params.weight_semantic
    return w * semantic + (1.0 - w) * coverage


def score_pair(
    raw_semantic: float | None, coverage: float, params: ScoringParams
) -> float:
    """Raw cosine + coverage -> final score on 0..1 (rescale, then combine)."""
    if raw_semantic is None:
        return combine(None, coverage, params)
    scaled = rescale(raw_semantic, params.anchor_low, params.anchor_high)
    return combine(scaled, coverage, params)


def verdict_from_score(score: float, params: ScoringParams) -> str:
    """Fit / Partial Fit / No-Fit. A score exactly on a threshold goes UP."""
    if score >= params.fit_threshold:
        return "Fit"
    if score >= params.partial_threshold:
        return "Partial Fit"
    return "No-Fit"
