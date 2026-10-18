"""
The analyzer: one resume + one job description -> AnalysisResult (Phase 5).

WHAT HAPPENS, STEP BY STEP
    1. parse both texts                      (Phase 2)
    2. compare skills with set logic         (Phase 3) -> matched / missing / extra
    3. requirement-line semantic score       (Phase 4) -> mean best-line cosine
    4. final score = rescale + weighted sum  (scoring.py)
    5. verdict from the thresholds           (scoring.py)
    6. missing skills ranked by importance   (gaps.py)

The three collaborators (section matcher, skill extractor, parse function) are
PARAMETERS, so tests can use tiny fakes and need no model or download.

WHAT THE RESULT REPORTS (kept as the schemas.py comments say)
    semantic_score : the RAW cosine of the requirement-line method, clipped to
                     0..1. The score itself uses the RESCALED value.
    skill_coverage : plain matched / required (Phase 3). The score itself uses
                     the must-have-weighted coverage.

Until the Phase 5 tuning script has produced values, the numbers in
ScoringParams are PLACEHOLDERS and the output must not be quoted as a result.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

from ..schemas import AnalysisResult
from ..skills.comparison import compare_documents
from .gaps import rank_missing_skills
from .scoring import (
    MUST_HAVE_SECTIONS,
    ScoringParams,
    score_pair,
    verdict_from_score,
    weighted_coverage,
)

# The JD section whose lines are scored by the requirement-line method.
REQUIREMENTS_SECTION = MUST_HAVE_SECTIONS[0]


class AnalysisError(ValueError):
    """The pair cannot be analysed (empty text, or nothing at all to score)."""


def _document_warnings(document: Any, label: str) -> list[str]:
    """Copy the parser's warnings for one document, prefixed with its name."""
    warnings = getattr(document, "warnings", None) or []
    return [f"{label}: {warning}" for warning in warnings]


class Analyzer:
    """Analyse resume-vs-JD pairs.

    section_matcher : object with match_documents(resume_doc, jd_doc) returning
                      something with mean_score(section) -> float | None
                      (normally matching.section_match.SectionMatcher).
    extractor       : skill extractor, or None for the default one.
    params          : ScoringParams (anchors, weight, thresholds).
    parse_fn        : parse_fn(text, kind) -> parsed document; None means the
                      real Phase 2 parse_text.
    """

    def __init__(
        self,
        section_matcher: Any,
        extractor: Any = None,
        params: ScoringParams | None = None,
        parse_fn: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.section_matcher = section_matcher
        self.extractor = extractor
        self.params = params if params is not None else ScoringParams()
        if parse_fn is None:
            from ..parsing.parser import parse_text

            parse_fn = parse_text
        self.parse_fn = parse_fn

    def analyze(self, resume_text: str, jd_text: str) -> AnalysisResult:
        """Compare one resume with one job description."""
        if not resume_text or not resume_text.strip():
            raise AnalysisError("The resume text is empty.")
        if not jd_text or not jd_text.strip():
            raise AnalysisError("The job description text is empty.")

        resume_doc = self.parse_fn(resume_text, "resume")
        jd_doc = self.parse_fn(jd_text, "jd")

        comparison = compare_documents(resume_doc, jd_doc, self.extractor)
        section_result = self.section_matcher.match_documents(resume_doc, jd_doc)
        raw_semantic = section_result.mean_score(REQUIREMENTS_SECTION)

        warnings = _document_warnings(resume_doc, "Resume")
        warnings += _document_warnings(jd_doc, "Job description")

        has_skills = comparison.has_jd_skills
        if raw_semantic is None and not has_skills:
            raise AnalysisError(
                "The job description has no requirement lines and no recognised "
                "skills, so there is nothing to score."
            )
        if raw_semantic is None:
            warnings.append(
                "No requirement lines were found in the job description; the "
                "score uses skill coverage only."
            )

        params = self.params
        if has_skills:
            coverage_used = weighted_coverage(
                comparison.matched, comparison.missing, comparison.jd_sources
            )
        else:
            warnings.append(
                "No recognised skills were found in the job description; the "
                "score uses the semantic match only."
            )
            coverage_used = 0.0
            params = dataclasses.replace(params, weight_semantic=1.0)

        score = score_pair(raw_semantic, coverage_used, params)

        if raw_semantic is None:
            semantic_reported = 0.0
        else:
            semantic_reported = min(1.0, max(0.0, float(raw_semantic)))

        return AnalysisResult(
            overall_score=score * 100.0,
            verdict=verdict_from_score(score, params),
            semantic_score=semantic_reported,
            skill_coverage=comparison.coverage,
            matched_skills=list(comparison.matched),
            missing_skills=rank_missing_skills(comparison.missing, comparison.jd_sources),
            extra_skills=list(comparison.extra),
            warnings=warnings,
        )
