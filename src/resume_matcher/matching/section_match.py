"""
Section-level matching: for each JD requirement, find the best resume line.

THE IDEA, WITH A TINY EXAMPLE
    JD requirement:   "Hands-on experience with pandas and NumPy"
    Resume lines:     "Cleaned a telecom dataset with pandas"      -> 0.62
                      "B.Tech, Computer Science, 2027"             -> 0.08
    The best resume line for this requirement scores 0.62 (numbers invented
    for illustration). Doing this for every requirement gives one best score
    per requirement; their average is the section-level score.

WHY IT IS USEFUL
    * It is explainable: for any score we can show WHICH resume line answered
      WHICH requirement, and which requirements were left unanswered.
    * It does not let a long resume drown a short JD (or the reverse), because
      every requirement is judged on its own.

WHAT COUNTS AS A REQUIREMENT / A RESUME LINE
    * JD lines come from the sections "requirements" and "nice_to_have"
      (Phase 2 section names). Other JD sections (header, about,
      responsibilities, benefits) are not requirements.
    * Resume lines come from every section except "header" (name and contact
      details) and "other" (unrecognised headings), one line or sentence at a
      time. A line longer than the model limit is cut into pieces.
    * A list marker ("- ", "* ") at the start of a line is removed first.
    * The score is cosine similarity, NOT a probability or a percentage.

PHASE 4 SCOPE
    This module only measures and explains. Turning these numbers into a
    final score or a verdict is Phase 5.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from resume_matcher.matching.chunking import split_units
from resume_matcher.matching.sbert import SbertEmbedder
from resume_matcher.matching.similarity import cosine_matrix

MUST_HAVE_SECTION = "requirements"
NICE_TO_HAVE_SECTION = "nice_to_have"
DEFAULT_JD_SECTIONS = (MUST_HAVE_SECTION, NICE_TO_HAVE_SECTION)
DEFAULT_SKIP_RESUME_SECTIONS = ("header", "other")

# A list marker at the start of a line: "- ", "* ", a bullet dot or a dash.
# It must be followed by a space (or end the line), so "-5 degrees" is left
# alone, and a line holding only a marker becomes empty and is dropped. The
# Phase 2 cleaner removes bullet dots but keeps "- ", so we strip it here.
_LIST_MARKER = re.compile(r"^[-*•–—]+(?:\s+|$)")


@dataclass(frozen=True)
class RequirementMatch:
    """The best resume line found for one JD requirement.

    Attributes:
        requirement: the JD line.
        jd_section: "requirements" or "nice_to_have".
        score: cosine similarity of the best resume line (0.0 if the resume
            has no usable line at all).
        resume_text: that best resume line, or None if there was none.
        resume_section: the resume section it came from, or None.
    """

    requirement: str
    jd_section: str
    score: float
    resume_text: str | None
    resume_section: str | None


@dataclass(frozen=True)
class SectionMatchResult:
    """All requirement matches for one resume-JD pair."""

    matches: tuple[RequirementMatch, ...]

    def for_section(self, jd_section: str) -> tuple[RequirementMatch, ...]:
        """Only the matches of one JD section."""
        return tuple(m for m in self.matches if m.jd_section == jd_section)

    def mean_score(self, jd_section: str = MUST_HAVE_SECTION) -> float | None:
        """Average best score over one JD section; None if it has no lines."""
        scores = [m.score for m in self.for_section(jd_section)]
        return float(np.mean(scores)) if scores else None

    def weakest(self, n: int = 3, jd_section: str | None = None) -> list[RequirementMatch]:
        """The `n` requirements with the lowest best score (the likely gaps)."""
        pool = self.matches if jd_section is None else self.for_section(jd_section)
        return sorted(pool, key=lambda m: (m.score, m.requirement))[: max(n, 0)]


class SectionMatcher:
    """Matches JD requirement lines against resume lines with Sentence-BERT.

    Args:
        embedder: a SbertEmbedder (the same one can be shared by all code).
        jd_sections: which JD sections count as requirements.
        skip_resume_sections: resume sections that are ignored.
    """

    def __init__(
        self,
        embedder: SbertEmbedder,
        jd_sections: Iterable[str] = DEFAULT_JD_SECTIONS,
        skip_resume_sections: Iterable[str] = DEFAULT_SKIP_RESUME_SECTIONS,
    ):
        self.embedder = embedder
        self.jd_sections = tuple(jd_sections)
        self.skip_resume_sections = frozenset(skip_resume_sections)

    # ------------------------------------------------------------ the lines
    def _lines(
        self,
        sections: Mapping[str, str],
        wanted: Iterable[str] | None = None,
        skipped: Iterable[str] = (),
    ) -> list[tuple[str, str]]:
        """(section, line) pairs; long lines are cut to fit the model."""
        wanted_set = None if wanted is None else set(wanted)
        skipped_set = set(skipped)
        lines: list[tuple[str, str]] = []
        for name, body in sections.items():
            if name in skipped_set:
                continue
            if wanted_set is not None and name not in wanted_set:
                continue
            for unit in split_units(body):
                unit = _LIST_MARKER.sub("", unit).strip()
                if not unit:
                    continue
                for chunk in self.embedder.chunk(unit, section=name):
                    lines.append((name, chunk.text))
        return lines

    # ------------------------------------------------------------- matching
    def match(
        self, resume_sections: Mapping[str, str], jd_sections: Mapping[str, str]
    ) -> SectionMatchResult:
        """Find the best resume line for every JD requirement line."""
        requirements = self._lines(jd_sections, wanted=self.jd_sections)
        if not requirements:
            return SectionMatchResult(matches=())

        resume_lines = self._lines(resume_sections, skipped=self.skip_resume_sections)
        if not resume_lines:
            return SectionMatchResult(
                matches=tuple(
                    RequirementMatch(text, section, 0.0, None, None)
                    for section, text in requirements
                )
            )

        requirement_vectors = self.embedder.embed_texts([text for _, text in requirements])
        resume_vectors = self.embedder.embed_texts([text for _, text in resume_lines])
        similarity = cosine_matrix(requirement_vectors, resume_vectors)
        best = similarity.argmax(axis=1)

        matches = []
        for row, (jd_section, requirement) in enumerate(requirements):
            resume_section, resume_text = resume_lines[best[row]]
            matches.append(
                RequirementMatch(
                    requirement=requirement,
                    jd_section=jd_section,
                    score=float(similarity[row, best[row]]),
                    resume_text=resume_text,
                    resume_section=resume_section,
                )
            )
        return SectionMatchResult(matches=tuple(matches))

    def match_documents(self, resume_doc, jd_doc) -> SectionMatchResult:
        """Convenience: match two Phase 2 ParsedDocument objects."""
        return self.match(resume_doc.sections, jd_doc.sections)
