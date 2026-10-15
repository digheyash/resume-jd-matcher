"""Skill comparison: set logic between a resume and a job description (Phase 3).

Given the skills found in the resume and in the JD (canonical names), plain
set operations answer three questions:

    matched = resume AND JD     what the resume already shows
    missing = JD minus resume   the skill gap
    extra   = resume minus JD   skills the JD did not ask for

    coverage = len(matched) / len(JD skills)

Tiny example
------------
    JD skills     {Python, SQL, Docker}
    Resume skills {Python, SQL, Git}
    matched {Python, SQL}   missing {Docker}   extra {Git}
    coverage = 2 / 3 = 0.67

This module does NOT rank the gap or decide must-have vs nice-to-have. It only
records, for every skill, the sections it came from ("requirements",
"nice_to_have", "skills", ...). Phase 5 uses that to set SkillGap.importance
without re-reading the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from resume_matcher.skills.extractor import (
    ExtractionResult,
    SkillExtractor,
    get_default_extractor,
)


def _sorted(names) -> tuple[str, ...]:
    """Sort canonical names alphabetically, ignoring case."""
    return tuple(sorted(names, key=str.lower))


@dataclass(frozen=True)
class SkillComparison:
    """Result of comparing resume skills with JD skills."""

    matched: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    jd_total: int  # number of distinct skills found in the JD
    # Sections where each JD skill (matched or missing) was found.
    jd_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Sections where each resume skill (matched or extra) was found.
    resume_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Share of JD skills the resume covers, from 0.0 to 1.0.

        If the JD contains no recognised skills there is nothing to cover and
        the value is 0.0; check has_jd_skills before trusting it.
        """
        if self.jd_total == 0:
            return 0.0
        return len(self.matched) / self.jd_total

    @property
    def coverage_percent(self) -> float:
        """Coverage as a percentage, from 0.0 to 100.0."""
        return self.coverage * 100.0

    @property
    def has_jd_skills(self) -> bool:
        """False when the JD yielded no skills (coverage is then meaningless)."""
        return self.jd_total > 0

    def missing_from(self, section: str) -> tuple[str, ...]:
        """Missing skills that the JD mentioned in a given section."""
        return tuple(s for s in self.missing if section in self.jd_sources[s])

    def summary(self) -> str:
        """One-line description for logs and demos."""
        return (
            f"matched {len(self.matched)}, missing {len(self.missing)}, "
            f"extra {len(self.extra)}, coverage {self.coverage_percent:.0f}% "
            f"of {self.jd_total} JD skills"
        )


def compare_skills(resume: ExtractionResult, jd: ExtractionResult) -> SkillComparison:
    """Compare two extraction results with set logic."""
    resume_skills = resume.skills()
    jd_skills = jd.skills()

    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    extra = resume_skills - jd_skills

    return SkillComparison(
        matched=_sorted(matched),
        missing=_sorted(missing),
        extra=_sorted(extra),
        jd_total=len(jd_skills),
        jd_sources={name: jd.sections_of(name) for name in _sorted(jd_skills)},
        resume_sources={
            name: resume.sections_of(name) for name in _sorted(resume_skills)
        },
    )


def compare_documents(
    resume_doc, jd_doc, extractor: SkillExtractor | None = None
) -> SkillComparison:
    """Extract skills from two ParsedDocument objects and compare them."""
    extractor = extractor if extractor is not None else get_default_extractor()
    return compare_skills(
        extractor.extract_from_parsed(resume_doc),
        extractor.extract_from_parsed(jd_doc),
    )
