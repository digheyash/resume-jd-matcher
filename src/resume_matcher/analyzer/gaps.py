"""
Ranked skill-gap list (Phase 5).

Turns the unordered set of MISSING skills (Phase 3 set logic) into an ordered
list of SkillGap objects, most important first.

Ranking rules (DESIGN CHOICES, not tuned on data):
  1. importance: must-have, then nice-to-have, then unknown
     (importance comes from the JD section, see scoring.importance_of)
  2. within the same importance: the skill mentioned in MORE distinct JD
     sections comes first (asked for twice is probably more central)
  3. then alphabetical (ignoring case), so the output is always the same
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..schemas import SkillGap
from .scoring import importance_of

# Lower number = ranked earlier.
_IMPORTANCE_ORDER = {"must-have": 0, "nice-to-have": 1, "unknown": 2}


def rank_missing_skills(
    missing: Iterable[str],
    jd_sources: Mapping[str, Sequence[str]],
) -> list[SkillGap]:
    """Return the missing skills as SkillGap objects, most important first.

    `missing` is SkillComparison.missing; `jd_sources` is
    SkillComparison.jd_sources (JD sections where each skill was found).
    A skill that has no entry in `jd_sources` counts as importance 'unknown'.
    Repeated names are listed once.
    """
    unique = list(dict.fromkeys(missing))

    def sort_key(skill: str):
        sources = tuple(jd_sources.get(skill, ()))
        return (
            _IMPORTANCE_ORDER[importance_of(sources)],
            -len(set(sources)),
            skill.lower(),
            skill,
        )

    return [
        SkillGap(skill=skill, importance=importance_of(jd_sources.get(skill, ())))
        for skill in sorted(unique, key=sort_key)
    ]
