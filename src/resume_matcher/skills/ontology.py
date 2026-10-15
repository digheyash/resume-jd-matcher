"""Skill ontology loader (Phase 3).

An ontology here is an organised skill dictionary: every skill has ONE
canonical name, a category, and a list of aliases (other ways to write it).

    {"canonical": "JavaScript", "aliases": ["JS", "ECMAScript"], ...}

This module only LOADS and VALIDATES data/skills.json and offers fast
look-ups. Finding skills inside text is the job of extractor.py.

Terms used in this file
-----------------------
canonical   the one agreed spelling, for example "Scikit-learn"
alias       any other spelling, for example "sklearn"
ambiguous   an alias that is also an ordinary word or letter ("R", "Go").
            The extractor accepts it only in a skill-like context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


class OntologyError(Exception):
    """Raised when skills.json is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Skill:
    """One skill entry from skills.json."""

    canonical: str
    category: str
    aliases: tuple[str, ...] = ()
    # Lower-cased spellings (canonical or alias) that need context to count.
    ambiguous: frozenset[str] = frozenset()

    def spellings(self) -> list[str]:
        """Canonical name followed by every alias, without duplicates."""
        seen: set[str] = set()
        result: list[str] = []
        for text in (self.canonical, *self.aliases):
            key = text.lower()
            if key not in seen:
                seen.add(key)
                result.append(text)
        return result


@dataclass(frozen=True)
class Spelling:
    """One searchable spelling and the skill it stands for."""

    text: str  # as written in skills.json, for example "sklearn"
    canonical: str  # the skill it maps to, for example "Scikit-learn"
    ambiguous: bool  # True if it only counts in a skill-like context


@dataclass(frozen=True)
class Ontology:
    """The whole skill dictionary plus context hints for ambiguous names."""

    skills: dict[str, Skill]
    context_words: frozenset[str] = frozenset()
    context_sections: frozenset[str] = frozenset()
    # lower-cased spelling -> canonical name
    _index: dict[str, str] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.skills)

    def __contains__(self, canonical: object) -> bool:
        return canonical in self.skills

    def canonical_names(self) -> list[str]:
        """All canonical skill names, in file order."""
        return list(self.skills)

    def get(self, canonical: str) -> Skill | None:
        """Return the Skill for a canonical name, or None."""
        return self.skills.get(canonical)

    def category_of(self, canonical: str) -> str | None:
        """Category of a canonical skill, or None if unknown."""
        skill = self.skills.get(canonical)
        return skill.category if skill else None

    def resolve(self, text: str) -> str | None:
        """Map any spelling (case-insensitive) to its canonical name."""
        return self._index.get(text.strip().lower())

    def spellings(self) -> list[Spelling]:
        """Every searchable spelling, longest first.

        Longest-first order matters: the extractor must try "Scikit-learn"
        before "learn", and "Hugging Face Transformers" before "Transformers".
        """
        items: list[Spelling] = []
        for skill in self.skills.values():
            for text in skill.spellings():
                items.append(
                    Spelling(
                        text=text,
                        canonical=skill.canonical,
                        ambiguous=text.lower() in skill.ambiguous,
                    )
                )
        items.sort(key=lambda s: (-len(s.text), s.text.lower()))
        return items


def _default_path() -> Path:
    """Path of data/skills.json: config.SKILLS_FILE if usable, else derived."""
    try:
        from resume_matcher import config  # local import keeps this module light

        candidate = Path(config.SKILLS_FILE)
        if candidate.exists():
            return candidate
    except (ImportError, AttributeError):
        pass
    # src/resume_matcher/skills/ontology.py -> project root is 3 levels up
    return Path(__file__).resolve().parents[3] / "data" / "skills.json"


def _require_str(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OntologyError(f"{where} must be a non-empty string")
    return value.strip()


def _string_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list):
        raise OntologyError(f"{where} must be a list")
    return [_require_str(item, f"{where} item") for item in value]


def parse_ontology(data: object) -> Ontology:
    """Validate already-parsed JSON data and build an Ontology.

    Checks: structure, duplicate canonical names, a spelling claimed by two
    different skills, and ambiguous entries that are not real spellings.
    """
    if not isinstance(data, dict) or "skills" not in data:
        raise OntologyError("skills.json must be an object with a 'skills' list")
    raw_skills = data["skills"]
    if not isinstance(raw_skills, list) or not raw_skills:
        raise OntologyError("'skills' must be a non-empty list")

    skills: dict[str, Skill] = {}
    index: dict[str, str] = {}

    for position, entry in enumerate(raw_skills, start=1):
        where = f"skills[{position}]"
        if not isinstance(entry, dict):
            raise OntologyError(f"{where} must be an object")
        canonical = _require_str(entry.get("canonical"), f"{where}.canonical")
        category = _require_str(entry.get("category"), f"{where}.category")
        aliases = _string_list(entry.get("aliases", []), f"{where}.aliases")
        ambiguous = _string_list(
            entry.get("ambiguous_aliases", []), f"{where}.ambiguous_aliases"
        )

        if canonical in skills:
            raise OntologyError(f"duplicate canonical name: {canonical!r}")

        skill = Skill(
            canonical=canonical,
            category=category,
            aliases=tuple(aliases),
            ambiguous=frozenset(a.lower() for a in ambiguous),
        )

        known = {s.lower() for s in skill.spellings()}
        unknown = skill.ambiguous - known
        if unknown:
            raise OntologyError(
                f"{canonical!r}: ambiguous_aliases not listed as spellings: "
                f"{sorted(unknown)}"
            )

        for text in skill.spellings():
            key = text.lower()
            owner = index.get(key)
            if owner is not None and owner != canonical:
                raise OntologyError(
                    f"spelling {text!r} is claimed by both "
                    f"{owner!r} and {canonical!r}"
                )
            index[key] = canonical

        skills[canonical] = skill

    hints = data.get("context_hints", {})
    if not isinstance(hints, dict):
        raise OntologyError("'context_hints' must be an object")
    words = _string_list(hints.get("words", []), "context_hints.words")
    sections = _string_list(hints.get("sections", []), "context_hints.sections")

    return Ontology(
        skills=skills,
        context_words=frozenset(w.lower() for w in words),
        context_sections=frozenset(s.lower() for s in sections),
        _index=index,
    )


def load_ontology(path: str | Path | None = None) -> Ontology:
    """Load and validate a skills.json file (default: data/skills.json)."""
    file_path = Path(path) if path is not None else _default_path()
    if not file_path.exists():
        raise OntologyError(f"skills file not found: {file_path}")
    try:
        with open(file_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise OntologyError(f"{file_path} is not valid JSON: {exc}") from exc
    return parse_ontology(data)


@lru_cache(maxsize=1)
def get_default_ontology() -> Ontology:
    """Load data/skills.json once and reuse it (like get_nlp() in Phase 2)."""
    return load_ontology()
