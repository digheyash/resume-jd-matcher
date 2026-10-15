"""Skill extractor (Phase 3).

Finds skills from the ontology (data/skills.json) inside CLEANED text.

How matching works
------------------
1. Every spelling of every skill becomes one regular expression.
2. Spellings are tried LONGEST FIRST ("Hugging Face Transformers" before
   "Transformers", "R programming" before "R"). A shorter match that overlaps
   a longer one already found is ignored.
3. Boundary rules instead of a plain \\b. A spelling must not be glued to a
   letter, digit, "+" or "#" on either side, and not to a "." on the left:
       "Java" is not found inside "JavaScript"
       "C"    is not found inside "C++" or "C#"
       ".NET" is not found inside "ASP.NET"
4. Ambiguous spellings ("R", "Go", "Excel", "React", ...) are stricter:
       a) they must be written with EXACTLY the capitalisation in skills.json
          ("excel in a team" is not a skill, "Excel" can be);
       b) they must not touch "&" ("R&D");
       c) they need context: either a hint word within two words on the same
          line ("experience with Go"), or they are an item in a list inside a
          Skills / Requirements / Nice-to-have section ("Python, R, SQL").

Input is the CLEANED text from Phase 2 (ParsedDocument.text / .sections),
never the normalized tokens. Leading "- " bullets are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Mapping

from resume_matcher.skills.ontology import Ontology, Spelling, get_default_ontology

# Same value as resume_matcher.parsing.sections.FULL_TEXT_KEY (a test checks it).
FULL_TEXT_KEY = "full_text"

# Sections that never contribute skills: resume "other" holds languages,
# hobbies and references; JD "benefits" holds perks.
SKIP_SECTIONS = frozenset({"other", "benefits"})

# Characters that must NOT touch a match (see module docstring, point 3).
_LEFT = r"(?<![A-Za-z0-9+#.])"
_RIGHT = r"(?![A-Za-z0-9+#])"
_LEFT_STRICT = r"(?<![A-Za-z0-9+#.&])"
_RIGHT_STRICT = r"(?![A-Za-z0-9+#&])"

_DASH_BULLET = re.compile(r"(?m)^[ \t]*-[ \t]+")
_WORD = re.compile(r"[A-Za-z']+")
_AND_OR_END = re.compile(r"\b(?:and|or)$", re.IGNORECASE)
_AND_OR_START = re.compile(r"(?:and|or)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"\.(?:\s|$)")

_LIST_LEFT_CHARS = "\n,;/|•:("
_LIST_RIGHT_CHARS = "\n,;/|•)"


def strip_dash_bullets(text: str) -> str:
    """Remove a leading "- " from every line (the cleaner keeps dash bullets)."""
    return _DASH_BULLET.sub("", text)


@dataclass(frozen=True)
class SkillMatch:
    """One accepted occurrence of a skill in a piece of text."""

    canonical: str  # for example "Scikit-learn"
    text: str  # as written in the document, for example "sklearn"
    start: int
    end: int
    section: str  # section key it was found in, for example "skills"


@dataclass(frozen=True)
class SkillHit:
    """Everything known about one skill in one document."""

    canonical: str
    category: str
    sections: tuple[str, ...]  # section keys where it appeared, first seen first
    count: int  # total accepted occurrences
    surfaces: tuple[str, ...]  # distinct spellings seen, as written

    def in_section(self, name: str) -> bool:
        return name in self.sections


@dataclass(frozen=True)
class ExtractionResult:
    """Skills found in one document, keyed by canonical name."""

    hits: dict[str, SkillHit] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.hits)

    def __contains__(self, canonical: object) -> bool:
        return canonical in self.hits

    def skills(self) -> set[str]:
        """Set of canonical skill names (the input to the set logic)."""
        return set(self.hits)

    def names(self) -> list[str]:
        """Canonical names sorted alphabetically (stable output for display)."""
        return sorted(self.hits, key=str.lower)

    def sections_of(self, canonical: str) -> tuple[str, ...]:
        """Section keys where a skill was found (empty if not found)."""
        hit = self.hits.get(canonical)
        return hit.sections if hit else ()


def _build_pattern(spelling: Spelling) -> re.Pattern[str]:
    """Compile the boundary-aware regex for one spelling."""
    words = spelling.text.split()
    body = r"\s+".join(re.escape(word) for word in words)
    if spelling.ambiguous:
        # Exact capitalisation and no touching "&".
        return re.compile(_LEFT_STRICT + body + _RIGHT_STRICT)
    return re.compile(_LEFT + body + _RIGHT, re.IGNORECASE)


class SkillExtractor:
    """Find ontology skills in cleaned text. Build once, reuse many times."""

    def __init__(self, ontology: Ontology | None = None) -> None:
        self.ontology = ontology if ontology is not None else get_default_ontology()
        # Longest spelling first (Ontology.spellings() guarantees the order).
        self._patterns: list[tuple[Spelling, re.Pattern[str]]] = [
            (spelling, _build_pattern(spelling))
            for spelling in self.ontology.spellings()
        ]

    # ------------------------------------------------------------ context

    @staticmethod
    def _is_list_item(text: str, start: int, end: int) -> bool:
        """True if the match stands as an item of a list, e.g. 'Python, R, SQL'."""
        left = text[max(0, start - 20) : start].rstrip(" \t")
        right = text[end : end + 20].lstrip(" \t")
        left_ok = (
            left == ""
            or left[-1] in _LIST_LEFT_CHARS
            or bool(_AND_OR_END.search(left))
        )
        right_ok = (
            right == ""
            or right[0] in _LIST_RIGHT_CHARS
            or bool(_AND_OR_START.match(right))
            or bool(_SENTENCE_END.match(right))
        )
        return left_ok and right_ok

    def _near_hint_word(self, text: str, start: int, end: int) -> bool:
        """True if a context hint word is within two words, on the same line."""
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)
        before = _WORD.findall(text[line_start:start])[-2:]
        after = _WORD.findall(text[end:line_end])[:2]
        hints = self.ontology.context_words
        return any(word.lower() in hints for word in before + after)

    def _context_ok(self, text: str, start: int, end: int, section: str) -> bool:
        """Decide whether an AMBIGUOUS spelling really is a skill here."""
        if section in self.ontology.context_sections and self._is_list_item(
            text, start, end
        ):
            return True
        return self._near_hint_word(text, start, end)

    # ----------------------------------------------------------- matching

    def find_matches(
        self, text: str, section: str = FULL_TEXT_KEY
    ) -> list[SkillMatch]:
        """Return every accepted skill occurrence in one piece of text."""
        text = strip_dash_bullets(text)
        claimed: list[tuple[int, int]] = []
        matches: list[SkillMatch] = []

        for spelling, pattern in self._patterns:
            for found in pattern.finditer(text):
                start, end = found.span()
                if any(start < c_end and c_start < end for c_start, c_end in claimed):
                    continue  # overlaps a longer match already accepted
                if spelling.ambiguous and not self._context_ok(
                    text, start, end, section
                ):
                    continue
                claimed.append((start, end))
                matches.append(
                    SkillMatch(
                        canonical=spelling.canonical,
                        text=found.group(),
                        start=start,
                        end=end,
                        section=section,
                    )
                )

        matches.sort(key=lambda m: m.start)
        return matches

    # ---------------------------------------------------------- extraction

    def _collect(self, matches: Iterable[SkillMatch]) -> ExtractionResult:
        """Group matches by skill into SkillHit objects."""
        sections: dict[str, list[str]] = {}
        surfaces: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        for match in matches:
            counts[match.canonical] = counts.get(match.canonical, 0) + 1
            sec_list = sections.setdefault(match.canonical, [])
            if match.section not in sec_list:
                sec_list.append(match.section)
            surf_list = surfaces.setdefault(match.canonical, [])
            if match.text not in surf_list:
                surf_list.append(match.text)

        hits = {
            name: SkillHit(
                canonical=name,
                category=self.ontology.category_of(name) or "unknown",
                sections=tuple(sections[name]),
                count=counts[name],
                surfaces=tuple(surfaces[name]),
            )
            for name in counts
        }
        return ExtractionResult(hits=hits)

    def extract_from_text(
        self, text: str, section: str = FULL_TEXT_KEY
    ) -> ExtractionResult:
        """Extract skills from one block of text."""
        return self._collect(self.find_matches(text, section))

    def extract_from_sections(self, sections: Mapping[str, str]) -> ExtractionResult:
        """Extract skills from a {section_key: text} mapping (Phase 2 output).

        Sections in SKIP_SECTIONS are ignored. Each skill remembers the
        sections it appeared in, so Phase 5 can tell must-have from
        nice-to-have without re-reading the text.
        """
        all_matches: list[SkillMatch] = []
        for key, body in sections.items():
            if key in SKIP_SECTIONS:
                continue
            all_matches.extend(self.find_matches(body, key))
        return self._collect(all_matches)

    def extract_from_parsed(self, document) -> ExtractionResult:
        """Extract skills from a ParsedDocument (uses document.sections)."""
        sections = document.sections or {FULL_TEXT_KEY: document.text}
        return self.extract_from_sections(sections)


@lru_cache(maxsize=1)
def get_default_extractor() -> SkillExtractor:
    """Build the extractor for data/skills.json once and reuse it."""
    return SkillExtractor()
