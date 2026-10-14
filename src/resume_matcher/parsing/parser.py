"""Parser: the single entry point that runs load -> clean -> sections.

The rest of the project should use this module instead of calling the
loader, cleaner and section detector one by one.

Main entry points
-----------------
parse_document(path_or_bytes, kind) -> ParsedDocument   (files, PDF bytes)
parse_text(text, kind)              -> ParsedDocument   (pasted text)

`kind` is "resume" or "jd". It only changes which section headings are
recognised and which warnings are produced.

Errors: a file that cannot be read at all raises
loader.DocumentLoadError. Softer problems (scanned PDF, very short text,
no headings) never raise; they become strings in ParsedDocument.warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from resume_matcher.parsing.cleaner import clean_text
from resume_matcher.parsing.loader import (
    LoadedDocument,
    from_text,
    load_document,
    load_pdf,
)
from resume_matcher.parsing.normalizer import normalize
from resume_matcher.parsing.sections import FULL_TEXT_KEY, detect_sections

VALID_KINDS = ("resume", "jd")


@dataclass
class ParsedDocument:
    """Everything the later phases need to know about one document.

    Attributes:
        kind: "resume" or "jd".
        source: how the text was read ("pdfplumber", "pymupdf", "text", ...).
        page_count: number of PDF pages (0 for plain text).
        raw_text: the text exactly as extracted, before cleaning.
        text: the cleaned text. Skill matching and Sentence-BERT use this.
        sections: section name -> section text (see sections.py).
        removed: what the cleaner removed (emails, phones, urls, bullets).
        warnings: human-readable problems, for the UI and AnalysisResult.
    """

    kind: str
    source: str
    page_count: int
    raw_text: str
    text: str
    sections: dict[str, str] = field(default_factory=dict)
    removed: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def normalized_text(self) -> str:
        """Lowercased lemmas joined by spaces, for TF-IDF (Phase 4).

        Computed only when asked, because loading spaCy is not free.
        """
        return normalize(self.text).text


def _check_kind(kind: str) -> None:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}, not {kind!r}")


def _section_warnings(sections: dict[str, str], kind: str) -> list[str]:
    """Warnings about the section structure (empty text is warned elsewhere)."""
    if not sections:
        return []
    if FULL_TEXT_KEY in sections:
        return [
            "No section headings were recognised, so the whole text is "
            "treated as one block."
        ]
    if kind == "resume" and "skills" not in sections:
        return [
            "No Skills section was found in the resume; skills will be "
            "searched for in the whole text."
        ]
    if kind == "jd" and not ({"requirements", "nice_to_have"} & set(sections)):
        return [
            "No Requirements or Nice-to-have section was found in the job "
            "description; must-have vs nice-to-have weighting is not available."
        ]
    return []


def _build(loaded: LoadedDocument, kind: str) -> ParsedDocument:
    """Clean a loaded document, split it into sections, collect warnings."""
    cleaned = clean_text(loaded.text)
    sections = detect_sections(cleaned.text, kind=kind)
    warnings = list(loaded.warnings)
    warnings.extend(_section_warnings(sections, kind))
    return ParsedDocument(
        kind=kind,
        source=loaded.source,
        page_count=loaded.page_count,
        raw_text=loaded.text,
        text=cleaned.text,
        sections=sections,
        removed=cleaned.removed,
        warnings=warnings,
    )


def parse_document(source: Union[str, Path, bytes], kind: str = "resume") -> ParsedDocument:
    """Parse a file (.pdf, .txt, .md) or raw PDF bytes.

    Args:
        source: a file path, or the bytes of a PDF (e.g. a web upload).
        kind: "resume" or "jd".

    Raises:
        ValueError: if kind is invalid.
        DocumentLoadError: if the file is missing, unsupported or unreadable.
    """
    _check_kind(kind)
    if isinstance(source, bytes):
        loaded = load_pdf(source)
    else:
        loaded = load_document(source)
    return _build(loaded, kind)


def parse_text(text: str, kind: str = "resume") -> ParsedDocument:
    """Parse text that is already in memory, such as a pasted JD."""
    _check_kind(kind)
    return _build(from_text(text), kind)
