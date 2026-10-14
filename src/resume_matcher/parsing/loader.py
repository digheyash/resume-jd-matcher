"""Document loader: turns a PDF or text file into raw text.

This module only READS. It does not clean or analyse the text; that is
the job of the later files in this package.

Main entry points
-----------------
load_document(path)  -> LoadedDocument   (decides by file extension)
load_pdf(path|bytes) -> LoadedDocument   (pdfplumber, fallback PyMuPDF)
load_text_file(path) -> LoadedDocument
from_text(text)      -> LoadedDocument   (for pasted text, e.g. a JD)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pdfplumber
import pymupdf

from resume_matcher.config import MIN_TEXT_CHARS

# A PDF can be given as a path (str/Path) or as raw bytes (Streamlit upload).
PathOrBytes = Union[str, Path, bytes]


class DocumentLoadError(Exception):
    """Raised when a document cannot be read at all."""


@dataclass
class LoadedDocument:
    """Result of loading one document.

    Attributes:
        text: the raw extracted text (may be empty).
        source: how the text was obtained, e.g. "pdfplumber", "pymupdf", "text".
        page_count: number of pages (0 for plain text).
        warnings: human-readable problems, shown later in the UI.
    """

    text: str
    source: str
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _length_warnings(text: str) -> list[str]:
    """Return warnings if the text is empty or suspiciously short."""
    stripped = text.strip()
    if not stripped:
        return [
            "No text could be extracted. If this is a PDF, it is probably a "
            "scanned image. Please upload a text-based PDF or paste the text."
        ]
    if len(stripped) < MIN_TEXT_CHARS:
        return [
            f"Very little text found ({len(stripped)} characters; "
            f"at least {MIN_TEXT_CHARS} expected). Results may be unreliable."
        ]
    return []


def _extract_with_pdfplumber(data: PathOrBytes) -> tuple[str, int]:
    """Extract text from every page using pdfplumber."""
    stream = io.BytesIO(data) if isinstance(data, bytes) else data
    with pdfplumber.open(stream) as pdf:
        # extract_text() returns None for a page with no text layer.
        pages = [(page.extract_text() or "") for page in pdf.pages]
    return "\n".join(pages), len(pages)


def _extract_with_pymupdf(data: PathOrBytes) -> tuple[str, int]:
    """Extract text from every page using PyMuPDF (the fallback)."""
    if isinstance(data, bytes):
        doc = pymupdf.open(stream=data, filetype="pdf")
    else:
        doc = pymupdf.open(str(data))
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n".join(pages), len(pages)


# ----------------------------------------------------------------------
# Public loaders
# ----------------------------------------------------------------------
def load_pdf(data: PathOrBytes) -> LoadedDocument:
    """Load a PDF, trying pdfplumber first and PyMuPDF as a fallback.

    Args:
        data: a file path, or the raw bytes of the PDF.

    Raises:
        DocumentLoadError: if the file is missing or both readers fail.
    """
    if not isinstance(data, bytes) and not Path(data).is_file():
        raise DocumentLoadError(f"File not found: {data}")

    warnings: list[str] = []
    best_text, best_source, best_pages = "", "", 0
    failures = 0

    for name, extractor in (
        ("pdfplumber", _extract_with_pdfplumber),
        ("pymupdf", _extract_with_pymupdf),
    ):
        # Only try the fallback if the first reader gave too little text.
        if best_text.strip() and len(best_text.strip()) >= MIN_TEXT_CHARS:
            break
        try:
            text, pages = extractor(data)
        except Exception as exc:  # noqa: BLE001 - any reader error is non-fatal
            failures += 1
            warnings.append(f"{name} could not read this PDF ({type(exc).__name__}).")
            continue
        # Keep whichever reader found more text.
        if len(text.strip()) > len(best_text.strip()):
            best_text, best_source, best_pages = text, name, pages

    if failures == 2:
        raise DocumentLoadError(
            "Could not open this file as a PDF. It may be corrupted or password-protected."
        )

    warnings.extend(_length_warnings(best_text))
    return LoadedDocument(
        text=best_text,
        source=best_source or "pdf",
        page_count=best_pages,
        warnings=warnings,
    )


def load_text_file(path: Union[str, Path]) -> LoadedDocument:
    """Load a plain-text file, trying common encodings in order."""
    path = Path(path)
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")

    raw = path.read_bytes()
    # utf-8-sig handles files saved by Notepad with a BOM; cp1252 is the
    # usual Windows fallback; latin-1 can decode any byte, so it never fails.
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    return LoadedDocument(text=text, source="text", warnings=_length_warnings(text))


def from_text(text: str) -> LoadedDocument:
    """Wrap already-available text (e.g. a pasted JD) as a LoadedDocument."""
    return LoadedDocument(text=text, source="text", warnings=_length_warnings(text))


def load_document(path: Union[str, Path]) -> LoadedDocument:
    """Load a .pdf, .txt or .md file, chosen by its extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".txt", ".md"):
        return load_text_file(path)
    raise DocumentLoadError(
        f"Unsupported file type '{suffix}'. Please use a PDF or a .txt file."
    )
