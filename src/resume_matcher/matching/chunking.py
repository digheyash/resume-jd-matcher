"""
Split text into chunks that fit the Sentence-BERT token limit.

WHY THIS EXISTS
    all-MiniLM-L6-v2 reads at most 256 tokens (measured on this machine in
    Phase 4). Anything after that is silently ignored. A real resume is longer,
    so we cut it into pieces, embed each piece, and combine the results.

HOW IT WORKS
    1. A text is cut into small "units": one per line, and a long line is
       cut further at sentence ends (. ! ?).
    2. Units are packed greedily into chunks. A chunk is closed as soon as
       adding the next unit would exceed `max_tokens`.
    3. A single unit that is still too long is cut by words.

    We default to 200 tokens, not 256, to leave room for the two special
    tokens the model adds ([CLS] and [SEP]) and for small counting differences.

THE TOKEN COUNTER IS A PARAMETER
    This module does not import any model. The caller passes a function
    `count_tokens(text) -> int`. In production this is the real Sentence-BERT
    tokenizer; in tests it can be a simple word counter. That keeps these tests
    fast and independent of any download.

KNOWN LIMIT
    A single "word" longer than `max_tokens` (for example a huge URL) cannot
    be split further and is kept as one oversized chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

TokenCounter = Callable[[str], int]

DEFAULT_MAX_TOKENS = 200
FULL_TEXT_KEY = "full_text"

# Split after . ! ? when followed by whitespace ("Node.js" is not split).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One piece of text that fits the model limit.

    Attributes:
        section: the section name it came from (for example "skills").
        text: the chunk text (units joined by newlines).
        n_tokens: token count of `text`, from the counter that was passed in.
    """

    section: str
    text: str
    n_tokens: int


def split_units(text: str) -> list[str]:
    """Cut text into small units: lines, then sentences inside long lines."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in _SENTENCE_END.split(line):
            part = part.strip()
            if part:
                units.append(part)
    return units


def _split_by_words(unit: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Last resort: cut one over-long unit into word groups that fit."""
    pieces: list[str] = []
    current = ""
    for word in unit.split():
        candidate = f"{current} {word}" if current else word
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _pack(units: Iterable[str], count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Greedily join units (with newlines) into texts of at most max_tokens."""
    texts: list[str] = []
    current = ""
    for unit in units:
        # A unit that cannot fit even alone is cut by words first.
        if count_tokens(unit) > max_tokens:
            parts = _split_by_words(unit, count_tokens, max_tokens)
        else:
            parts = [unit]
        for part in parts:
            candidate = f"{current}\n{part}" if current else part
            if current and count_tokens(candidate) > max_tokens:
                texts.append(current)
                current = part
            else:
                current = candidate
    if current:
        texts.append(current)
    return texts


def chunk_text(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    section: str = FULL_TEXT_KEY,
) -> list[Chunk]:
    """Split one text into chunks of at most `max_tokens` tokens.

    Empty or whitespace-only text gives an empty list.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    texts = _pack(split_units(text), count_tokens, max_tokens)
    return [Chunk(section=section, text=t, n_tokens=count_tokens(t)) for t in texts]


def chunk_sections(
    sections: Mapping[str, str],
    count_tokens: TokenCounter,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    skip_sections: Iterable[str] = (),
) -> list[Chunk]:
    """Chunk every section separately, so a chunk never mixes two sections.

    Sections named in `skip_sections` are ignored. Order follows the mapping.
    """
    skip = set(skip_sections)
    chunks: list[Chunk] = []
    for name, body in sections.items():
        if name in skip:
            continue
        chunks.extend(chunk_text(body, count_tokens, max_tokens, section=name))
    return chunks
