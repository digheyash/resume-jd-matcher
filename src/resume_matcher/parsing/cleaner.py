"""Regex-based text cleaning for resumes and job descriptions.

What this module does
---------------------
* Fixes line endings and odd Unicode characters that PDFs produce.
* Removes contact details and links (emails, phone numbers, URLs).
* Removes bullet symbols at the start of lines.
* Tidies whitespace and leftover separators such as " | | ".
* RECORDS everything it removed, so nothing disappears silently.

What it deliberately does NOT do
--------------------------------
* No lowercasing, tokenizing or lemmatizing. That is normalizer.py.
* No section detection. That is sections.py.

Main entry point
----------------
clean_text(text) -> CleanedText
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# A phone number must contain this many digits to be removed. This keeps
# year ranges such as "2022-2026" (8 digits) safe.
PHONE_MIN_DIGITS = 10
PHONE_MAX_DIGITS = 13

# ----------------------------------------------------------------------
# Patterns (compiled once, reused for every document)
# ----------------------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

# Only links that start with http/www, or that point at well-known profile
# sites. A generic "word.word" pattern would wrongly delete Node.js, ASP.NET.
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b(?:github|gitlab|bitbucket|linkedin|kaggle|medium)\.com/\S*",
    re.IGNORECASE,
)

# Candidate phone numbers: digits mixed with spaces, dots, dashes, brackets.
# Spaces only (not newlines), so a match can never span two lines.
PHONE_CANDIDATE_RE = re.compile(r"(?<![\w.])\+?\d[\d ().\-]{8,}\d(?!\w)")

# Bullet symbols at the start of a line, plus the "private use" characters
# that PDF fonts often use for bullets.
BULLET_RE = re.compile(
    r"^[ ]*([\u2022\u25cf\u25aa\u25ab\u25e6\u2023\u2219\u00b7\u25a0\u25a1"
    r"\u25c6\u25c7\u25b6\u25ba\u27a2\u27a4\u2713\u2714\u2726\u2756*"
    r"\ue000-\uf8ff]+)[ ]*",
    re.MULTILINE,
)

PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")

# Character-level replacements. A value of None deletes the character.
_CHAR_MAP = {
    ord("\t"): " ",
    ord("\u00a0"): " ",  # non-breaking space
    ord("\u2002"): " ",
    ord("\u2003"): " ",
    ord("\u2009"): " ",
    ord("\u202f"): " ",
    ord("\x0c"): "\n",  # form feed (page break)
    ord("\u200b"): None,  # zero-width space
    ord("\u200c"): None,
    ord("\u200d"): None,
    ord("\u2060"): None,
    ord("\ufeff"): None,  # byte-order mark
    ord("\u00ad"): None,  # soft hyphen
    ord("\u2018"): "'",
    ord("\u2019"): "'",
    ord("\u201c"): '"',
    ord("\u201d"): '"',
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
}


@dataclass
class CleanedText:
    """Result of cleaning one document.

    Attributes:
        text: the cleaned text.
        removed: what was taken out, by category, e.g.
            {"emails": [...], "phones": [...], "urls": [...], "bullets": [...]}.
    """

    text: str
    removed: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line description of what was removed (handy for logs and tests)."""
        parts = [f"{len(items)} {name}" for name, items in self.removed.items() if items]
        return "removed " + ", ".join(parts) if parts else "nothing removed"


def clean_text(text: str) -> CleanedText:
    """Clean raw resume or JD text and record what was removed.

    Args:
        text: raw text from the loader (or pasted by the user).

    Returns:
        CleanedText with the cleaned text and a record of removed items.
    """
    removed: dict[str, list[str]] = {
        "emails": [],
        "urls": [],
        "phones": [],
        "bullets": [],
    }

    # 1. Unicode and line endings --------------------------------------
    # NFKC turns PDF ligatures (the single character "fi") into plain letters.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_CHAR_MAP)

    # 2. Bullets (before anything else touches line starts) ------------
    def _bullet_sub(match: re.Match) -> str:
        removed["bullets"].append(match.group(1))
        return ""

    text = BULLET_RE.sub(_bullet_sub, text)
    text = PRIVATE_USE_RE.sub("", text)

    # 3. Emails, then URLs, then phones. -------------------------------
    # Order matters: an email or URL can contain digits that would
    # otherwise look like a phone number.
    def _email_sub(match: re.Match) -> str:
        removed["emails"].append(match.group(0))
        return " "

    text = EMAIL_RE.sub(_email_sub, text)

    def _url_sub(match: re.Match) -> str:
        raw = match.group(0)
        url = raw.rstrip(".,;:)]}'\"")  # keep sentence punctuation in the text
        removed["urls"].append(url)
        return " " + raw[len(url):]

    text = URL_RE.sub(_url_sub, text)

    def _phone_sub(match: re.Match) -> str:
        digits = sum(ch.isdigit() for ch in match.group(0))
        if PHONE_MIN_DIGITS <= digits <= PHONE_MAX_DIGITS:
            removed["phones"].append(match.group(0).strip())
            return " "
        return match.group(0)  # not a phone number (e.g. a year range)

    text = PHONE_CANDIDATE_RE.sub(_phone_sub, text)

    # 4. Whitespace and leftover separators ----------------------------
    cleaned_lines = []
    for line in text.split("\n"):
        line = re.sub(r" {2,}", " ", line)
        line = re.sub(r"(?:\s*\|\s*){2,}", " | ", line)  # " |  | " -> " | "
        line = line.strip(" |,;")
        if not re.search(r"\w", line):  # nothing but punctuation left
            line = ""
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return CleanedText(text=text, removed=removed)
