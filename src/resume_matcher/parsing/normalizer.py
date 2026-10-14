"""Text normalization with spaCy: tokenize, drop stop words, lemmatize.

IMPORTANT: the output of this module is meant for TF-IDF similarity
(Phase 4). It is NOT used for skill matching, because lemmatizing damages
skill names ("pandas" -> "panda", "C++" -> "c"). Skill matching uses the
cleaned text from cleaner.py instead.

Main entry points
-----------------
tokenize(text)   -> list[str]       lowercase tokens, nothing removed
normalize(text)  -> NormalizedText  lowercase lemmas, noise removed
get_nlp()        -> spaCy pipeline  (loaded once, then cached)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import spacy
from spacy.language import Language

MODEL_NAME = "en_core_web_sm"


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Load the spaCy English pipeline once and reuse it.

    The parser and the named-entity recogniser are switched off because we
    do not need them; this makes loading and processing faster. The tagger
    and lemmatizer stay on, since lemmatization needs part-of-speech tags.
    """
    try:
        return spacy.load(MODEL_NAME, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            f"The spaCy model '{MODEL_NAME}' is not installed. "
            f"Run: python -m spacy download {MODEL_NAME}"
        ) from exc


@dataclass
class NormalizedText:
    """Result of normalizing one text.

    Attributes:
        tokens: lowercase lemmas, with stop words, punctuation, symbols
            and numbers removed.
    """

    tokens: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The tokens joined by single spaces (what TF-IDF will read)."""
        return " ".join(self.tokens)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens. Nothing is removed except spaces.

    Uses only spaCy's tokenizer (no tagging), so it is fast.
    """
    doc = get_nlp().make_doc(text)
    return [token.text.lower() for token in doc if not token.is_space]


def normalize(text: str, remove_stopwords: bool = True) -> NormalizedText:
    """Lowercase, lemmatize and filter text for similarity scoring.

    Steps for every token:
      1. skip whitespace and number-like tokens ("2026", "ten")
      2. optionally skip stop words ("the", "and", "is")
      3. take the lemma (base form) and lowercase it
      4. skip it if it contains no letter or digit (e.g. "-", "++")

    Args:
        text: cleaned text (ideally the output of cleaner.clean_text).
        remove_stopwords: set False to keep stop words.
    """
    if not text.strip():
        return NormalizedText()

    tokens: list[str] = []
    for token in get_nlp()(text):
        if token.is_space or token.like_num:
            continue
        if remove_stopwords and token.is_stop:
            continue
        lemma = token.lemma_.lower().strip()
        if not any(ch.isalnum() for ch in lemma):
            continue
        tokens.append(lemma)
    return NormalizedText(tokens=tokens)
