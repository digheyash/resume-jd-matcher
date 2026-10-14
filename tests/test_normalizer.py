"""Tests for resume_matcher.parsing.normalizer.

These tests load the spaCy model, so the first test is a little slower
than the rest. The assertions are deliberately loose (for example "build"
is in the tokens) so they do not break on small spaCy model differences.
"""

from resume_matcher.config import SAMPLES_DIR
from resume_matcher.parsing.cleaner import clean_text
from resume_matcher.parsing.loader import load_text_file
from resume_matcher.parsing.normalizer import get_nlp, normalize, tokenize


# ------------------------------------------------------------------ tokenize
def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("Built ML models.") == ["built", "ml", "models", "."]


def test_tokenize_ignores_extra_whitespace():
    assert tokenize("a   b\n c") == ["a", "b", "c"]


# ----------------------------------------------------------------- normalize
def test_normalize_lemmatizes_words():
    tokens = normalize("I built models and studied results").tokens
    assert "build" in tokens
    assert "model" in tokens
    assert "study" in tokens
    assert "result" in tokens


def test_normalize_removes_stop_words_by_default():
    tokens = normalize("I built models and studied results").tokens
    assert "and" not in tokens
    assert "i" not in tokens


def test_normalize_can_keep_stop_words():
    assert "the" not in normalize("the quick fox").tokens
    assert "the" in normalize("the quick fox", remove_stopwords=False).tokens


def test_normalize_drops_punctuation_symbols_and_numbers():
    tokens = normalize("Python, SQL; 2026 - ten years!").tokens
    assert "python" in tokens
    assert "sql" in tokens
    assert "2026" not in tokens
    assert "ten" not in tokens
    # No token may be pure punctuation or symbols.
    assert all(any(ch.isalnum() for ch in tok) for tok in tokens)


def test_normalize_output_is_lowercase():
    assert normalize("Python PYTHON python").tokens == ["python", "python", "python"]


def test_normalize_text_property_joins_tokens():
    assert normalize("Python SQL").text == "python sql"


def test_normalize_empty_input():
    assert normalize("").tokens == []
    assert normalize("   \n  ").text == ""


# ------------------------------------------------------- whole sample resume
def test_sample_resume_normalizes_after_cleaning():
    raw = load_text_file(SAMPLES_DIR / "sample_resume.txt").text
    tokens = normalize(clean_text(raw).text).tokens

    assert len(tokens) > 20
    assert "python" in tokens
    assert "sql" in tokens
    assert "the" not in tokens
    assert "and" not in tokens
    assert all(tok == tok.lower() for tok in tokens)


# -------------------------------------------------------------------- model
def test_spacy_model_is_loaded_only_once():
    assert get_nlp() is get_nlp()
