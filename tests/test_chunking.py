"""
Tests for resume_matcher.matching.chunking.

These tests use a simple word counter instead of the real Sentence-BERT
tokenizer, so they are fast and need no model download.
"""
import pytest

from resume_matcher.matching.chunking import (
    DEFAULT_MAX_TOKENS,
    FULL_TEXT_KEY,
    Chunk,
    chunk_sections,
    chunk_text,
    split_units,
)


def words(text: str) -> int:
    """Test token counter: one token per whitespace-separated word."""
    return len(text.split())


# ---------------------------------------------------------------- split_units
class TestSplitUnits:
    def test_one_unit_per_line(self):
        assert split_units("alpha\nbeta\ngamma") == ["alpha", "beta", "gamma"]

    def test_blank_lines_and_spaces_are_dropped(self):
        assert split_units("  alpha  \n\n   \nbeta") == ["alpha", "beta"]

    def test_long_line_is_cut_at_sentence_ends(self):
        text = "Built a model. Deployed it! Did it work? Yes."
        assert split_units(text) == [
            "Built a model.",
            "Deployed it!",
            "Did it work?",
            "Yes.",
        ]

    def test_dot_inside_a_name_is_not_a_sentence_end(self):
        assert split_units("Used Node.js and ASP.NET daily") == [
            "Used Node.js and ASP.NET daily"
        ]

    def test_empty_text_gives_no_units(self):
        assert split_units("") == []
        assert split_units("   \n\n  ") == []


# ----------------------------------------------------------------- chunk_text
class TestChunkText:
    def test_empty_text_gives_empty_list(self):
        assert chunk_text("", words) == []
        assert chunk_text("  \n  ", words) == []

    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("one two three", words, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0].text == "one two three"
        assert chunks[0].n_tokens == 3

    def test_every_chunk_respects_the_limit(self):
        text = "\n".join(f"line {i} has five words" for i in range(20))
        chunks = chunk_text(text, words, max_tokens=12)
        assert len(chunks) > 1
        assert all(c.n_tokens <= 12 for c in chunks)

    def test_n_tokens_matches_the_counter(self):
        text = "\n".join(f"item number {i}" for i in range(10))
        for c in chunk_text(text, words, max_tokens=7):
            assert c.n_tokens == words(c.text)

    def test_no_words_are_lost_and_order_is_kept(self):
        text = "\n".join(f"word{i} extra{i}" for i in range(30))
        chunks = chunk_text(text, words, max_tokens=9)
        rebuilt = " ".join(c.text for c in chunks).split()
        assert rebuilt == text.split()

    def test_lines_stay_together_when_they_fit(self):
        text = "aaa bbb\nccc ddd\neee fff"
        chunks = chunk_text(text, words, max_tokens=4)
        assert [c.text for c in chunks] == ["aaa bbb\nccc ddd", "eee fff"]

    def test_over_long_line_is_cut_by_words(self):
        text = " ".join(f"w{i}" for i in range(25))
        chunks = chunk_text(text, words, max_tokens=10)
        assert [c.n_tokens for c in chunks] == [10, 10, 5]

    def test_over_long_sentence_inside_a_text_is_cut(self):
        text = "Short one.\n" + " ".join(["long"] * 15)
        chunks = chunk_text(text, words, max_tokens=10)
        assert all(c.n_tokens <= 10 for c in chunks)
        assert sum(c.n_tokens for c in chunks) == words(text)

    def test_one_oversized_word_is_kept_whole(self):
        # A single word cannot be split further (documented limit).
        counter = lambda s: 50 if "hugeword" in s else len(s.split())
        chunks = chunk_text("hugeword", counter, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0].text == "hugeword"

    def test_default_section_label(self):
        chunks = chunk_text("alpha beta", words)
        assert chunks[0].section == FULL_TEXT_KEY

    def test_custom_section_label(self):
        chunks = chunk_text("alpha beta", words, section="skills")
        assert chunks[0].section == "skills"

    @pytest.mark.parametrize("bad", [0, -1, -50])
    def test_max_tokens_below_one_is_rejected(self, bad):
        with pytest.raises(ValueError):
            chunk_text("alpha", words, max_tokens=bad)

    def test_default_limit_is_200(self):
        assert DEFAULT_MAX_TOKENS == 200

    def test_default_limit_is_used_when_not_given(self):
        text = " ".join(["tok"] * 450)
        chunks = chunk_text(text, words)
        assert all(c.n_tokens <= DEFAULT_MAX_TOKENS for c in chunks)
        assert sum(c.n_tokens for c in chunks) == 450

    def test_chunks_are_immutable(self):
        chunk = chunk_text("alpha", words)[0]
        assert isinstance(chunk, Chunk)
        with pytest.raises(Exception):
            chunk.text = "changed"


# -------------------------------------------------------------- chunk_sections
class TestChunkSections:
    def test_each_section_is_chunked_separately(self):
        sections = {"skills": "python sql", "experience": "built things"}
        chunks = chunk_sections(sections, words, max_tokens=50)
        assert [(c.section, c.text) for c in chunks] == [
            ("skills", "python sql"),
            ("experience", "built things"),
        ]

    def test_a_chunk_never_mixes_two_sections(self):
        # Both would fit in one chunk of 50, but must stay apart.
        sections = {"a": "one", "b": "two"}
        chunks = chunk_sections(sections, words, max_tokens=50)
        assert len(chunks) == 2

    def test_skip_sections_are_ignored(self):
        sections = {"skills": "python", "other": "noise", "education": "bsc"}
        chunks = chunk_sections(
            sections, words, max_tokens=50, skip_sections=["other"]
        )
        assert [c.section for c in chunks] == ["skills", "education"]

    def test_empty_sections_give_no_chunks(self):
        chunks = chunk_sections({"skills": "", "other": "  "}, words)
        assert chunks == []

    def test_long_section_gives_several_chunks_with_same_label(self):
        body = "\n".join(f"project {i} did something useful" for i in range(10))
        chunks = chunk_sections({"projects": body}, words, max_tokens=10)
        assert len(chunks) > 1
        assert {c.section for c in chunks} == {"projects"}
        assert all(c.n_tokens <= 10 for c in chunks)

    def test_empty_mapping(self):
        assert chunk_sections({}, words) == []

    def test_order_follows_the_mapping(self):
        sections = {"z_last": "one", "a_first": "two"}
        chunks = chunk_sections(sections, words)
        assert [c.section for c in chunks] == ["z_last", "a_first"]
