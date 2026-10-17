"""
Tests for resume_matcher.matching.section_match.

A tiny fake model replaces Sentence-BERT: it "embeds" a text as word counts
over a small vocabulary, so every expected score can be worked out by hand:
    "python and pandas"  vs  "python pandas sql"
    -> vectors (1,1,0,0,0,0) and (1,1,0,0,0,1) -> 2 / (sqrt(2) * sqrt(3)) = 0.8165
No PyTorch and no download are needed.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resume_matcher.matching.sbert import SbertEmbedder
from resume_matcher.matching.section_match import (
    DEFAULT_JD_SECTIONS,
    DEFAULT_SKIP_RESUME_SECTIONS,
    RequirementMatch,
    SectionMatcher,
    SectionMatchResult,
)

VOCAB = ["python", "pandas", "java", "spring", "docker", "sql"]


class FakeModel:
    max_seq_length = 256

    def __init__(self):
        self.encode_calls = []

    def tokenizer(self, text):
        return {"input_ids": list(range(len(text.split()) + 2))}

    def encode(self, texts, **kwargs):
        self.encode_calls.append(list(texts))
        return np.array(
            [[t.lower().split().count(w) for w in VOCAB] for t in texts],
            dtype=float,
        )


RESUME = {
    "header": "python python python",  # would be a perfect hit, but is ignored
    "summary": "likes docker",
    "skills": "python pandas sql",
    "experience": "built java spring service\nused docker daily",
    "other": "pandas pandas",  # ignored
}

JD = {
    "header": "java java",
    "about": "python python",
    "responsibilities": "docker docker",
    "requirements": "python and pandas\njava spring\nsql and docker",
    "nice_to_have": "docker",
    "benefits": "python",
}


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def matcher(model):
    return SectionMatcher(SbertEmbedder(model=model, max_tokens=20))


# ------------------------------------------------------------- construction
class TestConstruction:
    def test_defaults(self, matcher):
        assert matcher.jd_sections == DEFAULT_JD_SECTIONS == ("requirements", "nice_to_have")
        assert matcher.skip_resume_sections == frozenset(DEFAULT_SKIP_RESUME_SECTIONS)
        assert set(DEFAULT_SKIP_RESUME_SECTIONS) == {"header", "other"}


# ----------------------------------------------------------------- matching
class TestMatch:
    def test_one_match_per_requirement_line_in_jd_order(self, matcher):
        result = matcher.match(RESUME, JD)
        assert [m.requirement for m in result.matches] == [
            "python and pandas",
            "java spring",
            "sql and docker",
            "docker",
        ]

    def test_jd_section_label_is_kept(self, matcher):
        result = matcher.match(RESUME, JD)
        assert [m.jd_section for m in result.matches] == [
            "requirements", "requirements", "requirements", "nice_to_have",
        ]

    def test_best_line_and_its_section_are_returned(self, matcher):
        first, second, third, fourth = matcher.match(RESUME, JD).matches
        assert (first.resume_text, first.resume_section) == ("python pandas sql", "skills")
        assert (second.resume_text, second.resume_section) == (
            "built java spring service", "experience",
        )
        assert (third.resume_text, third.resume_section) == ("likes docker", "summary")
        assert (fourth.resume_text, fourth.resume_section) == ("likes docker", "summary")

    def test_scores_match_hand_calculations(self, matcher):
        first, second, third, fourth = matcher.match(RESUME, JD).matches
        assert first.score == pytest.approx(2 / (math.sqrt(2) * math.sqrt(3)))  # 0.8165
        # "built" and "service" are not in the fake vocabulary, so the resume
        # line is (java, spring) only: the same direction as the requirement.
        assert second.score == pytest.approx(1.0)
        assert third.score == pytest.approx(1 / math.sqrt(2))  # 0.7071
        assert fourth.score == pytest.approx(1.0)

    def test_score_is_a_plain_float(self, matcher):
        assert type(matcher.match(RESUME, JD).matches[0].score) is float

    def test_header_and_other_resume_sections_are_ignored(self, matcher):
        result = matcher.match(RESUME, JD)
        used = {m.resume_section for m in result.matches}
        assert used.isdisjoint({"header", "other"})

    def test_other_jd_sections_are_not_requirements(self, matcher):
        result = matcher.match(RESUME, JD)
        texts = [m.requirement for m in result.matches]
        for ignored in ("java java", "python python", "docker docker", "python"):
            assert ignored not in texts

    def test_a_sentence_boundary_starts_a_new_requirement(self, matcher):
        jd = {"requirements": "python pandas. java spring"}
        result = matcher.match(RESUME, jd)
        assert [m.requirement for m in result.matches] == ["python pandas.", "java spring"]

    def test_an_identical_line_scores_one(self, matcher):
        result = matcher.match({"skills": "python pandas"}, {"requirements": "python pandas"})
        assert result.matches[0].score == pytest.approx(1.0)

    def test_custom_jd_sections(self, model):
        custom = SectionMatcher(SbertEmbedder(model=model, max_tokens=20),
                                jd_sections=("responsibilities",))
        result = custom.match(RESUME, JD)
        assert [m.requirement for m in result.matches] == ["docker docker"]

    def test_custom_skip_sections_can_allow_the_header(self, model):
        custom = SectionMatcher(SbertEmbedder(model=model, max_tokens=20),
                                skip_resume_sections=())
        result = custom.match({"header": "python python python"}, {"requirements": "python"})
        assert result.matches[0].resume_section == "header"


# --------------------------------------------------------------- long lines
class TestLongLines:
    def test_a_line_longer_than_the_limit_is_cut_into_several_requirements(self, model):
        # max_tokens=6 allows 4 words (the fake adds 2 special tokens).
        small = SectionMatcher(SbertEmbedder(model=model, max_tokens=6))
        jd = {"requirements": "python pandas sql docker java spring"}
        result = small.match({"skills": "python"}, jd)
        assert [m.requirement for m in result.matches] == [
            "python pandas sql docker",
            "java spring",
        ]


# --------------------------------------------------------------- edge cases
class TestListMarkers:
    """A leading list marker is removed from every line before embedding."""

    @pytest.mark.parametrize("marker", ["- ", "* ", "• ", "– ", "— ", "-  "])
    def test_marker_is_stripped_from_requirements(self, matcher, marker):
        jd = {"requirements": f"{marker}python pandas"}
        result = matcher.match({"skills": "python pandas"}, jd)
        assert result.matches[0].requirement == "python pandas"

    def test_marker_is_stripped_from_resume_lines(self, matcher):
        resume = {"projects": "- built java spring service"}
        result = matcher.match(resume, {"requirements": "java spring"})
        assert result.matches[0].resume_text == "built java spring service"

    def test_lines_holding_only_a_marker_are_dropped(self, matcher):
        jd = {"requirements": "-\n--\n- python pandas\n*"}
        result = matcher.match({"skills": "python pandas"}, jd)
        assert [m.requirement for m in result.matches] == ["python pandas"]

    def test_a_resume_section_of_only_markers_has_no_usable_lines(self, matcher):
        result = matcher.match({"skills": "-\n--"}, {"requirements": "python"})
        assert result.matches[0].score == 0.0
        assert result.matches[0].resume_text is None

    def test_a_dash_glued_to_a_word_or_number_is_kept(self, matcher):
        resume = {"projects": "-5 degrees python"}
        result = matcher.match(resume, {"requirements": "python"})
        assert result.matches[0].resume_text == "-5 degrees python"

    def test_a_dash_inside_a_line_is_kept(self, matcher):
        resume = {"projects": "python - pandas"}
        result = matcher.match(resume, {"requirements": "python"})
        assert result.matches[0].resume_text == "python - pandas"

    def test_the_marker_does_not_change_the_score(self, matcher):
        plain = matcher.match({"skills": "python sql"}, {"requirements": "python pandas"})
        marked = matcher.match({"skills": "- python sql"}, {"requirements": "- python pandas"})
        assert marked.matches[0].score == pytest.approx(plain.matches[0].score)


class TestEdgeCases:
    def test_no_requirement_lines_gives_no_matches(self, matcher, model):
        result = matcher.match(RESUME, {"about": "python", "benefits": "x"})
        assert result.matches == ()
        assert result.mean_score() is None
        assert model.encode_calls == []  # the model was never called

    def test_resume_without_usable_lines_scores_zero(self, matcher, model):
        result = matcher.match({"header": "name", "other": "x"}, JD)
        assert len(result.matches) == 4
        assert all(m.score == 0.0 for m in result.matches)
        assert all(m.resume_text is None and m.resume_section is None for m in result.matches)
        assert model.encode_calls == []

    def test_empty_inputs(self, matcher):
        assert matcher.match({}, {}).matches == ()
        assert all(m.score == 0.0 for m in matcher.match({}, JD).matches)

    def test_the_model_is_called_exactly_twice(self, matcher, model):
        matcher.match(RESUME, JD)
        assert len(model.encode_calls) == 2  # all requirements, then all resume lines


# ------------------------------------------------------------------ results
class TestResult:
    def test_for_section(self, matcher):
        result = matcher.match(RESUME, JD)
        assert len(result.for_section("requirements")) == 3
        assert len(result.for_section("nice_to_have")) == 1
        assert result.for_section("benefits") == ()

    def test_mean_score_of_must_haves_is_the_default(self, matcher):
        result = matcher.match(RESUME, JD)
        # requirement scores: 0.8165, 1.0 and 0.7071
        expected = (2 / (math.sqrt(2) * math.sqrt(3)) + 1.0 + 1 / math.sqrt(2)) / 3
        assert result.mean_score() == pytest.approx(expected)

    def test_mean_score_of_nice_to_have(self, matcher):
        assert matcher.match(RESUME, JD).mean_score("nice_to_have") == pytest.approx(1.0)

    def test_mean_score_of_a_missing_section_is_none(self, matcher):
        assert matcher.match(RESUME, JD).mean_score("responsibilities") is None

    def test_weakest_lists_lowest_scores_first(self, matcher):
        weakest = matcher.match(RESUME, JD).weakest(2)
        scores = [m.score for m in weakest]
        assert len(weakest) == 2
        assert scores == sorted(scores)
        assert weakest[0].score <= weakest[1].score <= 0.8166

    def test_weakest_can_be_limited_to_one_section(self, matcher):
        weakest = matcher.match(RESUME, JD).weakest(5, jd_section="nice_to_have")
        assert [m.requirement for m in weakest] == ["docker"]

    @pytest.mark.parametrize("n", [0, -2])
    def test_weakest_with_nothing_requested(self, matcher, n):
        assert matcher.match(RESUME, JD).weakest(n) == []

    def test_results_are_immutable(self, matcher):
        result = matcher.match(RESUME, JD)
        assert isinstance(result, SectionMatchResult)
        assert isinstance(result.matches[0], RequirementMatch)
        with pytest.raises(Exception):
            result.matches[0].score = 0.0


# -------------------------------------------------------------- documents
class TestMatchDocuments:
    def test_uses_the_sections_of_two_parsed_documents(self, matcher):
        resume_doc = SimpleNamespace(sections=RESUME)
        jd_doc = SimpleNamespace(sections=JD)
        from_documents = matcher.match_documents(resume_doc, jd_doc)
        assert from_documents == matcher.match(RESUME, JD)
