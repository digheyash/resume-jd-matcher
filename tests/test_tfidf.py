"""
Tests for resume_matcher.matching.tfidf.

Tiny invented corpora are used so every expected number can be worked out by
hand. scikit-learn's smoothed IDF is  idf = ln((1 + n_docs) / (1 + df)) + 1,
where df is the number of corpus documents that contain the word.
"""
import math

import numpy as np
import pytest

from resume_matcher import config
from resume_matcher.matching.tfidf import TfidfMatcher

CORPUS = ["python pandas python", "java spring", "python sql"]  # 3 docs, 5 words


@pytest.fixture
def matcher():
    return TfidfMatcher().fit(CORPUS)


# ------------------------------------------------------------ construction
class TestConstruction:
    def test_default_max_features_comes_from_config(self):
        assert TfidfMatcher().max_features == config.TFIDF_MAX_FEATURES

    def test_custom_max_features(self):
        assert TfidfMatcher(max_features=50).max_features == 50

    def test_not_fitted_at_first(self):
        assert TfidfMatcher().is_fitted is False


# --------------------------------------------------------------- not fitted
class TestNotFitted:
    @pytest.mark.parametrize(
        "call",
        [
            lambda m: m.transform(["python"]),
            lambda m: m.similarity("python", "java"),
            lambda m: m.top_terms("python"),
            lambda m: m.vocabulary_size,
        ],
    )
    def test_using_before_fit_raises(self, call):
        with pytest.raises(RuntimeError, match="fit"):
            call(TfidfMatcher())


# ---------------------------------------------------------------------- fit
class TestFit:
    def test_fit_returns_self_and_marks_fitted(self):
        m = TfidfMatcher()
        assert m.fit(CORPUS) is m
        assert m.is_fitted is True

    def test_vocabulary_size_counts_distinct_words(self, matcher):
        assert matcher.vocabulary_size == 5  # python pandas java spring sql

    def test_accepts_any_iterable(self):
        m = TfidfMatcher().fit(text for text in CORPUS)
        assert m.vocabulary_size == 5

    @pytest.mark.parametrize("bad", [[], [""], ["", "   "]])
    def test_empty_corpus_is_rejected(self, bad):
        with pytest.raises(ValueError, match="no words|empty"):
            TfidfMatcher().fit(bad)

    def test_refit_replaces_the_vocabulary(self, matcher):
        matcher.fit(["alpha beta"])
        assert matcher.vocabulary_size == 2
        assert matcher.similarity("python", "python") == 0.0  # now unknown

    def test_max_features_limits_the_vocabulary(self):
        m = TfidfMatcher(max_features=2).fit(["a b c d a a b"])
        assert m.vocabulary_size == 2


# ---------------------------------------------------------------- transform
class TestTransform:
    def test_shape_is_texts_by_vocabulary(self, matcher):
        assert matcher.transform(["python", "java", "sql"]).shape == (3, 5)

    def test_rows_have_length_one(self, matcher):
        rows = matcher.transform(["python pandas", "java spring sql"]).toarray()
        assert np.linalg.norm(rows, axis=1) == pytest.approx([1.0, 1.0])

    def test_unknown_words_give_a_zero_row(self, matcher):
        row = matcher.transform(["zzz qqq"]).toarray()[0]
        assert not row.any()


# --------------------------------------------------------------- similarity
class TestSimilarity:
    def test_same_text_scores_one(self, matcher):
        assert matcher.similarity("python pandas", "python pandas") == pytest.approx(1.0)

    def test_no_shared_words_scores_zero(self, matcher):
        assert matcher.similarity("python pandas", "java spring") == 0.0

    def test_partial_overlap_matches_a_hand_calculation(self, matcher):
        # n=3 documents; python is in 2 of them, pandas and sql in 1 each.
        idf_python = math.log(4 / 3) + 1
        idf_rare = math.log(4 / 2) + 1  # pandas and sql
        # "python pandas" -> (idf_python, idf_rare); "python sql" likewise.
        expected = idf_python**2 / (idf_python**2 + idf_rare**2)
        assert matcher.similarity("python pandas", "python sql") == pytest.approx(expected)

    def test_symmetric(self, matcher):
        a, b = "python pandas", "python sql"
        assert matcher.similarity(a, b) == pytest.approx(matcher.similarity(b, a))

    def test_word_order_does_not_matter(self, matcher):
        assert matcher.similarity("python pandas", "pandas python") == pytest.approx(1.0)

    def test_capitalisation_does_not_matter(self, matcher):
        assert matcher.similarity("Python", "python") == pytest.approx(1.0)

    def test_text_with_only_unknown_words_scores_zero(self, matcher):
        assert matcher.similarity("zzz qqq", "python") == 0.0

    def test_returns_plain_float(self, matcher):
        assert type(matcher.similarity("python", "python sql")) is float

    def test_a_word_outside_the_corpus_is_ignored(self, matcher):
        with_extra = matcher.similarity("python pandas zzz", "python pandas")
        assert with_extra == pytest.approx(1.0)


# ---------------------------------------------------------------- tokenizer
class TestTokenizer:
    def test_whitespace_tokens_keep_symbols_whole(self):
        m = TfidfMatcher().fit(["b.tech c++ c# python", "java"])
        names = [term for term, _ in m.top_terms("b.tech c++ c# python")]
        assert set(names) == {"b.tech", "c++", "c#", "python"}

    def test_single_letter_words_are_kept(self):
        m = TfidfMatcher().fit(["r python", "java"])
        assert "r" in [term for term, _ in m.top_terms("r python")]


# ---------------------------------------------------------------------- idf
class TestIdfEffect:
    def test_rare_word_outweighs_common_word(self):
        m = TfidfMatcher().fit(["common alpha", "common beta", "common gamma"])
        weights = dict(m.top_terms("common alpha"))
        assert weights["alpha"] > weights["common"]

    def test_fitting_on_only_the_pair_flattens_idf(self):
        # The design point of this module: with just two documents, a word in
        # both gets idf 1.0 and a word in one gets ln(1.5) + 1 = about 1.405,
        # so the weights barely differ.
        m = TfidfMatcher().fit(["python pandas sql", "python java"])
        weights = dict(m.top_terms("python pandas sql"))
        ratio = weights["python"] / weights["pandas"]
        assert ratio == pytest.approx(1 / (math.log(1.5) + 1))
        assert ratio > 0.7  # flat: shared and unshared words weigh almost alike


# ---------------------------------------------------------------- top_terms
class TestTopTerms:
    def test_heaviest_first(self, matcher):
        weights = [w for _, w in matcher.top_terms("python pandas sql")]
        assert weights == sorted(weights, reverse=True)

    def test_limit_n(self, matcher):
        assert len(matcher.top_terms("python pandas sql", n=2)) == 2

    def test_n_zero_or_negative_gives_empty_list(self, matcher):
        assert matcher.top_terms("python", n=0) == []
        assert matcher.top_terms("python", n=-3) == []

    def test_unknown_words_are_not_listed(self, matcher):
        names = [term for term, _ in matcher.top_terms("python zzz")]
        assert names == ["python"]

    def test_text_with_no_known_word_gives_empty_list(self, matcher):
        assert matcher.top_terms("zzz qqq") == []

    def test_returns_plain_str_and_float(self, matcher):
        term, weight = matcher.top_terms("python")[0]
        assert type(term) is str
        assert type(weight) is float
