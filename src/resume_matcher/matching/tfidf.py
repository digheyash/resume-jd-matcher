"""
TF-IDF baseline: compare two texts by the WORDS they share.

THE IDEA, WITH A TINY EXAMPLE
    Corpus:  doc1 = "python pandas python"   doc2 = "java spring"   doc3 = "python sql"
    TF  (term frequency)          how often a word appears in ONE text.
    IDF (inverse doc. frequency)  how RARE the word is across the corpus:
                                  "python" is in 2 of 3 documents, "spring" in
                                  1 of 3, so "spring" gets the higher IDF.
    TF-IDF = TF x IDF. Common filler words get a low weight, rare and
    informative words get a high one. Every text becomes a vector with one
    number per word of the corpus, and two texts are compared by cosine
    similarity of those vectors.

    TF-IDF only knows WORDS. "ML" and "machine learning" are unrelated to it.
    That is exactly why it is the baseline to compare Sentence-BERT against.

WHY FIT AND TRANSFORM ARE KEPT SEPARATE (important design point)
    IDF compares a word with OTHER documents. If we fitted on just one resume
    and one JD, every word would be in 1 or 2 of 2 documents and IDF would
    carry almost no information (scikit-learn's IDF would range only from 1.0
    to about 1.4). So:
        fit(corpus)      learn the vocabulary and IDF from a REFERENCE corpus
                         (for example all resumes and JDs we have);
        similarity(a,b)  then only TRANSFORMS the two texts with that
                         frozen vocabulary.
    Words that are not in the corpus are ignored. A text with no known word
    gives the zero vector, and its similarity is 0.0.

INPUT
    ParsedDocument.normalized_text() from Phase 2: lowercased lemmas, stop
    words and punctuation already removed. Because that text is already
    tokenised, we split on whitespace only (token_pattern \\S+). The default
    scikit-learn tokenizer would turn "b.tech" into "tech".

THE NUMBER IS NOT A PROBABILITY
    A cosine of 0.3 is not "30% fit". Calibration is Phase 5.
"""
from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from resume_matcher import config
from resume_matcher.matching.similarity import cosine_similarity


class TfidfMatcher:
    """TF-IDF vectors with a vocabulary learned once from a reference corpus.

    Args:
        max_features: keep only the most frequent words of the corpus.
            Defaults to config.TFIDF_MAX_FEATURES (a PLACEHOLDER value).
            With a small corpus the vocabulary is far below this limit, so
            the limit has no effect.
    """

    def __init__(self, max_features: int | None = None):
        self.max_features = (
            config.TFIDF_MAX_FEATURES if max_features is None else max_features
        )
        self._vectorizer: TfidfVectorizer | None = None

    # ------------------------------------------------------------ fitting
    def fit(self, corpus: Iterable[str]) -> "TfidfMatcher":
        """Learn vocabulary and IDF from the reference corpus. Returns self.

        Raises:
            ValueError: if the corpus has no words at all.
        """
        texts = list(corpus)
        vectorizer = TfidfVectorizer(
            token_pattern=r"\S+",
            max_features=self.max_features,
        )
        try:
            vectorizer.fit(texts)
        except ValueError as error:
            raise ValueError(
                "cannot fit TF-IDF: the corpus is empty or has no words"
            ) from error
        self._vectorizer = vectorizer
        return self

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None

    def _require_fitted(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            raise RuntimeError("TfidfMatcher is not fitted: call fit(corpus) first")
        return self._vectorizer

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct words learned from the corpus."""
        return len(self._require_fitted().vocabulary_)

    # ---------------------------------------------------------- vectors
    def transform(self, texts: Iterable[str]):
        """TF-IDF vectors for the texts, as a sparse matrix (n_texts, vocab).

        Each row has length 1 (L2-normalised). Unknown words are ignored.
        """
        return self._require_fitted().transform(list(texts))

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity of two texts in the fitted TF-IDF space."""
        matrix = self.transform([text_a, text_b]).toarray()
        return cosine_similarity(matrix[0], matrix[1])

    # ----------------------------------------------------- explainability
    def top_terms(self, text: str, n: int = 10) -> list[tuple[str, float]]:
        """The `n` highest-weighted words of a text, heaviest first.

        Useful to explain a score: these are the words the text is "about"
        in this corpus. Words with weight 0 are never returned.
        """
        vectorizer = self._require_fitted()
        row = vectorizer.transform([text]).toarray()[0]
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(
            ((str(terms[i]), float(row[i])) for i in range(len(terms)) if row[i] > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[: max(n, 0)]
