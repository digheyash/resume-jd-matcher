"""
Tests for resume_matcher.matching.sbert.

A tiny fake model replaces Sentence-BERT, so these tests are fast and need no
PyTorch and no download. The fake "embeds" a text as word counts over a small
vocabulary, which makes every expected value checkable by hand. The real model
is checked separately with a short experiment script, not in this test file.
"""
import sys
import types

import numpy as np
import pytest

from resume_matcher import config
from resume_matcher.matching.sbert import SbertEmbedder

VOCAB = ["python", "pandas", "java", "spring", "docker", "sql"]


class FakeModel:
    """Test double with the two things SbertEmbedder needs."""

    def __init__(self, max_seq_length=256):
        self.max_seq_length = max_seq_length
        self.encode_calls = []  # (texts, kwargs) for every encode call

    def tokenizer(self, text):
        # One token per word plus two special tokens, like [CLS] and [SEP].
        return {"input_ids": list(range(len(text.split()) + 2))}

    def encode(self, texts, **kwargs):
        self.encode_calls.append((list(texts), kwargs))
        return np.array(
            [[t.lower().split().count(w) for w in VOCAB] for t in texts],
            dtype=float,
        )


class NoLimitModel(FakeModel):
    """A model that does not report max_seq_length."""

    def __init__(self):
        super().__init__()
        del self.max_seq_length


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def embedder(model):
    return SbertEmbedder(model=model, max_tokens=8)


# ------------------------------------------------------------ construction
class TestConstruction:
    def test_default_model_name_comes_from_config(self):
        assert SbertEmbedder().model_name == config.SBERT_MODEL_NAME

    def test_custom_model_name(self):
        assert SbertEmbedder(model_name="my-model").model_name == "my-model"

    def test_model_is_not_loaded_by_the_constructor(self):
        assert SbertEmbedder()._model is None

    def test_given_model_is_used_as_it_is(self, model):
        assert SbertEmbedder(model=model).model is model

    def test_lazy_load_uses_sentence_transformers_on_cpu(self, monkeypatch):
        created = []

        class FakeSentenceTransformer:
            def __init__(self, name, device=None):
                created.append((name, device))

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        emb = SbertEmbedder(model_name="some-model")
        assert created == []  # nothing loaded yet
        first = emb.model
        second = emb.model
        assert created == [("some-model", "cpu")]  # loaded once only
        assert first is second


# ------------------------------------------------------------ count_tokens
class TestCountTokens:
    def test_uses_the_model_tokenizer(self, embedder):
        assert embedder.count_tokens("a b c") == 5  # 3 words + 2 special


# ------------------------------------------------------------- embed_texts
class TestEmbedTexts:
    def test_shape_is_texts_by_dimension(self, embedder):
        result = embedder.embed_texts(["python pandas", "java"])
        assert result.shape == (2, len(VOCAB))

    def test_values_come_from_the_model(self, embedder):
        result = embedder.embed_texts(["python python sql"])
        assert result[0].tolist() == [2, 0, 0, 0, 0, 1]

    def test_empty_input_gives_empty_array_and_no_model_call(self, model):
        emb = SbertEmbedder(model=model)
        assert emb.embed_texts([]).shape == (0, 0)
        assert model.encode_calls == []

    def test_accepts_any_iterable(self, embedder):
        result = embedder.embed_texts(t for t in ["python", "java"])
        assert result.shape[0] == 2

    def test_passes_batch_size_and_quiet_flags(self, model):
        emb = SbertEmbedder(model=model, batch_size=7)
        emb.embed_texts(["python"])
        _, kwargs = model.encode_calls[0]
        assert kwargs["batch_size"] == 7
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["show_progress_bar"] is False


# ------------------------------------------------------------------- chunk
class TestChunk:
    def test_every_chunk_fits_max_tokens(self, embedder):
        text = "\n".join(["python pandas sql"] * 12)
        chunks = embedder.chunk(text)
        assert len(chunks) > 1
        assert all(c.n_tokens <= 8 for c in chunks)

    def test_section_label_is_passed_on(self, embedder):
        assert embedder.chunk("python", section="skills")[0].section == "skills"

    def test_max_tokens_above_model_limit_is_rejected(self, model):
        emb = SbertEmbedder(model=model, max_tokens=300)
        with pytest.raises(ValueError, match="max_seq_length"):
            emb.chunk("python")

    def test_max_tokens_equal_to_model_limit_is_allowed(self):
        emb = SbertEmbedder(model=FakeModel(max_seq_length=10), max_tokens=10)
        assert emb.chunk("python pandas")

    def test_model_without_a_reported_limit_is_accepted(self):
        emb = SbertEmbedder(model=NoLimitModel(), max_tokens=500)
        assert emb.chunk("python")


# --------------------------------------------------------- embed_document
class TestEmbedDocument:
    def test_short_document_is_one_chunk_vector(self, embedder):
        vector = embedder.embed_document("python pandas")
        assert vector.shape == (len(VOCAB),)
        assert vector.tolist() == [1, 1, 0, 0, 0, 0]

    def test_chunks_are_averaged_weighted_by_token_count(self):
        # max_tokens=5: line 1 (3 words) = 5 tokens, line 2 (1 word) = 3 tokens.
        # Together they would be 6 tokens, so they become two chunks.
        emb = SbertEmbedder(model=FakeModel(), max_tokens=5)
        vector = emb.embed_document("python python python\njava")
        # chunk vectors: python=3 and java=1; weights 5 and 3.
        assert vector[0] == pytest.approx(3 * 5 / 8)  # 1.875 (plain mean: 1.5)
        assert vector[2] == pytest.approx(1 * 3 / 8)  # 0.375 (plain mean: 0.5)

    def test_all_chunks_are_encoded(self, model):
        emb = SbertEmbedder(model=model, max_tokens=8)
        text = "\n".join(["python pandas sql"] * 12)
        n_chunks = len(emb.chunk(text))
        model.encode_calls.clear()
        emb.embed_document(text)
        encoded_texts = [t for texts, _ in model.encode_calls for t in texts]
        assert len(encoded_texts) == n_chunks

    @pytest.mark.parametrize("empty", ["", "   ", "\n\n  \n"])
    def test_empty_text_is_rejected(self, embedder, empty):
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_document(empty)

    def test_zero_token_counts_fall_back_to_plain_average(self):
        class ZeroTokens(FakeModel):
            def tokenizer(self, text):
                return {"input_ids": []}

        emb = SbertEmbedder(model=ZeroTokens(), max_tokens=8)
        vector = emb.embed_document("python pandas")
        assert vector.tolist() == [1, 1, 0, 0, 0, 0]


# -------------------------------------------------------------- similarity
class TestSimilarity:
    def test_identical_documents_score_one(self, embedder):
        assert embedder.similarity("python pandas", "python pandas") == pytest.approx(1.0)

    def test_documents_with_no_shared_words_score_zero(self, embedder):
        assert embedder.similarity("python pandas", "java spring") == pytest.approx(0.0)

    def test_partial_overlap_is_between_zero_and_one(self, embedder):
        value = embedder.similarity("python pandas", "python java")
        assert value == pytest.approx(0.5)

    def test_symmetric(self, embedder):
        a, b = "python pandas sql", "python docker"
        assert embedder.similarity(a, b) == pytest.approx(embedder.similarity(b, a))

    def test_returns_plain_float(self, embedder):
        assert type(embedder.similarity("python", "python")) is float

    def test_empty_text_is_rejected(self, embedder):
        with pytest.raises(ValueError):
            embedder.similarity("", "python")
