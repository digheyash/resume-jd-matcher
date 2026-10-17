"""
Sentence-BERT embeddings: turn a document into ONE vector that captures meaning.

PIPELINE FOR ONE DOCUMENT
    cleaned text
      -> chunking.chunk_text   (pieces of at most max_tokens tokens)
      -> model.encode          (one 384-number vector per piece)
      -> weighted average      (one vector for the whole document)
    Two documents are then compared with cosine similarity.

WHY AVERAGE, AND WHY WEIGHTED
    The model reads only 256 tokens at a time (measured in Phase 4), so a long
    resume must be cut into chunks. The document vector is the average of its
    chunk vectors, weighted by chunk length in tokens, so a 10-token chunk
    counts less than a 200-token one. This is a design choice; the plain
    average is the obvious alternative, and Phase 5 can compare them.

WHAT THE INPUT SHOULD BE
    CLEANED text (ParsedDocument.text), not the lemmatised text used for
    TF-IDF. Sentence-BERT was trained on normal sentences.

LOADING THE MODEL
    * The model loads lazily, on first use, so importing this module is
      instant and does not import PyTorch.
    * A ready-made model (or a test double with .encode and .tokenizer) can
      be passed in. The tests do this, so they need no download.
    * The first real run downloads about 90 MB, then it is cached.

THE NUMBER THIS GIVES IS NOT A PROBABILITY
    A cosine similarity of 0.6 does not mean "60% fit". Calibration into a
    score and verdict is Phase 5.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from resume_matcher import config
from resume_matcher.matching.chunking import DEFAULT_MAX_TOKENS, Chunk, chunk_text
from resume_matcher.matching.similarity import cosine_similarity


class SbertEmbedder:
    """Embeds texts and whole documents with a Sentence-BERT model.

    Args:
        model_name: Hugging Face model name. Defaults to
            config.SBERT_MODEL_NAME.
        max_tokens: chunk size in tokens (default 200, below the model's 256).
        model: an already-loaded model, or a test double. It needs
            `.encode(list_of_texts, ...)` and `.tokenizer(text)["input_ids"]`.
        batch_size: how many chunks are encoded at once.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model=None,
        batch_size: int = 32,
    ):
        self.model_name = model_name or config.SBERT_MODEL_NAME
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self._model = model

    # ------------------------------------------------------------ the model
    @property
    def model(self):
        """The Sentence-BERT model, loaded on first access (CPU)."""
        if self._model is None:
            # Imported here so that importing this module stays fast.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def count_tokens(self, text: str) -> int:
        """Token count by the model's own tokenizer (special tokens included)."""
        return len(self.model.tokenizer(text)["input_ids"])

    # ------------------------------------------------------------- chunking
    def chunk(self, text: str, section: str = "full_text") -> list[Chunk]:
        """Split `text` into chunks that fit the model.

        Raises:
            ValueError: if max_tokens is larger than the model can read, since
                the model would then silently cut the chunks.
        """
        limit = getattr(self.model, "max_seq_length", None)
        if limit is not None and self.max_tokens > limit:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) is larger than the model's "
                f"max_seq_length ({limit}); text would be silently cut off"
            )
        return chunk_text(text, self.count_tokens, self.max_tokens, section=section)

    # ----------------------------------------------------------- embeddings
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """One vector per text, as an array of shape (n_texts, dimension).

        The texts are encoded as they are: no chunking. The caller must make
        sure each one fits the model (see `chunk`). An empty input gives an
        empty array of shape (0, 0).
        """
        texts = list(texts)
        if not texts:
            return np.empty((0, 0))
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=float)

    def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        """One vector per chunk, in the same order."""
        return self.embed_texts([c.text for c in chunks])

    def embed_document(self, text: str) -> np.ndarray:
        """One vector for a whole document of any length.

        Raises:
            ValueError: if the text is empty (there is nothing to embed).
        """
        chunks = self.chunk(text)
        if not chunks:
            raise ValueError("cannot embed an empty text")
        vectors = self.embed_chunks(chunks)
        weights = np.array([c.n_tokens for c in chunks], dtype=float)
        if weights.sum() == 0.0:
            weights = np.ones_like(weights)
        return np.average(vectors, axis=0, weights=weights)

    # ----------------------------------------------------------- similarity
    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity of two whole documents (not a probability)."""
        return cosine_similarity(
            self.embed_document(text_a), self.embed_document(text_b)
        )
