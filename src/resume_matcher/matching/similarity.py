"""
Cosine similarity for dense vectors (used with Sentence-BERT embeddings).

WHAT COSINE SIMILARITY MEASURES
    The angle between two vectors, ignoring their length.
        same direction  -> 1.0
        at right angles -> 0.0   (unrelated)
        opposite        -> -1.0
    Tiny example:  (1, 0) vs (1, 1)  ->  0.7071 (a 45 degree angle).

IMPORTANT
    A cosine value is NOT a probability or a percentage. A value of 0.6 does
    NOT mean "60% fit". Turning similarity into a score and a verdict is the
    job of Phase 5 (calibration against human ratings).

TWO FUNCTIONS
    cosine_similarity(a, b)  one number for two vectors.
    cosine_matrix(A, B)      every row of A against every row of B at once.
                             Used later to match JD requirement lines against
                             resume sentences.

EDGE CASES
    A zero vector has no direction, so its similarity is defined as 0.0
    (instead of crashing with a division by zero).
    Vectors of different lengths are a bug in the caller, so they raise
    ValueError.
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(a, b) -> float:
    """Cosine similarity of two 1-D vectors, as a plain float in [-1, 1]."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    if va.ndim != 1 or vb.ndim != 1:
        raise ValueError("cosine_similarity expects two 1-D vectors")
    if va.shape != vb.shape:
        raise ValueError(
            f"vectors must have the same length, got {va.shape[0]} and {vb.shape[0]}"
        )
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    value = float(np.dot(va, vb) / (norm_a * norm_b))
    # Rounding can give 1.0000000000000002; keep the result inside [-1, 1].
    return max(-1.0, min(1.0, value))


def cosine_matrix(rows_a, rows_b) -> np.ndarray:
    """Cosine similarity of every row in `rows_a` with every row in `rows_b`.

    Returns an array of shape (len(rows_a), len(rows_b)). Zero rows give 0.0.
    """
    a = np.asarray(rows_a, dtype=float)
    b = np.asarray(rows_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("cosine_matrix expects two 2-D arrays (rows of vectors)")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"rows must have the same length, got {a.shape[1]} and {b.shape[1]}"
        )
    norms_a = np.linalg.norm(a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True)
    # Replace zero norms by 1 so the division is safe; those rows are all
    # zeros, so their dot products (and results) are 0.0 anyway.
    unit_a = a / np.where(norms_a == 0.0, 1.0, norms_a)
    unit_b = b / np.where(norms_b == 0.0, 1.0, norms_b)
    return np.clip(unit_a @ unit_b.T, -1.0, 1.0)
