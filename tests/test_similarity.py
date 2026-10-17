"""
Tests for resume_matcher.matching.similarity.

Only tiny hand-made vectors are used, so every expected value can be checked
by hand (for example (1, 0) vs (1, 1) is 45 degrees, cosine 1/sqrt(2)).
"""
import math

import numpy as np
import pytest

from resume_matcher.matching.similarity import cosine_matrix, cosine_similarity


# ---------------------------------------------------------- cosine_similarity
class TestCosineSimilarity:
    def test_same_direction_is_one(self):
        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_length_does_not_matter(self):
        assert cosine_similarity([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)

    def test_right_angle_is_zero(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_is_minus_one(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_forty_five_degrees(self):
        assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(1 / math.sqrt(2))

    def test_symmetric(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 0.5, -1.0]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_accepts_numpy_arrays_and_tuples(self):
        assert cosine_similarity(np.array([1, 1]), (1, 1)) == pytest.approx(1.0)

    def test_returns_plain_float(self):
        value = cosine_similarity([1, 2], [2, 1])
        assert type(value) is float

    def test_zero_vector_gives_zero(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0
        assert cosine_similarity([1, 1], [0, 0]) == 0.0
        assert cosine_similarity([0, 0], [0, 0]) == 0.0

    def test_result_stays_inside_minus_one_and_one(self):
        # Rounding errors could give 1.0000000000000002 without the clamp.
        v = [0.1, 0.2, 0.3, 0.7, 0.9]
        value = cosine_similarity(v, v)
        assert -1.0 <= value <= 1.0

    def test_different_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            cosine_similarity([1, 2], [1, 2, 3])

    def test_non_1d_input_is_rejected(self):
        with pytest.raises(ValueError):
            cosine_similarity([[1, 2]], [1, 2])
        with pytest.raises(ValueError):
            cosine_similarity([1, 2], [[1, 2]])


# -------------------------------------------------------------- cosine_matrix
class TestCosineMatrix:
    def test_shape_is_rows_of_a_by_rows_of_b(self):
        result = cosine_matrix([[1, 0], [0, 1]], [[1, 0], [0, 1], [1, 1]])
        assert result.shape == (2, 3)

    def test_values_on_a_small_example(self):
        result = cosine_matrix([[1, 0], [0, 1]], [[1, 0], [1, 1]])
        expected = np.array([[1.0, 1 / math.sqrt(2)], [0.0, 1 / math.sqrt(2)]])
        assert np.allclose(result, expected)

    def test_agrees_with_cosine_similarity_cell_by_cell(self):
        a = [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]
        b = [[3.0, 1.0, 0.0], [-1.0, 4.0, 2.0], [1.0, 1.0, 1.0]]
        result = cosine_matrix(a, b)
        for i, row_a in enumerate(a):
            for j, row_b in enumerate(b):
                assert result[i, j] == pytest.approx(cosine_similarity(row_a, row_b))

    def test_zero_rows_give_zero(self):
        result = cosine_matrix([[0, 0], [1, 0]], [[1, 0], [0, 0]])
        assert np.allclose(result, [[0.0, 0.0], [1.0, 0.0]])

    def test_values_stay_inside_minus_one_and_one(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(6, 8))
        result = cosine_matrix(data, data)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_diagonal_is_one_when_comparing_a_set_with_itself(self):
        data = [[1, 2], [3, 1], [-2, 5]]
        result = cosine_matrix(data, data)
        assert np.allclose(np.diag(result), 1.0)

    def test_inputs_are_not_modified(self):
        a = np.array([[3.0, 4.0]])
        b = np.array([[1.0, 0.0]])
        cosine_matrix(a, b)
        assert a.tolist() == [[3.0, 4.0]]
        assert b.tolist() == [[1.0, 0.0]]

    def test_1d_input_is_rejected(self):
        with pytest.raises(ValueError):
            cosine_matrix([1, 2], [[1, 2]])
        with pytest.raises(ValueError):
            cosine_matrix([[1, 2]], [1, 2])

    def test_different_row_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            cosine_matrix([[1, 2]], [[1, 2, 3]])
