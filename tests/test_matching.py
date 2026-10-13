import numpy as np
import pytest

from enlbp import cosine_nearest_neighbor, cosine_similarity


def test_cosine_similarity_hand_computed_and_zero_norm():
    queries = np.array([[1, 0], [1, 1], [0, 0]])
    gallery = np.array([[1, 0], [0, 1], [-1, 0], [0, 0]])
    half_root = 1 / np.sqrt(2)
    expected = [[1, 0, -1, 0], [half_root, half_root, -half_root, 0], [0, 0, 0, 0]]
    actual = cosine_similarity(queries, gallery)
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == np.float64
    assert actual.shape == (3, 4)


def test_nearest_neighbor_labels_scores_and_first_tie():
    predicted, scores, indices = cosine_nearest_neighbor(
        [[0, 1], [1, 1], [0, 0]], [[1, 0], [0, 1], [1, 0]], ["a", "b", "duplicate_a"]
    )
    np.testing.assert_array_equal(predicted, ["b", "a", "a"])
    np.testing.assert_array_equal(indices, [1, 0, 0])
    np.testing.assert_allclose(scores, [1, 1 / np.sqrt(2), 0])


def test_similarity_is_scale_invariant_without_large_value_overflow():
    baseline = cosine_similarity([[3, 4]], [[3, 4], [-4, 3]])
    np.testing.assert_allclose(cosine_similarity([[3e307, 4e307]], [[3e307, 4e307], [-4e307, 3e307]]), baseline, atol=1e-15)
    np.testing.assert_allclose(cosine_similarity([[3e-300, 4e-300]], [[3e-300, 4e-300], [-4e-300, 3e-300]]), baseline, atol=1e-15)


def test_zero_gallery_can_outrank_negative_similarities():
    labels, scores, indices = cosine_nearest_neighbor([[1, 0]], [[-1, 0], [0, 0]], [10, 20])
    np.testing.assert_array_equal(labels, [20])
    np.testing.assert_array_equal(scores, [0])
    np.testing.assert_array_equal(indices, [1])


@pytest.mark.parametrize("features", [[], [1, 2], np.zeros((0, 2)), np.zeros((2, 0)), [[np.nan, 1]], [[np.inf, 1]], [[1j, 0]], [["x", "y"]]])
def test_invalid_feature_matrices(features):
    with pytest.raises(ValueError):
        cosine_similarity(features, [[1, 2]])
    with pytest.raises(ValueError):
        cosine_similarity([[1, 2]], features)


def test_feature_dimensions_and_gallery_label_count_are_validated():
    with pytest.raises(ValueError, match="same feature dimension"):
        cosine_similarity([[1, 2]], [[1, 2, 3]])
    for labels in [[], ["only_one"], [["a", "b"]]]:
        with pytest.raises(ValueError, match="one label per gallery row"):
            cosine_nearest_neighbor([[1, 2]], [[1, 2], [2, 1]], labels)
