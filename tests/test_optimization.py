"""Guard semantic equivalence, batch boundaries, and reusable-index ownership."""
import numpy as np
import pytest

from enlbp import CosineIndex, cosine_nearest_neighbor, cosine_similarity, lbp, lbp_histogram
from benchmarks.reference_v1.enlbp import lbp as reference_lbp, lbp_histogram as reference_histogram


@pytest.mark.parametrize("shape,grid", [((19, 23), (3, 4)), ((112, 92), (8, 8)), ((250, 250), (8, 8))])
@pytest.mark.parametrize("comparison", ["ge", "gt"])
def test_vectorized_histogram_preserves_uneven_region_membership(shape, grid, comparison):
    pixels = np.random.default_rng(28).integers(0, 256, shape, dtype=np.uint8)
    np.testing.assert_array_equal(lbp(pixels, comparison), reference_lbp(pixels, comparison))
    np.testing.assert_array_equal(lbp_histogram(pixels, grid, comparison), reference_histogram(pixels, grid, comparison))
    # The uint8 fast path must agree with the general float64 path.
    np.testing.assert_array_equal(lbp(pixels, comparison), lbp(pixels.astype(float), comparison))


@pytest.mark.parametrize("batch_size", [1, 2, 7, 256])
def test_top1_blocks_preserve_dense_results_and_input_arrays(batch_size):
    rng = np.random.default_rng(71)
    gallery = rng.normal(size=(13, 19))
    gallery[5] = gallery[0]
    queries = np.concatenate((rng.normal(size=(21, 19)), gallery[:1], np.zeros((1, 19))))
    copies = gallery.copy(), queries.copy()
    dense = cosine_similarity(queries, gallery)
    expected = dense.argmax(axis=1)
    labels, scores, indices = cosine_nearest_neighbor(queries, gallery, np.arange(13), batch_size=batch_size)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_allclose(scores, dense[np.arange(len(queries)), expected], atol=2e-15, rtol=0)
    np.testing.assert_array_equal(gallery, copies[0])
    np.testing.assert_array_equal(queries, copies[1])


def test_reusable_index_owns_gallery_and_keeps_zero_and_negative_conventions():
    gallery = np.array([[-1., 0.], [0., 0.], [0., 1.]])
    labels = np.array([10, 20, 30])
    index = CosineIndex(gallery, labels)
    gallery[:] = 2
    labels[:] = 99
    predicted, scores, nearest = index.query([[1, 0], [0, 0], [0, 1]], batch_size=1)
    np.testing.assert_array_equal(predicted, [20, 10, 30])
    np.testing.assert_array_equal(nearest, [1, 0, 2])
    np.testing.assert_allclose(scores, [0, 0, 1])
    np.testing.assert_array_equal(index.query([[0, 1]])[0], [30])


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "256"])
def test_invalid_batch_sizes_are_rejected(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        cosine_nearest_neighbor([[1, 2]], [[1, 2]], [0], batch_size=batch_size)
