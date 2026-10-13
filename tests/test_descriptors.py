import numpy as np
import pytest

from enlbp import block_mean, enlbp, extract_feature, lbp, lbp_histogram


def test_paper_figure_one_is_237():
    # Figure 1: center=112; bits p0..p7 = [1,0,1,1,0,1,1,1].
    # 1 + 4 + 8 + 32 + 64 + 128 = 237 (binary 11101101).
    pixels = np.array([[167, 152, 94], [77, 112, 120], [170, 143, 210]], dtype=np.uint8)
    actual = lbp(pixels)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, [[237]])


def test_paper_figure_two_requires_strict_comparison():
    # Figure 2 mean grid: center=3; > bits [0,1,1,0,1,1,1,1].
    # 2 + 4 + 16 + 32 + 64 + 128 = 246; >= adds NW's bit 8.
    means = np.array([[3, 5, 6], [4, 3, 1], [6, 7, 9]], dtype=np.float64)
    original = np.repeat(np.repeat(means, 3, axis=0), 3, axis=1)
    np.testing.assert_allclose(block_mean(original), means)
    np.testing.assert_array_equal(lbp(means, comparison="gt"), [[246]])
    np.testing.assert_array_equal(lbp(means, comparison="ge"), [[254]])
    np.testing.assert_array_equal(enlbp(original), [[246]])
    np.testing.assert_array_equal(enlbp(original, comparison="ge"), [[254]])


def test_figure_two_integer_block_mean_equality_is_exact():
    means = np.array([[3, 5, 6], [4, 3, 1], [6, 7, 9]], dtype=np.uint8)
    original = np.repeat(np.repeat(means, 3, axis=0), 3, axis=1)
    # Figure 2's NW block and the uniform central block both sum to 27.
    original[:3, :3] = [[1, 2, 3], [4, 5, 6], [1, 2, 3]]
    actual_means = block_mean(original)
    assert actual_means[0, 0] == actual_means[1, 1] == 3.0
    np.testing.assert_array_equal(actual_means, means)
    np.testing.assert_array_equal(enlbp(original), [[246]])


def test_equal_integer_block_sums_remain_equal_under_pixel_permutation():
    block = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]], dtype=np.uint8)
    # These layouts previously rounded to different means when pixels were
    # individually divided by 9 before summing.
    permuted = np.array([[1, 1, 2], [2, 3, 3], [4, 5, 6]], dtype=np.uint8)
    pixels = np.concatenate((block, permuted, np.full((3, 3), 3, dtype=np.uint8)), axis=1)
    np.testing.assert_array_equal(block_mean(pixels), [[3.0, 3.0, 3.0]])


def test_equal_neighbors_and_unsigned_input():
    np.testing.assert_array_equal(lbp(np.ones((3, 3), dtype=np.uint8)), [[255]])
    np.testing.assert_array_equal(lbp(np.ones((3, 3)), comparison="gt"), [[0]])
    # Subtracting uint8 neighbors from 255 would wrap; direct comparison must not.
    dark_neighbors = np.zeros((3, 3), dtype=np.uint8)
    dark_neighbors[1, 1] = 255
    np.testing.assert_array_equal(lbp(dark_neighbors), [[0]])
    bright_neighbors = np.full((3, 3), 255, dtype=np.uint8)
    bright_neighbors[1, 1] = 0
    np.testing.assert_array_equal(lbp(bright_neighbors), [[255]])


def test_each_neighbor_has_the_specified_bit_weight():
    positions = [(1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    for bit, position in enumerate(positions):
        pixels = np.zeros((3, 3))
        pixels[1, 1] = 1
        pixels[position] = 2
        assert lbp(pixels, comparison="gt").item() == 2**bit


def test_stride_three_matches_independent_nonoverlapping_blocks():
    pixels = np.arange(8 * 10).reshape(8, 10)
    expected = np.array(
        [[lbp(pixels[r : r + 3, c : c + 3]).item() for c in (0, 3, 6)] for r in (0, 3)],
        dtype=np.uint8,
    )
    assert lbp(pixels).shape == (6, 8)
    np.testing.assert_array_equal(lbp(pixels, stride=3), expected)


def test_block_mean_crops_remainders_and_preserves_float_means():
    pixels = np.arange(5 * 7).reshape(5, 7)
    expected = np.array([[pixels[:3, :3].mean(), pixels[:3, 3:6].mean()]])
    actual = block_mean(pixels)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected)
    pixels[3:, :] = -999
    pixels[:, 6:] = -999
    np.testing.assert_allclose(block_mean(pixels), expected)
    np.testing.assert_allclose(block_mean(np.full((3, 3), 1e308)), [[1e308]])


def test_explicit_variants_and_multiple_levels():
    pixels = np.arange(81 * 81, dtype=np.float64).reshape(81, 81)
    assert enlbp(pixels, "mean_pool_lbp").shape == (25, 25)
    assert enlbp(pixels, "mean_pool_lbp", levels=2).shape == (6, 6)
    assert enlbp(pixels, "mean_pool_block_lbp").shape == (9, 9)
    assert enlbp(pixels, "mean_pool_block_lbp", levels=2).shape == (1, 1)
    assert enlbp(pixels, "lbp_reduce").shape == (26, 26)
    assert enlbp(pixels, "lbp_reduce", levels=2).shape == (8, 8)
    np.testing.assert_array_equal(
        enlbp(pixels, "lbp_reduce"), lbp(lbp(pixels, comparison="gt"), comparison="gt", stride=3)
    )


def test_histogram_uses_all_codes_and_normalizes_each_nonempty_region():
    # Affine plane: E,S,SW,SE beat the center; NE,N,NW,W do not -> 225.
    rows, columns = np.mgrid[:7, :9]
    pixels = 3 * rows + columns
    histogram = lbp_histogram(pixels, grid=(2, 3)).reshape(6, 256)
    np.testing.assert_allclose(histogram.sum(axis=1), 1)
    np.testing.assert_allclose(histogram[:, 225], 1)
    assert np.count_nonzero(histogram) == 6
    assert histogram.dtype == np.float64


def test_feature_defaults_select_equation_one_for_lbp_and_strict_enlbp():
    pixels = np.full((9, 9), 5, dtype=np.uint8)
    assert extract_feature(pixels, "raw").shape == (81,)
    np.testing.assert_array_equal(extract_feature(pixels, "lbp_raw"), np.full(49, 255))
    np.testing.assert_array_equal(extract_feature(pixels, "lbp_raw", comparison="gt"), np.zeros(49))
    np.testing.assert_array_equal(extract_feature(pixels), [0])
    np.testing.assert_array_equal(extract_feature(pixels, comparison="ge"), [255])
    assert extract_feature(pixels, "lbp_hist", grid=(2, 2)).shape == (1024,)
    assert extract_feature(pixels).dtype == np.float64


@pytest.mark.parametrize("pixels", [np.zeros((2, 3)), np.zeros((3, 2)), [], [1, 2, 3], np.zeros((2, 2, 2)), [[np.nan]], [[np.inf]], [[1j]], [["a"]]])
def test_invalid_lbp_images_are_rejected(pixels):
    with pytest.raises(ValueError):
        lbp(pixels)


@pytest.mark.parametrize("parameter", [0, -1, 1.5, True, "3"])
def test_invalid_positive_integer_options(parameter):
    pixels = np.zeros((9, 9))
    with pytest.raises(ValueError):
        lbp(pixels, stride=parameter)
    with pytest.raises(ValueError):
        block_mean(pixels, block_size=parameter)
    with pytest.raises(ValueError):
        enlbp(pixels, levels=parameter)


def test_invalid_options_and_small_intermediates():
    with pytest.raises(ValueError, match="comparison"):
        lbp(np.zeros((3, 3)), comparison="invalid")
    with pytest.raises(ValueError, match="variant"):
        enlbp(np.zeros((9, 9)), variant="invalid")
    with pytest.raises(ValueError, match="level 1"):
        enlbp(np.zeros((8, 9)))
    with pytest.raises(ValueError, match="level 2"):
        enlbp(np.zeros((9, 9)), levels=2)
    with pytest.raises(ValueError, match="complete block"):
        block_mean(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="method"):
        extract_feature(np.zeros((9, 9)), method="invalid")
    for grid in [(0, 1), (1, 1.5), (1,), (1, 1, 1), "bad", (8, 8)]:
        with pytest.raises(ValueError):
            lbp_histogram(np.zeros((9, 9)), grid=grid)
