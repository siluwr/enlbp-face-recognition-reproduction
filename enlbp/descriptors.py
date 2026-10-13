"""LBP and three explicit interpretations of the paper's ambiguous EnLBP.

These implementations are a reconstruction, not the authors' original code.
Neighbors use p0=E, p1=NE, p2=N, p3=NW, p4=W, p5=SW, p6=S,
p7=SE, as shown in Figure 1. No padding or interpolation is performed.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _image(image: np.ndarray, *, preserve_uint8: bool = False) -> np.ndarray:
    """Validate and convert to finite, real float64 without unsigned subtraction."""
    array = np.asarray(image)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("image must be a nonempty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("image must contain real numeric values")
    if preserve_uint8 and array.dtype == np.uint8:
        # uint8 values are finite and direct comparisons need no conversion.
        return array
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("image must contain finite values representable as float64")
    return array


def _comparison(comparison: str) -> str:
    if comparison not in ("ge", "gt"):
        raise ValueError("comparison must be 'ge' (>=) or 'gt' (>)")
    return comparison


def lbp(image: np.ndarray, comparison: str = "ge", stride: int = 1) -> np.ndarray:
    """Return 8-neighbor uint8 LBP codes for valid 3x3 neighborhoods.

    ``ge`` follows Equation 1; ``gt`` follows the strict comparison in the
    EnLBP prose and reproduces Figure 2. Stride 3 samples the centers of
    nonoverlapping 3x3 blocks; incomplete bottom/right blocks are discarded.
    With any positive stride, the output shape is
    ``(ceil((H - 2) / stride), ceil((W - 2) / stride))``.
    """
    array = _image(image, preserve_uint8=True)
    comparison = _comparison(comparison)
    stride = _positive_integer(stride, "stride")
    height, width = array.shape
    if height < 3 or width < 3:
        raise ValueError("LBP requires an image of at least 3x3 pixels")

    center = array[1 : height - 1 : stride, 1 : width - 1 : stride]
    neighbors = (
        array[1 : height - 1 : stride, 2:width:stride],       # E
        array[0 : height - 2 : stride, 2:width:stride],       # NE
        array[0 : height - 2 : stride, 1 : width - 1 : stride],  # N
        array[0 : height - 2 : stride, 0 : width - 2 : stride],  # NW
        array[1 : height - 1 : stride, 0 : width - 2 : stride],  # W
        array[2:height:stride, 0 : width - 2 : stride],       # SW
        array[2:height:stride, 1 : width - 1 : stride],       # S
        array[2:height:stride, 2:width:stride],               # SE
    )
    operator = np.greater_equal if comparison == "ge" else np.greater
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        # Compare values directly: uint8 subtraction can wrap below zero.
        codes |= operator(neighbor, center).astype(np.uint8) << bit
    return codes


def block_mean(image: np.ndarray, block_size: int = 3) -> np.ndarray:
    """Mean each nonoverlapping square block, cropping bottom/right remainder.

    The output has shape ``(H // block_size, W // block_size)`` and dtype
    float64. Partial blocks are discarded instead of padded.
    """
    array = _image(image)
    block_size = _positive_integer(block_size, "block_size")
    rows = array.shape[0] // block_size
    columns = array.shape[1] // block_size
    if rows < 1 or columns < 1:
        raise ValueError("image must contain at least one complete block")
    cropped = array[: rows * block_size, : columns * block_size]
    blocks = cropped.reshape(rows, block_size, columns, block_size)
    area = block_size * block_size
    # Sum first so equal integer block sums produce exactly equal means.
    # Dividing every pixel first can introduce order-dependent roundoff that
    # would change a strict LBP comparison at an equality boundary.
    with np.errstate(over="ignore", invalid="ignore"):
        sums = np.sum(blocks, axis=(1, 3))
        means = sums / area
        overflow = ~np.isfinite(sums)
        if overflow.any():
            # Only extreme-valued blocks need the safer scaled-sum fallback.
            selected_blocks = blocks.transpose(0, 2, 1, 3)[overflow]
            means[overflow] = np.sum(selected_blocks / area, axis=(1, 2))
    if not np.isfinite(means).all():
        raise ValueError("block means could not be represented as finite float64")
    return means


def enlbp(
    image: np.ndarray,
    variant: str = "mean_pool_lbp",
    levels: int = 1,
    comparison: str = "gt",
) -> np.ndarray:
    """Return uint8 EnLBP codes under an explicitly chosen interpretation.

    ``mean_pool_lbp``: at each level, average nonoverlapping 3x3 blocks and
    compute dense valid LBP on the resulting mean grid (Section 4 prose).
    ``mean_pool_block_lbp``: at each level, average blocks and then compute
    stride-3 LBP on the mean grid, also reducing the code neighborhoods.
    ``lbp_reduce``: first compute dense LBP on the original image, then apply
    stride-3 LBP ``levels`` times (literal Algorithm 1 interpretation).

    Repeated levels act on the previous encoded image. The paper does not
    uniquely specify the variant, border policy, or number of repetitions.
    Too-small intermediate images raise ValueError with the failing level.
    """
    array = _image(image)
    levels = _positive_integer(levels, "levels")
    comparison = _comparison(comparison)
    variants = ("mean_pool_lbp", "mean_pool_block_lbp", "lbp_reduce")
    if variant not in variants:
        raise ValueError(f"variant must be one of {variants}")

    if variant == "lbp_reduce":
        array = lbp(array, comparison=comparison)
    for level in range(1, levels + 1):
        try:
            if variant == "lbp_reduce":
                array = lbp(array, comparison=comparison, stride=3)
            else:
                means = block_mean(array)
                stride = 1 if variant == "mean_pool_lbp" else 3
                array = lbp(means, comparison=comparison, stride=stride)
        except ValueError as error:
            raise ValueError(f"{variant} failed at level {level}: {error}") from error
    return array


def _grid(grid: tuple[int, int]) -> tuple[int, int]:
    try:
        if len(grid) != 2:
            raise ValueError
        rows = _positive_integer(grid[0], "grid rows")
        columns = _positive_integer(grid[1], "grid columns")
    except (TypeError, IndexError, ValueError) as error:
        raise ValueError("grid must contain two positive integers") from error
    return rows, columns


def lbp_histogram(
    image: np.ndarray,
    grid: tuple[int, int] = (8, 8),
    comparison: str = "ge",
) -> np.ndarray:
    """Concatenate row-major, per-region L1-normalized 256-bin LBP histograms.

    ``grid=(8, 8)`` means 8 rows by 8 columns of regions, an explicit choice
    where the paper's "8 X 8 regions" wording leaves room for ambiguity.
    ``numpy.array_split`` distributes unequal region sizes. Every region
    must contain at least one valid LBP code; output length is 256*rows*cols.
    """
    codes = lbp(image, comparison=comparison)
    rows, columns = _grid(grid)
    if rows > codes.shape[0] or columns > codes.shape[1]:
        raise ValueError("grid must leave at least one LBP code in every region")
    offsets, sizes = _histogram_layout(*codes.shape, rows, columns)
    # Encode (region, code) as a single integer and count every bin in one pass.
    histogram = np.bincount((offsets + codes).ravel(), minlength=rows * columns * 256)
    return (histogram.reshape(-1, 256) / sizes[:, None]).ravel()


@lru_cache(maxsize=8)
def _histogram_layout(height: int, width: int, rows: int, columns: int):
    """Cache image-independent region indices, exactly matching array_split."""
    row_sizes = np.full(rows, height // rows, dtype=np.intp)
    row_sizes[:height % rows] += 1
    column_sizes = np.full(columns, width // columns, dtype=np.intp)
    column_sizes[:width % columns] += 1
    row_ids = np.repeat(np.arange(rows, dtype=np.intp), row_sizes)
    column_ids = np.repeat(np.arange(columns, dtype=np.intp), column_sizes)
    offsets = (row_ids[:, None] * columns + column_ids[None, :]) * 256
    sizes = (row_sizes[:, None] * column_sizes[None, :]).ravel()
    offsets.setflags(write=False)
    sizes.setflags(write=False)
    return offsets, sizes


def extract_feature(
    image: np.ndarray,
    method: str = "enlbp",
    variant: str = "mean_pool_lbp",
    levels: int = 1,
    comparison: str | None = None,
    grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Return a flat float64 vector for raw/LBP/EnLBP cosine matching.

    Supported methods: ``raw``, ``lbp_raw``, ``lbp_hist``, ``enlbp``.
    With ``comparison=None``, LBP baselines use Equation 1's >= and EnLBP
    uses Figure 2's strict >. An explicit ``ge`` or ``gt`` overrides this.
    Only method-relevant options are used (e.g. grid for lbp_hist).
    """
    if method not in ("raw", "lbp_raw", "lbp_hist", "enlbp"):
        raise ValueError("method must be 'raw', 'lbp_raw', 'lbp_hist', or 'enlbp'")
    if comparison is None:
        comparison = "gt" if method == "enlbp" else "ge"
    comparison = _comparison(comparison)
    if method == "raw":
        feature = _image(image)
    elif method == "lbp_raw":
        feature = lbp(image, comparison=comparison)
    elif method == "lbp_hist":
        feature = lbp_histogram(image, grid=grid, comparison=comparison)
    else:
        feature = enlbp(image, variant=variant, levels=levels, comparison=comparison)
    return np.asarray(feature, dtype=np.float64).ravel().copy()
