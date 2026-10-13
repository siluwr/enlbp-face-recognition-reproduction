"""Cosine nearest-neighbor matching for explicitly extracted feature vectors."""

from __future__ import annotations

import numpy as np


def _features(features: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(features)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional feature matrix")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values representable as float64")
    return array


def _normalized_rows(features: np.ndarray) -> np.ndarray:
    # One owned matrix, in-place division, and a row reduction without x*x.
    # Scaling still protects extreme finite inputs and leaves zero rows zero.
    normalized = np.array(features, dtype=np.float64, copy=True)
    scale = np.maximum(np.max(normalized, axis=1), -np.min(normalized, axis=1))[:, None]
    np.divide(normalized, scale, out=normalized, where=scale != 0)
    norm = np.sqrt(np.einsum("ij,ij->i", normalized, normalized))[:, None]
    np.divide(normalized, norm, out=normalized, where=norm != 0)
    return normalized


def cosine_similarity(query2d: np.ndarray, gallery2d: np.ndarray) -> np.ndarray:
    """Return float64 (number_of_queries, number_of_gallery_images) similarity.

    Zero vectors have similarity 0 to every vector, including other zeros.
    This convention avoids NaN; it does not imply that two zero descriptors
    represent a face match. Normalized dot products are clipped to [-1, 1].
    """
    queries = _features(query2d, "query2d")
    gallery = _features(gallery2d, "gallery2d")
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query2d and gallery2d must have the same feature dimension")
    return np.clip(_normalized_rows(queries) @ _normalized_rows(gallery).T, -1.0, 1.0)


def cosine_nearest_neighbor(
    query2d: np.ndarray,
    gallery2d: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predicted_labels, best_similarities, gallery_indices).

    A tie chooses the first gallery row. In particular, zero queries choose
    row 0 with similarity 0. A zero gallery row can outrank negative cosine
    similarities because its similarity is defined as 0.
    """
    return CosineIndex(gallery2d, gallery_labels).query(query2d, batch_size=batch_size)


class CosineIndex:
    """Reusable, owned gallery normalization for exact cosine top-1 search.

    Query blocks bound temporary memory to O(batch_size * (dimension +
    gallery_count)), instead of normalizing every query at once. Ties still
    choose the first gallery row. Input arrays are never mutated or retained.
    This index performs no approximate search or float32 conversion.
    """

    def __init__(self, gallery2d: np.ndarray, gallery_labels: np.ndarray):
        gallery = _features(gallery2d, "gallery2d")
        labels = np.asarray(gallery_labels)
        if labels.ndim != 1 or labels.size != gallery.shape[0]:
            raise ValueError("gallery_labels must have one label per gallery row")
        self._gallery = _normalized_rows(gallery)
        self._labels = labels.copy()
        self._gallery.setflags(write=False)
        self._labels.setflags(write=False)

    def query(self, query2d: np.ndarray, *, batch_size: int = 256):
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        queries = _features(query2d, "query2d")
        if queries.shape[1] != self._gallery.shape[1]:
            raise ValueError("query2d and gallery2d must have the same feature dimension")
        indices = np.empty(len(queries), dtype=np.intp)
        scores = np.empty(len(queries), dtype=np.float64)
        for start in range(0, len(queries), batch_size):
            stop = min(start + batch_size, len(queries))
            similarities = _normalized_rows(queries[start:stop]) @ self._gallery.T
            np.clip(similarities, -1.0, 1.0, out=similarities)
            local = np.argmax(similarities, axis=1)
            indices[start:stop] = local
            scores[start:stop] = similarities[np.arange(stop - start), local]
        return self._labels[indices].copy(), scores, indices
