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
    # Scaling prevents overflow in sum(x*x), even for values near 1e308.
    scale = np.max(np.abs(features), axis=1, keepdims=True)
    scaled = np.divide(features, scale, out=np.zeros_like(features), where=scale != 0)
    norm = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm != 0)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predicted_labels, best_similarities, gallery_indices).

    A tie chooses the first gallery row. In particular, zero queries choose
    row 0 with similarity 0. A zero gallery row can outrank negative cosine
    similarities because its similarity is defined as 0.
    """
    similarities = cosine_similarity(query2d, gallery2d)
    labels = np.asarray(gallery_labels)
    if labels.ndim != 1 or labels.size != similarities.shape[1]:
        raise ValueError("gallery_labels must have one label per gallery row")
    indices = np.argmax(similarities, axis=1)
    scores = similarities[np.arange(similarities.shape[0]), indices]
    return labels[indices].copy(), scores, indices
