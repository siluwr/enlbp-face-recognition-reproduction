"""Frozen pre-optimization encoder; only its descriptor import is redirected."""
import numpy as np
from run_experiment import METHODS

def encode(images: list[np.ndarray], method_name: str, grid: tuple[int, int]) -> np.ndarray:
    """Build float64 feature rows, rejecting inconsistent or invalid vectors."""
    from benchmarks.reference_v1.enlbp.descriptors import extract_feature
    configuration = METHODS[method_name]
    vectors = []
    for image in images:
        if method_name == "raw":
            feature = image.astype(np.float64).ravel()
        else:
            keywords = {key: value for key, value in configuration.items() if value is not None}
            keywords["grid"] = grid
            feature = np.asarray(extract_feature(image, **keywords), dtype=np.float64).ravel()
        if feature.size == 0 or not np.isfinite(feature).all():
            raise ValueError(f"{method_name} returned an empty or nonfinite feature.")
        vectors.append(feature)
    if len({vector.size for vector in vectors}) != 1:
        raise ValueError(f"{method_name}: feature dimensions differ; specify --resize WIDTH HEIGHT.")
    return np.stack(vectors)
