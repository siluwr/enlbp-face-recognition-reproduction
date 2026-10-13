"""Image discovery and reproducible, image-disjoint train/test membership.

The split is *within* each person: training and test identities overlap, but
image paths do not. This is closed-set face recognition, not verification or
open-set recognition. Paths are naturally sorted before random sampling so
``s2`` precedes ``s10`` and ``2.pgm`` precedes ``10.pgm``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = frozenset({".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})


def natural_key(value: str | Path) -> tuple:
    """Return an ordering key with numeric runs compared as integers."""
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", str(value)))


@dataclass(frozen=True)
class ImageSample:
    """One original image, with its directory/name-derived identity."""

    path: Path
    relative_path: str
    label: str

    def as_record(self) -> dict:
        return {"path": self.relative_path, "label": self.label}


@dataclass(frozen=True)
class DatasetSplit:
    seed: int
    selected_labels: tuple[str, ...]
    eligible_labels: tuple[str, ...]
    train: tuple[ImageSample, ...]
    test: tuple[ImageSample, ...]
    train_per_person: int
    min_images: int

    def as_record(self) -> dict:
        return {
            "seed": self.seed,
            "rng": "numpy.random.default_rng / PCG64",
            "selected_labels": list(self.selected_labels),
            "eligible_labels": list(self.eligible_labels),
            "train_per_person": self.train_per_person,
            "min_images": self.min_images,
            "train_count": len(self.train),
            "test_count": len(self.test),
            "train": [sample.as_record() for sample in self.train],
            "test": [sample.as_record() for sample in self.test],
        }


def discover_images(data_dir: str | Path, dataset: str = "folder") -> list[ImageSample]:
    """Discover images without altering, aligning, or replacing the data.

    All datasets accept immediate ``identity/image`` directories, including
    nested image subdirectories under each identity. For ``lfwa`` only, a
    flat root with ``identity_0001.jpg`` names is also supported. Each file is
    counted once; combining directory and flat layouts is rejected to avoid
    silently combining datasets. The caller must verify LFW-a provenance;
    selecting its name cannot establish that arbitrary supplied images have
    the alignment used by the paper.
    """
    if dataset not in {"orl", "lfwa", "folder"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")

    # The author's LFW-a archive contains lfw2/lfw2 wrappers. Unwrap only
    # homogeneous directory levels with no immediate images, preserving
    # relative paths from the user's original data_dir in exported records.
    image_root = root
    if dataset == "lfwa":
        while True:
            immediate_files = [p for p in image_root.iterdir()
                               if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES]
            child_directories = [p for p in image_root.iterdir() if p.is_dir()]
            if immediate_files or len(child_directories) != 1:
                break
            child = child_directories[0]
            if any(p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES for p in child.iterdir()):
                break
            if not any(p.is_dir() for p in child.iterdir()):
                break
            image_root = child

    result: list[ImageSample] = []
    for directory in sorted((p for p in image_root.iterdir() if p.is_dir()), key=natural_key):
        files = sorted((p for p in directory.rglob("*")
                        if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES), key=natural_key)
        result.extend(ImageSample(p.resolve(), p.relative_to(root).as_posix(), directory.name)
                      for p in files)

    flat = sorted((p for p in image_root.iterdir()
                   if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES), key=natural_key)
    if flat:
        if result:
            raise ValueError("Mixed flat and identity-directory image layouts are ambiguous.")
        if dataset != "lfwa":
            raise ValueError("Expected identity/image directories; flat files are supported only for lfwa.")
        for path in flat:
            match = re.fullmatch(r"(.+)_(\d+)", path.stem)
            if match is None:
                raise ValueError(f"Cannot infer LFW-a identity from {path.name!r}; expected person_0001.jpg.")
            result.append(ImageSample(path.resolve(), path.relative_to(root).as_posix(), match.group(1)))

    if not result:
        raise ValueError(f"No supported images found under {root}.")
    return sorted(result, key=lambda sample: (natural_key(sample.label), natural_key(sample.relative_path)))


def group_by_label(samples: Sequence[ImageSample]) -> dict[str, list[ImageSample]]:
    grouped: dict[str, list[ImageSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)
    return {label: sorted(grouped[label], key=lambda sample: natural_key(sample.relative_path))
            for label in sorted(grouped, key=natural_key)}


def make_split(samples: Sequence[ImageSample], *, seed: int = 0,
               train_per_person: int = 5, min_images: int = 10,
               subjects: int | None = None) -> DatasetSplit:
    """Select identities, then sample training images without replacement.

    Eligibility requires at least ``max(min_images, train_per_person + 1)``
    images. All remaining images of selected people become test images.
    The same seed is deterministic for a given sorted image inventory and
    NumPy RNG version; the exported split JSON is the authoritative record.
    Returned partitions are naturally sorted after membership is sampled.
    """
    if train_per_person < 1:
        raise ValueError("train_per_person must be positive.")
    if min_images < 1:
        raise ValueError("min_images must be positive.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    if subjects is not None and subjects < 1:
        raise ValueError("subjects must be positive or None for all.")
    grouped = group_by_label(samples)
    effective_min = max(min_images, train_per_person + 1)
    eligible = tuple(label for label, images in grouped.items() if len(images) >= effective_min)
    if not eligible:
        raise ValueError(f"No identity has at least {effective_min} images.")
    count = len(eligible) if subjects is None else subjects
    if count > len(eligible):
        raise ValueError(f"Requested {count} subjects, but only {len(eligible)} have at least {effective_min} images.")

    rng = np.random.default_rng(seed)
    if subjects is None:
        selected = eligible
    else:
        selected = tuple(sorted((str(label) for label in rng.choice(eligible, count, replace=False)), key=natural_key))
    train: list[ImageSample] = []
    test: list[ImageSample] = []
    for label in selected:
        images = grouped[label]
        permutation = rng.permutation(len(images))
        train.extend(images[int(i)] for i in permutation[:train_per_person])
        test.extend(images[int(i)] for i in permutation[train_per_person:])
    sort_key = lambda sample: (natural_key(sample.label), natural_key(sample.relative_path))
    train.sort(key=sort_key)
    test.sort(key=sort_key)
    train_paths = {sample.relative_path for sample in train}
    test_paths = {sample.relative_path for sample in test}
    if len(train_paths) != len(train) or len(test_paths) != len(test) or train_paths & test_paths:
        raise ValueError("Duplicate image paths or intersecting train/test membership.")
    return DatasetSplit(seed, selected, eligible, tuple(train), tuple(test), train_per_person, effective_min)


def read_grayscale(sample: ImageSample | str | Path,
                   resize: tuple[int, int] | None = None) -> np.ndarray:
    """Read uint8 grayscale; optional resize is (width, height), bilinear.

    Pillow's ``convert('L')`` is the only default preprocessing. There is no
    histogram equalization, automatic cropping, alignment, or augmentation.
    """
    path = sample.path if isinstance(sample, ImageSample) else Path(sample)
    with Image.open(path) as image:
        image = image.convert("L")
        if resize is not None:
            if len(resize) != 2 or min(resize) < 1:
                raise ValueError("resize must contain positive width and height.")
            image = image.resize(tuple(resize), resample=Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8).copy()


def image_manifest(samples: Sequence[ImageSample]) -> list[dict]:
    """Hash original file bytes, recording original image size and mode."""
    records: list[dict] = []
    for sample in samples:
        digest = hashlib.sha256()
        with sample.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        with Image.open(sample.path) as image:
            width, height = image.size
            mode = image.mode
        records.append({**sample.as_record(), "sha256": digest.hexdigest(),
                        "bytes": sample.path.stat().st_size,
                        "original_shape_hw": [height, width], "original_mode": mode})
    return records
