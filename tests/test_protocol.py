"""Meaningful safeguards for random membership and experiment reporting."""

from pathlib import Path
from contextlib import contextmanager
import shutil
import unittest
import uuid

import numpy as np
from PIL import Image

from enlbp.datasets import ImageSample, discover_images, make_split, natural_key, read_grayscale
from run_experiment import summarize_results


@contextmanager
def temporary_images():
    """Use ordinary Windows directory ACLs in sandboxed Python runtimes."""
    test_root = Path(__file__).resolve().parent
    temporary = (test_root / f"protocol_tmp_{uuid.uuid4().hex}").resolve()
    if temporary.parent != test_root:
        raise ValueError("Temporary test directory escaped its parent.")
    temporary.mkdir()
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary)


class ProtocolTests(unittest.TestCase):
    def sample_inventory(self):
        return [ImageSample(Path(f"/{label}/{number}.pgm"), f"{label}/{number}.pgm", label)
                for label in ("s1", "s2", "s10", "s20") for number in range(1, 11)]

    def test_natural_ordering(self):
        self.assertEqual(sorted(["s10", "s2", "s1"], key=natural_key), ["s1", "s2", "s10"])
        self.assertEqual(sorted(["10.pgm", "2.pgm", "1.pgm"], key=natural_key), ["1.pgm", "2.pgm", "10.pgm"])

    def test_split_reproducible_and_inventory_order_independent(self):
        samples = self.sample_inventory()
        left = make_split(samples, seed=17, subjects=3)
        right = make_split(list(reversed(samples)), seed=17, subjects=3)
        self.assertEqual(left.as_record(), right.as_record())
        self.assertNotEqual(left.train, make_split(samples, seed=18, subjects=3).train)

    def test_split_no_leakage_and_every_remaining_image_used(self):
        split = make_split(self.sample_inventory(), seed=0, subjects=2)
        train_paths = {sample.relative_path for sample in split.train}
        test_paths = {sample.relative_path for sample in split.test}
        self.assertFalse(train_paths & test_paths)
        self.assertEqual(len(train_paths), 10)
        self.assertEqual(len(test_paths), 10)
        for label in split.selected_labels:
            self.assertEqual(sum(sample.label == label for sample in split.train), 5)
            self.assertEqual(sum(sample.label == label for sample in split.test), 5)
        self.assertEqual({sample.label for sample in split.train}, {sample.label for sample in split.test})

    def test_ineligible_identity_excluded_and_impossible_subject_request_rejected(self):
        samples = self.sample_inventory() + [ImageSample(Path("/s99/1.pgm"), "s99/1.pgm", "s99")]
        split = make_split(samples, min_images=10)
        self.assertNotIn("s99", split.eligible_labels)
        with self.assertRaisesRegex(ValueError, "only 4"):
            make_split(samples, subjects=5)
        with self.assertRaisesRegex(ValueError, "No identity"):
            make_split(samples, train_per_person=10)

    def test_folder_and_flat_lfwa_loading(self):
        with temporary_images() as temporary:
            root = Path(temporary)
            (root / "s2").mkdir()
            Image.fromarray(np.arange(35, dtype=np.uint8).reshape(5, 7)).save(root / "s2" / "10.png")
            Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8)).save(root / "s2" / "2.png")
            samples = discover_images(root, "orl")
            self.assertEqual([sample.relative_path for sample in samples], ["s2/2.png", "s2/10.png"])
            image = read_grayscale(samples[0])
            self.assertEqual(image.shape, (5, 7))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(read_grayscale(samples[0], (8, 6)).shape, (6, 8))
        with temporary_images() as temporary:
            root = Path(temporary)
            Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(root / "Mary_Jane_0001.jpg")
            self.assertEqual(discover_images(root, "lfwa")[0].label, "Mary_Jane")
            with self.assertRaisesRegex(ValueError, "flat files"):
                discover_images(root, "folder")

    def test_lfwa_archive_wrappers_preserve_relative_paths(self):
        with temporary_images() as temporary:
            root = Path(temporary)
            for label in ("Person_One", "Person_Two"):
                directory = root / "lfw2" / "lfw2" / label
                directory.mkdir(parents=True)
                Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(directory / f"{label}_0001.jpg")
            samples = discover_images(root, "lfwa")
            self.assertEqual([sample.label for sample in samples], ["Person_One", "Person_Two"])
            self.assertTrue(all(sample.relative_path.startswith("lfw2/lfw2/") for sample in samples))

    def test_aggregate_sample_std_and_percent_units(self):
        base = {"method": "raw", "feature_dimension": 100, "encoding_seconds": 1.0,
                "matching_seconds": 0.1, "macro_subject_accuracy_percent": 50.0}
        single = summarize_results([{**base, "accuracy_percent": 60.0}])[0]
        self.assertIsNone(single["accuracy_percent_std"])
        pair = summarize_results([{**base, "accuracy_percent": 60.0},
                                  {**base, "accuracy_percent": 80.0}])[0]
        self.assertEqual(pair["accuracy_percent_mean"], 70.0)
        self.assertAlmostEqual(pair["accuracy_percent_std"], np.sqrt(200))
        self.assertEqual(pair["accuracy_unit"], "percent")


if __name__ == "__main__":
    unittest.main()
