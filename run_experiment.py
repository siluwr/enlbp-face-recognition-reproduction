"""Reproducible EN-LBP paper experiments on supplied face images.

Example::

    python run_experiment.py --dataset orl --data-dir data/orl \
        --output-dir results/orl --seeds 0 1 2 3 4

This is a closed-set nearest-neighbor recognition experiment. Every method
uses exactly the same exported membership for each seed. No paper results
are injected into measured outputs. The method names distinguish plausible
interpretations of the paper's underspecified enhancement algorithm.

Default preprocessing is grayscale only. Image reading and file hashing are
excluded from encoding/matching timing. Encoding includes extracting both
gallery and query features, and matching uses the core cosine implementation.
Timing is measured once per method and seed, without warm-up or repetitions;
it is machine-specific and cannot directly verify the paper's timing claims.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import PIL

from enlbp.datasets import discover_images, group_by_label, image_manifest, make_split, read_grayscale


METHODS = {
    "raw": {"method": "raw", "variant": None, "levels": 0, "comparison": None},
    "lbp_raw": {"method": "lbp_raw", "variant": None, "levels": 0, "comparison": "ge"},
    "lbp_hist": {"method": "lbp_hist", "variant": None, "levels": 0, "comparison": "ge"},
    "enlbp_mean_gt": {"method": "enlbp", "variant": "mean_pool_lbp", "levels": 1, "comparison": "gt"},
    "enlbp_mean_ge": {"method": "enlbp", "variant": "mean_pool_lbp", "levels": 1, "comparison": "ge"},
    "enlbp_mean_l2": {"method": "enlbp", "variant": "mean_pool_lbp", "levels": 2, "comparison": "gt"},
    "enlbp_mean_block": {"method": "enlbp", "variant": "mean_pool_block_lbp", "levels": 1, "comparison": "gt"},
    "enlbp_algorithm": {"method": "enlbp", "variant": "lbp_reduce", "levels": 1, "comparison": "ge"},
    "enlbp_algorithm_l2": {"method": "enlbp", "variant": "lbp_reduce", "levels": 2, "comparison": "ge"},
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results: list[dict]) -> list[dict]:
    """Average seeds; sample SD uses ddof=1 (None for a single seed).

    Accuracy is reported in percent throughout JSON and CSV. Macro accuracy
    is the mean of per-identity test accuracies within a seed. Seed-averaged
    accuracy is an unweighted mean; it is not a pooled-image estimate.
    """
    summaries = []
    for method in dict.fromkeys(row["method"] for row in results):
        selected = [row for row in results if row["method"] == method]
        row = {"method": method, "seed_count": len(selected), "accuracy_unit": "percent",
               "timing_unit": "seconds", "std_ddof": 1,
               "feature_dimensions": ";".join(str(value) for value in sorted({r["feature_dimension"] for r in selected}))}
        for metric in ("accuracy_percent", "macro_subject_accuracy_percent", "encoding_seconds", "matching_seconds"):
            values = np.asarray([item[metric] for item in selected], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summaries.append(row)
    return summaries


def encode(images: list[np.ndarray], method_name: str, grid: tuple[int, int]) -> np.ndarray:
    """Build float64 feature rows, rejecting inconsistent or invalid vectors."""
    from enlbp.descriptors import extract_feature
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


def parse_subjects(value: str) -> int | None:
    if value.casefold() == "all":
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("subjects must be a positive integer or 'all'.") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("subjects must be positive.")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("orl", "lfwa", "folder"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-per-person", type=int, default=5)
    parser.add_argument("--min-images", type=int, help="Default: ORL=10, LFW-a=11, folder=train+1.")
    parser.add_argument("--subjects", type=parse_subjects, default=argparse.SUPPRESS,
                        help="Positive count or all. Default: LFW-a=14; other datasets=all.")
    parser.add_argument("--methods", choices=list(METHODS), nargs="+", default=list(METHODS))
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Optional bilinear resize; default keeps supplied dimensions.")
    parser.add_argument("--grid", type=int, nargs=2, default=[8, 8], metavar=("ROWS", "COLS"),
                        help="LBP histogram grid (default: 8 x 8).")
    return parser


def run(args: argparse.Namespace) -> dict:
    """Run all configured methods/seeds and persist auditable measured data.

    The original-file SHA256 manifest covers the complete discovered dataset.
    The output directory is expected to identify one experiment; files with
    matching names are overwritten if a run is repeated there.
    """
    from enlbp.matching import cosine_nearest_neighbor

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique so aggregation does not double-count a split.")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Methods must be unique.")
    if min(args.grid) < 1:
        raise ValueError("Grid sizes must be positive.")
    if args.resize is not None and min(args.resize) < 1:
        raise ValueError("Resize dimensions must be positive.")
    minimum = args.min_images
    if minimum is None:
        minimum = {"orl": 10, "lfwa": 11, "folder": args.train_per_person + 1}[args.dataset]
    subjects = getattr(args, "subjects", 14 if args.dataset == "lfwa" else None)
    resize = tuple(args.resize) if args.resize is not None else None
    grid = tuple(args.grid)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    samples = discover_images(data_dir, args.dataset)
    grouped = group_by_label(samples)
    splits = [make_split(samples, seed=seed, train_per_person=args.train_per_person,
                         min_images=minimum, subjects=subjects) for seed in args.seeds]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = image_manifest(samples)
    write_json(output_dir / "image_manifest.json", manifest)

    environment = {"python": sys.version, "numpy": np.__version__, "Pillow": PIL.__version__,
                   "platform": platform.platform(), "machine": platform.machine(),
                   "processor": platform.processor()}
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "data_dir": data_dir.name, "environment": environment,
        "seeds": args.seeds, "subjects_requested": subjects if subjects is not None else "all",
        "train_per_person": args.train_per_person, "min_images_requested": minimum,
        "min_images_effective": max(minimum, args.train_per_person + 1),
        "discovered_image_count": len(samples), "discovered_subject_count": len(grouped),
        "images_per_subject": {label: len(images) for label, images in grouped.items()},
        "methods": {method: METHODS[method] for method in args.methods},
        "preprocessing": {"grayscale": "Pillow Image.convert('L')", "storage_dtype": "uint8",
                          "resize_wh": list(resize) if resize is not None else None,
                          "resize_resampling": "bilinear" if resize is not None else None,
                          "histogram_equalization": False, "crop": None,
                          "automatic_alignment": False, "lbp_hist_grid_rc": list(grid)},
        "units": {"accuracy": "percent", "timing": "seconds", "dimension": "scalar feature entries"},
        "explicit_assumptions": [
            "Identity is the directory under the image root, or the prefix before the trailing numeric suffix in flat LFW-a files. Homogeneous LFW-a archive wrapper directories are automatically traversed.",
            "LFW-a provenance must be established separately; supplied data are never substituted or automatically aligned.",
            "Random persons and train images are selected without replacement using NumPy default_rng/PCG64; the paper does not provide its exact random memberships.",
            "All remaining images of each selected identity are test images; each identity has at least one test image.",
            "Each method shares the same split within a seed; image-file SHA256 and split paths are exported.",
            "Reported means average seed accuracies equally; sample standard deviations use ddof=1 and are null for one seed.",
            "Named enhancement variants are explicit interpretations of underspecified paper operations, not confirmed author source code.",
            "Encoding times include gallery and query extraction; disk I/O, image decoding, hashing and preprocessing are excluded.",
            "Times are single wall-clock measurements per method/seed without warm-up, and depend on hardware and implementation.",
            "No original-paper table entry is used as a measured result.",
        ],
    }
    write_json(output_dir / "experiment.json", metadata)
    records: list[dict] = []
    predictions: list[dict] = []
    for split in splits:
        write_json(output_dir / f"split_seed_{split.seed}.json", split.as_record())
        selected_samples = list(split.train) + list(split.test)
        # Bound memory to one seed's selected people, rather than retaining
        # the full 13,233-image LFW-a inventory or a growing cross-seed union.
        images = [read_grayscale(sample, resize) for sample in selected_samples]
        shapes = sorted({tuple(image.shape) for image in images})
        if len(shapes) != 1:
            raise ValueError(f"Selected image dimensions differ: {shapes}. Specify --resize WIDTH HEIGHT.")
        training_count = len(split.train)
        true_labels = np.asarray([sample.label for sample in split.test])
        gallery_labels = np.asarray([sample.label for sample in split.train])
        for method in args.methods:
            start = perf_counter()
            features = encode(images, method, grid)
            encoding_seconds = perf_counter() - start
            gallery = features[:training_count]
            queries = features[training_count:]
            start = perf_counter()
            pred_labels, scores, nearest = cosine_nearest_neighbor(queries, gallery, gallery_labels)
            matching_seconds = perf_counter() - start
            pred_labels = np.asarray(pred_labels)
            scores = np.asarray(scores)
            nearest = np.asarray(nearest, dtype=int)
            if scores.ndim == 2:
                scores = scores[np.arange(len(nearest)), nearest]
            if pred_labels.shape != true_labels.shape or scores.shape != true_labels.shape or nearest.shape != true_labels.shape:
                raise ValueError("Cosine matcher must return one prediction, score and index per query.")
            correct_mask = pred_labels == true_labels
            subject_results = {
                label: {"correct": int(correct_mask[true_labels == label].sum()),
                        "test_count": int((true_labels == label).sum()),
                        "accuracy_percent": float(100 * correct_mask[true_labels == label].mean())}
                for label in split.selected_labels
            }
            record = {
                "method": method, "seed": split.seed, "subject_count": len(split.selected_labels),
                "train_count": training_count, "test_count": len(split.test),
                "correct": int(correct_mask.sum()), "accuracy_percent": float(100 * correct_mask.mean()),
                "macro_subject_accuracy_percent": float(np.mean([value["accuracy_percent"] for value in subject_results.values()])),
                "feature_dimension": int(features.shape[1]), "image_shape_hw": list(shapes[0]),
                "gallery_feature_shape": list(gallery.shape), "query_feature_shape": list(queries.shape),
                "encoding_seconds": encoding_seconds, "matching_seconds": matching_seconds,
                "subject_results": subject_results,
            }
            records.append(record)
            for i, sample in enumerate(split.test):
                predictions.append({"method": method, "seed": split.seed, "test_path": sample.relative_path,
                                    "true_label": sample.label, "predicted_label": str(pred_labels[i]),
                                    "correct": int(correct_mask[i]), "cosine_top_score": float(scores[i]),
                                    "matched_train_path": split.train[int(nearest[i])].relative_path})
            print(f"seed={split.seed} {method:20s} {record['correct']:3d}/{record['test_count']} "
                  f"accuracy={record['accuracy_percent']:.2f}% dim={record['feature_dimension']} "
                  f"encode={encoding_seconds:.4f}s match={matching_seconds:.4f}s", flush=True)
    aggregate = summarize_results(records)
    write_json(output_dir / "results.json", records)
    flat_results = [{key: value for key, value in row.items() if key != "subject_results"} for row in records]
    write_csv(output_dir / "results.csv", flat_results)
    write_json(output_dir / "aggregate.json", aggregate)
    write_csv(output_dir / "aggregate.csv", aggregate)
    write_csv(output_dir / "predictions.csv", predictions)
    print(f"Saved measured results and split/manifest records to {output_dir}", flush=True)
    return {"metadata": metadata, "results": records, "aggregate": aggregate}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
