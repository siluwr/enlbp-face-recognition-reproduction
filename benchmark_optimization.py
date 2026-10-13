"""Paired before/after benchmark with a frozen reference and exported splits.

One warmup and alternating implementation order limit order effects. Timing
uses medians of repeated operations; allocation peaks are measured separately
with tracemalloc and are not process RSS. Disk I/O is outside measured regions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import tracemalloc

# Establish a common BLAS thread budget before importing NumPy.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from PIL import Image
from benchmarks.reference_v1.encode import encode as reference_encode
from benchmarks.reference_v1.enlbp.matching import cosine_nearest_neighbor as reference_match
from enlbp.matching import CosineIndex, cosine_nearest_neighbor
from run_experiment import encode, write_csv, write_json

METHODS = ("lbp_raw", "lbp_hist", "enlbp_mean_gt", "enlbp_mean_ge")


def operation_peak(function):
    gc.collect()
    tracemalloc.start()
    result = function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return peak


def paired_times(functions, repeats, alternate):
    # Warm up each implementation, then alternate its position each repeat.
    for function in functions:
        result = function()
        del result
    times = [[], []]
    for repeat in range(repeats):
        order = (0, 1) if (repeat + alternate) % 2 == 0 else (1, 0)
        for version in order:
            start = perf_counter()
            result = functions[version]()
            elapsed = perf_counter() - start
            times[version].append(elapsed)
            del result
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optimization"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("At least two repeats and unique seeds are required.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a new, empty output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, timings, hashes, comparisons, reuse_records = [], [], {}, [], []
    for dataset in ("orl", "lfwa"):
        data = Path("data") / dataset
        experiments = Path("outputs") / ("lbp_enlbp_" + dataset)
        for seed in args.seeds:
            split_path = experiments / f"split_seed_{seed}.json"
            hashes[split_path.as_posix()] = hashlib.sha256(split_path.read_bytes()).hexdigest()
            split = json.loads(split_path.read_text(encoding="utf-8"))
            samples = split["train"] + split["test"]
            images = []
            for sample in samples:
                with Image.open(data / sample["path"]) as image:
                    images.append(np.asarray(image.convert("L")))
            n_train = len(split["train"])
            labels = np.asarray([s["label"] for s in split["train"]])
            true_labels = np.asarray([s["label"] for s in split["test"]])
            for method in METHODS:
                before = reference_encode(images, method, (8, 8))
                features = encode(images, method, (8, 8))
                exact_features = np.array_equal(before, features)
                if not exact_features:
                    raise AssertionError(f"Feature mismatch: {dataset}/{seed}/{method}")
                del before
                gallery, queries = features[:n_train], features[n_train:]
                old_prediction = reference_match(queries, gallery, labels)
                new_prediction = cosine_nearest_neighbor(queries, gallery, labels)
                labels_equal = np.array_equal(old_prediction[0], new_prediction[0])
                indices_equal = np.array_equal(old_prediction[2], new_prediction[2])
                score_error = float(np.max(np.abs(old_prediction[1] - new_prediction[1])))
                if not labels_equal or not indices_equal or score_error > 1e-12:
                    raise AssertionError(f"Prediction mismatch: {dataset}/{seed}/{method}, {score_error}")
                comparisons.append({"dataset": dataset, "seed": seed, "method": method,
                                    "image_count": len(images), "query_count": len(queries),
                                    "features_bitwise_equal": exact_features, "labels_equal": labels_equal,
                                    "indices_equal": indices_equal, "max_score_abs_difference": score_error})
                encoders = (lambda: reference_encode(images, method, (8, 8)), lambda: encode(images, method, (8, 8)))
                matchers = (lambda: reference_match(queries, gallery, labels), lambda: cosine_nearest_neighbor(queries, gallery, labels))
                encoding_times = paired_times(encoders, args.repeats, seed)
                matching_times = paired_times(matchers, args.repeats, seed)
                for version, name in enumerate(("before", "after")):
                    row = {"dataset": dataset, "seed": seed, "method": method, "version": name,
                           "train_count": n_train, "query_count": len(queries), "image_count": len(images),
                           "feature_dimension": features.shape[1],
                           "accuracy_percent": 100 * float(np.mean(new_prediction[0] == true_labels)),
                           "encoding_seconds_median": float(np.median(encoding_times[version])),
                           "matching_seconds_median": float(np.median(matching_times[version])),
                           "encoding_peak_bytes": operation_peak(encoders[version]),
                           "matching_peak_bytes": operation_peak(matchers[version])}
                    row["compute_seconds"] = row["encoding_seconds_median"] + row["matching_seconds_median"]
                    records.append(row)
                    for operation, measurements in (("encoding", encoding_times), ("matching", matching_times)):
                        for repeat, seconds in enumerate(measurements[version]):
                            timings.append({"dataset": dataset, "seed": seed, "method": method,
                                            "version": name, "operation": operation, "repeat": repeat, "seconds": seconds})
                if seed == args.seeds[0] and method == "enlbp_mean_gt":
                    chunks = [queries[i:i+16] for i in range(0, len(queries), 16)]
                    def rebuild_each_request():
                        return [cosine_nearest_neighbor(chunk, gallery, labels) for chunk in chunks]
                    def reuse_gallery():
                        index = CosineIndex(gallery, labels)
                        return [index.query(chunk) for chunk in chunks]
                    reuse_times = paired_times((rebuild_each_request, reuse_gallery), args.repeats, seed)
                    reuse_records.append({"dataset": dataset, "seed": seed, "method": method,
                                          "query_chunk_size": 16, "request_count": len(chunks),
                                          "rebuild_seconds_median": float(np.median(reuse_times[0])),
                                          "reuse_seconds_median": float(np.median(reuse_times[1])),
                                          "reuse_includes_one_index_build": True})
                print(f"{dataset} seed={seed} {method}: features/predictions agree; compute {records[-2]['compute_seconds'] / records[-1]['compute_seconds']:.2f}x", flush=True)
                del features, gallery, queries, encoders, matchers
            del images
    summary = []
    for dataset in ("orl", "lfwa"):
        for method in METHODS:
            old = [r for r in records if r["dataset"] == dataset and r["method"] == method and r["version"] == "before"]
            new = [r for r in records if r["dataset"] == dataset and r["method"] == method and r["version"] == "after"]
            entry = {"dataset": dataset, "method": method, "seed_count": len(old), "repeats_per_seed": args.repeats,
                     "accuracy_percent": float(np.mean([r["accuracy_percent"] for r in new])),
                     "feature_dimension": new[0]["feature_dimension"]}
            for metric in ("encoding_seconds_median", "matching_seconds_median", "compute_seconds", "encoding_peak_bytes", "matching_peak_bytes"):
                for version, rows in (("before", old), ("after", new)):
                    entry[metric + "_" + version] = float(np.mean([r[metric] for r in rows]))
                entry[metric + "_ratio"] = entry[metric + "_before"] / entry[metric + "_after"]
            summary.append(entry)
    source_paths = [Path(__file__), Path("run_experiment.py"), *Path("enlbp").glob("*.py"),
                    Path("benchmarks/reference_v1/encode.py"), *Path("benchmarks/reference_v1/enlbp").glob("*.py")]
    manifest = {"date_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "numpy": np.__version__, "platform": platform.platform(), "processor": platform.processor(),
                "threads": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
                "seeds": args.seeds, "repeats": args.repeats, "warmup": 1,
                "timing": "alternating implementation order; per-seed median of repeats, then unweighted mean of seed medians",
                "compute_seconds": "sum of encoding median and matching median; excludes I/O and validation",
                "memory": "tracemalloc peak per operation after input allocation, separate from timing; not process RSS; histogram layout cache warm",
                "source_sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "split_sha256": hashes,
                "limitations": "One machine, fixed BLAS threads, overlapping seed splits; results are descriptive, not confidence intervals or cross-hardware guarantees."}
    for name, value in (("results", records), ("summary", summary), ("equivalence", comparisons), ("gallery_reuse", reuse_records)):
        write_json(args.output_dir / f"{name}.json", value)
        write_csv(args.output_dir / f"{name}.csv", value)
    write_csv(args.output_dir / "timing_repeats.csv", timings)
    write_json(args.output_dir / "benchmark_manifest.json", manifest)
    print(f"Saved paired benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
