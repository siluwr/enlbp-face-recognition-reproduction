"""Render the declared EnLBP pipeline and measured seed-level comparisons.

Input images are unmodified except for the algorithm's declared block means
and binary coding. All image panels share intensity/code limits [0,255].
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from enlbp import block_mean, enlbp, lbp

LABELS = {
    "raw": "Raw pixels",
    "lbp_raw": "LBP codes",
    "lbp_hist": "LBP histograms",
    "enlbp_mean_gt": "EnLBP mean / >",
    "enlbp_mean_ge": "EnLBP mean / >=",
    "enlbp_mean_l2": "EnLBP mean / 2 levels",
    "enlbp_mean_block": "Mean + block LBP",
    "enlbp_algorithm": "Algorithm 1 / 1 level",
    "enlbp_algorithm_l2": "Algorithm 1 / 2 levels",
}


def pipeline_figure(sample: Path, output: Path):
    with Image.open(sample) as im:
        image = np.asarray(im.convert("L"))
    panels = [("Original", image), ("3x3 block means", block_mean(image)),
              ("Dense LBP (>=)", lbp(image)),
              ("EnLBP (>), level 1", enlbp(image)),
              ("EnLBP (>), level 2", enlbp(image, levels=2))]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), layout="constrained")
        for ax, (title, values) in zip(axes, panels):
            ax.imshow(values, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{title}\n{values.shape[0]} x {values.shape[1]} = {values.size:,}")
            ax.set_axis_off()
        fig.suptitle("Declared reconstruction: block means, then valid LBP; no image enhancement")
        fig.savefig(output / "encoding_pipeline.png", dpi=180, facecolor="white")
        fig.savefig(output / "encoding_pipeline.svg", facecolor="white")
        plt.close(fig)
    manifest = {
        "source_image": sample.as_posix() if not sample.is_absolute() else sample.name, "transformations": [
            "Pillow grayscale conversion", "nonoverlapping 3x3 block means",
            "valid dense LBP, no padding, E/NE/N/NW/W/SW/S/SE bit order",
            "EnLBP strict > comparison, 1 and 2 levels"],
        "panels": [{"label": title, "shape": list(a.shape), "dimension": a.size}
                   for title, a in panels],
        "display": "grayscale [0,255], nearest pixel rendering; no contrast adjustment",
        "destination": "general classroom explanation; no publisher-specific claims",
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def plot_comparison(rows: list[dict], dataset: str, output: Path):
    methods = list(dict.fromkeys(row["method"] for row in rows))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        for index, method in enumerate(methods):
            values = np.array([float(r["accuracy_percent"]) for r in rows if r["method"] == method])
            offsets = np.linspace(-0.13, 0.13, len(values))
            ax.scatter(values, index + offsets, color="#555555", marker="x", s=24,
                       alpha=0.7, label="Each fixed seed" if index == 0 else None)
            sd = values.std(ddof=1) if len(values) > 1 else 0.0
            ax.errorbar(values.mean(), index, xerr=sd, fmt="o", color="#006da3",
                        capsize=4, markersize=6, label="Mean +/- seed SD" if index == 0 else None)
        ax.set_yticks(range(len(methods)), [LABELS.get(m, m) for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Closed-set top-1 identification accuracy (%)")
        ax.set_title(f"{dataset}: independently reconstructed methods, same splits within each seed")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.legend(loc="lower right")
        fig.savefig(output / f"{dataset.lower()}_comparison.png", dpi=180, facecolor="white")
        fig.savefig(output / f"{dataset.lower()}_comparison.svg", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/orl/s1/1.pgm"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--metrics", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_figure(args.sample, args.output_dir)
    for path in args.metrics:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        plot_comparison(rows, path.parent.name.upper(), args.output_dir)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
