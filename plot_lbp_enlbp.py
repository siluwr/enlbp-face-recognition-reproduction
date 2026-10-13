"""Chinese reference figures from paired, measured LBP/EnLBP experiments.

No measurements are synthesized. Sources, split membership, predictions,
means, sample SDs and timing normalization are checked before plotting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

from enlbp import block_mean, enlbp, lbp
from run_experiment import METHODS, write_csv, write_json

METHOD_ORDER = ("lbp_raw", "lbp_hist", "enlbp_mean_gt", "enlbp_mean_ge")
LABELS = ("LBP 编码\n≥ 阈值", "LBP 直方图\n≥ 阈值",
          "EnLBP\n> 阈值", "EnLBP\n≥ 阈值对照")
COLORS = ("#0072B2", "#5B5285", "#A94D00", "#007A5E")
HATCHES = ("", "//", "..", "xx")
MARKERS = ("o", "s", "^", "D")
PAIRS = (("enlbp_mean_gt", "lbp_raw", "EnLBP (>) − LBP 编码 (≥)"),
         ("enlbp_mean_gt", "lbp_hist", "EnLBP (>) − LBP 直方图 (≥)"),
         ("enlbp_mean_ge", "lbp_raw", "EnLBP (≥) − LBP 编码 (≥)"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_name(path: Path) -> str:
    """Keep public manifests free of machine-specific absolute directories."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def chinese_font() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                   "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS"):
        if family in installed:
            return family
    raise ValueError("Chinese font required: install Microsoft YaHei or Noto Sans CJK SC.")


def sample_sd(values: np.ndarray) -> float | None:
    return float(values.std(ddof=1)) if len(values) > 1 else None


def load_experiment(directory: Path) -> dict:
    meta_path, data_path = directory / "experiment.json", directory / "results.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    preprocessing = meta["preprocessing"]
    if preprocessing["lbp_hist_grid_rc"] != [8, 8] or any(preprocessing[key] for key in
            ("resize_wh", "histogram_equalization", "crop", "automatic_alignment")):
        raise ValueError(f"{directory}: reference figures expect original-size grayscale images and an 8x8 histogram grid.")
    seeds = sorted(meta["seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{directory}: nonempty, unique seeds required.")
    for method in METHOD_ORDER:
        if meta["methods"].get(method) != METHODS[method]:
            raise ValueError(f"{directory}: missing or unexpected method {method}.")
    selected = [r for r in rows if r["method"] in METHOD_ORDER]
    indexed = {(r["method"], r["seed"]): r for r in selected}
    if len(selected) != len(METHOD_ORDER) * len(seeds) or len(indexed) != len(selected):
        raise ValueError(f"{directory}: missing or duplicate method/seed rows.")
    sources = [meta_path, data_path, directory / "predictions.csv"]
    import csv
    with sources[-1].open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    grouped_predictions = {}
    for prediction in predictions:
        key = (prediction["method"], int(prediction["seed"]))
        grouped_predictions.setdefault(key, []).append(prediction)
    for seed in seeds:
        split_path = directory / f"split_seed_{seed}.json"
        sources.append(split_path)
        split = json.loads(split_path.read_text(encoding="utf-8"))
        expected = {item["path"]: item["label"] for item in split["test"]}
        train_paths = {item["path"] for item in split["train"]}
        if len(expected) != len(split["test"]) or train_paths & set(expected):
            raise ValueError(f"{directory}: duplicate test paths or train/test overlap.")
        for method in METHOD_ORDER:
            row = indexed[method, seed]
            observed = grouped_predictions.get((method, seed), [])
            if len(observed) != len(expected) or {p["test_path"]: p["true_label"] for p in observed} != expected:
                raise ValueError(f"{directory}: predictions do not match split: {method}/{seed}.")
            correct = sum(p["true_label"] == p["predicted_label"] for p in observed)
            if row["test_count"] != len(expected) or row["train_count"] != len(train_paths) or row["correct"] != correct:
                raise ValueError(f"{directory}: prediction/count mismatch: {method}/{seed}.")
            if not np.isclose(row["accuracy_percent"], 100 * correct / len(expected), rtol=0, atol=1e-10):
                raise ValueError(f"{directory}: accuracy/count mismatch.")
            if row["feature_dimension"] <= 0 or any(not np.isfinite(row[k]) or row[k] < 0 for k in ("encoding_seconds", "matching_seconds")):
                raise ValueError(f"{directory}: invalid dimensions or timings.")
    for method in METHOD_ORDER:
        if len({indexed[method, seed]["feature_dimension"] for seed in seeds}) != 1:
            raise ValueError(f"{directory}: varying dimensions require a different figure.")
    dataset = {"orl": "ORL", "lfwa": "LFW-a"}.get(meta["dataset"], meta["dataset"])
    return {"directory": directory, "dataset": dataset, "metadata": meta,
            "seeds": seeds, "indexed": indexed,
            "sources": [{"path": source_name(p), "sha256": digest(p)} for p in sources]}


def values(experiment: dict, method: str, metric: str) -> np.ndarray:
    result = []
    for seed in experiment["seeds"]:
        row = experiment["indexed"][method, seed]
        if metric == "encoding_ms_per_image":
            value = 1000 * row["encoding_seconds"] / (row["train_count"] + row["test_count"])
        elif metric == "matching_ms_per_query":
            value = 1000 * row["matching_seconds"] / row["test_count"]
        else:
            value = row[metric]
        result.append(value)
    return np.asarray(result, dtype=float)


def format_stat(array: np.ndarray, decimals: int = 2) -> str:
    sd = sample_sd(array)
    return f"{array.mean():.{decimals}f}" + (f" ± {sd:.{decimals}f}" if sd is not None else "（仅一次）")


def accuracy_description(experiments: list) -> str:
    descriptions = []
    for experiment in experiments:
        results = [f"{label.replace(chr(10), ' ')} {values(experiment, method, 'accuracy_percent').mean():.2f}%"
                   for method, label in zip(METHOD_ORDER, LABELS)]
        descriptions.append(experiment["dataset"] + "：" + "，".join(results))
    return "；".join(descriptions) + "。"


def decorate(ax, ylabel: str, ylim=None):
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(4), LABELS)
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.65)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)


def bars(ax, experiment: dict, metric: str, ylim, digits: int, show_seeds: bool = False):
    arrays = [values(experiment, method, metric) for method in METHOD_ORDER]
    for i, array in enumerate(arrays):
        sd = sample_sd(array)
        ax.bar(i, array.mean(), width=0.57, color=COLORS[i], edgecolor="#263238",
               hatch=HATCHES[i], linewidth=0.7, alpha=0.88)
        if sd is not None:
            ax.errorbar(i, array.mean(), yerr=sd, fmt="none", ecolor="#18222D", capsize=4, linewidth=1.5)
        if show_seeds:
            ax.scatter(i + np.linspace(-0.19, 0.19, len(array)), array,
                       marker=MARKERS[i], facecolors="white", edgecolors="#18222D",
                       s=25, linewidths=0.8, zorder=4)
        top = max(array.max() if show_seeds else 0, array.mean() + (sd or 0))
        ax.text(i, top + ylim[1] * 0.025, f"{array.mean():.{digits}f}", ha="center", va="bottom", fontsize=12)
    ax.set_ylim(*ylim)


def canvas(title: str, subtitle: str, rows: int = 1, figsize=(13.6, 6.9)):
    fig, axes = plt.subplots(rows, 2, figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.76 if rows == 1 else 0.82,
                        bottom=0.22 if rows == 1 else 0.17, wspace=0.24, hspace=0.58)
    fig.text(0.06, 0.955, title, fontsize=21, weight="bold", va="top", color="#172B3A")
    fig.text(0.06, 0.889, subtitle, fontsize=12, va="top", color="#34495A")
    return fig, axes


def export(fig, name: str, alt: str, output: Path, entries: list):
    files = []
    for extension in ("png", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white", transparent=False)
        if extension == "png":
            with Image.open(path) as image:
                image.convert("RGB").save(path, dpi=(300, 300))
        files.append({"path": path.name, "sha256": digest(path)})
    entries.append({"name": name, "alt_text": alt, "size_inches": fig.get_size_inches().tolist(), "files": files})
    plt.close(fig)


def accuracy_figure(experiments: list, output: Path, entries: list):
    fig, axes = canvas("01  哪种方法识别得更准？", "同一种子共享训练 / 测试图像；四种方法均使用余弦最近邻；准确率坐标统一为 0–100%。")
    for ax, experiment in zip(axes[0], experiments):
        bars(ax, experiment, "accuracy_percent", (0, 100), 2, True)
        decorate(ax, "识别准确率（%）")
        ax.set_title(f"{experiment['dataset']}  |  {len(experiment['seeds'])} 次划分", fontsize=15, pad=18)
    fig.text(0.06, 0.085, "柱顶数字 = 各划分准确率的等权均值；误差线 = 样本标准差 SD；空心点 = 每次划分的实测结果。", fontsize=11)
    fig.text(0.06, 0.045, "LBP 编码与 LBP 直方图的表示不同；EnLBP 的 ≥ 版本用于控制阈值差异。SD 不代表置信区间。", fontsize=11)
    export(fig, "01_accuracy", accuracy_description(experiments) + "误差线为划分间 SD，空心点显示全部划分。", output, entries)


def paired_figure(experiments: list, output: Path, entries: list):
    all_differences = [values(e, a, "accuracy_percent") - values(e, b, "accuracy_percent")
                       for e in experiments for a, b, _ in PAIRS]
    limit = max(10, int(np.ceil(max(np.abs(d).max() for d in all_differences) / 5)) * 5 + 5)
    fig, axes = canvas("02  换一批训练图像，结论还成立吗？", "在同一次划分中计算 EnLBP − LBP；0 左侧表示 EnLBP 较低，右侧表示 EnLBP 较高。", figsize=(14.8, 7.4))
    fig.subplots_adjust(left=0.055, right=0.975, wspace=0.16, top=0.76, bottom=0.22)
    for ax, experiment in zip(axes[0], experiments):
        ax.axvline(0, color="#18222D", linewidth=1.5)
        for i, (enhanced, baseline, label) in enumerate(PAIRS):
            differences = values(experiment, enhanced, "accuracy_percent") - values(experiment, baseline, "accuracy_percent")
            y = 2 - i
            ax.scatter(differences, y + np.linspace(-0.1, 0.1, len(differences)),
                       facecolors="white", edgecolors=COLORS[i], marker=MARKERS[i], s=37, linewidths=1.2, zorder=3)
            ax.errorbar(differences.mean(), y, xerr=sample_sd(differences), fmt="D",
                        color="#172B3A", capsize=5, markersize=7, linewidth=1.6, zorder=4)
            wins = int((differences > 0).sum())
            ax.text(-limit + 1, y + 0.27, label, fontsize=11, va="bottom")
            ax.text(limit - 1, y + 0.27, f"均值 {differences.mean():+.2f}；胜 {wins}/{len(differences)}", fontsize=10, ha="right", va="bottom")
        ax.set(xlim=(-limit, limit), ylim=(-0.25, 2.8), xlabel="准确率差（百分点）：EnLBP − LBP", yticks=[])
        ax.grid(axis="x", color="#D7DCE2", linewidth=0.65)
        ax.set_title(experiment["dataset"], fontsize=15, pad=18)
    fig.text(0.06, 0.08, "空心点 = 同一种子的配对差；黑色菱形与误差线 = 配对差均值 ± SD；胜 = 差值严格大于 0 的划分数。", fontsize=11)
    fig.text(0.06, 0.04, "最下行双方都用 ≥，保持编码展开与匹配方式一致，用于观察均值压缩的影响；未进行显著性推断。", fontsize=11)
    export(fig, "02_paired_difference", "配对差值图完整显示三个比较的全部划分；正值表示 EnLBP 准确率较高，负值表示较低，图中标有差值均值和获胜划分数。下行保持双方 ≥ 阈值一致。", output, entries)


def dimension_figure(experiments: list, output: Path, entries: list):
    fig, axes = canvas("03  EnLBP 把特征压缩了多少？", "每个数据集内以 LBP 编码的维数为 100%；柱顶同时标出绝对维数和比例；数值越小，存储越少。")
    for ax, experiment in zip(axes[0], experiments):
        reference = values(experiment, "lbp_raw", "feature_dimension")[0]
        for i, method in enumerate(METHOD_ORDER):
            dimension = int(values(experiment, method, "feature_dimension")[0])
            percentage = dimension / reference * 100
            ax.bar(i, percentage, width=0.57, color=COLORS[i], hatch=HATCHES[i], edgecolor="#263238", linewidth=0.7)
            ax.text(i, percentage + 4, f"{dimension:,} 维\n{percentage:.1f}%", ha="center", va="bottom", fontsize=12)
        ax.axhline(100, linestyle="--", linewidth=1, color="#4C5662")
        decorate(ax, "相对 LBP 编码的特征维数（%）", (0, 200))
        ax.set_title(experiment["dataset"], fontsize=15, pad=18)
    fig.text(0.06, 0.085, "所有特征均以 float64 参与匹配；维数是每张图像的标量个数，不是编码图像的文件字节数。", fontsize=11)
    fig.text(0.06, 0.045, "维数减少与准确率提高是不同问题，应结合图 01 阅读。相同存储精度下，维数比例也等于向量字节数比例。", fontsize=11)
    dimensions = "；".join(f"{e['dataset']} 的 LBP 编码 {int(values(e, 'lbp_raw', 'feature_dimension')[0]):,} 维，EnLBP {int(values(e, 'enlbp_mean_gt', 'feature_dimension')[0]):,} 维" for e in experiments)
    export(fig, "03_feature_dimension", "特征维数以 LBP 编码为百分比基准，标有绝对维数。" + dimensions + "。", output, entries)


def timing_figure(experiments: list, output: Path, entries: list):
    fig, axes = canvas("04  特征提取与匹配各需要多久？", "每次实测总耗时先按图像数 / 查询数归一化，再对划分等权平均；每行两个数据集使用相同坐标。", rows=2, figsize=(13.6, 9.7))
    for r, (metric, ylabel) in enumerate((("encoding_ms_per_image", "平均每张图像的编码耗时（ms）"),
                                        ("matching_ms_per_query", "平均每张查询的匹配耗时（ms）"))):
        upper = max(values(e, m, metric).mean() + (sample_sd(values(e, m, metric)) or 0)
                    for e in experiments for m in METHOD_ORDER) * 1.28
        for ax, experiment in zip(axes[r], experiments):
            bars(ax, experiment, metric, (0, upper), 3)
            decorate(ax, ylabel)
            ax.set_title(experiment["dataset"] + ("：特征提取" if r == 0 else "：余弦最近邻匹配"), fontsize=14, pad=14)
    fig.text(0.06, 0.075, "编码包含训练图像与测试图像；匹配包含归一化和批量相似度计算。计时不包含磁盘读取、解码或预处理。", fontsize=10.5)
    fig.text(0.06, 0.038, "每方法每划分仅计时一次，无预热；柱顶为均值，误差线为 SD。归一化数值不是单图延迟基准，跨硬件不能直接比较。", fontsize=10.5)
    export(fig, "04_timing", "两行分别显示每张图像平均编码耗时和每张查询平均匹配耗时，均为总耗时归一化后的均值与 SD。每方法每划分仅计时一次、无预热；结果依赖机器和批量大小。", output, entries)


def pipeline_figure(sample: Path, output: Path, entries: list) -> dict:
    with Image.open(sample) as image_file:
        image = np.asarray(image_file.convert("L"))
    panels = (("输入灰度图", image), ("普通 LBP 编码（≥）", lbp(image)),
              ("3×3 非重叠均值", block_mean(image)), ("EnLBP 编码（>）", enlbp(image)))
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 6.6))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.75, bottom=0.31, wspace=0.35)
    fig.text(0.06, 0.955, "05  LBP 与 EnLBP 到底怎样处理同一张脸？", fontsize=21, weight="bold", va="top", color="#172B3A")
    fig.text(0.06, 0.889, "从原始图像出发，显示实际算法输出；低分辨率结果仅以最近邻方式放大显示。", fontsize=12, va="top")
    for ax, (title, array) in zip(axes, panels):
        ax.imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{title}\n{array.shape[0]} × {array.shape[1]}", fontsize=13, pad=14)
        ax.set_axis_off()
    fig.text(0.06, 0.22, f"LBP：输入 → LBP 编码图 → 编码展开（{panels[1][1].size:,} 维）或分区直方图（8 × 8 × 256 = 16,384 维）", fontsize=12)
    fig.text(0.06, 0.155, f"EnLBP：输入 → 3 × 3 均值压缩 → LBP 编码图 → 编码展开（{panels[3][1].size:,} 维）；≥ 对照只改变阈值。", fontsize=12)
    fig.text(0.06, 0.075, "显示范围均为 0–255，无对比度增强；编码值表示局部二进制模式，不等同于原图灰度。", fontsize=10.5)
    credit = "AT&T Laboratories Cambridge / ORL" if sample.resolve() == Path("data/orl/s1/1.pgm").resolve() else source_name(sample)
    fig.text(0.06, 0.038, f"示例图像来源：{credit}；本图用于说明既定算法的处理过程。", fontsize=10.5)
    export(fig, "05_encoding_process", "同一示例图像的原图、普通 LBP 编码、均值压缩和 EnLBP 编码，并解释编码展开与直方图表示。所有图像使用相同 0–255 显示范围，无对比度调整。", output, entries)
    return {"path": source_name(sample), "sha256": digest(sample),
            "panels": [{"label": label, "shape_hw": list(array.shape)} for label, array in panels]}


def report(experiments: list, entries: list, output: Path):
    lines = ["# LBP 与 EnLBP 对比实验参考图", "", "本报告使用独立重建程序的真实实验结果；不是论文作者代码的原始结果。", "",
             "## 实验设置", "", "四种方法共享同一种子的训练与测试图像，使用余弦最近邻和原始图像尺寸。普通 LBP 使用 ≥；EnLBP 分别使用 > 和 ≥。后者是保持比较阈值一致的控制组。",
             "", "- LBP 编码：8 邻域、valid、逐像素编码后展开。",
             "- LBP 直方图：同一 LBP 编码，8×8 分区，每区 256 bins，L1 归一化并拼接。",
             "- EnLBP：非重叠 3×3 块取均值，然后一层 dense LBP，编码展开。",
             "", "原文存在流程与阈值歧义；这些是公开假设下的重建。对比 EnLBP(≥) 与 LBP 编码(≥)，可以保持表示和阈值一致，观察均值压缩的影响。对比 LBP 直方图时，表示方式也发生了变化。", ""]
    table_rows, seed_rows, differences = [], [], []
    for experiment in experiments:
        n = len(experiment["seeds"])
        meta = experiment["metadata"]
        sample_row = experiment["indexed"]["lbp_raw", experiment["seeds"][0]]
        counts = values(experiment, "lbp_raw", "test_count").astype(int)
        lines += [f"### {experiment['dataset']}", "",
                  f"{n} 次固定种子划分：{experiment['seeds']}。每人 {meta['train_per_person']} 张训练；每次 {sample_row['subject_count']} 人、{sample_row['train_count']} 张训练图像。测试数范围 {counts.min()}–{counts.max()}。图片尺寸（高×宽）：{'×'.join(map(str, sample_row['image_shape_hw']))}。",
                  "", "| 方法 | 准确率（%）均值±SD | 人均准确率（%）均值±SD | 维数 | 编码（ms/图）均值±SD | 匹配（ms/查询）均值±SD |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method, label in zip(METHOD_ORDER, LABELS):
            record = {"dataset": experiment["dataset"], "method": method, "label": label.replace("\n", " "), "seed_count": n}
            for metric in ("accuracy_percent", "macro_subject_accuracy_percent", "encoding_ms_per_image", "matching_ms_per_query"):
                array = values(experiment, method, metric)
                record[metric + "_mean"] = float(array.mean())
                record[metric + "_sd"] = sample_sd(array)
            record["feature_dimension"] = int(values(experiment, method, "feature_dimension")[0])
            table_rows.append(record)
            lines.append(f"| {record['label']} | {format_stat(values(experiment, method, 'accuracy_percent'))} | {format_stat(values(experiment, method, 'macro_subject_accuracy_percent'))} | {record['feature_dimension']:,} | {format_stat(values(experiment, method, 'encoding_ms_per_image'), 3)} | {format_stat(values(experiment, method, 'matching_ms_per_query'), 3)} |")
            for seed in experiment["seeds"]:
                row = experiment["indexed"][method, seed]
                seed_rows.append({"dataset": experiment["dataset"], **{k: v for k, v in row.items() if k != "subject_results"},
                                  "encoding_ms_per_image": 1000 * row["encoding_seconds"] / (row["train_count"] + row["test_count"]),
                                  "matching_ms_per_query": 1000 * row["matching_seconds"] / row["test_count"]})
        lines += ["", "配对差异：", ""]
        for enhanced, baseline, label in PAIRS:
            array = values(experiment, enhanced, "accuracy_percent") - values(experiment, baseline, "accuracy_percent")
            lines.append(f"- {label}：平均 {array.mean():+.2f} 个百分点，SD = {sample_sd(array):.2f}；胜 / 平 / 负 = {(array > 0).sum()} / {(array == 0).sum()} / {(array < 0).sum()}。" if n > 1 else f"- {label}：{array.mean():+.2f} 个百分点，仅一次划分。")
            for seed, difference in zip(experiment["seeds"], array):
                differences.append({"dataset": experiment["dataset"], "seed": seed, "enhanced": enhanced,
                                    "baseline": baseline, "accuracy_difference_pp": float(difference)})
        lines += [""]
    lines += ["## 如何解释", "", accuracy_description(experiments), ""]
    for experiment in experiments:
        enhanced = values(experiment, "enlbp_mean_gt", "accuracy_percent")
        for baseline, label in (("lbp_raw", "LBP 编码"), ("lbp_hist", "LBP 直方图")):
            difference = enhanced - values(experiment, baseline, "accuracy_percent")
            lines.append(f"- {experiment['dataset']}：EnLBP(>) 相比 {label} 的平均准确率差为 {difference.mean():+.2f} 个百分点，{len(difference)} 次中胜 {(difference > 0).sum()} 次、负 {(difference < 0).sum()} 次。")
        ratio = values(experiment, "enlbp_mean_gt", "feature_dimension")[0] / values(experiment, "lbp_raw", "feature_dimension")[0]
        lines.append(f"- {experiment['dataset']}：EnLBP 保留 LBP 编码 {100 * ratio:.2f}% 的维数。准确率、维数和耗时应一起比较；不能仅凭一项指标概括成 EnLBP 全面优于 LBP。")
    lines += ["- 图 02 下行保持双方 ≥ 和编码展开一致；两种 EnLBP 阈值版本均完整保留，没有选择有利结果后隐藏另一组。",
              "", "均值按种子等权计算；误差线是划分之间的样本 SD（ddof=1），不是标准误或置信区间。划分共享图像，不能当作十组独立数据；未做显著性检验。LFW-a 每次身份集合与测试数量可能改变，提供了人均准确率避免只看多图身份占主导的总准确率。",
              "", "编码 ms/图 = 每次编码总秒数 ×1000 /（训练数+测试数）；匹配 ms/查询 = 每次匹配总秒数 ×1000 / 测试数，然后对种子取等权均值。匹配使用批量计算，数值不是独立单图延迟。时间只测一次且无预热，依赖本机、线程数、实现与图库大小；不用于验证论文计时。",
              "", "## 图与原始数据", ""]
    for entry in entries:
        name = entry["name"]
        lines += [f"### {name}", "", f"![{entry['alt_text']}]({name}.png)", "", entry["alt_text"], "", f"[PNG]({name}.png) · [SVG]({name}.svg)", ""]
    lines += ["[汇总表](comparison_summary.csv) · [逐种子数据](comparison_seed_results.csv) · [配对差](paired_differences.csv) · [图源与哈希](figure_manifest.json)", "", "## 可视化方法来源", "", "图形设计采用 Scientific Agent Skills 的科学可视化流程：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。它是可视化方法来源，不是 EnLBP 算法来源。", ""]
    write_csv(output / "comparison_summary.csv", table_rows)
    write_csv(output / "comparison_seed_results.csv", seed_rows)
    write_csv(output / "paired_differences.csv", differences)
    write_json(output / "comparison_summary.json", table_rows)
    (output / "参考图说明.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orl", type=Path, default=Path("outputs/lbp_enlbp_orl"))
    parser.add_argument("--lfwa", type=Path, default=Path("outputs/lbp_enlbp_lfwa"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reference_figures"))
    parser.add_argument("--sample", type=Path, default=Path("data/orl/s1/1.pgm"))
    parser.add_argument("--skip-process", action="store_true", help="Draw measured results without a source face image.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace previously generated figures in this directory.")
    args = parser.parse_args()
    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise ValueError("Output directory is nonempty; choose a new directory or pass --overwrite.")
        experiments = [load_experiment(args.orl), load_experiment(args.lfwa)]
        if [e["metadata"]["dataset"] for e in experiments] != ["orl", "lfwa"]:
            raise ValueError("--orl and --lfwa must refer to their respective datasets.")
        if not args.skip_process and not args.sample.is_file():
            raise ValueError("Sample image missing: download ORL or use --skip-process to plot measured results only.")
        font = chinese_font()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        with plt.rc_context({"font.family": font, "font.size": 11, "svg.fonttype": "path",
                             "axes.unicode_minus": False, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.labelpad": 10}):
            accuracy_figure(experiments, args.output_dir, entries)
            paired_figure(experiments, args.output_dir, entries)
            dimension_figure(experiments, args.output_dir, entries)
            timing_figure(experiments, args.output_dir, entries)
            sample = None if args.skip_process else pipeline_figure(args.sample, args.output_dir, entries)
        report(experiments, entries, args.output_dir)
        write_json(args.output_dir / "figure_manifest.json", {
            "destination": "Chinese classroom/report reference figures; no journal-specific claims",
            "script": {"path": source_name(Path(__file__)), "sha256": digest(Path(__file__))},
            "font": font, "matplotlib": matplotlib.__version__, "numpy": np.__version__,
            "png_dpi": 300, "svg_fonttype": "path", "background": "opaque white",
            "uncertainty": "sample SD across paired overlapping seed splits, ddof=1; not a CI",
            "selection": "all seeds and four declared configurations; other variants stay in original experiment folders",
            "timing_transform": "divide each seed's measured total by corresponding image/query count, then equal-weight seed mean",
            "normalization": "feature dimensions / same dataset's LBP-code dimension * 100",
            "sources": [source for e in experiments for source in e["sources"]],
            "image": sample, "figures": entries,
            "checks": "prediction membership and accuracy counts verified against exported paired split files"})
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(entries)} reference figures, source tables and explanations to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
