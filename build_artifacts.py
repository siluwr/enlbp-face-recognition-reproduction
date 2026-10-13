"""Build the Chinese measured-results report, explanatory notebook and provenance.

Run after the two experiment directories and figures exist. Numbers are read
from the experiment JSON files, never copied from the paper's reported scores.
The notebook is executed separately to verify its runnable cells.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NAMES = {
    "raw": "原始灰度像素",
    "lbp_raw": "普通 LBP 编码展开",
    "lbp_hist": "普通 LBP 分区直方图",
    "enlbp_mean_gt": "EnLBP 均值后编码，严格 >，一层（默认）",
    "enlbp_mean_ge": "EnLBP 均值后编码，>=，一层",
    "enlbp_mean_l2": "EnLBP 均值后编码，严格 >，两层",
    "enlbp_mean_block": "均值后非重叠编码，一层",
    "enlbp_algorithm": "Algorithm 1 流程解释，一层",
    "enlbp_algorithm_l2": "Algorithm 1 流程解释，两层",
}


def read(dataset: str, filename: str):
    return json.loads((ROOT / "outputs" / dataset / filename).read_text(encoding="utf-8"))


def summary_table(dataset: str) -> str:
    rows = ["| 方法 | Top-1 准确率均值 ± SD（%） | 每人准确率等权均值（%） | 维数 |",
            "|---|---:|---:|---:|"]
    for r in read(dataset, "aggregate.json"):
        rows.append(f"| {NAMES[r['method']]} | {r['accuracy_percent_mean']:.2f} ± "
                    f"{r['accuracy_percent_std']:.2f} | "
                    f"{r['macro_subject_accuracy_percent_mean']:.2f} | {r['feature_dimensions']} |")
    return "\n".join(rows)


def build_report():
    orl = read("orl", "aggregate.json")
    lfwa = read("lfwa", "aggregate.json")
    by_orl = {r["method"]: r for r in orl}
    by_lfw = {r["method"]: r for r in lfwa}
    meta_lfw = read("lfwa", "experiment.json")
    default_lfw_runs = [r for r in read("lfwa", "results.json") if r["method"] == "enlbp_mean_gt"]
    selected_test_counts = [r["test_count"] for r in default_lfw_runs]
    timings = ["| ORL 方法 | 编码全部 400 张（ms） | 匹配 200 查询×200 图库（ms） |",
               "|---|---:|---:|"]
    for name in ["raw", "lbp_raw", "lbp_hist", "enlbp_mean_gt", "enlbp_mean_l2"]:
        r = by_orl[name]
        timings.append(f"| {NAMES[name]} | {1000*r['encoding_seconds_mean']:.2f} | "
                       f"{1000*r['matching_seconds_mean']:.2f} |")
    seed_table = ["| seed | 人数 | 训练数 | 测试数 | 默认 EnLBP 正确数 | 准确率（%） |",
                  "|---:|---:|---:|---:|---:|---:|"]
    for r in default_lfw_runs:
        seed_table.append(f"| {r['seed']} | {r['subject_count']} | {r['train_count']} | "
                          f"{r['test_count']} | {r['correct']} | {r['accuracy_percent']:.2f} |")
    seed_table_text = "\n".join(seed_table)
    timings_text = "\n".join(timings)
    text = f"""# EnLBP 实验结果与复现边界

运行日期：2026-10-13（Asia/Shanghai）。本报告的实测值均来自 `outputs/orl/` 和 `outputs/lfwa/` 的 JSON、CSV。代码重建了论文方法的多个解释，**没有恢复作者原代码，也没有复现到原文报告的准确率**。

## 1. 核心结果

默认实现是先取非重叠 3×3 均值，再做 dense valid LBP，严格 `>`，一层，编码图按行展开，使用余弦最近邻。十次固定种子的 ORL 准确率为 **{by_orl['enlbp_mean_gt']['accuracy_percent_mean']:.2f}% ± {by_orl['enlbp_mean_gt']['accuracy_percent_std']:.2f} 个百分点**；LFW-a 为 **{by_lfw['enlbp_mean_gt']['accuracy_percent_mean']:.2f}% ± {by_lfw['enlbp_mean_gt']['accuracy_percent_std']:.2f} 个百分点**。± 是这些随机划分的样本标准差，不是置信区间。

| 数据集 | 本次默认 EnLBP 均值（%） | 论文报告（%） | 能否确认同一实验？ |
|---|---:|---:|---|
| ORL | {by_orl['enlbp_mean_gt']['accuracy_percent_mean']:.2f} | 99.07 | 不能：作者未给随机划分、次数和完整实现细节 |
| LFW-a | {by_lfw['enlbp_mean_gt']['accuracy_percent_mean']:.2f} | 61.428 | 不能：作者身份子集与测试分母未知 |

原报告值来自用户提供的论文第 4–5 页、印刷页 99–100。上表只呈现差距，不把两种未确认相同的协议当作严格性能排名。没有针对论文目标分数筛选 seed 或调整参数。

## 2. 已实际运行的协议

- ORL：40 人×10 张，共 400 张，原始 112×92；每人随机取 5 张训练、5 张测试，每次 200/200。
- LFW-a：作者页面下载的 13,233 张、5,749 人，原始 250×250；其中 {sum(n >= 11 for n in meta_lfw['images_per_subject'].values())} 人拥有至少 11 张图片。每个 seed 随机选 14 人，再每人取 5 张训练，剩余全部测试。图库固定 70 张，本次测试数从 {min(selected_test_counts)} 到 {max(selected_test_counts)} 张。
- 固定种子 0–9，NumPy `default_rng` / PCG64，无放回抽样。每个 seed 的 9 种方法使用相同图片划分；不同 seed 的图片与身份可能重叠。
- 仅做 Pillow 灰度读取，不缩放、直方图均衡化、裁脸或重新对齐。保留原始图像，池化只丢弃底部和右侧不完整块；LBP 使用 valid 邻域。
- 所有方法使用余弦相似度最大者的身份；零向量相似度设为 0，同分取图库第一张。这里是封闭集识别，分数不是概率。

LFW-a 沿用原数据目录标签，不使用普通 LFW、funneled、deep-funneled 或 scikit-learn 内置 LFW 作为替代。这是依据原文文字重建的身份识别实验，不是 LFW 的标准配对验证协议。

## 3. ORL 全部测量结果

{summary_table('orl')}

各 seed 测试数量相同、每人测试数量相同，所以这里图像准确率与每人等权准确率一致。默认 EnLBP 的向量从 10,304 个原像素缩为 980 个编码元素；普通 LBP 分区直方图基线在本次配置下为 {by_orl['lbp_hist']['accuracy_percent_mean']:.2f}%，高于默认 EnLBP。这里的普通 LBP 配置也是本项目明确选择，不能视为原文 LBP 基线的精确复现。

![ORL 九种配置；叉号为每次 seed，蓝点与横线为均值和样本SD](outputs/figures/orl_comparison.png)

## 4. LFW-a 全部测量结果

{summary_table('lfwa')}

图像数不均衡，所以同时给出每人等权准确率。表中每列先对每个 seed 计算，再对 10 个 seed 等权平均；不能把所有 seed 的正确数简单相加后除以测试数当成这个均值。

双层均值版本在这些划分上为 {by_lfw['enlbp_mean_l2']['accuracy_percent_mean']:.2f}%，属于事先列出的流程/层数敏感性分析；它不能被事后认定为作者的真实算法。不同流程的分数明显不同，说明原文缺失的实现选择会改变实验结论。

![LFW-a 九种配置；叉号为每次 seed，蓝点与横线为均值和样本SD](outputs/figures/lfwa_comparison.png)

默认 EnLBP 每次的实际分母：

{seed_table_text}

## 5. 运行时间与编码维数

{timings_text}

上述时间为各 seed 单次 wall-clock 测量的均值，编码包含图库与测试特征提取；匹配包含行归一化与矩阵相乘，排除读图、哈希和预处理。没有预热或独立重复计时，不能将其当作严格微基准。实验串行运行，启动时 OPENBLAS/MKL/OMP 线程变量均设为 1；环境版本保存在 `experiment.json`。

本次可见维数下降伴随匹配时间下降；不能据此验证原文“比 2DPCANet 更快”的说法，因为本项目没有运行 2DPCANet。编码图展开是把二进制代码作为数值向量，位权、边界和空间位置会影响余弦匹配；它与直方图表示的含义不同。

![ORL 原图、均值图、普通LBP及一层/两层EnLBP；所有面板固定灰度或编码范围0–255](outputs/figures/encoding_pipeline.png)

## 6. 为什么未达到论文分数

能够直接确认的差异是：正文先求均值，Algorithm 1 却先求 LBP；式 (1) 用 `>=`，图 2 需要 `>`；作者未给出层数、边界政策、直方图选项及预处理。LFW-a 身份名单、图片清单、测试数量和种子未公开。ORL 单次 200 张测试的准确率按 0.5 个百分点变化，99.07% 需要未知的重复平均或其它汇总解释。

本次严格按公开假设运行，没有补写原文缺失的流程来追逐目标值。可能还存在作者未说明的裁脸范围、尺寸、特征表示或评估差异，但这些只是解释候选，未被本次实验逐项证实。当前结果不足以判定论文数字必然错误，也不足以支持“已精确复现作者分数”。

## 7. 验证与复查

46 项自动测试已通过，覆盖两幅论文示例、八个位权、相等比较、无符号像素、浮点均值的相等关系、stride/多层尺寸、余弦边界和随机划分协议。另有对导出结果的独立重算与内容重复检查，见 `outputs/validation_audit.json`。这只核验实现和保存的实验记录，不证明与作者程序相同。

原 ZIP 来源、SHA-256 和下载记录见 `data/orl_source.json`、`data/lfwa_source.json`。每张原图的 SHA-256、每个 seed 的图片清单、逐图预测与命中图库路径均已保存。论文及最终源代码指纹见 `paper/source.json` 和 `outputs/reproduction_manifest.json`。

复查顺序建议：先读 [论文解析](paper/论文解析.md)，再打开 [Notebook](reproduction.ipynb)演算图 1/2，最后比较 `aggregate.csv` 与 `predictions.csv`。运行命令与全部参数见 [README](README.md)。

## 数据与方法来源

1. 原论文：Srinivasa Perumal R., Nadesh R. K., Senthil Kumar N. C. (2018). *Robust Face Recognition Using Enhanced Local Binary Pattern*. BEEI, 7(1), 96–101. [DOI](https://doi.org/10.11591/eei.v7i1.761)。正文、图 1/2 和原表数字来源于用户提供 PDF。
2. [AT&T/剑桥大学 ORL 官方说明与存档](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)，图片署名 AT&T Laboratories Cambridge。
3. [Tal Hassner 的 LFW-a 作者页面](https://talhassner.github.io/home/projects/lfwa/)。该页要求遵循原 LFW 的使用说明，并引用相关论文，例如 Wolf, Hassner & Taigman (2011), *Effective Face Recognition by Combining Multiple Descriptors and Learned Background Statistics*, TPAMI, 33(10)。
4. 分析/绘图工作流程参考 Kassis, Agarwal, He, Patel & Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。它不提供 EnLBP 方法或实验数据。
"""
    (ROOT / "实验结果.md").write_text(text, encoding="utf-8")


def build_notebook():
    notebook = nbf.v4.new_notebook()
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    notebook.cells = [
        md("# EnLBP：论文图示演算与独立复现\n\n依据用户提供的 2018 年论文；这是明确假设下的重建代码。先均值后 LBP 是默认正文解释，不能保证还原作者程序。完整分析见 `paper/论文解析.md`，实测报告见 `实验结果.md`。"),
        code("from pathlib import Path\nimport json\nimport numpy as np\nfrom IPython.display import display, Markdown, Image\nfrom enlbp import lbp, block_mean, enlbp, extract_feature, cosine_nearest_neighbor\nROOT = Path.cwd()\nprint('NumPy', np.__version__)"),
        md("## 1. 图 1：普通 LBP\n\n中心112，按右、右上、上、左上、左、左下、下、右下分配权重 $2^0$ 至 $2^7$；采用邻居≥中心。"),
        code("pixels = np.array([[167,152,94],[77,112,120],[170,143,210]], dtype=np.uint8)\nc = int(lbp(pixels).item())\nassert c == 237\nprint(pixels)\nprint('LBP =', c, 'binary =', format(c, '08b'))"),
        md("## 2. 图 2：相等规则决定 246 或 254\n\n左上真实3×3小块的总和27，均值3；中心块也均值3。原图其余块在这里用其图示均值填满，以重建该局部例子。"),
        code("means = np.array([[3,5,6],[4,3,1],[6,7,9]], dtype=np.uint8)\noriginal = np.repeat(np.repeat(means,3,axis=0),3,axis=1)\noriginal[:3,:3] = [[1,2,3],[4,5,6],[1,2,3]]\nprint('Block means:\\n',block_mean(original))\nstrict = int(enlbp(original, comparison='gt').item())\nnonstrict = int(enlbp(original, comparison='ge').item())\nassert strict == 246 and nonstrict == 254\nprint('> :', strict, format(strict,'08b'))\nprint('>=:', nonstrict, format(nonstrict,'08b'))"),
        md("## 3. 余弦最近邻\n\n余弦分数不是身份概率；本实验选择图库中分数最大的身份，未加入陌生人拒识。"),
        code("gallery = np.array([[1.,0.],[0.,1.]])\nquery = np.array([[0.8,0.2],[0.1,0.9]])\nlabels,scores,indices = cosine_nearest_neighbor(query,gallery,np.array(['person_a','person_b']))\nprint('labels:',labels)\nprint('scores:',scores)\nprint('gallery indices:',indices)"),
        md("## 4. 编码过程和维数\n\n图片未增强，展示时灰度/编码统一固定为0–255；ORL 112×92原图经一层均值-LBP后为35×28=980，第二层为9×7=63。"),
        code("display(Image(filename=str(ROOT/'outputs/figures/encoding_pipeline.png'), alt='ORL original 112x92; block means 37x30; LBP 110x90; one-level EnLBP 35x28; two-level EnLBP 9x7.'))"),
        md("## 5. 两个数据集的实际实验\n\n每方法10个固定种子，所有方法共享每个seed的图片划分。均值±样本SD表示这些划分的波动；并非置信区间。LFW-a每seed从至少11张图片的人中随机选14人，70张训练、剩余全部测试。"),
        code("for dataset in ['orl','lfwa']:\n    rows=json.loads((ROOT/'outputs'/dataset/'aggregate.json').read_text(encoding='utf-8'))\n    table=['| method | accuracy mean ± SD (%) | dimension |','|---|---:|---:|']\n    for r in rows:\n        table.append(f\"| {r['method']} | {r['accuracy_percent_mean']:.2f} ± {r['accuracy_percent_std']:.2f} | {r['feature_dimensions']} |\")\n    display(Markdown('### '+dataset.upper()+'\\n\\n'+'\\n'.join(table)))"),
        code("display(Image(filename=str(ROOT/'outputs/figures/orl_comparison.png'), alt='ORL measured seed accuracies, means and sample standard deviations; values are in the preceding table.'))\ndisplay(Image(filename=str(ROOT/'outputs/figures/lfwa_comparison.png'), alt='LFW-a measured seed accuracies, means and sample standard deviations; values are in the preceding table.'))"),
        md("## 6. 复现结论\n\n论文图示已经逐值核验，算法与实验可重跑。默认实现未达到原文的ORL 99.07%或LFW-a 61.428%；算法文本冲突和协议遗漏限制了精确数值复现。不能通过事后挑选流程或随机种子把某个分数称为作者的唯一结果。\n\n若需重跑：`python run_experiment.py --dataset orl --data-dir data/orl --output-dir outputs/new_orl --seeds 0 1 2`。完整命令见README。"),
    ]
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11.7"}}
    nbf.write(notebook, ROOT / "reproduction.ipynb")


def build_manifests(source_pdf: Path | None = None):
    source_record = ROOT / "paper/source.json"
    record = json.loads(source_record.read_text(encoding="utf-8")) if source_record.exists() else {}
    record.update({"title": "Robust Face Recognition Using Enhanced Local Binary Pattern",
              "authors_as_pdf": ["Srinivasa Perumal R.", "Nadesh R. K.", "Senthil Kumar N. C."],
              "year": 2018, "doi": "10.11591/eei.v7i1.761", "pdf_pages": 6,
              "printed_pages": "96-101", "source_type": "user-provided published paper"})
    record.pop("local_path", None)
    if source_pdf is not None:
        record["source_filename"] = source_pdf.name
        record["sha256"] = hashlib.sha256(source_pdf.read_bytes()).hexdigest()
    source_record.write_text(json.dumps(record, indent=2), encoding="utf-8")
    files = sorted([*ROOT.glob("*.py"), *ROOT.glob("enlbp/*.py"), *ROOT.glob("tests/test_*.py"), ROOT/"requirements.txt"])
    manifest = {"purpose": "independent reconstruction; not original author source code",
                "client_run_date": "2026-10-13", "client_timezone": "Asia/Shanghai",
                "executed_seeds": list(range(10)), "datasets": ["orl", "lfwa"],
                "methods_per_dataset": 9,
                "blas_thread_environment_used": {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                "test_result": {"passed": 46, "command": "python -m pytest -q"},
                "code_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "paper_source_record": "paper/source.json",
                "independent_result_audit": "outputs/validation_audit.json"}
    (ROOT / "outputs/reproduction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, help="Optional local PDF for source hash verification; its absolute path is not published")
    args = parser.parse_args()
    build_report()
    build_notebook()
    build_manifests(args.source_pdf)
    print("Created report, notebook and provenance manifests.")
