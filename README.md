# EnLBP 论文解析与复现

针对 *Robust Face Recognition Using Enhanced Local Binary Pattern*（2018，DOI: 10.11591/eei.v7i1.761）的独立 Python 重建实现。

本项目已下载官方来源的 ORL 和作者发布的 LFW-a，运行 9 种配置、每种 10 个固定种子的真实实验。论文的方法描述存在冲突，因此实现明确区分三种解释。代码不是作者公开的原始程序，也不保证得到原文分数。

## 先看哪些文件

- [论文解析](paper/论文解析.md)：研究问题、数学流程、图示手算、实验协议、优缺点和课堂汇报建议。
- [实验结果](实验结果.md)：实际运行结果及其与原文的差异。
- [交互式 Notebook](reproduction.ipynb)：图 1/2 演算、编码过程和实验表。
- `enlbp/descriptors.py`：LBP、均值池化、EnLBP 和直方图。
- `enlbp/matching.py`：余弦相似度和最近邻匹配。
- `run_experiment.py`：数据划分、实验运行及结果导出。

## 环境和快速运行

本次验证环境为 Python 3.11.7、NumPy 1.26.4、Pillow 10.2.0。核心算法只需要 NumPy，图像读取使用 Pillow；Matplotlib 负责生成图，pytest 负责测试，nbformat 用于 Notebook 文件。

在本目录打开 PowerShell：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python download_data.py orl
python run_experiment.py --dataset orl --data-dir data/orl --output-dir outputs/my_orl --seeds 0 1 2
```

公开仓库保留代码、报告、Notebook 和实验记录，原始人脸数据集与论文全文提取文件保存在本地。克隆仓库后需要单独下载数据；下载脚本验证 ZIP、拒绝越界路径，仅解压图像和说明文本：

```powershell
python download_data.py orl
python download_data.py lfwa
```

ORL 来自 [剑桥大学官方存档](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)，使用时注明 AT&T Laboratories Cambridge。LFW-a 来自 [Tal Hassner 作者页面](https://talhassner.github.io/home/projects/lfwa/)，使用与引用要求见该页。脚本使用作者页提供的 Google Drive 文件；若下载链接失效，可从作者页取得 ZIP 后运行 `python download_data.py lfwa --archive 路径.zip`。本项目不以其它对齐版本替换 LFW-a。

## 完整复现实验

本次运行未缩放、裁脸或做直方图均衡化，使用全部原始尺寸。ORL 每人 5 张训练、5 张测试。LFW-a 每次在图片数大于 10 的人中随机取 14 人，每人 5 张训练，其余全部测试。种子改变时，LFW-a 的身份集合也改变。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python run_experiment.py --dataset orl --data-dir data/orl --output-dir outputs/orl --seeds 0 1 2 3 4 5 6 7 8 9
python run_experiment.py --dataset lfwa --data-dir data/lfwa --output-dir outputs/lfwa --seeds 0 1 2 3 4 5 6 7 8 9
python visualize_results.py --metrics outputs/orl/results.csv outputs/lfwa/results.csv
```

同一输出目录重跑会覆盖同名实验文件；保留对比实验时请使用新的目录。图脚本重跑会更新同名图。

每个实验目录包括：

| 文件 | 用途 |
|---|---|
| `experiment.json` | 数据、环境、算法参数、预处理和假设 |
| `image_manifest.json` | 原图路径、标签、尺寸及 SHA-256 |
| `split_seed_N.json` | 每次的身份与训练/测试图片清单 |
| `results.json`、`results.csv` | 每种方法每次运行的正确数、分母、准确率、维数及耗时 |
| `aggregate.json`、`aggregate.csv` | 各种子的均值和样本标准差 |
| `predictions.csv` | 逐张预测身份、余弦分数及命中的训练图像 |

`accuracy_percent` 是该次测试中正确图片数/测试图片数；`macro_subject_accuracy_percent` 先计算各人的准确率再等权平均。汇总中的 mean 是各 seed 的等权平均，std 使用 `ddof=1`；种子划分重叠，标准差只描述这些划分的波动，不能当作独立样本置信区间。

## 9 种配置的含义

| CLI 方法名 | 处理流程 |
|---|---|
| `raw` | 灰度像素展开 |
| `lbp_raw` | 普通 dense LBP，`>=`，编码图展开 |
| `lbp_hist` | 普通 LBP，8×8 网格，每格 256 bins，格内 L1 归一化后拼接 |
| `enlbp_mean_gt` | 默认正文解释：非重叠 3×3 均值 → dense LBP，严格 `>`，一层 |
| `enlbp_mean_ge` | 同上，比较规则改成 `>=` |
| `enlbp_mean_l2` | 默认均值解释，重复两层 |
| `enlbp_mean_block` | 均值 → 步长 3 的非重叠 LBP，一层 |
| `enlbp_algorithm` | 伪代码解释：原图 dense LBP → 步长 3 再编码，`>=`，一层 |
| `enlbp_algorithm_l2` | 同上，两层降维 |

所有配置使用相同的余弦最近邻。位序固定为右、右上、上、左上、左、左下、下、右下，权重依次为 1、2、4、8、16、32、64、128。均值保留 float64；池化裁掉底部/右侧不完整块，LBP 使用 valid 邻域。整数图先求和再除以面积，保留相同块总和的相等关系。

代码已验证论文图 1 编码为 237；图 2 严格 `>` 为 246，`>=` 为 254。论文未说明边界、编码窗口重叠方式、重复层数、完整实验划分及基线参数，所以这些配置是公开假设下的重建和敏感性分析。未实现 LPQ、Fisher Faces、Multiradii LBP、2DPCANet；原表中这些方法的数字只作为论文报告值引用。

## 使用自己的图片

图片按身份放在子目录中：

```text
my_faces/
  person_a/1.png
  person_a/2.png
  person_b/1.png
  person_b/2.png
```

```powershell
python run_experiment.py --dataset folder --data-dir my_faces --output-dir outputs/custom --train-per-person 1 --min-images 2 --methods raw enlbp_mean_gt --resize 92 112 --seeds 0
```

`--resize` 顺序是宽、高；默认不缩放。每人必须至少留一张测试图。封闭集算法会把查询分配给图库中的某一人，余弦分数不是概率；本项目没有陌生人拒识或自动人脸对齐模块。

## 分析过程来源

本项目采用 PDF 阅读、科学批判思维与科学可视化技能，分别用于核对公式图示、审查证据与呈现真实测量；相关方法来源已在解析和实验报告中列出：Kassis et al. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。它不是 EnLBP 算法来源。
