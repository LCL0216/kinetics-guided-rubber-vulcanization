# 硫化曲线的动力学引导机器学习预测

本文代码对应文章 **Kinetics-Guided Machine Learning for Interpretable Prediction of Rubber Vulcanization Curves**，包含特征提取、动力学拟合、回归器比较、曲线重构、消融分析、SHAP 与置换重要性、直接模型对照、六条独立曲线验证及正文和 SI 绘图。

## 快速运行

推荐 Python 3.12。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_pipeline.py --stages 09 --no-plots
```

程序先校验输入和模型的哈希，再解压到独立的 `work/` 工作目录。该命令加载已用 88 条建模曲线训练并冻结的模型，只预测 6 条独立验证曲线，不重新训练。

输出位于 `work/Stage9_一键预测/Stage9_Output/`。需要同时画出六条曲线及误差图时，去掉 `--no-plots`。

## 分阶段运行

```bash
# 从原始建模曲线提取特征、拟合动力学并建立曲线级数据表
python run_pipeline.py --stages 01 02 03 --no-plots

# 按依赖顺序运行完整分析和正文、SI 绘图
python run_pipeline.py --all

# 仅查看命令，不执行
python run_pipeline.py --all --dry-run

# 校验输入、代码语法和六条曲线的模型推理结果
python checks/verify_release.py
```

完整流程包含重复交叉验证、嵌套验证、置换重要性和高分辨率出图，耗时明显长于六条曲线预测。脚本中的少折数、少重复次数选项只用于调试，不代表论文正式结果。可用 `--work-dir work-new` 选择另一个工作目录。

## 文件组织

- `scripts/`：17 份文章分析与绘图代码，使用可读的英文名称并保留阶段编号。
- `exploratory/`：两份探索性神经网络和幅值目标对照代码，不在默认论文流程中运行；运行神经网络版本还需安装 `requirements-exploratory.txt`。
- `data/`：六份科学输入、冻结模型及 SHA-256 来源记录。原始逐点训练数据使用 gzip 压缩，数值保持原样。
- `reference/`：23 份原始数值结果表，用于对照，不作为新计算结果。
- `docs/stage_mapping.csv`：原文件名与当前文件名对应关系。
- `docs/code_provenance.json`：原代码和整理后代码的哈希，以及保持原样的科学函数清单。
- `checks/`：验证脚本与实际验证结果。

重新生成的中间文件、模型、图像及日志保存在工作目录中，不重复提交到仓库。

## 与文章一致的使用原则

88 条建模曲线和 6 条独立验证曲线分别编号，两个数据集的相同编号不表示同一条曲线。验证曲线不参与模型训练。七个工艺输入预测 `ML`、`MH`、`ti`、`k1`、`n1`，其中 `k1` 使用 log10 变换。前向预测中的诱导边界值来自训练数据，而非待预测曲线的实测值。

曲线模型对照以整条曲线划分五折；使用实测 ML/MH 的分析属于诊断，不能与完整前向预测成绩混用。固定 `n1=1` 的对照会重新拟合速率常数。

数据集构建阶段检出：88 条建模记录包含 33 组不同的时间—扭矩曲线，其中 67 条记录属于有重复曲线的组。归档的 Stage 04/08 按曲线编号划分，而非按 `curve_group_id` 分组，因此已有成绩不能证明对完全未见曲线组的泛化。原代码中的提示和现有结果均保留；按组重新评价属于后续独立的科学分析。

本次整理保留科学算法、模型设置与原始数据，调整文件名、模块导入、帮助文字、本地缓存路径和绘图工作目录。保留原有结果文件夹命名，便于核对数据依赖。不同软件版本的重拟合或归因结果可能有差异，原脚本中的数值一致性检查仍然保留。

冻结模型自带原保存环境信息；依赖文件记录本次代码包验证使用的环境。此次发布未重新运行完整交叉验证和全部图件，实际完成的检查记录在 `checks/verification.json`。

文稿、Zotero 数据库、历史备份和本地缓存不在本代码仓库中。
