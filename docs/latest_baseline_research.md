# 近期公开基线初筛（截至 2026-10-11）

范围：只核对 2025–2026 作者论文、作者仓库及其核心配置/数据入口，关注 MOSEI 与 CH-SIMS v2.0 的真实支持、代码/权重可得性、许可证、监督信息和评测类别。此报告用于选择可复现路径；论文分数使用不同标签映射时不可直接横比 JevAffect 五分类结果。

## 建议

建议先以 **CMC 的 pseudo-label 配置作为可审计主复现锚点**，并将它适配到共同的五分类协议；CMC 的代码采用 MIT，仓库明确包含 SIMS-v2 与 MOSEI 的数据路径和训练入口。把 CMC-GT 单独列为额外监督参照，不能与普通训练结果混称。CMC 原论文主要报告二分类指标，因此其已发表成绩不能直接当作五分类成绩。

TA-OEM 与 SentiLLM 是更近期、且代码中存在 v2 路径的候选，但当前没有明确代码许可证；TA-OEM 默认配置还只使用 2,272 条训练样本。暂不应把任一项当成权利与数据设置均已核清的主复现方案。BC-MLF 是最接近 Jev 分支校准思路的近期作者实现，但仓库目前配置旧版 SIMS，不是 SIMS v2。

## 候选比较

| 方法 | 作者代码/版本和数据支持 | 权利与权重 | 标签、训练和评测核查 | 结论 |
|---|---|---|---|---|
| **CMC** | [作者仓库](https://github.com/gw-zhong/CMC)，最新检查到 commit [`c695f719`](https://github.com/gw-zhong/CMC/commit/c695f719f9cd0136e52dc2f7a9e75a40aabd37ad)（2025-10-24）；README 给出 `python main.py --dataset SIMS-v2`，支持 SIMS-v2、MOSEI。数据入口读取 CH-SIMS-v2 的 `Processed/unaligned.pkl`。 | [仓库 LICENSE](https://github.com/gw-zhong/CMC/blob/main/LICENSE) 为 MIT。README 提供百度网盘预训练权重链接（提取码 `2rtn`），但仓库 MIT 不自动覆盖这些权重或数据的权利；下载权重可用性和其单独条款仍需实机核验。 | [作者论文](https://arxiv.org/html/2510.20256v1)报告 CH-SIMS v2 labeled split 为 train/valid/test = 2,722/647/1,034；未使用 v2 的 10,161 无标签部分。MOSEI 为 16,326/1,871/4,659。默认模型头为 3 类，论文主要报告 Acc2；CH-SIMS/v2 的二分类按非正（≤0）/正（>0），MOSEI/MOSI 的 Acc2 与 F1 也不是五分类结果。论文实验配置：Python 3.10、PyTorch 2.7.1、单 RTX 4090D、最多 100 epochs、patience 10、batch 64、初始 LR 1e-3，并调层数/头数/dropout。`--is_pseudo` 启用伪单模态监督；关闭后为 CMC-GT，读取真实 T/A/V unimodal labels，形成额外标签优势，应分开报告。 | **推荐首个复现锚点**：代码许可清楚且确实有两数据集路径；锁定 pseudo 配置作为主对照，CMC-GT 作为独立额外监督对照。为共同五分类协议适配后重新评测，不能引用论文 Acc2 作为同协议结果。 |
| **TA-OEM** | [作者仓库](https://github.com/lintonao/TA-OEM)，最新检查到 commit [`3834b379`](https://github.com/lintonao/TA-OEM/commit/3834b3798c85b8531711ad94c6079c74a00dff31)（2026-01-06）；README 写有 CH-SIMS v2 `unaligned.pkl` 与 MOSEI 数据；`run.py`/dataloader 中确有 `sims3l` 读取 `data_set/ch_simsv2/unaligned.pkl` 的路径。 | 未找到明确 LICENSE，不能假定可再分发或改编。README 要求用户自行准备数据和中英文 BERT；未核实仓库中有可直接使用的模型权重。论文：[Knowledge-Based Systems DOI](https://doi.org/10.1016/j.knosys.2025.115149)。 | 配置以 `sims` 命名而非 `simsv2`；当前 config 的 `train_samples` 与 `supvised_nums` 是 2,272，低于官方 v2 train 的 2,722。默认是回归，评估代码提供 Acc5/Acc3/Acc2、F1、MAE、Corr 等。SIMS loader 读取 `regression_labels_T/A/V`，所以其训练路径可接触真实单模态标签，需作为监督条件记录。论文摘要的 CH-SIMS 表述与代码 README 的 v2 说法不完全等价；本次未从论文正文核实全套 v2 实验。 | 近期候选，但许可证和训练子集未解决前不锁为主基线。若适配，需先核对论文实验配置、完整训练行数和 T/A/V 标签使用。 |
| **SentiLLM** | [作者仓库](https://github.com/especiallyW/SentiLLM)，配置包含 MOSEI 和 CH-SIMS v2；论文 [arXiv HTML](https://arxiv.org/html/2607.27790)（2026-07-30）。 | 仓库未见 LICENSE，也未见打包模型权重；需另行取得数据及 Hugging Face/ModelScope LLM checkpoint，权利条件需分别核验。 | v2 配置使用 Qwen2.5-1.5B，LLM 冻结，30 epochs、batch 16、gradient accumulation 2、seed 42；MOSEI 配置 30 epochs。训练目标为回归，报告 Acc5 等多种指标。论文报告 RTX 5090 32GB。与项目五分类仍须核验分箱定义并统一映射后重算。 | 有相关近期论文和双数据配置，适合作为前沿结果候选；当前许可、外部权重和可复现完整性不足以作为已清权利的主复现基线。 |

## 重要近邻与排除项

- **BC-MLF**：[作者论文](https://arxiv.org/abs/2609.37100)（2026-09-29）、[作者仓库](https://github.com/sunyulin0421/BC-MLF)，最新检查 commit [`428b973b`](https://github.com/sunyulin0421/BC-MLF/commit/428b973be580fd3cf09dd51bb61c448f2a26b2c9)，仓库 MIT。其 prediction-layer branch calibration 与 Jev-inspired 校准最接近；但作者当前配置实际是 MOSEI + 旧 SIMS（`SIMS/Processed/unaligned_39.pkl`），没有 SIMS-v2 配置。作者 README 称已投 ICASSP’27；截至检索日不能视为已发表接收。适合作为架构参考或后续移植，不是原样 v2 基线。
- **GMEL (AAAI 2026)**：[作者仓库](https://github.com/zjutangk/GMEL)支持 README 所列 MOSEI/MOSI/CH-SIMS，但未发现 SIMS-v2 配置；主要方案是预训练后做 test-time adaptation，评测协议也不同，故不作为同协议主对照。
- **TS-DTF (ACM MM 2026)**：[作者仓库](https://github.com/gaosen1/ts-dtf)面向文本+音频，不含视觉模态；仓库示例配置以 MOSI 为主，不满足三模态的 MOSEI + SIMS-v2 对照要求。

## 决策边界

1. 先跑 CMC pseudo 的可重复公开方案，并将 CMC-GT 单列；确认其适配版输出能以项目统一离散映射重新评分。
2. 若将 TA-OEM 或 SentiLLM 作为候选复现，先解决其代码许可证/权重条件，并固定实际使用的训练样本和是否使用真实 unimodal labels。
3. 所有结果需在相同 split、标签定义和五分类评测器下报告；Acc2、Acc5 的论文结果不可跨映射直接宣称优劣。

本次没有下载数据或模型、运行训练/GPU，亦未检查数据分发许可。仓库中的权重链接不等于核验过下载内容、哈希、许可证或可复现性。
