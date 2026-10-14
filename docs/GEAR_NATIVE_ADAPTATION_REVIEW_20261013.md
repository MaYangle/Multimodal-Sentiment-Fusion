# GEAR 在原生 M 回归上的适配审查

日期：2026-10-13。只读审查；没有改代码、训练、调用 GPU、读取数据集 TEST，或检查 TEST 结果。本审查以论文方程、作者 GitHub 在审查日可见的最新 commit、以及现有 TRAIN source-CV 诊断为依据。

## 实施判断

GEAR 是与当前问题最接近的已发表 MSA 多模态去偏方法，值得作为成熟方法对照；但作者链接的仓库在可核实的最新提交里没有论文所述 GEAR 的 GMAE/IPW 实现。不要把该仓库里的 IFL_ES 代码或普通 GroupDRO/DANN 结果标为 GEAR。若要实验，建议另写一份明确标成“按论文公式独立重实现的 GEAR adaptation”，只训练轻量的冻结特征适配器；不能称为作者代码复现。该适配在 RTX 5070 Ti 16 GB 上可行，且不需要模态专属真值或人工 OOD 训练数据。

现有 CH-SIMS source-OOF 结果给出采用 GEAR 的动机，但没有证明 GEAR 会改善来源迁移：text-256 MAE 为 `.39477/.39431/.39171`，text-640 为 `.39639/.39607/.39712`，fusion-256 为 `.41941/.41489/.40540`（seed 13/42/2026）。Fusion 相对 text-640 的额外 MSE 平均为 `.011707`，其中来源均值偏移仅 `.003112`（26.6%），组内误差变化为 `.008595`（73.4%）。GEAR 是按样本、按模态偏置预测构造权重，不是固定 source-video 截距校正；应把它看作针对跨模态伪相关的来源留出基线，而非已知症状的直接解法。source group 是视频 ID，不是经过核实的说话人。

## 论文中的 GEAR 机制

[原论文](https://arxiv.org/html/2307.10511v2)把任务定义为输入 `(T,A,V)`、监督一个多模态情感分数 `Y` 的连续回归。核心不是“去掉来源 ID”，而是估计每条样本依赖模态偏置线索的程度，并降低这类样本对鲁棒预测器的训练影响。

1. **模态内鲁棒/偏置分路。** 每个模态各有一个鲁棒提取器 `E_R^m` 和一个偏置提取器 `E_B^m`，输出同维 `d_s` 向量。论文用预训练 BERT 的最后层 `[CLS]` 表示文本，用一层 LSTM 编码手工音频/视觉序列。把两向量拼接后，再由不同线性层映射为 `f_R^m` 和 `f_B^m`。
2. **同批潜向量交换。** 在一个 mini-batch 内随机替换样本的偏置向量，保留它自己的鲁棒向量，形成 `[v_R^m; v_B^{m,donor}]`。偏置头对原样本和交换样本都用 GMAE 学习：原样本监督目标为该样本的 `Y`，交换偏置向量的监督目标为 donor 的 `Y_donor`。鲁棒融合头在交换视图上仍以原样本 `Y` 为目标。论文在若干 epoch 之后才启动交换。
3. **GMAE 偏置头。** 对绝对误差 `e=|Y-Y_B^m|`，论文定义 `L_GMAE(e)=-2 log(exp(e)+1)+2e`，等价于数值稳定的 `-2 softplus(-e)`。它让偏置预测器更重视早期“容易”预测的样本；论文给出的梯度权重为 `2/(1+exp(e))`。这是回归版的偏置建模损失，不是普通 MAE，也不需要人工偏置标签。
4. **样本级偏置估计与 IPW。** 从三种偏置头的误差得到 `ψ_min=1/min_m |Y-Y_B^m|` 或 `ψ_avg=1/avg_m |Y-Y_B^m|`。论文写出 `P(x|Bias(x)) ∝ ψ · |Y-Ŷ|`，其中鲁棒预测误差不回传梯度，再用 `1/(P+1)` 缩小 IPW-MAE。方程中的 `∝` 没给出唯一归一化常数；独立实现必须固定并记录这个系数、`ε` 和数值裁剪规则，所有统计量只能从当前训练折估计。
5. **只用鲁棒表示做预测。** 将三个 `f_R^m` 堆叠，对模态维做 multi-head self-attention，再拼接并用线性回归头预测原生连续 `M`。总目标为 `L_IPW + λ L_GMAE + β(L̂_IPW + λ L̂_GMAE)`；交换分支整体由 `β` 控制，`λ` 控制偏置学习项。

因此，GEAR 与来源 GroupDRO/DANN 的优化量不同：GroupDRO 改变来源组的风险权重；DANN 用来源判别器作对抗不变性；GEAR 用三条模态偏置支路、GMAE、同批潜变量交换和样本级 IPW。删去 GMAE、偏置估计、交换或 IPW 中关键环节后，只能称作 GEAR-inspired 消融，不能报告为完整 GEAR。

## 论文机制与本地适配边界

| 层面 | GEAR 论文机制 | 冻结特征适配如何保持忠实 | 何时只能称为 GEAR-inspired |
|---|---|---|---|
| 输入与目标 | 三模态 T/A/V 输入；一个共享的连续情感目标 Y | 复用本地冻结 T/A/V 表示，只监督原生 M | 改用只有 source ID 的域标签，或把任务改成分类 |
| 偏置估计 | 每模态 biased branch + GMAE + 同批 latent swap；从三个偏置误差估计样本级 ψ | 保留三个偏置分支、GMAE、donor swap、ψ 与 IPW 权重 | 只做来源均衡采样、GroupDRO、DANN，或只按模态不确定性加权 |
| 鲁棒预测 | 三个 robust representation 经多头 self-attention 融合，使用样本 IPW-MAE | 保留 robust fusion、原样本和交换视图两项回归损失 | 去掉 IPW/GMAE/swap 后仍把简化模型报告为 GEAR |
| 评测 | 论文为 MOSI/MOSEI 构造 modality/OOD 测试集 | 使用 source-video 外层留出作为本项目自己的迁移协议 | 把 source-video 留出结果说成论文 OOD benchmark 复现 |

冻结特征只替换编码器、输入表示和数据划分；GMAE、交换、偏置估计、IPW 与鲁棒融合这条机制链必须保留。按论文公式独立实现仍不是作者代码复现；而只借其中一两项则要按 GEAR-inspired 另行命名。

## 作者仓库与授权核验

审查日 GitHub default branch `main` 最新可见提交是 [cab45ac62bd7f7880b5abadd15ff5136cad91761](https://github.com/Teng-Sun/GEAR/commit/cab45ac62bd7f7880b5abadd15ff5136cad91761)（2023-08-05，提交内容只更新 README）。[该 commit 的 src tree](https://github.com/Teng-Sun/GEAR/tree/cab45ac62bd7f7880b5abadd15ff5136cad91761/src)只有 `IFL_ES.py`、`IFL_LS.py`、MMSA 数据/训练工具等；没有以 GEAR 命名的模型/训练模块。代码搜索该默认分支也未找到 `GMAE`、`IPW` 或 `bias_weight`。README 仍称仓库为 General Debiasing for MSA，但代码入口 [src/run.py](https://github.com/Teng-Sun/GEAR/blob/cab45ac62bd7f7880b5abadd15ff5136cad91761/src/run.py) 导入 `models.IFL_ES.IFL_ES`；[src/models/IFL_ES.py](https://github.com/Teng-Sun/GEAR/blob/cab45ac62bd7f7880b5abadd15ff5136cad91761/src/models/IFL_ES.py)开头明确标注来源为 [thuiar/Self-MM](https://github.com/thuiar/Self-MM)，实现含 GCE、模型分路与交换逻辑，但没有论文的 GMAE 样本权重和 IPW。作者 [src/trains.py](https://github.com/Teng-Sun/GEAR/blob/cab45ac62bd7f7880b5abadd15ff5136cad91761/src/trains.py)也会在训练流程中加载 `test_nn_tav` 做逐 epoch 评估；不能原样用于本项目严格的外层 source-heldout 实验。

仓库包含 [MIT LICENSE](https://github.com/Teng-Sun/GEAR/blob/cab45ac62bd7f7880b5abadd15ff5136cad91761/LICENSE)，正文署名为 `Copyright (c) 2020 iyuge2`，并要求在复制或实质性副本中保留版权与许可声明。该文本与其显式来源 [Self-MM LICENSE](https://github.com/thuiar/Self-MM/blob/main/LICENSE)一致。仓库列出了 [requirements.txt](https://github.com/Teng-Sun/GEAR/blob/cab45ac62bd7f7880b5abadd15ff5136cad91761/requirements.txt)（含旧版 PyTorch 1.11、Transformers 4.3.0 等）；仓库 MIT 文本不自动授权数据、预训练权重或其他依赖，也不能仅凭根 LICENSE 推断所有子文件作者和来源。若复制上游实现，保留 MIT 和 Self-MM 归属；若实现论文算法，明确引用论文并把这次实现记为本项目重实现。

## 最小忠实可行的冻结特征版本

建议固定为**三模态、单一原生 M 回归**，使用现有 TRAIN 的冻结 T/A/V 特征；不微调 MacBERT、音频或视觉编码器。对文本保留现有冻结特征的 CLS 表示；音频和视觉若缓存为时间序列，使用与现有特征长度掩码一致的一层小 LSTM，否则使用现成 segment 表示。所有对照必须读取同一特征字节、同一原生 `M∈[-1,1]`，不对标签做跨数据集标准化。

每个模态设置独立的鲁棒和偏置投影，将固定输入映射到 `d_s=32`；保留论文的拼接线性 `L_R/L_B`、三个偏置回归头、四头模态 attention 和鲁棒回归头。偏置支路使用原论文 GMAE；同一训练 batch 内用随机 permutation 交换偏置 latent，偏置头以 donor 标签监督，鲁棒头仍以 receiver 标签监督；达到预定 warm-up epoch 才开始交换。用原始样本的三个偏置绝对误差计算 `ψ_min` 或 `ψ_avg`，与 detach 的鲁棒预测误差相乘，再按 `1/(1+P)` 重加权点态 MAE。验证时不交换；验证、OOF 和最终推理都使用鲁棒支路与鲁棒融合头。

论文 MOSI/MOSEI 起始设置为 batch 32、`d_s=32`、4 个 attention heads、`β=.3`、早停 patience 8；MOSI/MOSEI 报告的 `λ` 分别为 18/10、swap 起始 epoch 为 11/8。论文优化器给 BERT `5e-5`、其他模块 `1e-3`。冻结特征版不含 BERT 更新，可从其他模块 `1e-3` 起步。CH-SIMS 标签范围较窄，不能照搬 `λ=10/18` 当作最优；应在外层训练源内部，以按 source video 留出的 inner validation 选择 `λ`、`ψ_min/avg`、swap 起始 epoch及是否微调 `β`。结构先固定为 `d_s=32`、4 heads、batch 32，避免小型 CV 上扩大搜索。保留论文数值作起点，并报告所有本地改动；将 `P` 比例常数、`ε` 和权重裁剪写入配置。

这不是官方 GEAR 代码复现：输入表示从 BERT/手工序列变为本地 frozen feature，标签量纲、数据、训练折和评价协议也都不同。若结果进入论文或发布，名称应为“GEAR paper-based frozen-feature adaptation / our reimplementation”，并说明配置与差异。

## 公平评测和来源泄露控制

- 与现有 sourcefold3、seed 13/42/2026 及 source-OOF 划分完全相同；每个外层折仅用剩余 source video 训练。early stop、学习率、`λ`、`β`、`ψ`、交换起始 epoch 和归一化都在该外层训练来源内部的 inner source split 决定，绝不查看 outer-held source 的标签。
- 偏置 donor 只从同一 outer-fold 的训练 batch 抽取。held-out 来源不能进入 donor 池；不得把所有外折 OOF 预测拼在一起再训练一个 GEAR 校准器或 IPW 头。source ID 只用于折划分、按组汇总和整组 bootstrap，不输入模型或作为偏置标签。
- 不使用官方 VALID/TEST 选择任何超参；外层 OOF 是来源迁移主结果。冻结特征固定来源可以复用，但任何特征标准化、数据统计或估计的权重尺度必须在每个 outer train 内单独拟合。上游 `src/trains.py` 的 test 评估路径不适用于这里。
- 比较至少包括同一输入/折/种子的 Text-ERM、Fusion-ERM、参数量或更新预算匹配的融合 ERM、完整 paper-based GEAR；再做去掉 IPW、去掉 swap、去掉 GMAE 的消融以确认收益来自哪些模块。主 agent 正在实现的 GroupDRO/DANN 应作为单独来源去偏基线，同样匹配 sourcefold、seed、原生 M 和内层选择预算。报告总可训练参数、实际更新数、每折 MAE 和整组 bootstrap 区间，不能把计算量更大的模型提升单独归因于 GEAR 机制。

现有缓存特征下，训练模块只是 6 个 32 维分支投影、偏置头、小型融合网络以及（若缓存保留序列）浅层音视频 LSTM；每批只对 3 个模态 token 做 4-head attention。预计显存远低于全量微调 BERT，16 GB 足以进行这类冻结特征训练。这里是结构估计，没有在该机器上运行或测时；仍应先做单折 smoke，再按授权配置正式运行。

## Primary sources

- [GEAR 原论文 arXiv HTML](https://arxiv.org/html/2307.10511v2)，尤其 §3.3–3.5 的分路、swap、GMAE、IPW 与目标函数，以及 §4.1.5 的超参。
- [作者仓库在审查日最新 commit](https://github.com/Teng-Sun/GEAR/commit/cab45ac62bd7f7880b5abadd15ff5136cad91761)、[该 commit 的 src tree](https://github.com/Teng-Sun/GEAR/tree/cab45ac62bd7f7880b5abadd15ff5136cad91761/src)、[IFL_ES 模型](https://github.com/Teng-Sun/GEAR/blob/cab45ac62bd7f7880b5abadd15ff5136cad91761/src/models/IFL_ES.py)、[训练入口](https://github.com/Teng-Sun/GEAR/blob/cab45ac62bd7f7880b5abadd15ff5136cad91761/src/trains.py)、[MIT LICENSE](https://github.com/Teng-Sun/GEAR/blob/cab45ac62bd7f7880b5abadd15ff5136cad91761/LICENSE)。
- [Self-MM 来源仓库与 MIT LICENSE](https://github.com/thuiar/Self-MM)；其许可证与 GEAR 仓库根 LICENSE 同文。
- 本地 [source-CV 容量结果](NATIVE_SOURCE_CAPACITY_FINDINGS_V1.md)提供上文数值和 MSE 误差分解依据。
