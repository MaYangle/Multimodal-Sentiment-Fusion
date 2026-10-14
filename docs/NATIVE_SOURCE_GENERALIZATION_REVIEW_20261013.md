# AV 来源干扰与未见来源情感回归：机制及先例审查

日期：2026-10-13。只读审查本地已完成的 AnchorCDF/source-CV 证据与指定原始论文、作者代码；没有运行训练、没有读取或评估 TEST，也没有修改训练模块。CH-SIMS 原生标签范围 `[-1,1]` 与 MOSEI `[-3,3]` 在本审查中始终分开。

本文的 source group 指片段 ID 中的 source video（如 `video_xxxx/clip`），不是已核实的说话者身份；没有独立 speaker ID 时，实验结论只能称 source-video-heldout，不能称 speaker generalization。

## 判断

现在还不能把“AV source nuisance 导致未见来源回归变差”写成已证实机制，也没有证据支撑一个新的方法贡献。现有 TRAIN source-OOF 比较提供了值得查证的线索：同一目标 M、同一 source folds/seed 和相同 MLP 架构下，融合相对 text-only 的 MAE 差为 `+0.02464 / +0.02058 / +0.01369`（三个种子均变差；text 为 `.39477/.39431/.39171`，fusion 为 `.41941/.41489/.40540`）。但这只说明这些冻结特征和拟合条件下的 source-heldout 平均误差不同，尚未隔离 AV 的来源风格、训练样本量差异、优化与选择误差。

官方 VALID 上 fusion MAE 为 `.42132/.42279/.42209`；source-inner-selected fullText 对应 `.44115/.44085/.44087`。官方 VALID 的646/647条与 TRAIN 同源，故这组相反方向的差异不能当作“融合只在已见来源有效”的因果证据。TRAIN source-OOF 中 held source 确实从对应折模型中隔离；但校准器重用整张 OOF 表时，内层来源仍可能已进入其他折教师的训练。若要声称端到端来源迁移，需在外层 held-source pipeline 内重新生成教师、拟合与选择。

AnchorCDF 也不支持把模态分歧当来源风险代理：三个种子的 source-OOF spread 与点误差 Spearman 约为 `−.054/−.038/−.042`，VALID 约为 `+.052/−.014/+.009`。disagreement CDF 的原生分箱 log score 三种子均略差于同预算 feature-only mixture：`1.32677/1.32411/1.32859` 对 `1.32656/1.32343/1.32823`（越低越好）。这不是 AV 来源效应的证伪，但足以拒绝用 teacher 分歧 gate 代替来源诊断。

## 最窄、可证伪的候选机制

先做来源分解诊断；若融合 OOF 的额外误差主要落在每个 source video 的**组均值有符号误差**，而同组内情感变化的误差并未同步变差，才值得用 CPU 做一个受限实验：训练时对同一来源的视频片段对增加来源内预测差分监督，检验它能否抑制固定来源截距/录制风格对回归的牵引，同时保留点态目标对绝对分数的锚定。

对 source `g` 内两个片段 `i,j`，以现有点态损失 `L_point` 加一项：

```text
L_within = mean_g mean_{i<j, i,j∈g} ρ( [f(x_i)−f(x_j)] − [y_i−y_j] )
L = L_point + λ L_within
```

假设是：若背景、说话者/录制风格在同一来源内近似固定，它们会在标签差分中抵消；加权后的训练信号更强调同一来源中随情感改变的证据。这个目标**不会**自动让模型 source-invariant，也不会消除随片段变化的来源风格交互。它只检验 source-matched 差分监督是否能让同一 MLP 学到对未见 source 更稳定的变化关系。该假设**不等于**证明 source effect 是因果 nuisance：同源片段的标签跨度、片段数量与主题结构也可能不同。点态损失必须保留，否则只学差分不能定出新来源的绝对偏移。每组等权，并限制每组每批配对数，避免大视频的 O(n²) 配对垄断训练。

这只是可检验的 baseline 候选，不应现在命名为原创算法。within-group 去均值/差分是固定效应的经典变换；通用 pairwise regression 也已有先例。更可辩护的研究问题是：在 MSA 的源视频分组下，source-matched 的连续分数差分是否比普通点态回归及成熟来源去偏方法更能改善**端到端未见 source video**的 M native MAE。即使通过一次 CH-SIMS 检验，仍需要在独立 MOSEI `[-3,3]` 协议复验；不得把目标缩放后合并训练或把跨数据集分数当同一量纲。

## 最接近的原始先例与边界

| 先例 | 原论文与作者代码 | 与候选的关系 |
|---|---|---|
| **S²LIF, “Learning in Order!”** | [arXiv 2409.04473](https://arxiv.org/html/2409.04473)，[ACM MM 2024 DOI](https://doi.org/10.1145/3664647.3681056)。论文采用文本先行的稀疏特征选择，再条件化地选视觉特征；跨 MOSI/MOSEI/MELD 的单源/多源 OOD，三分类极性任务。它把“先文本、再受文本引导的非文本信息”作为明确方法主张，不能将 text-priority 或 sequential fusion 改名原创。它的 source domain 是数据集/语料层级，不是 CH-SIMS 的单条 YouTube source video；不含 audio。论文/arXiv 页仍称代码将在录用后公开，当前审查未找到作者链接的官方仓库，不能声称已复现其代码。 |
| **GEAR, General Debiasing for MSA** | [原论文 HTML](https://arxiv.org/html/2307.10511v2)，[作者仓库](https://github.com/Teng-Sun/GEAR)，[作者训练代码](https://github.com/Teng-Sun/GEAR/blob/main/src/trains.py)。论文使用鲁棒/偏置模态提取器、偏置表示 swap、GMAE 学偏置样本及 IPW 降低高偏置样本权重；目标是 MSA OOD 去偏。代码明确以 regression 模式运行并读取 M 标签。这是本问题最直接的 MSA 多模态 debiasing baseline，任何“AV nuisance weighting/disentangling”都必须先对它。它没有利用同一源视频内的监督差分；论文 OOD 构造与本项目 leave-source-video-out 协议也需对齐后再比较。 |
| **MISA** | [原论文](https://arxiv.org/abs/2005.03545)，[作者代码](https://github.com/declare-lab/MISA)。学习模态共同与模态特有子空间并融合，覆盖 MOSI/MOSEI 的连续回归及分类。它处理的是模态差异和共有/私有表示，不是来源视频不变性；“让 A/V 表示 invariant/private”本身不是新的 source nuisance 方法。 |
| **DANN / conditional domain invariance** | [DANN 原论文](https://arxiv.org/abs/1505.07818)，[DANN/CDANN/GroupDRO 等作者实现集合 DomainBed](https://github.com/facebookresearch/DomainBed)；[条件不变 adversarial DG](https://openaccess.thecvf.com/content_ECCV_2018/html/Ya_Li_Deep_Domain_Generalization_via_Conditional_Invariant_Adversarial_Networks_ECCV_2018_paper.html)，[条件不变表示](https://ojs.aaai.org/index.php/AAAI/article/view/11682)。DANN 对 source ID 做 adversarial 去信息；conditional 版本对表示与标签的联合分布/类别条件做对齐。候选差异是不用域判别器逼 representation 边缘不变，而以已知同视频成对连续标签变化训练回归差分。可是 conditional alignment 已直接覆盖“给定标签移除来源信息”，而连续标签如何条件化也必须做成熟对照，不能仅凭换损失宣称原创。 |
| **IRM / GroupDRO** | [IRM 原文](https://arxiv.org/abs/1907.02893)，[GroupDRO 原文](https://arxiv.org/abs/1911.08731)，[GroupDRO 作者代码](https://github.com/kohpangwei/group_DRO)。IRM 约束各环境最优预测规则一致；GroupDRO 优化预定义组的最坏风险且依赖足够正则化。两者不构造同源片段差分，但直接针对 train source groups，必须是来源迁移的已知基线。将 129 个 source video 作为小样本环境时要报告组大小/有效样本数；不得把每组 MAE 波动简单解释为 predictor mechanism change。 |
| **线性 nuisance/nullspace 去除** | [INLP 原论文](https://aclanthology.org/2020.acl-main.647/)，[作者代码](https://github.com/shauli-ravfogel/nullspace_projection)；[LEACE 原论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/d066d21c619d0a78c5b557fa3291a8f4-Paper-Conference.pdf)，[作者代码](https://github.com/EleutherAI/concept-erasure)。可在训练折表示上移除线性可预测的 source ID，作为廉价 post-hoc 基线。标准版本去除的是边缘可读 source 信息；label-conditional 版本须按连续 M 的预注册 native bins 分层拟合/验证，否则容易一并抹掉与情感真实相关的跨组信息。 |
| **Pairwise regression / domain regression** | [DARC: Multi-domain adaptation for regression under conditional distribution shift](https://doi.org/10.1016/j.eswa.2023.119907) 用 `|y_i−y_j|` 定义 pair similarity，并同时保留 within-domain 与 cross-domain pair；其两域实验使用有限标注 target，另有多域实验。它不是本候选的等价 loss/outer-source 协议（PSP 对齐表示距离，本候选直接监督标量预测差），但已经覆盖“域内 pair + 连续标签差”的关键思路，不能说 source-matched pairing 本身新颖。文章/正式页面未核实到作者公开实现。[GAR, ICML 2025](https://proceedings.mlr.press/v267/zhu25x.html) 使用成对标签差分损失来对齐回归梯度，范围明确为无噪 clean regression；[PADRE](https://doi.org/10.1021/acs.jcim.1c00670) 将一般回归改写为样本对差分预测。候选最多检验“source-video 内 signed scalar-difference loss 在无目标组标签的 MSA 外层留源下，是否优于这些已知 pairwise 思路”，不能主张 pairwise regression 或 within-domain pair 本身新颖。 |

因此，候选与 DANN/GroupDRO/IRM 的**优化对象**不同，与 S²LIF 的**任务切分和输入**不同，与 MISA 的**不变性对象**不同，也与 DARC/GAR 的**距离式 pair objective/迁移设定**不同；这些差异定义了实验问题，但单靠差异不足以确立原创。DARC 已有 within-domain 和 cross-domain pair，所以 source-video 同源成对本身不构成新意；可能留下的实证问题只是同源约束用于 MSA native-score regression 的收益能否在严格外层来源留出中独立成立。GEAR 是最接近的多模态去偏先例；DARC/GAR 是差分监督最接近的通用回归先例。

## 最小公平验证与否决条件

1. **先诊断，不训练。** 仅对官方 TRAIN 的 source-OOF Text 与 Fusion 预测按 source video 分组，报告每组样本量、均值标签跨度、组均值有符号误差方差、组内残差/绝对误差与 source 内 pairwise difference MAE；同时看融合相对文本的优势/损失是否主要由少数 source 均值偏移驱动。采用整组 bootstrap。若组间偏移不解释当前 OOF 差异，或误差没有可重复的来源结构，停止这个机制方向。
2. **若通过诊断，再跑嵌套来源留出。** 外层按 source video 划分 TRAIN；每个外层只在其余源上做内层 checkpoint 与超参选择，并在外层训练源内重新拟合归一化、source-OOf 特征和配对样本。被留出 source 的标签不得进入教师生成、早停、配对构造、lambda 选择或任何后处理。官方 VALID 只报告其既有同源混合 cohort 指标，不能代替外层 source-transfer 终点。
3. **对照与预算。** 固定同一 encoder/特征字节、target M、source fold、MLP 主干、三种子、优化器、点态 MAE、更新数和内层搜索次数。最少比较 Text-ERM、Fusion-ERM、Fusion+source GroupDRO、Fusion+source-ID DANN，以及 Fusion+`L_within`；再加 GEAR（原作者回归代码若能按同一 native M/划分适配）和通用 cross-source pairwise/GAR 式对照。给 INLP/LEACE 做一次仅训练折拟合的线性 post-hoc probe/remove 参考。S²LIF 若缺少官方代码，记录为方法学对照/自行忠实适配并披露，不伪称作者实现。报告参数量、有效梯度参数、算力/更新预算及每个来源的 MAE，避免更大模型或更多更新解释效果。
4. **结果门槛。** 预注册 CH 原生 MAE 的最小实用改善阈值、source 聚类 bootstrap 区间、三个种子方向一致要求；同时报告平均来源 MAE、最差来源 MAE（先定最小组大小）、Corr、组间/组内误差分解与同源差分 MAE。只有 `L_within` 在外层 source-heldout 上超过 Fusion-ERM、Text-ERM、GroupDRO/DANN 与通用 pairwise baseline，且收益在各外层来源稳定、没有把总体绝对分数拖坏，才进入原创性复查。只胜于 ERM 或只在官方 VALID 胜出，记为 baseline 结果。
5. **跨数据集保持 native。** 先完成 CH source-heldout 诊断与小规模确认，再独立用 MOSEI 原生 `[-3,3]` 目标、其自己的 ID/source split、native MAE/Corr 复验；各自选超参、各自报告分数，不做共同标准化或跨数据集合并的标签差分。

当前研究判断：可批准“来源误差分解诊断”；若诊断支持组间来源偏移，允许把 `L_within` 当作一个低成本、明确可证伪的候选实验；**不批准现在对外称它为原创 MSA 方法**。现有结果更像是要求把强 ERM、GEAR、S²LIF 风格文本先行以及 source-DG 基线放在同一严格外层来源评测下。旧的 reliability、dependence、annotation residual 和 utility/gate 负结果都不因这项假设而撤销。

## 本地证据

- [AnchorCDF 预注册方案](NATIVE_ANCHOR_CDF_PLAN_V1.md)与[正式校准结果](native_anchor_calibration_formal_20261013.json)、[source OOF 诊断](native_anchor_calibration_diagnostics_20261013.json)：约束点预测不变；disagreement 未优于同预算 feature-only mixture，且 source-OOF spread 与点误差关联接近零。
- [可靠性机制负结果](NATIVE_RELIABILITY_FINDINGS_V1.md)、[相关性机制负结果](NATIVE_DEPENDENCE_FINDINGS_V1.md)、[annotation-reference 机制负结果](NATIVE_REFERENCE_RESIDUAL_FINDINGS_V1.md)保留为已测证据，不把下一候选写成这些机制的换名版本。
- [前一轮独立机制审查](NEXT_MECHANISM_INDEPENDENT_REVIEW_20261013.md)已将现阶段优先事项放在 MOSEI ALMT padding/坐标正确性上；本文件仅补充 source-video transfer 审查，不能把它升级为贡献。
