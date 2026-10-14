# 原生标注观测似然与跨模态 atom 质量：独立研究审查（2026-10-13）

本审查只做文献和实验设计分析；未改代码、未用 GPU，也未读取或解析 TEST 标签/分数。MOSEI 的 `433/1869` exact-zero VALID 数和 CH-SIMS 的原生支持、已知 `-0.1` 例外，均只按当前任务说明使用。

## 裁定

**拒绝当前候选的广义原创资格。** 多模态情感分布、标注者差异、跨模态冲突和零/边界膨胀分布都有直接先例。仍有一个较窄、值得至多做一次预注册验证的研究问题：在正确的原生标注区间似然下，预测到的跨模态分歧能否把概率质量更准确地分给中性/边界观测值，并在结构、参数和训练预算相同的 hurdle 模型上带来额外收益。通过下述门槛前，它只能称为一个待证伪的 CH-SIMS v2 观测似然实验，不能称为新机制。

候选的窄差异不是“概率输出”“标注分布”或“增加中性类别”。唯一可能的差异是**由模型可用的跨模态分歧条件化离散观测原子概率**，并针对真实评分/量化过程比较观测概率。这个差异仍与近期 MSA 分布学习高度接近，原创性风险高。

## 观测过程与解释边界

CH-SIMS v2 论文报告：七位标注者分别给出多模态和单模态评分，后处理去掉最高和最低评分，再平均其余五个并映射到目标范围；单模态评分还采用模态隔离。项目当前 native 任务使用 `[-1,1]` 综合分数，已有材料指出其分数有有限网格、许多 `0` 和端点 `±1`。这些事实支持检查一个观测似然，却**不证明 `0` 是“所有人都判中性”的潜在状态**：平均评分可以由不同方向的评分抵消。端点也可能来自受限评分尺度。没有原始逐标注者评分时，不应声称模型恢复了人类个体分歧或真正的中性心理状态。[CH-SIMS v2 原论文](https://arxiv.org/abs/2209.02604)；[原始 CH-SIMS 的评分与聚合说明](https://aclanthology.org/2020.acl-main.343.pdf)。

MOSEI 原论文说明每条样本由三位众包标注者依 `[-3,3]` 七点 Likert 量表标注；本项目 VALID 中 exact zero 有 `433/1869`。这值得检查零值是否超过连续/量化观测模型的预测，但计数本身不能说明零值有独立生成机制。MOSEI 没有 CH-SIMS 的真实 `T/A/V` 注释通道，因此 MOSEI 上的 atom 预测不能称为“跨模态标注冲突”的复现。[MOSEI 原论文](https://aclanthology.org/P18-1208.pdf)。

将每个训练集中出现过的唯一数值当作类别会丢失连续支持，也会错误处理已知的有效 `-0.1`。观测区间必须根据标注/量化协议预先固定，覆盖整个原生范围，使任何合法分数都能进入一个区间；不能由 TRAIN unique values 生成类别表，也不能把 TEST 例外用于拟合或挑 bin。标注分辨率若无法从公开协议中确定，就先不宣称得到了“真实标注观测模型”。

## 近期先例与边界

| 先例 | 已覆盖的内容 | 尚未等同于候选的部分 |
|---|---|---|
| [零/一膨胀 Beta 回归，Ospina & Ferrari](https://arxiv.org/abs/1103.2372) | 在连续有界结果上，以点质量加 Beta 连续部分建模边界值；混合离散/连续似然是成熟统计模型。平移到 `[-1,1]` 并加中心原子是常规扩展。 | 不涉及多模态情感或由模态分歧预测原子质量。不能把“中心零膨胀”本身说成新贡献。 |
| [CORAL rank-consistent ordinal regression](https://arxiv.org/abs/1901.07884) | 为有序强度分箱建模阈值概率；应作为固定阈值分类概率基线。MSA 本身也长期报告 5/7 级离散准确率。 | 分箱概率不是一个点质量加连续强度的观测似然，也不自动处理标注量化与连续原生 MAE。 |
| [MoNIG，NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/371bce7dc83817b7893bcdeed13799b5-Abstract.html) | 直接在 MOSI/MOSEI 情感任务做多模态概率回归和不确定性融合；项目已有 MoNIG 本地负结果。 | 使用 NIG/Student-t 连续预测，不显式建模精确观测点质量或量化单元。已有概率输出不构成新颖性。 |
| [DFMU，IJCAI 2025](https://www.ijcai.org/proceedings/2025/0917.pdf) | 明确将主观标注偏差视为 MSA 的不确定性来源；在各模态表示上学习高斯分布，采用不确定性正则，并通过多点采样模拟多标注者预测。也建模跨模态特征关系。 | 主要是特征空间高斯分布与加权 MSE，不是共同测度下的离散观测单元概率，也没有证明“模态分歧 → 观测 atom 质量”的独立收益。 |
| [Prompt-based Distribution Learning (PDL)，Information Fusion，2026-09](https://www.sciencedirect.com/science/article/pii/S1566253526001727) 与[作者代码](https://github.com/Dongri404/PDL) | 最接近的近期 MSA 先例：用提示模拟不同标注者，针对显式模态冲突和隐式标注分歧学习潜在情感分布，并为音频/视觉构造模态感知分布；评估包含 MOSI、MOSEI、CH-SIMS。 | 论文摘要未显示它采用固定原生评分单元上的“离散 atom + 有界连续内部分布”观测似然。候选最多只能争取这个观测层面的差异，不能重新包装“用分布表示情感歧义和跨模态冲突”。 |

因此，“观测分布 + 冲突分配 atom”的组合**尚未在本次检索中找到完全相同的实现**，但近邻覆盖很强。仅换成三原子、增加一个 gate 或改用 CH-SIMS 数据，均不足以支持原创资格。

## 唯一建议检验的核心假设

**假设：** 在 CH-SIMS v2 上，使用模型在推理时可得的 T/A/V 预测分歧，条件化 `M=0` 和端点观测质量；相对同结构但不读取该分歧的 hurdle 模型，它能改善固定原生评分单元的概率评分和 atom 校准，同时不损害原生连续标量 MAE。

把它限定在 CH-SIMS v2，因为它有真实的 `M/T/A/V` 训练目标。若使用这些目标训练辅助 heads，候选和所有控制必须使用完全相同的辅助目标/权重；推理时 gate 只能读取模型预测的分歧，不能读取 `label_T/A/V` 或任何目标值。即使通过，该结论也只是“预测观测到的聚合评分”，不能宣称恢复了逐人判断或跨模态因果贡献。MOSEI 可另测一般的固定单元似然，但因缺少真实 T/A/V 注释，不能作为这条冲突机制的复制实验。

## 公平的最小对照与共同似然

核心是嵌套三臂；再用 ordinal/异方差家族提供必要外部参照：

1. **连续量化基线：** 同一骨干输出有界异方差连续分布，对固定观测单元用 `CDF(upper)-CDF(lower)` 计算概率。
2. **无冲突 hurdle：** 相同骨干和 head 路径，分配 `0/±boundary` 原子质量，其余质量给有界连续内部分布；atom logits 只读融合表示。
3. **冲突条件 hurdle（唯一候选）：** 与第 2 臂完全同构，只让 atom logits 额外读入由 T/A/V heads 预测的分歧量。通过禁用这一路输入得到机制移除对照。

另报告固定阈值 CORAL 概率基线和已有 native L1 标量基线。所有 `T/A/V` 辅助任务、原生 M 目标、特征字节、样本 ID、划分、随机种子、优化器、更新数、早停/验证选择规则及可训练参数量保持一致；不能把增益归给额外单模态监督或计算预算。优先同一 frozen encoder/backbone、三种固定种子；报告每种子更新数与额外推理成本。

不要直接比较连续密度 NLL 与点质量 NLL。先根据评分/量化协议定义一组固定单元 `C_j`，所有模型都输出同一组 `P(Y∈C_j|x)`。连续分布的概率通过 CDF 积分；混合模型的概率是其 atom 落入该单元的概率质量，加上连续部分在该单元内的积分。若端点评分意味着截断/饱和，端点单元必须按该观测规则处理。主要概率指标用共同的离散 log score，辅以 Brier/校准误差；原生标量仍报告 MAE/Corr，混合分布用其预测中位数形成 MAE 点预测。density NLL 只可作为模型内诊断，不用于不同参考测度模型间排序。

区间必须由公开注释规则预先确定，覆盖全原生分数范围；“训练唯一值分类”不合格。任何合法分数（包括已知的 `-0.1`）都必须按固定区间规则得到概率，但不得读取、拟合或调参于 TEST 标签。验证集只用于预注册选择；如需校准，应另做 TRAIN 内交叉拟合或预留校准部分。

还需区分“观察到 `Y=0` 的概率”与“潜在结构零 atom 的概率”。若零值所在观测单元也接收连续密度，数据只观察到 `p(C_0)=π_0+(1−Σπ)∫_{C_0}f`，没有逐标注者原始分数时，无法非参数地把两项分开。此分解只能是带分布假设的预测参数化；即使单元概率校准，也不能据此声称识别出了标注者的独立中性状态或端点裁剪事件。若研究目标必须是 latent atom 的真实成因，现有聚合标签不足以验证该目标，应停止或取得有授权的逐标注者标签。

## 拒绝门槛

- 若原子 hurdle 只胜过连续分布，却没有胜过 ordinal/fixed-bin 基线，结论是观测似然或分箱选择有用，不是新机制。
- 若冲突条件 hurdle 在共同单元 log score、精确 atom Brier/校准及三种子配对方向上都没有稳定胜过**无冲突 hurdle**，拒绝“冲突分配 atom 质量”的原创主张。
- 若改进只来自额外 `T/A/V` loss、参数、更新数、单独后校准，或仅来自不公平的连续密度 NLL 与离散概率比较，拒绝。
- 若不同合理的连续内部分布给出相近的总单元概率、但 atom 分量 `π_0/π_{±B}` 大幅互换，说明 latent atom 分配没有被聚合标签识别；只报告总观测单元概率，不得报告“冲突分配 atom 质量”的机制结论。
- 要保留为候选，建议至少要求：三个种子的共同单元 log score 改善方向一致、source-group 配对区间支持改善；atom 预测校准也改善；原生 MAE 对冻结的最强 native L1 基线在预注册的 `0.003` 非劣界内。`0.003` 是建议的决策门槛，不是既有实验结果。只改善 NLL 而显著损坏 MAE，不满足保留原生标量任务的要求。
- 即便通过，先按 PDL/DFMU 与 hurdle 的精确输出空间、似然和冲突定义做完整逐项检索/复核；若差异只剩“把普通 gate 接到 atom logits”，仍不够支持原创机制。

## 建议

可以在当前 CH-SIMS top-four baseline 完成并冻结后，做一次有界的三种子验证；**不要现在把它列为已有原创贡献，也不建议立即扩成 MOSEI × CH-SIMS 的大规模概率模型工程**。如果共同观测单元定义无法从公开标注流程固定，或冲突 gate 不能超过同预算无冲突 hurdle，就停止并保留为负结果。下一步若被接受，应把贡献表述为“对某种人工聚合标签的条件观测分布建模”，而不是一般性的可信概率输出或首个多模态分布学习方法。

## 修订候选：固定原生 point 的概率形状

本节只审查新的修订：不加入 latent atom、不解释真实标注者状态，也不再把前文 hurdle 假设带进来。将当前最强 simple-fusion 输出 `m`（按数据集 native 规则裁剪后的 point）冻结；所有概率模型都以 `m` 为**精确条件中位数**，只学习它周围的概率形状。这个修订更适合当前结果：top-four 的点 MAE `.4272` 弱于冻结 simple-fusion `.4221`，且 neutral/weak/conflict 分层更差；新的候选不能再动 point 来换表面收益。

**判定：可以做一次有限的 shape/gate 机制验证，但不把 median-preserving 代数当原创。** 本次有限检索未找到 MSA 论文直接采用“冻结现有 native 回归中位数 + 中心核混合 + 显式约束混合后中位数”的同一实现；但 mixture of quantile regressions 已直接建模条件中位数，回归概率校准也已是通用方法。MoNIG、DFMU 和 PDL 则覆盖了 MSA 分布/不确定性建模，PDL 还覆盖模态感知的冲突表示。因而最多可研究“跨模态预测分歧是否为固定 point 周围的概率形状提供独立且可校准的信息”，不能声称首个概率回归、首个 MSA 分布或新的人类情感分解。[Mixtures of quantile regressions](https://doi.org/10.1016/j.csda.2014.04.014)；[Kuleshov et al., calibrated regression](https://proceedings.mlr.press/v80/kuleshov18a.html)；[PDL 2026](https://www.sciencedirect.com/science/article/pii/S1566253526001727)；[DFMU 2025](https://www.ijcai.org/proceedings/2025/0917.pdf)。

### 参数化与边界条件

候选分布是

\[
F(y\mid x)=\pi(x)F_0(y)+(1-\pi(x))F_{\mathrm{loc}}(y;\mu(x),s(x)),
\]

其中 `F0` 是固定中心/固定尺度的 clipped logistic kernel，第二项是以 `m(x)` 为条件中位数的 logistic。对内部 `m`，令

\[
q(x)=\frac{0.5-\pi(x)F_0(m(x))}{1-\pi(x)},\qquad
\mu(x)=m(x)-s(x)\operatorname{logit}(q(x)).
\]

必须保证 `0<q<1`，即 `π < 0.5 / max(F0(m), 1−F0(m))`；在等号处 location 发散，所以实际实现要留严格数值余量。端点 `m=±B` 使用裁剪前 latent CDF 的单侧边界条件：上端令 `F_latent(B⁻)=0.5`，下端令 `F_latent(−B)=0.5`，再做相同的 native clip。不能把已 clip CDF 的 `F(B)=1` 直接代进内部公式。还需明确是 censor/clip 还是截断 logistic；若 clip 后边界出现概率质量，那是 native 变换的 pushforward 结果，不应命名成 latent annotation atom。实现须逐例核对混合分布的中位数与原 `m` 相等。

公式本身只是为 mixture 解一个中位数约束，常规代数；通用的 quantile mixture / conditional median regression 先例已经存在。能检验的增量是**形状或 disagreement gate 的条件概率是否有用**，不是这个公式。

### 最小对照与防泄漏

所有条件共用同一冻结 simple-fusion point `m`、特征、样本、评分边界和 native clip，至少比较：

1. **异方差 clipped logistic：** location 固定为 `m`，只由特征预测尺度。
2. **feature-only central-kernel mixture：** `π/s` 由同一融合特征预测，location 按上式校正以固定中位数。
3. **disagreement-conditioned mixture：** 与第 2 组相同，只把同一个 `π` head 输入扩为包含模型预测的 T/A/V common-M 分歧量。
4. **直接 regression-hit 校准基线：** 直接预测 `1[|Y−m|≤ε]`，单独比较该 hit probability/coverage；它是一个二值事件概率，不冒充完整作者 bin 分布。

训练 shape/gate 时必须新做 source-group CV OOF，覆盖 fusion point 及各单模态**共同 M 目标**预测；不可复用 TRAIN in-sample residual/预测。MOSEI 不提供真实 `T/A/V` 标注，不能拿 CH-SIMS 的 T/A/V targets 当等价输入；disagreement 只能由在相同 M 目标上训练的单模态预测计算。TRAIN 用 OOF point/分歧训练 shape head，VALID/TEST 用冻结基模型的 held-out predictions；须检查 OOF 与 held-out 的 point/分歧分布差异，若明显漂移，就不能声称校准可迁移。三种子、相同 OOF folds、相同形状头参数/更新预算；中位数预测保持不变，因此所有组 native MAE/Corr 应与冻结 simple fusion 完全一致（数值误差范围内），不能把它包装成点准确率提升。

所有组把 CDF 积分到作者既有 5/7 分类边界，使用共同的类别 log score/Brier score；另报预注册 `ε` 的 `P(|Y−m|≤ε)` 和 VALID 经验覆盖，并与直接 hit classifier 对照。不得从 TRAIN unique targets 临时创建新类别，也不读 TEST 结果选 shape、ε 或温度。

### 是否通过机制门

- 只有 disagreement-conditioned mixture 相对**同预算 feature-only mixture**在共同作者 bin 的 proper score 和 hit Brier/calibration 上跨三种子、source-group 配对一致改善，才支持“disagreement 帮助调整 point 周围的概率形状”。仅胜过异方差 logistic 说明 mixture family 有用；仅胜过简单 hit classifier 也不足以证明完整分布更好。
- 若 `π`/尺度模型和 Kuleshov-style 的 TRAIN 内/独立 calibration split CDF recalibration 得分相当，复杂 gate 没有独立价值。校准只能用独立 TRAIN source folds；不能把 VALID 同时用于 gate 调参与最终报告。
- 若 cross-fit 的单模态 common-M 分歧没有独立增益，或 OOF→held-out feature shift 明显，拒绝 gate 机制。不要复用 TRAIN in-sample 预测重做。
- 因 point 被硬性冻结，任何有效改进都只能称概率形状/校准或区间覆盖改进。`π` 是数学混合权重，不是中性概率、标注者分歧概率或“AV冲突概率”。

**最终建议：** 这比 atom/annotation-generation 解释更干净，足以支持一次具体的 shape/gate 试验；原创资格仍取决于它能否在新 sourceCV OOF 上超过相同分布族的 feature-only 对照和简单 CDF 校准。公式和固定 point 的做法本身不新，若 disagreement gate 不增加 proper-score 收益，就保留为一个校准基线而停止。
