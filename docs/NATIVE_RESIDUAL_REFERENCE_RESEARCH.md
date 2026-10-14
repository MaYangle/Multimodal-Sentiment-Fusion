# 原生回归的 annotation-aligned residual：文献边界与证伪方案

调研日期：2026-10-13。只读阅读项目已有记录、原论文和作者实现；未训练模型、未下载数据、未修改已有模块。本文只评估一个窄候选，不宣称它原创或有效。

## 候选定义与当前证据

CH-SIMS v2 的人工标签分别给出纯文本情感 `y_T` 和综合多模态情感 `y_M`。候选将二者重参数化为：

\[
t^*=y_T,\qquad d^*=y_M-y_T,\qquad \widehat y_M=t_\theta(x_T)+d_\phi(x_T,x_A,x_V).
\]

文本头预测文本单模态标注，完整模态头预测“综合标注相对文本标注的差”。`d*` 是两个真实 annotation targets 的差值；它不是 `y_M` 减掉当前文本模型的预测误差，也不是音频/视觉对人类情感的因果效应。它包含文本与综合标注之间的整体差异和 annotation noise，不能直接命名为“AV 的真实贡献”“AV 可靠性”或反事实治疗效应。

截至 2026-10-13，项目同步的 MOSEI 验证结果为 raw feature：text MAE .6697、simple fusion .6709、ALMT .6115；TRAIN 坐标标准化方案为 text .6516、fusion .6450。ALMT 的 neutral MAE 从 .370 降至 .270、`|y|≥2` MAE 从 1.756 降至 1.367，但 `0<|y|≤1` MAE 从 .467 升至 .521，Non0 accuracy 从 .786 降至 .768。其含义是强度/边界分区存在相反变化，不能据总 MAE 宣称各类样本均变好。项目此前中文原生结果为 text .4373、simple fusion .4221、ALMT .4271；NIG、joint-context 和 OOF dependence correction 没有竞争性收益。MOSEI 的 MultiBench 特征是已处理的 GloVe/audio/vision 特征包，不等于 raw encoder 端到端推理。

## 最接近的原论文和作者实现

| 工作 | 作者论文/实现中实际的输出与监督 | 与 `y_M-y_T` additive decoder 的边界 |
|---|---|---|
| **AV-MC，CH-SIMS v2**：[原论文](https://arxiv.org/abs/2209.02604)，[作者模型代码](https://github.com/thuiar/ch-sims-v2/blob/main/models/multiTask/V1.py)，[作者训练代码](https://github.com/thuiar/ch-sims-v2/blob/main/trains/multiTask/V1.py) | 作者模型有 `M/T/A/V` 四个独立标量输出头。训练器对 M、T、A、V 分别用对应真实人工标签做 L1；AV mixup 时还对被混合的 A/V label 和 M label 做 mixup consistency。模型最后从融合隐藏表示直接预测 `output_fusion`，另有各模态预测头。 | **最直接的 true-label multi-task 对照。** 它已使用真实 `y_T` 与 `y_M`，但不是将 `y_M-y_T` 当作预测头目标，也没有最终 `ŷ_M=ŷ_T+ḓ` 的加法解码；两组标签分别监督独立输出。候选不能把“首次利用真实文本单模态标注”当贡献，只能检验把已有两个监督值变换为有结构的 additive output 是否更有效。 |
| **Self-MM，AAAI 2021**：[原论文](https://arxiv.org/abs/2102.04830)，[作者代码](https://github.com/thuiar/Self-MM) | 最终输出是多模态 `ŷ_M`；另有三个单模态预测头和 ULGM 生成的 pseudo labels `y_s`。对单模态 L1 的权重为 `tanh(|y_s-y_M|)`，引导模型强调 pseudo unimodal 与 multimodal supervision 差异较大的样本。 | 差值参与 sample weight，pseudo `y_s` 是从多模态标注/表征产生；并非 `d*=真实 y_M-真实 y_T` 的残差头。论文还做 human-label 对照，但主算法仍是 pseudo label + 独立 uni heads。它已占据“按单模态/多模态情感差异训练”的宽泛概念。 |
| **CMC，2025 preprint / 作者代码**：[原论文](https://arxiv.org/abs/2510.20256)，[作者仓库](https://github.com/gw-zhong/CMC) | 模型输出 `pred` 与 `pred_t/pred_a/pred_v`；`--is_pseudo` 时用伪 uni labels，关掉时 CMC-GT 直接从数据读真实 T/A/V labels。M 和三个单模态预测分别用 cross-entropy 监督，最终预测为按路由权重求和的单模态 logits。 | 真实标签配置是现成的“multimodal+unimodal heads”参照，但目标是分类 logits 加权融合；不计算/预测 `y_M-y_T`，不加文本 score 与 residual score。不能把 CMC-GT 误报为 residual supervision，也不能将论文二分类数字当本项目 native regression 对照。 |
| **MTCL，2026**：[机构论文记录/摘要](https://orca.cardiff.ac.uk/id/eprint/185959/)，[作者实现](https://github.com/tiggers23/MTCL) | 公开训练代码直接读取 `label_M/T/A/V`，M head 用 MSE 监督 `y_M`。每模态有 fused task head 和 unimodal/private head；`Loss_Branch` 将 private head 对齐 `y_u`，并按分支表示的 JS discrepancy 划分样本：低 discrepancy 子集的 fused head 监督 `y_u`，高 discrepancy 子集则监督 `y_M`。代码使用的是真实标注张量，而不是只从文本/融合 score 拟造差异。 | 这是**最接近的近期“真实单模态与联合标注不一致时，针对模态分支调整监督目标”**对照。它不做 scalar `y_M-y_T` regression，也不把 M 作为 T+D 加法解码；它按 feature-disagreement 子集切换分支监督目标。机构页面列为 ACM TOMM 2026，仓库 README 仍留有“under review”提示，发表状态与代码备注不一致，应据 DOI 最终书目信息引用而不要复述 README 状态。 |
| **CDML，2024**：[作者论文](https://www.mdpi.com/1999-5903/16/6/213) | ULGM 从多模态标签和表示生成 `y_s`，并定义 `δ_M` 偏移；训练额外 unimodal outputs。其单模态辅助损失约为 `tanh(|y_s-y_M|)·|ŷ_s-y_s|`，最终预测仍取 `ŷ_M`。 | 明确把 uni/mm label difference 用于辅助训练，但 uni labels 是生成的，且差异作为 sample weight，不作为真实文本参考 residual score `d*`。没有核验到该论文的作者 repo。 |
| **H²CAN，2025**：[原文/作者文章](https://doi.org/10.1007/s40747-025-01806-y)，[论文 PDF](https://researchoutput.csu.edu.au/ws/portalfiles/portal/607905281/587562602_published_article.pdf) | 用 ULGM 生成 uni labels，以 `|y_M-y_s^{generated}|` 选出 discrepancy 最大的 biased modality。然后构造 factual/counterfactual fusion predictions `ŷ_c` 与 `ỹ_c`，`y_ef=ŷ_c-ỹ_c` 作为 polarity cross-entropy 的辅助监督；主 regression head 仍用 `y_M` 的 MAE。 | 有两个“差值”：pseudo label 与 M 的 discrepancy 选 modality，以及 counterfactual/factual prediction gap 做分类辅助目标；都不是人工 `y_M-y_T` 的连续分数 target，也不采用 `t+d` 主输出。未找到可核验的作者代码链接。 |

### 与残差融合直接相关的近邻

| 工作 | 它的 residual 到底是什么？ | 对候选的边界 |
|---|---|---|
| **MAG-BERT，ACL 2020**：[原论文](https://aclanthology.org/2020.acl-main.214/)，[作者实现](https://github.com/eshansurendra/MAG-BERT) | MAG 由 audio/vision 条件化一个 hidden-state shift，加到 BERT/XLNet 中间 embedding 再继续 Transformer；作者 API 输出 `logits`，默认 `num_labels=1` 可做 sentiment regression。监督是数据的 multimodal target。 | 早已存在 text-centered + AV residual/additive correction 的**representation-space**结构，但没有 `label_T` head 或 `y_M-y_T` score target。它限制“文本作锚点、非文本加法修正”这个宽泛表述的原创性，不能说它已覆盖候选 annotation target。 |
| **DUAL，2025/26 IEICE**：[作者论文](https://globals.ieice.org/en_transactions/fundamentals/10.1587/transfun.2025EAP1148/_f) | 分别把 A 映射到 V 空间、V 映射到 A 空间，形成 `F_A-α(F_V)` 与 `F_V-β(F_A)` 的跨模态**feature residual**；主结果由 AV features 融合后回归，文本在 decoder 才做 refinement。 | 明确 residual learning 在情感建模已有成熟做法，但差异发生于特征表示，不是人工标注 `y_M-y_T`。该论文是 residual-feature prior，不是 exact target match。 |
| **UCMR-Net，2026**：[原论文](https://doi.org/10.3390/app16147142)，[论文声明的匿名代码包](https://anonymous.4open.science/r/UCMR-Net-Code-B115) | 明确将文本隐藏表示设为 anchor，估计 `r_AV=F_r([h_T;h_A;h_V])`，以 learned weight `ω_r` 形成 `h_F=h_T+ω_r r_AV`，再用 regression head 预测连续 `y_M`。score/辅助目标是多模态标签的 regression、correlation、多任务和 masked-modality distillation；论文公式没有 `y_T` 或 `y_M-y_T` 的监督目标。其 reliability score 也被作者明确限定为非概率的 residual contribution controller。 | **非常接近 text-anchor + AV correction 的结构与目标语义**，但输出级建模是 latent correction，且不是 annotation-aligned score residual。文章声明匿名开源包，本次无法检查该包内容。 |
| **Visual-Acoustic Residual Compensation，2026-08**：[作者论文页](https://www.ecice06.com/EN/10.19678/j.issn.1000-3428.0260838) | 语言聚焦 fusion 先给出 base score；AV 表示进入独立标量 compensation branch，最终做受限输出级校准。文章明确有 residual supervision、clip threshold、forward compensation weight 和幅值正则，目标是 base prediction 对真实 `y_M` 的 remaining error（即 `y_M-ŷ_base` 方向），不是文本人工 annotation 与多模态人工 annotation 之差。MOSEI/MOSI 两数据都报告原生回归指标，论文给出 residual 训练路径消融。 | **最直接的 output-residual prior。** 它基本占据“AV 支路输出一项监督残差并加到情感预测上”的一般机制。候选最窄差异仅是改用 human `label_T` 作为参考，并把 residual 定义为 `y_M-y_T`，不再让 residual 拟合 base model error；作者页未提供代码链接，本次未找到公开仓库。 |

## 精确原创边界

1. **真实 `label_T` 和 `label_M` 同时训练，已有直接先例。** AV-MC 对 M/T/A/V 分别开 scalar heads 并各自用真实相应 label 做 L1；MTCL 的 author code 也读取真实 M/T/A/V 并依 discrepancy 区分模态 fused/private heads。故“增加真实文本 annotation 辅助头”不新。候选的剩余差别是以 `d*=y_M-y_T` 取代独立的 M head、并将主 score 强制写成 `t+d`。
2. **使用 label discrepancy 表征一致/差异，已有多条路线。** Self-MM 用 pseudo `y_s-y_M` 加权；CDML 用生成标签差异加权 aux L1；H²CAN 用生成 label difference 找被偏置模态；MTCL 根据分支分布差异切换样本级 supervision。它们没有确切做 `label_M-label_T` 的连续目标，但把“情感差异监督”作为贡献本身不可再泛称新。
3. **主预测作锚点、A/V 学输出 residual 已由 2026-08 论文直接覆盖。** 该论文残差 target 是 `y_M-ŷ_base`，而候选 `y_M-y_T` 是人工 annotation channel gap。更窄、尚未在上述文献中找到完全相同实现的部分，是“训练文本-only branch 去拟合真实 `label_T`，同时训练全模态 branch 去拟合真实 `label_M-label_T`，最终加和输出 `label_M`”。目前它更像 AV-MC/MTCL 真实多任务 label pair 的一个输出参数化变体，而不是自动成立的新监督理论。
4. **反事实/因果含义不能借用。** `y_M-y_T` 是两种人工标注协议的观察差值；它不是同一样本“有/无 AV”的模型潜在结果，也不是音视频造成的人类情绪因果效应。可称“跨 annotation reference 的差值目标”或“标注对齐的预测分解”，不能称 AV 的真实贡献/处理效应/可靠性。

## 最小、可证伪的设计

先限定在有真实单模态标注的 CH-SIMS v2。作者数据说明文本、音频、视觉、综合四种情感分别独立提供，`label_T` 可支持训练参照；MOSEI 不提供同种 `label_T`，不能凭 pseudo label 宣称复制了相同方法。[CH-SIMS v2 注释说明](https://arxiv.org/html/2209.02604)还明确 unimodal annotation 与统一 multimodal annotation 可以不同。

做一个预注册 `decoder structure × true-reference auxiliary targets` 2×2：

| 读出结构 | 不加 `T/D` 标签损失 | 加同样的真实 `label_T`、`d*=label_M-label_T` 损失 |
|---|---|---|
| Flat M | 标准直接 M head 输出 `ŷ_M` | 直接 M head 为主输出，T/D 两个同尺寸辅助头预测真实 `y_T`/`d*` |
| Additive | `ŷ_M=ŷ_T+ḓ`，只用主 `label_M` loss | 同一加和主 loss，另加与 Flat-M 完全相同权重的 T/D 辅助 L1 |

四个条件共用同一特征、文本/AV backbone、总 heads、优化器、种子、source-group folds 和训练预算。T/D/M 三个 scalar heads 在每个 cell 都存在；由于不同 decoder 令梯度进入各头的路径有意不同，要额外审计每个 cell 的 active-gradient 参数和更新数。若做不到参数配平，单独加“容量匹配 direct MLP head”对照，避免把多两个输出头误认为 residual decoder 的收益。辅助损失权重只在 train groups 内确定。优先使用已建立的 source-video-disjoint split 做诊断，保留官方 split 结果并披露 source overlap；三 seeds；final test 在方案冻结前不碰。

还需加三个直接对照：

- **AV-MC-style M/T multitask control：** 主 M head + 独立 text head 用真实 M/T labels；若首轮只测 text reference，可固定音频/视觉 aux 处理，避免将 A/V 附加监督差异混入主因子。它验证收益是否只是既有真实 uni label multitask。
- **Prediction-error residual control：** text-only model 预测 `y_M`，source-group OOF 生成 `d_err*=y_M-ŷ_T^{OOF}`；AV branch 预测 base model 剩余误差。它直接代表一般 residual correction / 2026 output-compensation prior，与 annotation residual `y_M-y_T` 明确分开。按相同 teacher/OOF 预算计费。
- **近期直接残差对照：** 至少按 [Visual-Acoustic Residual Compensation](https://www.ecice06.com/EN/10.19678/j.issn.1000-3428.0260838) 描述实现相匹配的 base-score + clipped/weighted AV residual；说明是文献适配，不宣称复现作者代码。UCMR-Net 可作为 feature-level text-anchor residual 对照。

除总体 native MAE/Corr 外，预先按真实 `|d*|`/符号、`label_T` polarity 和弱/中/强强度汇总 source-group 配对误差。候选的直接可证伪预测是：Additive 相对 Flat-M 在同样 `T/D` 监督下有稳定增益，且 AV correction 的收益集中在 `|d*|` 大的组；若只有 “aux on” 主效应、Additive 与 Flat-M 没差别，结论只能是“增加 M/T label tasks 有效”，不能归功于 residual structure。

MOSEI **不生成虚构 `label_T`**。可选的 proxy 实验另命名为 `OOF-text-model error residual`：在训练 source folds 上用 text-only `f_T` 的 OOF `ŷ_T` 得 `d_proxy*=y_M-ŷ_T^{OOF}`，AV correction 预测该 base error。每个比较组使用相同教师折数、模型更新/预测预算及冻结 text teacher；它检验 residual correction 的跨数据集作用，不是 annotation-reference residual 的 replication。除非这条 model-residual 变体本身超过 2026 直接 prior，否则没有必要将其称为第二个贡献。

## 弃用条件

- 在相同真实 T/D 辅助目标下，`t+d` 不优于 direct M head；或优势只来自 auxiliary-label 主效应，而不是结构主效应/稳定交互。
- Additive 只提高训练/重叠 source validation，在 source-disjoint 新视频上不改善；效果不跨 seeds 或只改善大 `|d*|` 极少数样本而总体/弱强度段退化。
- annotation residual `y_M-y_T` 不优于 OOF prediction-error residual `y_M-ŷ_T`、AV-MC-style multi-task、以及 V-A output residual prior；这说明标签差值换坐标并未带来额外建模收益。
- d-head 输出能跟踪 `d*` 但 `t_hat+d_hat` 的实际 MAE/Corr 不改善；或者表现全由额外参数/多一个 loss/不同训练预算解释。
- 候选若需要把 `d*` 解释成“AV 引起的真实情感变化”“文本预测偏差”或“模态可靠性”，应立即弃用这些解释；它只是真实两个注释值的差。
- MOSEI 没有真实 text-only annotation；用 `y_M-ŷ_T` 的 OOF proxy 取得的结果不能回填为“真实 `label_M-label_T` 方法已跨语种复现”。

**裁定：** 可把这个候选作为一次窄的结构证伪实验，前提是与 AV-MC/MTCL 的真实多任务损失和 2026 output-residual 方法逐项对照。宽泛的 text anchor + multimodal residual 已有 MAG-BERT、UCMR-Net 和 2026 视听输出补偿作为近邻；真实 `M/T` 分开监督已由 AV-MC、CMC-GT 和 MTCL 覆盖。唯一较窄的不同点是把真实的 annotation gap `y_M-y_T` 监督到加和的 scalar decoder 上。即使实验成功，第一轮也更适合报告为“CH-SIMS v2 annotation-aligned score decomposition 的任务级实证”，在检索更多同类输出残差方法并击败控制组前，不足以宣称普适新机制。
