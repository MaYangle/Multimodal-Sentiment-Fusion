# 原生回归下的文本优先模态获取：原创性边界与最小检验

调研日期：2026-10-12。只读检查现有研究记录并检索公开原论文/作者材料；本次没有训练、改动已有模块、下载数据或联系作者。本文评估的是候选想法，不将其登记为已成立的新机制。

## 结论先行

“先看文本，预测某样本再加入音视频能否降低误差，再决定是否获取”这一一般机制已有很近的直接先例。尤其是 2026 年 10 月 5 日公开的 SeMARC，明确为每个样本、每个已获取模态集合计算加入下一模态的正确率增量，并让控制器按样本决定继续获取或停止；2026 年的 EAMA 也学习给定已观察模态后的逐样本子集价值；RouteCert 则专门解决自适应 acquisition 产生的终止模态组如何校准和认证。因此，“Jev 启发的结构化收益分数 + 置信门控/路由”不能单独作为原创性主张。

目前唯一值得做的小型证伪试验，是把目标收窄到**源视频分组的 OOF 配对原生回归 MAE 改善预测**：

\[
\Delta_i = |y_i-\hat y_{T,i}|-|y_i-\hat y_{TAV,i}|,
\]

其中正值表示在同一个样本上，固定的全模态预测相对文本预测降低了 MAE。只用文本阶段可见的分数、不确定性和文本特征预测这项差值；再问该分数在相同音视频获取预算下，是否比文本不确定度和随机路由更会挑出 AV 真正有用的样本。这个差值目标比已有主要分类获取论文更贴合原生回归，也能利用成对样本减小比较噪声；但这仍是一个窄的任务适配假说，不能因为用了 OOF、校准或源视频分组就自动成为新方法。

若其价值预测在源视频隔离的验证集上没有稳定胜过不确定度/随机路由，建议弃用为核心贡献。若有效，再把贡献限制为“源分组的配对回归收益估计及预算受限的 AV 路由”，并直接复现或严谨适配 EAMA/SeMARC/RouteCert，而非声称首次做样本级模态效用估计。

## 本项目证据与适用边界

- 当前中文原生回归报告：文本 MAE .4373、简单融合 .4221、ALMT .4271。已测试的 12 个 NIG 辅助×gate、18 个 joint-context 组合、9 个 OOF 相关校正读数没有带来竞争性收益，分别保留在 [NATIVE_RELIABILITY_FINDINGS_V1.md](NATIVE_RELIABILITY_FINDINGS_V1.md)、[NATIVE_CONTEXT_NIG_FINDINGS_V1.md](NATIVE_CONTEXT_NIG_FINDINGS_V1.md) 与 [NATIVE_DEPENDENCE_FINDINGS_V1.md](NATIVE_DEPENDENCE_FINDINGS_V1.md)。旧五分类窗口 utility 路线也失败，见 [architecture_reassessment.md](architecture_reassessment.md)。这些结果说明新候选应检验**按样本是否需要付出 AV 成本**，不应把旧门控、可靠性权重或窗口干预换名重做。
- MOSEI 已获得并审计 MultiBench aligned-50 的作者链接特征包，含 GloVe 300、audio 74、vision 35 维序列；该包是处理后特征，不是原始媒体。文本、简单融合、ALMT 的三 seed 原生回归正在进行；文本 MAE 约 .67，ALMT 单 seed 暂优于 mean-pooled 融合，均不是最终对照。队列、划分差异和输入边界见 [NATIVE_MOSEI_BASELINE_PLAN_V1.md](NATIVE_MOSEI_BASELINE_PLAN_V1.md) 与 [MOSEI_MULTIBENCH_COHORT.md](MOSEI_MULTIBENCH_COHORT.md)。本候选应等三 seed 基线完成后再选目标，不能把当前单 seed 走势当成已确认效应。
- 中文标签尺度为 [-1,1]，MOSEI 为 [-3,3]，不能合并 delta、成本或置信阈值。CH-SIMS 官方 train/valid 有严重源视频交叠；除官方结果外，应报告源视频隔离敏感性评估，不能把它伪称官方 test。MOSEI MultiBench 当前核对到官方 SDK fold 有四条 train 记录差异，必须保持其已披露 cohort 名称。

## 最接近的公开方法

| 工作 | 与候选的实质重叠 | 重要差别 / 作者实现边界 |
|---|---|---|
| [SeMARC，Mohapatra 等，2026](https://arxiv.org/abs/2610.07466) | 最接近的近期直接先例。对不同已获取集合的逐样本预测计算加入某模态的边际正确率收益；用集合/类别均值作 acquisition prior，再由控制器按已观测证据、置信度和成本选择下一模态或停止。获取发生在 encoder 执行之前，论文测量 raw extraction、fusion、决策的端到端时延。 | 主要目标是分类正确率，而不是连续情感分数的配对绝对误差差值；用顺序式 bottleneck 融合和 fitted-Q residual。论文版本没有可核验的作者代码链接，本次未找到相同标题的作者公开仓库。由于其提出时间非常新，方法上的一般重合尤其强。 |
| [EAMA，When is More Better?，TMLR 匿名审稿稿](https://openreview.net/forum?id=gnJNCIPO6y)；作者论文材料可由 [MIT 论文库](https://dspace.mit.edu/handle/1721.1/174585)核对 | 逐样本学习已观察模态条件下，购买候选模态子集的预期任务价值；分阶段 acquisition，并在预期净价值不再为正时停止。其 utility/accuracy 变化预测与“预测加 AV 的收益再路由”高度相似。SeMARC 也将 EAMA 作为 concurrent acquisition 方法比较。 | EAMA 针对分类/模态子集价值及异质成本；本候选若不同，仅在回归 MAE 差、文本首阶段和源组 OOF / 校准的具体估计目标。匿名审稿稿未提供可归属作者的公开实现；不应将代码不可见误写成该算法不存在。 |
| [RouteCert / Conditional Validity for Adaptive Modality Acquisition，Baghi，2026](https://arxiv.org/abs/2608.15520) | 最接近“校准收益 + acquisition policy”。方法用缺失模态完成模型估算候选模态/小组缩小残余答案集的预期收益/成本；路由决定样本最终落入哪个模态模式，并给出对策略所产生终止模式的有限样本条件风险/覆盖认证，还处理在认证数据上选择完整策略族。 | 其收益代理是答案集缩小或相对参考模型的风险，不是 OOF 成对 MAE 差值；目标侧重分类与认证。论文明确说明代码包需向作者索取，没有公开仓库可直接核验。它已覆盖“自适应路由后应如何认证所选组”的核心统计难点。 |
| [A2MT，Kossen 等，TMLR 2023](https://openreview.net/forum?id=Gbu1bHQhEL) | 明确提出逐样本、序列时间步的多模态 active acquisition，优化预测损失与 acquisition cost；以 Perceiver IO/RL 学何时观察、何时停止。 | 论文报告真实 Kinetics/AudioSet 上学到了成本响应，但 ablation 未学到真正个体自适应行为。这是对“entropy/不确定度下降就代表边际损失下降”的重要警告。官方论文未给出本次可验证的作者代码 URL。 |
| [Joint-AFA，Shim 等，NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/e5841df2166dd424a57127423d276bbe-Abstract.html)；[作者实现](https://github.com/OpenXAIProject/Joint-AFA-Classification) | 已经把预测与代价敏感的“获取一项还是停止”联合学习，是学习型路由的经典强先例。 | 主要是表格/医学低维特征，不是 raw 音视频序列或深度 encoder 成本。代码是作者公开实现，环境较旧，需匹配适配后对照。 |
| [Mozannar & Sontag，Consistent Estimators for Learning to Defer，ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html)；[作者代码](https://github.com/clinicalml/learn-to-defer) | “自己预测或付出成本交给另一个专家”是结构化路由决策的成熟形式；有一致的 cost-sensitive surrogate，而非仅按 confidence 阈值。把 AV 分支看成更昂贵的专家时，这类 deferral 先例直接限制了决策头本身的原创性。 | 专家是人/外部预测器，论文不是主动传感器获取；但“输出一个概率再决定 defer”的一般性不是新点。 |
| [A Conformal Selection Framework for Individual Treatment Beneficiaries，Liu 等，2026](https://arxiv.org/abs/2606.31954) 与 [Conformal Risk Control，Angelopoulos 等](https://arxiv.org/abs/2208.02814) | 前者用交叉拟合收益伪结果、独立校准、conformal p-value 与 FDR 控制来选择可能受益个体；后者给任意有界单调风险的有限样本期望控制框架。共同说明“预测谁会受益 + 校准决策”已有成熟统计路线。 | 治疗效应里两个 potential outcome 不能同时观测；本项目中 train clip 已有文本/AV 特征，因此两个固定模型对同个 `y` 的误差差是可观测配对结果，不应误称为因果 ITE。也不能忽略其组选择/多重筛选校正问题。 |
| [MCCF，Conformal Fusion Under Missing Modalities，Moayedikia，2026](https://arxiv.org/abs/2608.07183)；[作者代码](https://github.com/amoayedikia/MCCF) | 使用 modality-presence mask 做 Mondrian conformal calibration，证明“按可见模态模式分别校准”已有近期直接实现。 | 静态缺失模式下校准融合分布，不学习是否值得购买模态；不能替代 RouteCert 的 policy-created terminal-group 认证。 |

## 原创边界判断

1. **已被占据的概念层：** “个体层面的模态价值估计”“根据已观察证据选择模态”“在预测前跳过未选 encoder”“达到阈值后停止”“价值减成本”的广义机制，分别由 SeMARC、EAMA、A2MT、Joint-AFA 与较新的 RouteCert 组合覆盖。用概率分数、置信度、校准或一个结构化 JSON 输出封装这个路由动作，也不构成单独贡献。
2. **仍可能有差异的具体目标：** 以同一样本、同一真实连续标签上的 `Δ MAE` 为监督目标，并用 source-video-group OOF 预测避免文本/全模态老师训练内过拟合；用文本可见信息预测该差值或 `P(Δ>0)`，回答文本阶段是否值得再取 AV。所查论文的主要 acquisition target 多为分类正确/accuracy gain、问答集合缩小或任务风险，而非这项已配对的 native-regression MAE gain。
3. **差异还不足以直接称原创：** 将分类准确率替换成回归 MAE、将样本按视频分组、或将 head 输出改为“收益概率/下界”，可能只是指标、交叉验证与实现适配。尤其是 OOF 是防止乐观偏差的估计设计；conformal/校准是成熟统计工具。核心新增必须来自可复现且非平凡的算法或统计保证，并且在最接近的逐样本 value-acquisition 基线之上产生独立收益。
4. **不能作因果宣传：** 该 `Δ` 是给定两种固定训练方案、同一标签和同一输入记录的**模型配对损失差**，不是音视频对人类情感的因果效应，也不是输入内容的模型无关“真实价值”。它可能随文本/多模态 backbone、训练集、损失和部署成本改变。

## 最小、可证伪的实验

**问题：** 对原生回归模型，文本阶段的样本级信息能否以稳定、预算受限的方式预测“加上全部 AV 后本模型 MAE 会不会降低”？

1. **固定两种基础预测。** 对每个 source group 做 OOF：`f_T` 仅用文本，`f_TAV` 使用文本/音频/视觉。成对样本、同折训练/验证组、同一标签、同一 checkpoint 选择政策。CH-SIMS 的 `Δ` 要按其原生 MAE 规则 clip prediction 和 label；MOSEI 使用 [-3,3] 原始分数 MAE。保存每样本分数与 fold/model hash。OOF 预测上得到的 `Δ` 只作为训练“收益预测器”的目标。
2. **收益预测器只能看文本阶段可得的量。** 可用 `f_T(x_T)`、其训练内可独立估计的不确定度/ensemble spread、文本 CLS/池化特征；不许输入 AV embedding、全模态预测或真实标签。分别评 `E[Δ|x_T]` 和 `P(Δ>0|x_T)`，可比较简单线性/小 MLP；先不叠加 MoNIG、额外 cross-attention 或新 fusion 网络。
3. **预算曲线做配对比较。** 在完全相同模型/特征/数据和 AV 预算下，对比：文本-only（0%）、全样本 always-all（100%）、随机抽取 `p%`、按文本回归不确定度挑前 `p%`、按预测 `Δ` 挑前 `p%`；可加 oracle `Δ` 排序作为不可部署的上界。报告 `p={10,25,50,75}%` 下 native MAE、平均配对 Δ、负收益获取率、按 source group bootstrap 的区间、`MAE vs. acquired cost` 曲线。正值 `Δ` 是 AV 有益；负值意味着路由付费后反而变差。所有 route 阈值只由训练/开发 groups 预先确定，test 只在机制冻结后评一次。
4. **与最近先例公平对比。** 至少实现 SeMARC/EAMA 的适配版作为方法比较：相同文本首阶段、相同基础模型及输入、同一 source-group split、同一可观察动作、同一预算/成本价格。逐项标明原算法保留部分和重写部分，不能把自己的 adaptation 说成对原论文方法的原样复现。Joint-AFA 可作为 cost-sensitive stop/acquire 历史基线；若不适合高维序列，说明其 tabular 输入边界。
5. **把预测质量与路由效用拆开。** 先报告 `Δ` 的 MAE/Corr、正收益概率 Brier/reliability、按 predicted-benefit 分箱的实际平均 `Δ`。再单独看政策的 `Δ`-budget 曲线。如果预测头校准改善但路由没有在同预算下改善平均 Δ/MAE，它没有实用决策收益；如果只赢 entropy 但不赢 EAMA/SeMARC，仍是已有方法的特定任务适配。

### 校准和组划分

视频内 clips 相关，普通按 clip 随机划分的 split-conformal/i.i.d. bootstrap 不能给出新的 source-video 保证。source-group OOF 防止预测重叠，却不等于已得到校准保证。应明确以“新 source group”为目标单位，独立保留 calibration groups；若目标是新组内所有 clips 同时满足收益下界，可按组聚合最大 residual 做保守校准，并披露 group 数量/大小；若只要平均 clip MAE，明确采用组加权还是 clip 加权风险。普通组聚类 bootstrap 只能支持不确定性估计，不能自动变成 conformal 保证。

若需要声称“所选路由子群内，收益误判率 ≤ α”，边界是更难的：按收益分数选出子群后，整体边际覆盖通常不推出所选子群条件覆盖；当同一 calibration data 还用于选择/调节 policy 时，终止模式本身也依赖校准集。RouteCert 的策略–终止模式联合认证正是这个问题的直接先例。最小试验先把“校准概率/下界质量”和“留出 source groups 上的经验政策效用”分开，不轻率声称个体保证。

## 成本对照与失效条件

- **MultiBench 当前特征包只能检验收益预测与输出头/路由开销。** GloVe/audio/vision 序列已经离线制作；如果先把全量 AV 包读取到内存或磁盘，再决定是否使用它们，路由没有节省 AV 原始采集、解码或特征提取成本。只能报告读取之后的 head/fusion FLOPs 或时延，而且要把“缓存后头部开销”准确命名。
- **端到端效率必须在 encoder 之前路由。** 实际量测要按样本分开记录文本输入/文本 encoder、AV 原始媒体获取、音频解码+WavLM、视频帧抽取/预处理+DINOv2、策略开销、fusion/head、总体 p50/p95 latency 与硬件能耗。完整系统与 text-first route 共用机器、预热和计时边界；route 中已付出的文本与路由开销都计入。若媒体已录制且不收费，节省的是编码算力，不是采集费用；若数据通过云端取回，也要区分网络/存储与 encoder 代价。
- **量纲需一致。** `Δ MAE` 是情感分数单位，不能直接减毫秒、美元或 FLOPs。固定获取率时可比较预算曲线；成本异质时需在目标函数内预先声明应用效用映射，或直接做 cost–error Pareto 曲线。不要靠任意选择 λ 得到想要的路由。
- **直接失效条件：** 预测收益与真实 OOF `Δ` 无关；同预算结果不胜过随机或文本不确定度；正收益概率未校准且下界经常选到负 `Δ`；收益只存在一个 seed / 一个相邻 split；训练组 OOF 与校准/评测组差异导致反转；收益由某个更弱 `f_TAV` 造成而非可部署增益；高价值 AV 样本上获取成本超过可获得收益；原始端到端测量计入 encoder 与 gate 后没有成本改善；或者当前 ALMT/MOSEI 基线完成后根本没有稳健的 AV 改善空间。

**决策建议：** 先等待 MOSEI 三 seed 原生基线完整结果，再用固定 backbone 和组划分做这项窄测试。不要扩展成新 fusion architecture。若收益预测/预算曲线失败，关闭方向；若成功，也先与 EAMA、SeMARC、RouteCert 做逐项差异与成本对照，再决定是否达到论文级方法贡献门槛。

## 补充：`Δ = m × r` 结构是否构成更窄的贡献？

2026-10-12 后续审查。对同一样本的文本点预测 `p_T` 和全模态点预测 `p_TAV`，令

\[
\Delta=|y-p_T|-|y-p_{TAV}|,\qquad m=|p_T-p_{TAV}|.
\]

反三角不等式给出 `|Δ|≤m`；`m=0` 时必有 `Δ=0`；若 `m>0`，定义 `r=Δ/m`，则 `r∈[-1,1]`。所以 `Δ=m r` 是精确的目标重参数化。对 CH-SIMS，`p_T` 和 `p_TAV` 必须先执行 native MAE 所用的 clip，`m` 和 `Δ` 才对应同一个受评估的损失；MOSEI 沿用未裁剪分数。该恒等式来自 L1 几何，不需要一个新学习算法来成立。

### 更近的回归 deferral 先例

[Mao、Mohri、Zhong 的 Regression with Multi-Expert Deferral（ICML 2024）](https://proceedings.mlr.press/v235/mao24d.html) 直接研究连续标签回归中的“当前 predictor 自己作答，或把样本交给一个更昂贵/更准确的预训练 expert”。论文接受一般有界回归损失，也包括 L1；两阶段方案先固定 predictor，再学习 deferral function；expert cost 通常写成 `L(g_j(x), y)+α_j`，其中 α 可表示推理代价。这样已有方法就能把文本预测器当作 `h`、文本+AV 预测器当作一个 fixed expert，并用训练标签计算两者损失，学习文本可见的 route。它不提出 `m×r` 这个具体分解，但已覆盖核心决策目标与“直接学习何时切换到回归 expert”的统计框架。它的理论要求 bounded loss/cost；如要援引该保证，MOSEI 的未裁剪输出需另行满足 boundedness，不能直接假定成立。论文页/所链接 PDF 未给出作者代码仓库；本次 exact-title 搜索也未核验到公开实现。

更早的 [controlled abstention regression](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002575) 已让回归网络输出预测值和不确定度，并以不确定度学习哪些预测应拒绝；它不估计 paired multimodal `Δ`。在决策论里，SeMARC/EAMA 的 acquisition gain 和 Q/action advantage 已经是“选这个 action 相比 stop 的收益”；按文本不确定度路由也是其标准基线之一。个体 treatment-benefit 文献则长期将收益方向与收益大小作为决策关心的两个属性，例如 [Causal Isotonic Calibration for Heterogeneous Treatment Effects](https://proceedings.mlr.press/v202/van-der-laan23a.html) 校准预测效应的 sign 与 magnitude；但这类因果效应并不等于同一标签、两种模型的 paired error gap。

本次在这些原论文及作者材料中**没有核验到同样的 `|p_T-p_TAV| × tanh(r)` 网络头或同一 reverse-triangle ratio target**。这只是“未找到相同编码”的有限文献结论，不能据此声称原创：两个 head、正幅值乘有界方向本身是通用的 sign–magnitude 参数化，而回归 deferral 已有直接风险差路由方法。

### 统计和优化含义

- **推理时的 cap 是预测量，不是证书。** 在 encoder 未运行时，真实 `m=|p_T-p_TAV|` 未知。`m_hat=softplus(z_m)`、`r_hat=tanh(z_r)` 使模型输出 `|Δ_hat|≤m_hat`，但这只约束预测器自己的输出；若 `m_hat` 低估真实模型间距，它就不保证 `|Δ_hat|≤m`，也不保证真实收益方向。除非 `p_TAV` 已执行，否则不能把 `m_hat` 称为“由 reverse triangle 确保的样本级收益界”。
- **无 `m/r` 辅助目标时，分解本身不增加表达能力。** 任意实数 `Δ_hat` 可由多种 `m_hat≥|Δ_hat|` 与 `r_hat=Δ_hat/m_hat` 得到；因子不可辨识，乘积模型与直接 Δ 回归主要差在优化偏置。带同样的 OOF `m`、ratio 监督后，能测试的是额外 teacher targets 是否改善收益估计；不应把它算成结构乘法带来的机制增益。
- **条件均值并不可分解：** 一般 `E[mr|x_T] ≠ E[m|x_T]E[r|x_T]`，差值由条件协方差决定。若分别把 `m_hat` 与 `r_hat` 当作两个条件均值，再相乘，通常不是 `E[Δ|x_T]`。若共同 Δ 损失让参数找到有效因子，则要用 OOF/留组验证证明，而不能从形式上推出。
- **ratio 尾部容易放大噪声。** `m≈0` 时 ratio 无定义或对小预测误差极敏感；遮掉 `m<ε` 后，`ε` 必须只由训练 folds 设定。无权重 ratio MSE 会让极小 `m` 的样本获得和高收益样本一样的方向监督；若使用此项，应在训练集上明确目标权重并单独做 sensitivity。`m≈0` 目标上的 `Δ` 已几乎没有路由价值，因而无监督 `r` 往往是更干净的控制。
- **辅助 targets 带来额外监督与计算。** `m` 和 `r` 都需要用 `f_TAV` 的 OOF 输出及训练标签形成。即使它们在推理时只看文本，训练成本也要计入每个 cell，并且 `m/r` 标注、样本 mask、head 参数、loss 权重、OOF folds 与 seeds 需要在 2×2 中固定。若 direct-Δ cell 没有等量 head/参数，结构效应和容量效应会混在一起。

### 最小结构检验与结论

若项目决定把它作为一个小型 ablation，2×2 的主问题应预注册为：

| Delta 输出结构 | 无额外 `m/r` supervision | 有同样 `m/r` supervision |
|---|---|---|
| 直接 Δ readout | 直接 paired-loss 回归基线 | 用容量匹配的辅助 heads 监督 OOF `m/r`，但 route 仍读取 direct-Δ head |
| `m_hat × r_hat` | 只用共同 Δ 目标更新乘积 | 共同 Δ loss + 固定的 `m`/ratio 辅助损失 |

四个 cell 使用相同 OOF 预测目标、基础预测器、文本输入、source-group folds、calibration groups、预算评估和 seeds；总参数量/输出 head 规模按配置配平，报告实际梯度更新与额外 teacher 成本。辅助 ratio loss 对 `m≈0` 的处理要在实验前写明。另加 `E[|y-p_T||x_T]-E[|y-p_TAV||x_T]` 的两风险相减控制，它回答配对差值监督是否比两个风险估计相减更稳；再比较回归 two-stage deferral surrogate（Mao et al.）、raw-Δ、text uncertainty、EAMA/SeMARC 适配及随机同预算路由。

主评标准仍是 held-out source groups 上同预算的实际 native MAE/平均 `Δ` 和负收益获取率。另报告 Δ 预测误差、`P(Δ>0)` 的 Brier/reliability、分组置信区间，以及 `m_hat` 对真实 `m` 的误差/覆盖图；这些指标能区分“收益预测变准”和“输出被硬约束得更窄”。独立 calibration groups 只评估/校准 `P(Δ>0)`，仍不能把 `m_hat` 解释为实际 `m` 的保证界，也不能由边际概率校准推出被路由子群内的条件误判率保证。

**原创性裁定：建议降级为参数化/估计器 ablation，不将 `m×r` 本身作为核心贡献。** 最接近的回归学习到 defer 已能直接用任意有界 loss difference 学切换决策；本结构目前没有新的统计保证，预测端的 `m_hat` 也不是可验证上界。若 product-only/no-aux 不胜 raw Δ，结构可以直接弃用。只有当 paired `m/r` OOF target 在 source-group 外部验证上稳定改善同预算路由，且显著优于回归 deferral、两风险相减、raw Δ 和 uncertainty baseline 时，才值得把“该目标/辅助监督在本任务中减少收益估计噪声”作为有限实证结果讨论；它仍不能自动变成普适新机制。
