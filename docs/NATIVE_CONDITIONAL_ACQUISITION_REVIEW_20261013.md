# 原生回归下的文本先行 AV 获取：条件风险、先例与最窄检验

调研日期：2026-10-13。本文只审查一个下一核心问题：只用文本阶段可见的信息，能否在实际运行 AV encoder 前判断哪些样本值得获取音频和视频，并在独立 source 评估中控制相对全模态模型的 native MAE 变化。只读研究；未改代码、未训练、未用 GPU、未读取 TEST。

## 判断

当前构想值得作为**小型、可证伪的系统实证**，但不能作为原创路由算法来命名或宣传。其损失差目标正是固定两个回归专家之间的逐样本风险差；回归 learning-to-defer 已经覆盖固定预测器加成本敏感选择，AdaMML、A2MT、EAMA 和 2026 年 10 月新公开的 SeMARC 则覆盖样本级、获取前的动态模态选择。通过 source-CV 造 OOF 标签、加入文本 confidence、按预测收益阈值分流，或再做独立校准，都没有单独产生新的算法原理。

仍有价值的具体问题是更窄的：**在这组原生连续情感回归任务中，严格留 source-video 后，文本可观察量能否预测同一样本上固定 text/fusion 模型的配对 MAE 差；这种排序能否在真实 AV encoder 前取得比强回归 defer/VOI 对照更好的风险—实际成本 Pareto 点？**若通过，适合先报告为特定 MSA 回归、source-heldout、text-first acquisition 的经验结果；需要另有新的可验证算法或风险保证，才有理由升格为方法贡献。

## 目标与当前证据

令两个基础模型在同一 source-heldout clip 上的预测分别为 text-only 的 \(\hat y_T\) 和 full-fusion 的 \(\hat y_F\)，先按各数据集 native MAE 的预测规则处理输出，再定义：

\[
r_i = |y_i-\hat y_{F,i}| - |y_i-\hat y_{T,i}|.
\]

正值表示文本误差更低，跳过 AV 能同时少用计算并降低该模型对的 MAE；负值表示 full fusion 更准确。若 \(a_i=1\) 表示跳过 AV，则混合路由相对固定 fusion 的配对误差差精确为

\[
R_{route}-R_F=-\mathbb{E}_{clip}[a_i r_i].
\]

因此，要求路由 MAE 不超过 full fusion 的总体约束是 \(\mathbb{E}_{clip}[a_i r_i]\ge 0\)；若接受最多 \(\delta\) 的 MAE 回退，则是 \(\mathbb{E}_{clip}[a_i r_i]\ge-\delta\)。CH native score 与预测都限制在 [-1,1] 时，\(r_i\in[-2,2]\)；对固定策略可令 \(z_i=(2-a_i r_i)/4\in[0,1]\)，回退条件等价于 \(\mathbb{E}_{clip}[z_i]\le(2+\delta)/4\)。这给出一个可研究的有界相对风险标量，但 threshold/策略选择、source 抽样单位和 calibration 依赖仍须单独满足认证前提；MOSEI 的未裁剪 native 预测不能直接沿用这个界。这个差值只是两个固定预测器之间的配对风险核算，不是 AV 对人类情感的因果效应、真实模态价值或个体保证。成本需另用原始数据测量并呈现 Pareto 曲线；不要把毫秒、焦耳、美元或 FLOPs 直接减进情感分数。

这个问题有实验证据支撑，但效应并未稳定定论。既有 raw source-OOF 三 seed 中，简单 fusion MAE 为 .41941/.41489/.40540，text 为 .39477/.39431/.39171；九个 source-fold 内坐标标准化控制中，fusion 改善为 .39265/.39846/.38961，约与 text 平均 .3936 相当，方向并不统一。随后作者原 TRAIN/VALID cohort 的六个标准化点预测实验里，fusion 为 .40805/.41567/.41084，text 为 .44360/.44296/.44091；fusion 三 seed 都较好，但这批结果仍处 root reload audit。见 [source capacity 结果](NATIVE_SOURCE_CAPACITY_FINDINGS_V1.md)、[标准化 source formal runs](native_source_standardized_formal_20261013.json) 与 [作者 cohort 标准化基线](native_simsv2_standardized_baselines_20261013.json)。这组差异说明应该固定同一训练协议、同一预测器对后再研究“哪些样本需要 AV”；不能用不同标准化设置的 text/fusion 预测拼成 regret 标签。作者 cohort 的 reload audit 完成前，新的 gate 结论也应暂缓。

## 最接近的先例与差异边界

| 先例 | 已覆盖的部分 | 对本候选的边界与代码状态 |
|---|---|---|
| [Regression with Multi-Expert Deferral，ICML 2024](https://proceedings.mlr.press/v235/mao24d.html) | 最直接的决策理论先例：给定预测器或专家，学习回归样本应由谁作答；支持有界回归损失和样本相关成本，并给出两阶段固定专家/学习 deferer 的一致性框架。把文本模型当便宜专家、full fusion 当昂贵专家，已经覆盖“文本可见时预测两个专家的损失差再决定是否转交”。 | 这是本试验最强的公平方法对照。直接拟合上述 r 后按阈值选动作是它的 plug-in 风险差版本，不能称新 defer 算法。PMLR 原论文页没有链接作者实现；按论文 surrogate 忠实适配时应标作复现性受限的适配，不伪称原样代码复现。 |
| [AdaMML，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Panda_AdaMML_Adaptive_Multi-Modal_Learning_for_Efficient_Video_Recognition_ICCV_2021_paper.pdf)，[作者实现](https://github.com/IBM/AdaMML) | 按样本/视频片段决定保留哪些模态，策略网络先运行、选择后才执行相应识别子网；论文还用低成本 RGB frame difference 作昂贵 optical flow 的 proxy，并明确报告策略开销和计算节省。它直接占据“输入条件的模态跳过以降低推理成本”这一概念。 | 原任务是 RGB/flow/audio 分类，不是文本先行 native MAE。其策略也使用低分辨率非文本预览，所以将它改成 text-only regret 预测并非原样复现；所有预览、解码和策略开销必须计入。代码仓库可查，但当前 GitHub 显示已归档。 |
| [A2MT，TMLR 2023](https://openreview.net/forum?id=I_CxADVLM8p) | 明确研究高维时间多模态数据的主动获取，以预测损失和模态获取成本训练序列策略；作者的消融也发现模型虽学会对成本反应，却未能稳定学到个体自适应策略。 | 是已知 acquisition/value 路线的直接对照。相比本文二选一静态 AV bundle，它做逐时间步多次 acquisition；不应为适配而把其计算成本从目标中省略。此处未核到作者维护的官方代码入口。 |
| [EAMA：Efficient Adaptive Modality Acquisition](https://openreview.net/forum?id=gnJNCIPO6y)，作者学位论文可由 [MIT DSpace](https://dspace.mit.edu/handle/1721.1/174585)核对 | 为部分已观察输入和候选未观测模态学习逐样本期望 task value，并最大化 value 减获取成本，支持多阶段获取和停止。它与“文本可见后预测追加 AV 的 MAE 收益”在价值估计层面高度重合。 | 当前候选的可能差异仅是 native continuous MAE、固定 T/F 配对损失差及留 source 的评估对象；这属于特定任务上的目标适配，不足以成为新算法。审查时未核实到 EAMA 作者公开实现。 |
| [SeMARC，2026-10-05](https://arxiv.org/abs/2610.07466) | 截至本次调研，最接近且非常新的获取前路由先例：逐次只用已获取证据，在未选 encoder 执行前决定下一模态或停止；控制器结合模态集合的边际 utility prior 与 residual fitted-Q。论文在六个分类数据集上比较 EAMA、AdaMML、Joint-AFA 等，并报告从原始输入到输出的 FLOPs、延迟和能耗，而不是只算 gate FLOPs。 | SeMARC 是分类、逐步部分融合；本候选是从 text-only 一次路由到全 AV 的连续 native MAE 风险差。这个差异值得做小适配，但不支撑“首次按样本估值/获取 AV”。本次未在论文材料里找到作者公开代码链接；若重写它，应标明自己的 regression adaptation。 |
| [Joint-AFA，NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/e5841df2166dd424a57127423d276bbe-Abstract.html) | 经典的成本敏感主动特征获取：联合预测与 acquire/stop，给出了学习型 policy 强对照。SeMARC 也将其纳入对比。 | 任务与输入偏低维分类；若不适合连续多模态时序，应将其视为需解释边界的已知路线，而不是略去所有 acquisition 方法后宣称新颖。 |
| [RouteCert，2026](https://arxiv.org/abs/2608.15520)，[作者代码](https://github.com/MelikaBaghi/RouteCert)；[Conformal Risk Control](https://arxiv.org/abs/2208.02814) | RouteCert 直接处理策略产生的终止模态组：路由后按实际终止 pattern 控制风险；当校准数据也选择策略时，校验完整 policy-pattern 组合。一般 CRC 可控制有界单调损失的期望。 | “再用独立/嵌套 calibration 保证 AV 路由不伤 MAE”本身已有成熟统计方向。若使用同一批校准源选择 threshold 和报告风险，普通 split conformal/CRC 不能自动覆盖这种自适应选择；需 RouteCert 类联合认证，或再留独立 source 评估组。RouteCert 代码公开，但它不是本项目的回归 MAE 结果复现；本项目的相对 MAE 差是有符号量，还需映射成预先声明的有界非负风险并证明所需控制，不能直接套用一条覆盖率结论。 |
| [Multimodal Routing，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.143/)，[作者仓库](https://github.com/martinmamql/multimodal_routing)；[EUAR，ACM MM 2024](https://openreview.net/forum?id=78TMql1c04)，[作者实现](https://anonymous.4open.science/r/EUAR-7BF6)；[3WD-DRT，2026](https://doi.org/10.1016/j.ins.2025.122704) | MSA 已有逐样本 modality importance、置信度/不确定度驱动的 gating、MoE 路由与回归模型。它们强烈限制“Jev 启发动态 confidence gate”本身的原创空间。 | 这些融合/专家门通常在 modality feature extraction 后给融合权重、筛内部 experts 或调整表示；它们不必然在原始 AV encoder 前跳过分支，故不能当作节省 WavLM/DINO 编码的效率证据。要按 acquisition 功能公平比较，还需统计它们已经支付的 encoder 成本。 |

原始材料进一步区分了“学会让分数/表示更依赖某模态”与“未用模态 encoder 根本没有运行”：SeMARC 特意把 post-encoding selection 和 before-encoder acquisition 分开，指出前者已付出被丢弃模态的特征获取成本；AdaMML 用便宜的 preview 做决策，但 preview 本身也有成本。故本候选若只输入纯文本预测、文本不确定度与文本表示，确实比一般 MSA 动态融合 gate 更清楚地满足“AV encoder 前决策”；这是部署约束上的差异，而非新的 risk-learning 原理。

## 一个窄而可证伪的实验

建议只检验一个二动作系统：先运行同一冻结 text-only predictor；若文本可见路由选择 stop，就输出文本预测；否则才读取原始音视频并运行同一冻结 full-fusion predictor。首轮只测试现有原生 CH-SIMS 目标及 source-video-heldout TRAIN 评估，不扩成新 fusion 网络、不合并 MOSEI 的 [-3,3] 目标。当前标准化作者 cohort 的 reload audit 完成后，选定一个固定的 text/fusion 配对方案，并在预注册里固定其训练源、种子、预测 clipping、标准化和 checkpoint 选择政策。每个样本的 regret 必须由**同一 fold、同一 seed、同一 native 标签上的两项 out-of-source 预测**计算。

### 必须完全嵌套

采用外层 held-source 作为最终经验评估；每个外层训练集内重做全部内层步骤。只可使用外层训练 sources 生成 gate 的 OOF 训练表、拟合 preprocessing/standardization、选择模型与成本价格。划出独立 calibration source groups 给一个已冻结的路由/threshold 使用，然后在完全未碰过的外层 source groups 评估。若 calibration 也参与选择 threshold 或策略族，应预先采用适用的 policy-pattern 联合风险认证，或再保留最终独立评估 sources。

**不能复用当前整张 cross-fit OOF 表当作嵌套 gate 的训练/校准数据。** 表中某 outer-held source 的标签可能已进入其他 fold teacher，后者又为 gate-training rows 产生预测；这会把 outer source 信息带进本应独立的管线。AnchorCDF 记录已明确揭示了这种“校准集看似 source-disjoint，但别的折教师曾使用校准来源”的问题；本实验必须在每个 outer split 里重新生成老师、OOF 特征、regret 标签、归一化、选择和路由。

推理时的 gate 只能读取当时已经有的文本输出、文本预测的不确定性和文本 encoder 的表示。不得输入 AV prediction、AV embedding、音频响度或视觉预览而仍宣称省掉 AV encoder：如果算门控前这些 AV 特征已被提取，节省额只能是后续模型部分。含 ASR 的文本入口也要明确是否把 transcript 当已提供；若需语音识别，ASR 的原始音频成本不能从系统总成本中消失。

### 最少对照与主要终点

| 对照/策略 | 比较目的 |
|---|---|
| 永远 full fusion、永远 text-only | 定义误差与成本两端点。 |
| 同预算随机 acquisition | 检查任何细分路由是否胜过无个体信息；随机分派应在 source 与实际 AV 成本上配平，并用多个随机种子。 |
| text confidence threshold | 检验常规 confidence/selective prediction 是否已足够；阈值只从内层 source/专用 calibration 源选。 |
| source-train 的 utility-mean baseline | 同时报告单一全局平均 regret 决策与低容量的 text-confidence 分箱均值路由：后者在训练 sources 上估计各预先固定文本置信分箱的平均 r，只用该 lookup 决定 text 或 fusion。用同预算 Pareto 点与复杂 regret gate 比较；全局单一决策在中间预算时对应同预算随机策略。 |
| 两阶段 regression learning-to-defer | 用 Mao–Mohri–Zhong 的固定专家、连续有界损失和实例成本设定作最强通用回归 gate 对照；使用同样的输入、source folds、外层风险和成本。 |
| 已知 VOI/acquisition route 适配 | 最少选一个 EAMA 或 SeMARC/A2MT 风格的 text-first value route；其 value target、动作约束和 cost 项必须忠实注明，不能称作作者实现复现。若本轮资源只支持一个强方法比较，优先先做 regression defer，再补一个 acquisition 专用适配后才讨论算法差异。 |
| oracle regret 排序 | 只作不可部署上界，禁止当作可用策略或调参目标。 |

首要报告每个 outer source fold/seed 上的 native MAE、配对 \(R_{route}-R_F\)、被获取比例及总量、source-cluster uncertainty、负收益选择率，以及 MAE–实际成本 Pareto。固定预算比较同时给出 AV 样本率与原始 encoder 实际时间/能耗预算；若样本时长不同，以 calibration sources 上选出的门槛匹配实际预算，在 outer sources 上冻结评估。source-bootstrap 区间只描述这些组的抽样不确定性，不能包装成有限样本校准保证。阈值选择必须只看训练内侧；TEST 不用于筛选策略族、置信区间或阈值。

### 原始端到端成本是结果的一部分

缓存 AV features 只能检验 gate 的质量和缓存读取后的分支开销。真实效率结论必须在可用的 raw-media route 上把决策放到 AV encoder 之前，并对全模型与路由模型用同一媒体、相同硬件、预热、批量/并发和计时边界，分别记：

- 文本准备/ASR（如适用）、文本 encoder 与 gate；
- 文件/网络获取、解封装与音频/视频解码；
- WavLM 输入处理与 encoder、视频抽帧/预处理与 DINO encoder；
- fusion/head、策略运行、端到端 p50/p95 latency 与设备能耗。

将未获取模态的 encoder 完全跳过，才计为 encoder 节省。媒体已经预先下载时，不能声称节省网络/采集费用；只可报告减少了本地解码和推理计算。当前本地原始媒体示例只有两条 competition-acquired TRAIN clip，原作者完整 raw media 仍不可用；这只够验证计时管线，不足以支持数据集级效率结论。也不能以一个 MLP 的 FLOPs 代替上述测量。

## 风险解释、否决线与旧结果

“校准后保证不超过 full fusion MAE”须先定义风险单位。若 deployment target 是随机 clip 的 native MAE，路由按 clip 求 \(\mathbb{E}[a r]\)；若 calibration 以每个 source 的均值作为一个 conformal 单位，则控制的是 source-equal 风险。不同 source 的 clip 数量不同，这两个目标不相同。source ID 当前表示 source video，不是已验证 speaker。要声称新 source 的风险控制，必须披露以 source 为抽样单位的交换性/抽样假设和组大小加权方式；它们不会因按 source 拆分或 bootstrap 自动成立。先报告外层 source-heldout 的经验结果，有限样本保证则需另行说明与验证假设。

继续条件应同时满足：文本 gate 的输入在实际决策时可获得；outer source 上的同预算 MAE–cost 曲线在 seed/fold 间稳定优于随机和 text confidence，并至少达到 regression defer / 一个忠实 VOI adaptation 的风险或成本前沿；相对 full fusion 的风险落在预先声明的零回退或 \(\delta\) 预算内；实际原始 AV 推理测到净节省。若只改善 pooled MAE、只在一个 seed/source fold 有利、或仅在缓存特征上省下少量 MLP FLOPs，均不够。若 gate 预测的是 r 但路由端的同预算 MAE 没改善，也没有决策价值。

旧的五分类 per-window utility/gate 失败必须原样保留，见 [architecture reassessment](architecture_reassessment.md)：已测窗口 utility 监督没有提升准确率，旧 gate 与观测到的 masking utility 几乎无相关。这里是不同的决策动作和终点——原始 AV encoder 前的 text-vs-full-Fusion 二选一，标签为 source-heldout native-L1 配对风险差；这不抹掉旧负结果，也不能当作旧窗口机制换名复活。若当前 route 候选失败，应停止将其包装成核心贡献。

## 来源与本地证据

本文补充 [10 月 12 日 acquisition review](NATIVE_ACQUISITION_RESEARCH.md)，集中更新本地标准化结果、2026 年 10 月 5 日公开的 SeMARC，以及 source-risk 和真实编码成本边界；不取代该 review 中对 EAMA、RouteCert、Joint-AFA、回归 defer 及旧路线的其他详细审查。

