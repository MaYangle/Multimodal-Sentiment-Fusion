# 下一项机制独立审查（2026-10-13）

只读审查指定的 MOSEI / reference-residual / scope 文档并检查近期原论文。没有修改代码或运行模型；没有使用 GPU。

## 裁定

目前没有足够证据支持再提出一个可称为原创的融合机制。最具体、优先级最高的下一步是**先修正并重新评估 MOSEI ALMT 适配中的 padding 坐标依赖**。这是基线正确性检查，不能算 JevAffect 的原创贡献。若修正后的误差分析仍显示稳定且未被近邻工作覆盖的现象，再从那里提出新机制。

理由来自已完成的冻结特征诊断：ALMT valid MAE 为 `0.6115 ± 0.0051`；保留有效向量与顺序、仅改为前置 padding 后是 `0.6699`（+0.0584）。联合有效步置换是 `0.6223`（+0.0108），联合反转是 `0.6190`（+0.0075），独立打乱 A/V 则是 `0.6108`（−0.0007）。这表明相对 padding 布局的依赖远大于观察到的顺序变化，而没有证据表明细粒度 AV 对齐是 ALMT 优势来源。当前项目适配明确保留了 ALMT 的无 mask padding 注意力。与此同时，ALMT 有约 305 万参数，简单融合约 10.5 万，训练更新数也不同；现有差值不能单独归因于新机制。

## 建议的下一项：修复后做同骨干坐标因子审查

保持相同 MOSEI 特征、ID、划分、原始 L1、优化器、三种子和验证选择规则，比较同一 ALMT 骨干的三组：

1. 已有无 padding mask 的适配，作为复现锚点；
2. 有效步 mask 加按有效序列顺序编号的位置索引，作为正确性修复；
3. 与第 2 组相同，但训练时再加入等价前置/后置 padding 视图的一致性项。

第三组与第二组每一步计算两种视图，匹配总前向/反向预算、优化器更新数、参数量和看到的有效训练样本。固定其余预算，三个种子均报告原生 valid MAE/Corr/Non0 Acc、前置 padding 的配对变化、联合反转/置换敏感度，以及既有 source-disjoint 诊断。final test 继续保持冻结。

预先拒绝门槛：若第 2 组已经把相同有效序列在两种 padding 布局下的 MAE 差压到 `0.01` 以内，且第 3 组对第 2 组没有跨三种子稳定的原生收益，就只记录为基线修复，不继续包装一致性项。若优势只相对第 1 组成立、在第 2 组消失，也不构成新机制。只有第 3 组相对第 2 组在三种子同向改善、配对区间支持改善、source-disjoint 结果不反转，并且自然序列顺序效应仍可区分，才值得扩大原创性检索。`0.01` 是建议的预注册效应门槛，不是既有统计结论。

**原创性边界：** 有效步 mask 与正确的位置编号是常识性实现修正；把“对 padding 位置稳定”本身称为新算法也站不住。可继续研究的窄问题只是：在正确的坐标语义上，显式对等价存储布局施加一致性训练，是否比已修复的 ALMT 提高原生回归和跨 source 稳定性。即使通过门槛，也须先检查更多直接的序列位置不变性先例，再决定能否称为方法贡献。

## 暂不建议的第二项

弱非零样本的 ALMT MAE 为 `0.5208`，高于 raw-text 的 `0.4669`；中性和强情感组则改善。这是真实的误差异质性，但目前不能据此推出需要“sign/intensity decomposition”。极性与强度辅助任务已有直接先例，PS-Mixer 也分别建模 polarity 与 strength；将常见分解重述为新机制会越过证据。建议将它保留为修正后基线的误差分层/对照，不晋升原创假设。12 次 T+D 结果已否定当前 annotation-reference additive decoder 的结构增益；不建议再换名或转做 MOSEI proxy。

## 近期原始文献边界

- [ALMT 原论文，EMNLP 2023](https://aclanthology.org/2023.emnlp-main.49.pdf)：每个模态先用 Transformer 压到固定 token 数，再以语言引导的 hyper-modality 学习和跨模态 Transformer 融合；这是当前 MOSEI 骨干最接近的先例。论文没有报告“同一有效序列只改变 batch padding 位置”的不变性目标。
- [ALMT 作者模型实现](https://github.com/Haoyu-ha/ALMT/blob/master/models/almt.py)：可用于核验参考实现的序列投影、固定 token 与融合路径。项目本地适配已声明保留无 mask padding 行为，因此需区分作者模型与本地适配。
- [Analyzing Modality Robustness in Multimodal Sentiment Analysis](https://arxiv.org/abs/2205.15465)：对缺失/噪声模态做诊断并研究训练扰动；支持把鲁棒性与变换范围预先注册，但不等于对等价 padding 布局的测试。
- [Polarity and Intensity: The Two Aspects of Sentiment Analysis](https://aclanthology.org/W18-3306/) 与 [PS-Mixer](https://doi.org/10.1016/j.ipm.2022.103229)：极性与强度拆分/辅助监督已有先例，故不把简单 sign/magnitude 头当原创。

## 给主 agent 的结论

先做 ALMT padding/位置语义修复及同骨干预算匹配的因子审查；在结果出来前，不主张新的原生情感融合机制。这个建议由 +0.0584 的 padding 敏感度和远小得多的顺序变化支撑，且将常规修复和可能的研究贡献分开。
