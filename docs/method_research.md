# JevAffect 方法研究：CMU-MOSEI + CH-SIMS v2 五分类

**检索日期：** 2026-10-11。只读查阅论文、作者仓库和官方数据页；未安装、训练或调用 GPU。

## 结论

MulT、SELF-MM、MMIM、TETFN 与 ALMT 提供了时序交互、模态特异监督、信息保留和语言引导融合等强参照。近两年已经出现不确定性路由、贝叶斯融合、冲突分解和校准共识工作；因此，“加一个可靠性门控/冲突分数/温度校准”不能单独作为 JevAffect 的新颖性。Jev 只借鉴结构化概率决策的思路，不能声称复现 RLCD：目前没有可公开核对的 RLCD 配方。

建议的竞争比较是 **ALMT + CMC**：ALMT 是公开作者实现的强时序融合基线；CMC 是直接覆盖 MOSEI 和 CH-SIMS v2、并专门研究模态不一致的近年基线。两篇的公开结果主要用二分类指标，必须在统一五分类协议下重跑，不能把论文中的 Acc-2 当成五分类成绩。

## 方法位置与最接近先例

| 方法 | 与本项目最相关的已公开设计 | 复现提示 |
|---|---|---|
| [MulT (ACL 2019)](https://aclanthology.org/P19-1656/) | 方向性跨模态注意力直接处理不同长度的时间序列；时序基线。 | [作者代码](https://github.com/yaohungt/Multimodal-Transformer)，MIT；老环境，优先用统一框架实现或移植其模块。 |
| [TETFN (Pattern Recognition 2023)](https://doi.org/10.1016/j.patcog.2022.109259) | 文本引导的跨模态注意力与单模态任务，兼顾共享/差异信息。 | [MMSA](https://github.com/thuiar/MMSA) 提供模型实现；MMSA 为 MIT，统一支持 MOSEI、SIMS、多个方法。先核实其 SIMS 特征是否为 v2。 |
| [SELF-MM (AAAI 2021)](https://arxiv.org/abs/2102.04830) | 从多模态标注生成单模态伪标签，再用多任务学习模态差异。 | [作者代码](https://github.com/thuiar/Self-MM)，MIT；原论文主要是强度回归，五分类需统一输出头。 |
| [MMIM (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.723/) | 分层最大化单模态间及融合与单模态间的信息，防止融合丢失任务信息。 | [作者代码](https://github.com/declare-lab/Multimodal-Infomax)，MIT；README 主要给 MOSI/MOSEI 流程。 |
| [ALMT (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.49/) | 用多尺度文本表示指导音视频超模态，明确抑制辅助模态中的无关/冲突信息，同时压缩序列。 | [作者代码](https://github.com/Haoyu-ha/ALMT)，MIT；有 MOSEI、CH-SIMS 配置，作者在 2025 年更新过实现并建议用统一特征公平比较。未明确承诺 CH-SIMS v2，适合作强基线但需对齐数据版本。 |
| [EUAR (ACM MM 2024)](https://doi.org/10.1145/3664647.3680949) | 专家不确定性和路由，目标是对不同噪声样本选较可靠专家。 | [作者代码](https://github.com/ZixianGao/EUAR) 自述为 partial implementation；仓库页未见许可证，不应假定可再分发。 |
| [TMSON (TCSVT 2024)](https://doi.org/10.1109/TCSVT.2024.3376564) | 估计单模态不确定性分布、贝叶斯融合，再用序数回归约束情感等级；与五分类概率决策很接近。 | 查到论文，未找到作者公开代码。 |
| [CMC (arXiv 2025)](https://arxiv.org/abs/2510.20256) | 伪单模态预训练、参数无关融合与共识路由，显式研究模态语义不一致。 | [作者代码](https://github.com/gw-zhong/CMC)，MIT；README 有 MOSEI 和 SIMS-v2 配置，论文报告重点是二分类。是最贴近的公开挑战基线。 |
| [MCAN (2025)](https://arxiv.org/abs/2502.09675) | 分层拆分一致/冲突表征，并在表征和输出层施加差异约束。 | 论文覆盖 MOSI/MOSEI；本次检索未找到作者仓库。说明“显式冲突分支”本身已有人做。 |
| [RUAL (2026)](https://doi.org/10.32604/cmc.2026.085382)、[ECA (Pattern Recognition 2026)](https://doi.org/10.1016/j.patcog.2026.113315) | RUAL 用注意力熵估计不确定性并调节跨模态残差；ECA 用冲突强度驱动一致/冲突双路径融合，评估 CH-SIMS 与 MOSEI。 | 都是必须纳入新颖性核对的近年先例；本次未找到作者公开实现。 |
| [HCF-Net (2026)](https://doi.org/10.1016/j.sasc.2026.200517) | 分层局部—全局时间编码，加对齐与冲突残差双分支；覆盖 MOSEI。 | [仓库](https://github.com/TangoRomeo80/HCF-Net) 标注 Apache-2.0，但论文/代码主要是 7 类、已对齐特征，非 CH-SIMS v2 五类的直接对照。它进一步说明“时序 + 冲突分支”也不能单独宣称新颖。 |

## 可证伪候选假设（暂不声称 novelty）

**假设：** 在相同的保时序骨干和五分类标签下，经过验证集温度校准后，按时间窗计算的“序数冲突”能比全局熵或全局 JSD 更好地预测连续时间片遮蔽造成的预测风险。序数冲突可定义为不同模态五类后验之间的期望等级距离（相邻等级分歧小于正负极性反转），再保留该分数在时间轴上的位置与集中度。若把它作为可靠性融合的一个输入，模型应在固定比例连续时间片遮蔽时减小测试集 accuracy 的下降和 NLL；干净样本 accuracy 应与只用全局熵/JSD 的门控相差不超过 1 个百分点。分别在 MOSEI 与 CH-SIMS v2 检验，按样本配对 bootstrap 给 95% 置信区间；任一数据集的 accuracy 改善区间跨零，或干净 accuracy 超出界限，即不支持该假设。

这把问题限定为“序数后验冲突在时间轴上的位置是否对遮蔽敏感性有增量预测力”，而非重复提出通用冲突分支或可靠性门控。它仍与 TMSON 的序数不确定性、CMC 的共识校准、HCF-Net 的局部时序冲突、RUAL/ECA 的可靠性/冲突融合相邻；**新颖性必须经更完整检索和实验消融确认**。

## 公平评估与实施预算建议

- **数据协议：** CH-SIMS v2 官方给出 NEG/WNEG/NEU/WPOS/POS 五类、监督样本及说话人无关测试拆分；只用官方监督训练/验证/测试。MOSEI 原始情感分数为约 −3 到 +3，先公开写定归一化和五类边界，再冻结映射；不要从测试分布调边界。逐项核实缓存确为 v2、split 与标签版本一致，MMSA 的 `SIMS` 名称本身不足以证明使用 v2。
- **时间与证据：** 缓存每模态 `[T,D]` 序列、有效长度与掩码，不做均值池化。测试端做单模态全遮蔽及 10/25/50% 连续时间块遮蔽；记录遮蔽前后五类概率、NLL、预测变化和不确定性。用均值/中性特征替换并报告随机窗口对照，避免把注意力权重直接当因果解释。
- **概率指标：** 主指标用竞赛要求的五分类 overall accuracy；同时报告 fixed-five-class Macro-F1、平衡准确率、NLL、Brier、分类别 ECE 和错误排序的 risk–coverage。每个数据集只用验证集拟合温度参数，测试集只评估。报告五分类映射和类频率。
- **同骨干预算：** 固定同一组预提取文本、音频、视觉序列，特征编码器、投影维度、时间长度处理、划分、类别权重、优化器及早停规则一致；融合模块比较需配套同骨干消融。每方法每数据集先 3 个种子排错，正式结果至少 5 个种子；最多相同训练轮数/更新数和同等验证集调参次数，预注册早停指标，绝不用测试集选 checkpoint。CMC 两阶段的总更新数也计入预算。
- **硬件：** RTX 5070 Ti 16GB + 64GB RAM 足以优先做特征级序列模型；缓存 `.npy`/HDF5，按长度分桶、混合精度、batch 16–64 起步。先冻结或预计算 BERT/音视频特征，避免原始视频端到端抽取成为主耗时。具体吞吐与显存需实际运行测量，本次未运行。

## 数据与核心来源

- [CH-SIMS v2 官方页](https://thuiar.github.io/sims.github.io/chsims)：5 类定义、4402 个监督样本、未标注扩展和官方拆分信息；[数据论文](https://arxiv.org/abs/2209.02604)。
- [MMSA 作者框架](https://github.com/thuiar/MMSA)：统一实现 MulT、SELF-MM、MMIM、TETFN、ALMT 等；MIT；MOSEI 与 SIMS 特征/配置文档。
- [ALMT 作者实现](https://github.com/Haoyu-ha/ALMT)：MIT，含 MOSEI/CH-SIMS 训练配置，适合竞争性强的可复现结构基线。
- [CMC 作者实现](https://github.com/gw-zhong/CMC)：MIT，明确提供 SIMS-v2 与 MOSEI 运行配置和一致/不一致子集，适合最接近的冲突校准基线。
- [CH-SIMS 原始论文](https://aclanthology.org/2020.acl-main.343/) 说明五级划分来源；MOSEI 五分类仍需在项目协议中明确映射，避免沿用不同论文的二分类结果。
