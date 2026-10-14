# 原生回归评测协议独立审查

审查对象：`docs/protocol_research.md`、`docs/EVALUATION_PROTOCOL.md`、`jevaffect/protocol.py`、`jevaffect/metrics.py`，并对照 CH-SIMS v2 作者论文/代码、CMU-MOSEI 作者论文/数据入口及 MMSA 作者评测代码。未启动训练、未使用 GPU；未修改实现。本文的结论是：新版研究任务若声称遵循 CH-SIMS v2 / CMU-MOSEI 原生回归协议，训练目标必须保留数据集原始数值标签并输出一个实数。已有五分类流程只能作为另一个竞赛适配任务或派生对照。

## CH-SIMS v2.0 的标签与划分

CH-SIMS 的原始标注由五位标注者分别给出 −1、0、+1，最终分数为五人的平均，因此多模态原始标签落在 `{-1.0, -0.8, …, 0.8, 1.0}`，步长为 0.2。CH-SIMS v2 延续这一细粒度标注方案；它的原生监督目标应是 `meta.csv` 中原始数值 `label`，保留浮点数值而不先把它替换成类别编号。参考 [CH-SIMS 标注论文](https://aclanthology.org/2020.acl-main.343.pdf) 和 [CH-SIMS v2 论文](https://arxiv.org/abs/2209.02604)。

作者 v2 论文 Table 1 给出 train **2,722**、valid **647**、test **1,034**，三者相加是 **4,403**；正文同时称有 4,402 个有监督样本，存在一条记录的来源计数不一致。数据清单/最终实际 CSV 应作为这次实验的计数依据；项目当前核对出的 v2 metadata 也是 2,722/647/1,034，共 4,403 行。论文报告总数时应显式写出这处差异，不能把正文的 4,402 与三 split 计数混写。来源：[作者论文](https://arxiv.org/abs/2209.02604)，[作者数据/代码仓库](https://github.com/thuiar/ch-sims-v2)。

作者 v2 论文基准表实际列出 Acc2、F1、Acc2_weak、MAE、Corr、R²，没有列 Acc3 或 Acc5。作者仓库的 [`utils/metricsTop.py`](https://github.com/thuiar/ch-sims-v2/blob/main/utils/metricsTop.py) 还实现 Acc3/Acc5，因此下面的分箱定义是“作者公开评测代码定义”，不应误称为论文 Table 2 的已报告指标。源码在计算前把预测及真值都裁剪到 `[-1, 1]`，再计算 MAE、Pearson Corr、加权 F1 与 R²。

| 指标 | 作者代码中的区间（左开右闭；端点按右闭桶归类） | 中性 / 弱情绪处理 |
|---|---|---|
| Acc2 | `(-1.01, 0]` 为负，`(0, 1.01]` 为正 | 原始零分被归到负类，不剔除。`F1_score` 是这组二分类的 support-weighted F1。 |
| Acc2_weak | 与 Acc2 同一二分类 | 只在真值 `y ∈ [-0.4, 0.4]` 的样本上计算，端点包括在内；零分仍计入负类。代码上方有一条过时的弱情绪注释写成 ±0.6，但实际筛选及论文文字都是 ±0.4。 |
| Acc3 | `(-1.01, -0.1]`、`(-0.1, 0.1]`、`(0.1, 1.01]` | 零分为中性；由于真值步长为 0.2，中间桶对真值只含 0。 |
| Acc5 | `(-1.01, -0.7]`、`(-0.7, -0.1]`、`(-0.1, 0.1]`、`(0.1, 0.7]`、`(0.7, 1.01]` | 中间桶是中性；真值的五桶恰好对应 NEG `{−1,−0.8}`、WNEG `{−0.6,−0.4,−0.2}`、NEU `{0}`、WPOS `{0.2,0.4,0.6}`、POS `{0.8,1}`。 |

这些是对真实值和回归预测值分别做同一阈值转换后计算的准确率，不是五个 logits 的 argmax。特别是 Acc2、Acc3、Acc5 的中性归属不同：Acc2 把 0 并入负类；Acc3/Acc5 把一个以 0 为中心的小区间作为中性。新实现 `native_metrics.py` 对 SIMS v2 用 `searchsorted(..., side="left")`；这等价于作者循环的 `value > left && value <= right`。我在当前环境用 NumPy 2.3.5 对 float32/float64 的每个边界值及相邻可表示值逐项比对，分箱结果与作者代码相同；未发现 float32 阈值误分。

选模证据来自作者实现而非论文表格：[`config_regression.py`](https://github.com/thuiar/ch-sims-v2/blob/main/config/config_regression.py) 将 `KeyEval` 设为 `Loss`、early-stop patience 设为 8；[`V1.py`](https://github.com/thuiar/ch-sims-v2/blob/main/trains/multiTask/V1.py) 在验证集按最小 `Loss` 保存最佳权重，连续 8 个 epoch 未改进后停止。官方 V1 是带多任务/多模态辅助损失的实现，其 `Loss` 不是单一 MAE 的同义词。当前单输出回归实验以验证集 MAE 选 checkpoint 是清楚且合理的协议，但应描述为本项目的单任务回归规则，不能宣称逐项复现了作者多任务 checkpoint 目标。作者 [`run.py`](https://github.com/thuiar/ch-sims-v2/blob/main/run.py) 的调参分支注释声称使用 valid，实际调用却把 `dataloader['test']` 传给评测；不应沿用该调参路径，也不能用任何 test 分数选模型。

## CMU-MOSEI 原生回归与指标

CMU-MOSEI 的 sentiment 原始目标是 `[-3, 3]` Likert 数值，七个整数锚点由 −3（强负向）到 0（中性）再到 +3（强正向）；应保留加载数据中实际提供的浮点值，不映射到项目竞赛五类后再训练。官方标准 split 为 train **16,326**、valid **1,871**、test **4,659**（合计 22,856）；CMU 数据主页/论文也报告更大的全体标注语料数 23,453，故需把“数据集总语料”与“标准评测 split 覆盖量”分开写。数据入口以 [CMU-MOSEI 官方主页](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) 与 [CMU Multimodal Data SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) 为准；标签范围及数据集说明见 [ACL 2018 原论文](https://aclanthology.org/P18-1208/)。

作者 MMSA [`metricsTop.py`](https://github.com/thuiar/MMSA/blob/master/src/MMSA/utils/metricsTop.py) 对 MOSEI regression 定义如下：

- **MAE、Corr**：直接在原始预测和原始真值上算，不裁剪。Corr 是 Pearson 相关。
- **Acc2**：同时报告 `Has0_acc_2` 与 `Non0_acc_2`。Has0 对真值和预测分别以 `>= 0` 为非负类，故零归非负；Non0 先按真值 `y != 0` 筛选，再用 `> 0` 判正负。对应 F1 是加权 F1。
- **Acc5、Acc7**：只在准确率的支路先分别将预测和真值裁剪到 `[-2,2]` 或 `[-3,3]`，再套 `np.round`，以四舍五入后的整数是否相等计分。MAE/Corr 不沿用该裁剪。Python/NumPy 的 `np.round` 在半整数处采用 ties-to-even，因此这段具体实现需保留，不能擅自换成不同的桶阈值算法。
- 该作者实现也计算了将分数裁剪到 `[-1,1]` 后的三类准确率，但返回结果字典没有暴露 `Mult_acc_3`。论文/表格中若要报告 MOSEI Acc3，应明确采用的外部定义；不要声称它是这份 MMSA regression evaluator 的输出指标。
- 这组 MOSEI regression evaluator 不返回 R²。不要把 CH-SIMS v2 的 R²字段复制进 MOSEI，除非将其明确标作本项目的补充指标。

MMSA 的 [regression 配置](https://github.com/thuiar/MMSA/blob/master/src/MMSA/config/config_regression.json)对 MOSEI 与 SIMSv2 都使用 `KeyEval: Loss`；作者的一般选模规则是只看 validation loss。新版实现可以用原值上的 L1 训练、以 valid MAE 最小值选模型，并且只在选择及配置冻结后评估 test。不要在同一套比较里让某个方法按 test 的 MAE/Acc2/Acc5 各自挑不同 epoch，再比较这些 test 最优值。

当前 [native_metrics.py](../jevaffect/native_metrics.py) 在上述目标上与这些作者定义相符：SIMS v2 的 Acc2/3/5 使用左右端点规则并先 clip；MOSEI 的 Acc5/7 使用 `np.round(clip(...))`，Has0/Non0 分开处理，MAE/Corr 用未裁剪数值。Constant-vector Corr 以 `None` 输出，适合 JSON 且比作者代码产生 `NaN` 更稳妥。额外的严格输入范围检查和 support 计数是实现防护信息，不改变已定义的分数。

## 与当前五分类路径的边界及函数建议

现有 [protocol.py](../jevaffect/protocol.py) 的 `map_mosei_sentiment`、`map_simsv2_label` 是把原始分数转换成竞赛五类的接口；`make_prediction_record`、`validate_probability_vector`、`export_predictions` 则要求五类概率并输出五类文本标签。它们适用于 `docs/protocol_research.md` 记载的竞赛任务，不能作为原生 regression 的训练标签转换或预测格式。

现有 [metrics.py](../jevaffect/metrics.py) 的 `_numeric_matrix` 固定要求 `[N,5]`，`_labels` 要求 0–4 整数，`classification_metrics` / `accuracy` / `macro_f1` 都评类别结果；`fit_temperature_on_validation` 只接受五类 logits。历史 [training.py](../jevaffect/training.py) 用 5 类交叉熵、要求模型输出 `(batch,5)`，且以验证 accuracy 选 checkpoint。这些函数不支持 CH-SIMS v2 / MOSEI 原生回归值。当前 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) 明文注册的是五类竞赛适配评估，应随新版任务更新目标与结果口径，或将其明确限定为 competition-only 文档。

建议保持两个清晰接口：

1. 原生训练入口直接读取数值 `label`，校验范围（SIMSv2 `[-1,1]`，MOSEI `[-3,3]`），不调用 `map_*`；单输出回归头使用浮点标量，回归损失和 checkpoint 选择用约定的 valid loss/MAE，最终 test 只用一次。
2. 把 `native_metrics.native_metrics` 作为原生评测的唯一主入口；五类映射仅在单独的竞赛导出或派生五类分类对照中调用。结果表明确区分“native regression”与“five-class competition adaptation”，不要把两个 `Acc5` 概念混成一个：MOSEI 的 Acc5 是整数舍入分箱，SIMSv2 的 Acc5 是固定阈值区间，竞赛模型的 Acc 是五类 softmax argmax。

## 已核实的依据

- [CH-SIMS v2 论文](https://arxiv.org/abs/2209.02604)：v2 数据规模、每 split 计数、作者原文回归/分类指标及弱情绪范围。
- [CH-SIMS v2 作者评测代码](https://github.com/thuiar/ch-sims-v2/blob/main/utils/metricsTop.py)、[配置](https://github.com/thuiar/ch-sims-v2/blob/main/config/config_regression.py)、[trainer](https://github.com/thuiar/ch-sims-v2/blob/main/trains/multiTask/V1.py)：SIMS clipping、Acc bins、KeyEval Loss 与 early stop。
- [CH-SIMS 原始标注论文](https://aclanthology.org/2020.acl-main.343.pdf)：五人均值产生的 0.2 步长分值与五级类别来源。
- [CMU-MOSEI 官方数据页](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) 与 [CMU SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)：官方数据入口与数据说明。
- [CMU-MOSEI ACL 2018 论文](https://aclanthology.org/P18-1208/) 和 [MMSA regression 指标代码](https://github.com/thuiar/MMSA/blob/master/src/MMSA/utils/metricsTop.py)：原始标签范围、作者 Acc/F1/MAE/Corr 定义。
- [MMSA regression 配置](https://github.com/thuiar/MMSA/blob/master/src/MMSA/config/config_regression.json)：MOSEI/SIMSv2 的 `KeyEval: Loss` 及标准训练样本数。
