# BC-MLF 代码实现审查：SIMS v2 五分类适配

审查对象：只读检查 `D:/MYL/JevAffect/work/BC-MLF`，HEAD 为 `428b973be580fd3cf09dd51bb61c448f2a26b2c9`，工作树干净。未改代码、未下载权重/数据、未运行 GPU。此结论针对该克隆版本，不是论文五分类结果；在 SIMS v2 和 JevAffect 协议上应称为 **BC-MLF protocol-adapted reproduction**。

关键源码：`MMSA/models/singleTask/MSALM.py`（模型与 BCHead），`MMSA/trains/singleTask/MSALM.py`（冻结、损失、评估），`MMSA/models/singleTask/MMGPT.py`（AV encoder 权重载入），`MMSA/trains/singleTask/BI_ENC.py`（AV 预训练），`MMSA/config/regression/bcmlf/sims/base_best_bchead.json`，`MMSA/config/regression/bcmlf/sims/bienc.json`，`MMSA/run.py`，`MMSA/utils/metricsTop.py`。

## BCHead 实际做什么

当前 SIMS 默认配置设 `use_contribution_calibrated_head=true`；`MSALM.__init__` 因此选 `branch_head_mode="adaptive"`。模型取三个分支表征：20 个 bottleneck fusion token 的平均表征、GPT 最后一个有效文本 token、池化后的 audio-visual 表征，并拼成 `task_inputs`。三个独立线性头分别给每个分支产生一个**标量回归预测**：`W_bn`（逐 fusion token 后取平均）、`W_text`、`W_av`。

Adaptive gate 是一个 MLP `task_inputs → d_task → 3`，对三个 gate logits 除以温度后做 softmax。配置温度为 1，最小权重 `m=0.08`；代码将权重变为 `w'=(1-3m)·softmax(g)+m`，从而每支至少 0.08 且总和为 1。最终预测是 `ŷ=w'_bn·ŷ_bn + w'_text·ŷ_text + w'_av·ŷ_av`。这是一种**由联合表征预测的样本级凸组合**。另外有 uniform / global / adaptive 三种模式；当前配置映射到 adaptive。

因此 BC 的 gate 值表示模型要给每个回归分支多少混合权重；它不是某分支预测正确的概率、不是不确定性估计，也不是经过可靠性校准的置信度。不要将 `Contribution_weights` 直接当作最终类别概率或 Jev 的 `choice_confidence`。Jev-inspired 的验证假设若是“按分支预测置信度/可靠性校准，再融合五类概率”，则与 BC 实际机制的区别是：BC gate 从融合表征学权重，并未用分支正确性或置信度监督；gate softmax 作用于**分支权重**，不是类别概率校准。

训练同时监督最终混合预测和分支预测。当前是 L1 回归：`L = L_task + l_bn·L_bn + l_av·L_av + l_t·L_text`；SIMS config 的三个辅助系数均为 1。融合特征另有 soft supervised contrastive loss，权重 0.005、温度 0.2、标签距离尺度 sigma 0.8。该损失以原始连续情感值算 `exp(-distance²/(2 sigma²))` 软正样本权重。BCHead 的分支混合和这个对比损失是两个独立机制。

实现细节：active adaptive 路径先算 `W_task(task_inputs)`，之后用 BCHead 分支混合覆盖该输出；所以当前损失不监督这个 legacy `W_task`，即便训练器把它解冻。做 port 时重点修改实际参与 forward/loss 的分支头；只有保留普通非 BC 路径或给 legacy head 单独定义辅助目标时才需让 `W_task` 输出 5 类。

## GPT 与可训练参数

SIMS 配置令 `lm=${BCMLF_MODEL_ROOT}/gpt2-chinese-cluecorpussmall`，代码用 `AutoTokenizer` 和 `AutoModelForCausalLM.from_pretrained` 加载；目录不在克隆仓库中，需预先准备。配置的多模态模块期望 768 维、`transformer.h` 层路径，插入层索引为 `[5,6,7,8,9,10,11]`。源码中的另一份 `gpt` 字典写了 1024/24/16，但 MSALM 这一路加载的是 `lm` 指向的 HF 本地 checkpoint；启动前应读取该本地模型 config 确认真实 hidden size/层数与 768 维 `mmgpt` 设置兼容，不要把字典尺寸误当成本地权重已核实。

训练器先对整个模型 `requires_grad_(False)`，之后只解冻指定多模态 cross-attention 层参数（attention、门控、BN embedding，以及 `use_lora=false` 时 cross-attention 内 `c_fc/c_proj`），全量解冻 AV encoder、task/branch heads 和 adaptive gate。基础 GPT 的原生 Transformer 权重和 token embedding保持冻结；这不是完全冻结 LLM 的推理式用法，因为梯度仍要经过被冻结的 decoder 运算回传到插入的跨模态层。AV encoder 配置 `finetune=true`，所以加载的 AV 初始化权重随后继续更新。

## 必须补齐的预训练依赖

`base_best_bchead.json` 的 `from_pretrained=true` 指向 `${BCMLF_CHECKPOINT_ROOT}/bienc-sims-test/bienc-sims-1990.pth`。仓库 README 明确没有数据、语言模型权重或 checkpoints；当前克隆也没有 checkpoint 目录。启动 MSALM 前必须先为**目标 SIMS v2 特征与官方 split**训练/生成 BI_ENC AV encoder checkpoint，并把配置路径改到确切产物。

`AV_Enc.from_pretrained` 创建同配置 AV encoder，再从 checkpoint 中复制带 `Model.` 前缀且形状相符的 state dict 参数；这里是 AV 编码器初始化，之后会继续 finetune。不要误认为这只是一个可选下载。仓库 README 的示例只展示了 MOSEI BI_ENC 训练；SIMS v2 适配需要自行跑第一阶段并记录数据路径、split、种子与 checkpoint hash。

## 测试集泄漏点

- `MSALM.do_train(..., return_epoch_results=True)` 每轮训练后会运行 `do_test` 并保存 test 结果。`run.py` 的 `_run` 通过 `from_sena` 参数打开该模式。正式实验保持 `from_sena=False`，不要逐 epoch 查看 test 曲线或按它选 checkpoint。
- `BI_ENC.do_train(..., return_epoch_results=True)` 也逐 epoch 调 test。预训练阶段同样禁用，或者只在阶段训练结束后对选定模型做一次 test。
- `run.py` 的 `is_tune` 分支注释写“use valid set to tune”，但执行的是 `trainer.do_test(... dataloader['test'])`。不要用此分支挑超参数；改为 valid，或禁用 tuning。否则会把测试集用于模型/配置选择。
- 普通 `_run` 是以 valid 的 `KeyEval="Loss"` 保存最佳权重，训练结束后加载 best checkpoint 再 test 一次，适合作为正式流程。最终 test 应只对固定配置、固定 seed 的最佳模型运行一次。可选 `dump_*_diagnostics` 若指向 TEST 也应避免训练期间查看。

## 五分类适配：仅设 `task_out=5` 和 CE 不够

当前 config 的 `task_out=1` 在 MSALM 模型中没有被读取；模型头输出维度在源码中写死为 1：`W_task` 最后一层、`W_bn`、`W_text`、`W_av`（以及 ULGM 分支头）都是 scalar。将配置改成 5 不会自动改变模型 shape。当前代码路径还被 `run.py` 多处强制设为 `train_mode='regression'`，训练标签 reshape 为 `[B,1]`，主任务和分支辅助损失都是 L1；`MetricsTop` 的 regression 评估会自行按其 MMSA 边界离散化。

至少需同步改以下内容：

1. **标签与 split**：用项目已核实的官方 SIMS v2 label 映射产生 0–4 整数目标；保持原始标签另存，严格保留官方 train/valid/test。loader/config 不能沿用旧 SIMS 的 `sims/Processed/unaligned_39.pkl` 和 `train_samples=1368`；需指向 v2 并核实全量训练样本数与模态维度。`num_classes` 也改为 5。
2. **输出 head**：active adaptive 路径中的 `W_bn`、`W_text`、`W_av` 输出改为 5。`W_bn` 每个 fusion token 输出 `[B,N,5]`，继续沿 token 平均为 `[B,5]`；gate 仍产生三个分支权重 `[B,3]`，按同样凸组合逐类混合三个 `[B,5]` branch logits，得到 final `[B,5]`。使用 logits 做 CE，推理时 softmax 得五类概率。若保留非 BC 路径，再把 `W_task` 改为 5；active BCHead 代码目前会覆盖 `W_task` 输出。这样保留了 BC 的“样本级 branch convex mixture”结构；将回归标量直接扩成 5 维或只改 `task_out` 不构成忠实移植。
3. **主/辅助 loss 和 label shape**：主 loss 对 `[B,5]` logits 与 `[B]` long class id 做 CE。三个辅助头也必须改成五类分类头并用 CE；BN token loss 要将 `[B,N,5]` 整形为 `[B·N,5]`、目标重复到 `[B·N]`，或明确先聚合再算损失。当前 `crit_bn/av/text` 固定是 L1；不能只替换主 criterion。保留 `l_bn/l_av/l_t` 系数，独立报告该辅助监督。
4. **损失中的原始标签**：当前融合对比损失的标签距离基于连续值。建议训练 batch 同时携带 `raw_y` 和离散 `class_y`：CE 使用 `class_y`，保留原始 `raw_y` 给原始 BC soft contrastive loss。若 v2 源标签本身只有离散五类，应明确按类别距离重定义对比权重，并将其记录为 protocol adaptation，而不要假称与连续标注版本完全一致。
5. **训练入口与评测**：修复 `run.py` 强制 regression 的入口；分类评测对 logits 使用 argmax，对真值使用同一固定映射，报告共同协议下五分类指标和混淆矩阵/每类指标。val `KeyEval` 可用 CE loss 或预先固定的五类指标做模型选择；test 仅最终运行。现成 `MetricsTop` regression 的 Acc5 是对连续预测和真值按 MMSA 规则分箱，不等同于五类 CE argmax 评测。

换言之，保留回归标量头、L1 训练、最终按 Jev 映射划分分数，是更小的**协议评分适配**，但模型本身依旧是回归；若目标明确为五类概率输出，需要以上多处变更。两种结果应分名报告。

## 16 GB 显存与训练预算

原 SIMS config 使用 `use_bf16=false`、batch 32、文本长度 39 加 20 个 fusion tokens，三模态输入使用预提取特征；配置含 3 层、30 维的 AV encoder 和 7 个多模态 GPT 注入层。基础 GPT 权重冻结，但仍需保存部分中间激活以训练插入层，且 AV encoder 与 gate/heads 更新。**16 GB 上有希望，但未经实测不能保证 batch 32 可跑**；免费 Kaggle 卡型/显存也不稳定。

建议第一次在目标卡做小型 forward/backward preflight，从 batch 8 或 16 起，检查峰值显存和速度，再逐步增大；`use_bf16` 仅在卡支持时开启。配置的 `update_epochs=8` 是梯度累积步数；当前 loop 多个 batch 的 loss 直接反传、没有除以累积步数，减小 microbatch 后仅增大 `update_epochs` 会同时改变有效样本数和梯度尺度，需相应调整 LR/累积实现或单独验证，不能宣称预算等价。

## 最小忠实 port 顺序

1. 先用 v2 官方 split 和确定的特征训练 BI_ENC，并锁定 checkpoint、数据/配置 hash。
2. 复制当前 SIMS adaptive BC 配置到单独的 `simsv2` 实验配置，保持 `branch_head_mode=adaptive`、温度 1、最小 branch 权重 0.08、gate 结构和 branch 辅助任务；更新数据路径、样本数、序列长度和 v2 split。
3. 做五类版时将回归分支头改成 5-logit，gate 按类别逐 logit 混合，分别给主头和三个 branch heads 加 CE；保留原始标签以支持对比损失，统一修改 train/eval 输出 shape 和指标。
4. 只根据 valid 选择 checkpoint；先禁用 `is_tune` test 分支与 `return_epoch_results` test 曲线；固定 seed/config 后最后一次性评分 test。

报告成绩名应写“BC-MLF, SIMS v2 protocol-adapted, 5-class”，并记录 checkpoint 与额外 AV 预训练阶段。不要将它写成作者论文在 SIMS v2 上的原样复现或引用论文旧 SIMS 分数作为 v2 成绩。
