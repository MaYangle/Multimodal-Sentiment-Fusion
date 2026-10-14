# BC-MLF SIMS v2 五分类 source port

本目录提供模型 source closure 和协议适配层，供主项目自己的 trainer 接入；不替换或修改主项目统一 trainer，也不携带数据、语言模型、BI_ENC checkpoint 或训练输出。

## 来源与归属

- 固定来源：作者仓库 [sunyulin0421/BC-MLF](https://github.com/sunyulin0421/BC-MLF)，commit `428b973be580fd3cf09dd51bb61c448f2a26b2c9`。
- 原 MIT 许可证及 `Copyright (c) 2026 Yulin Sun` 保留在 [`vendor/bcmlf/LICENSE`](../vendor/bcmlf/LICENSE)。
- 原作者 README 的 acknowledgment 保留：BC-MLF 使用了 [DeepMLF](https://github.com/efthymisgeo/deepmlf) 的相关模型代码。这里没有另行复制 DeepMLF 仓库文件。
- vendored 模型文件：`models/singleTask/MSALM.py`、`MMGPT.py`、Transformer encoder 的 attention/position/transformer 文件，以及 `transformations/pool/mmaug.py`。这保留了完整的 MSALM、audio-visual encoder、插入式 multimodal GPT cross-attention、bottleneck token 和序列增强依赖，不是单独拿 BCHead 接一个新建 MLP。
- 另加入作者 `models/singleTask/BI_ENC.py`，其 SoftPerm 和 positional embedding 导入均指向已 vendored 的相对依赖。它暴露作者的 AV 预训练模型类，供主项目 trainer 执行 MSALM 前置预训练；没有复制 MMSA runner、trainer 或 data loader。
- 所有源代码中的 `MMSA.*` 绝对导入改为 `vendor.bcmlf` 包内相对导入；调用路径示例：`from vendor.bcmlf.models.singleTask.MSALM import MSALM`。未创建同名顶层 `MMSA` 包，避免覆盖主项目模块。
- 上游 `models/singleTask/__init__.py` 会 eagerly import 未 vendored 的 `BI_ENC` trainer、UNI 和 TAFC 等模型；此处用惰性包初始化文件替换，避免导入缺失依赖。模型本体与 `MMGPT.py` 依赖闭包仍保留。

## 明确适配

1. `MSALM` 的 `task_out` 默认仍为 1，保持 native regression 可运行路径。指定 `task_out=5` 时，MSALM 的任务/分支头输出 5 个 logits，BC 自适应 gate 仍输出三个分支混合权重，按 logits 逐类做凸组合。active adaptive path 中 `W_task` 仍保留源结构，但其输出随后被 branch mixture 覆盖，损失 helper 不把它当作额外主头。
2. 三支预测在 branch mix 前检查维数与 shape 相同：regression `[B,1]` 或五类 `[B,5]`。gate 公式、温度及最小混合权重语义不变。
3. `adapters/five_class.py` 增加五类主头 CE、逐 bottleneck token CE、text/AV branch CE 和源 BC soft label contrastive loss。CE 使用离散 `class_targets`；对比损失仍使用独立保留的 numeric `raw_y`。不要把 class ID 静默传作 raw label 以替代原连续情感距离。
4. `five_class_forward_view` 将 final class logits、未校准 softmax 类概率、平均 fusion/text/AV 分支 logits 与 BC branch mixing weights 分开暴露。它不计算 correctness confidence，也不声称 softmax 已校准。
5. 为避免缺权重时自动联网，vendored MSALM 仅接受本地 GPT tokenizer/model 路径，使用 `local_files_only=True`；缺失时报清晰的 `FileNotFoundError`。这只改变权重发现策略，加载到位的本地 checkpoint 和 BC 模型结构不变。
6. 五类 config 保留 BC 默认结构和关键超参（7 个 GPT 插入层、20 个 bottleneck tokens、adaptive branch gate、3 层 AV encoder、batch 32、最多 100 epoch）。切换到 CH-SIMS v2 官方 split/五类协议，使用项目已核实的 label mapper；不是作者论文在旧 SIMS 上报告的原结果。
7. `config/bienc_simsv2_pretrain_v1.json` 单列第一阶段 AV encoder 的 raw-label regression 预训练设定，并保留源 BI_ENC 配置的 `maxlen=39`；下游 MSALM 使用源 base config 的 `maxlen=55`。MSALM 配置中的 `path_to_pretrained` 指向待生成 checkpoint。BI_ENC checkpoint 仍是硬前置，不能用随机 encoder 替代。
8. AV `from_pretrained` 按源 state-key 约定加载 `Model.<target-key>`，使用 `torch.load(map_location="cpu", weights_only=True)`，逐项检查所有目标 key 均存在且 shape 一致，并用 `strict=True` 加载。missing、shape/type mismatch 或非允许的额外权重会明确报错；仅允许目标以外的 `Model.clf.*` 头作为额外源 key。若主 trainer 直接训练 `BI_ENC` 而不沿用作者 `AMIO` wrapper，保存 checkpoint 时需将 state keys 加上 `Model.` 前缀，否则下游 strict loader 会拒绝。
9. 五分类 loss helper 补回配置默认开启的 CLM 辅助目标：next-token target 使用源 trainer 的左移一位、末位置 token ID 0、attention mask 为 false 的位置设为 ignore index -1，交叉熵乘 `gamma` 后加入总 loss。若 `use_clm=true`，缺 `lm_logits`、attention mask，或既无 `lm_targets` 也无 `input_ids` 时立即报错。
10. 两份配置不再写死 D 盘路径，改用 `${BCMLF_DATA_ROOT}`、`${BCMLF_MODEL_ROOT}`、`${BCMLF_CHECKPOINT_ROOT}`。调用模型前用 [`config_paths.py`](../vendor/bcmlf/adapters/config_paths.py) 的 `resolve_bcmlf_paths` 递归解析；默认对应 repo 的 `data/mmsa`、`models`、`checkpoints`，也可由环境变量或显式 `roots` 覆盖。未修改主项目统一 trainer。

## 配置和调用

- [`config/bcmlf_simsv2_5class_v1.json`](../vendor/bcmlf/config/bcmlf_simsv2_5class_v1.json)：protocol-adapted 五分类配置。dataset loader 内部保留原 MMSA 的 `sims` alias，metadata 明确指向 CH-SIMS v2。
- [`config/bcmlf_simsv2_native_regression_override_v1.json`](../vendor/bcmlf/config/bcmlf_simsv2_native_regression_override_v1.json)：叠加到上述配置时，切回 scalar regression (`task_out=1`)；由评测端按统一协议对 scalar 输出评分。不要把该结果标成五类概率模型。
- [`config/bienc_simsv2_pretrain_v1.json`](../vendor/bcmlf/config/bienc_simsv2_pretrain_v1.json)：为 MSALM 准备 AV encoder 初始化的预训练配置。
- 调用 `MSALM(config)` 或 `BI_ENC(config)` 前，对 JSON 用 `resolve_bcmlf_paths` 解析，并传入 EasyDict-like 配置。模型 `forward` 返回 BC 原生 dictionary，包括 `task_logits`、`bn_logits`、`text_logits`、`av_logits`、`lm_logits`、`Contribution_weights` 和 feature keys。五类适配可用 `bcmlf_five_class_loss(outputs, class_targets, raw_y, input_ids=..., attention_mask=..., gamma=...)`；或传入预先生成的 `lm_targets`。主 runner 每个 batch 必须提供：项目 mapper 产生的 `class_targets`、原始数值 `raw_y`、供 MSALM 的 `input_ids` 与 `attention_mask`，以及对应 audio/vision 特征；模型输出提供各 branch logits、`lm_logits` 和 `Feature_bn`。若配置关闭 CLM，可显式传 `use_clm=False`。
- 显式调用 BI_ENC 模型：`from vendor.bcmlf.models.singleTask.BI_ENC import BI_ENC`；主 trainer 可用其 raw-label regression 输出执行前置 AV encoder 训练，保存带 `Model.` 前缀的兼容 state dict。
- 原作者 `run.py` 会强制 `train_mode='regression'`，还含 test-every-epoch 和 tuning 时读 test 的路径；本目录不复制该 runner。主 trainer 必须显式设置 classification loss/shape/metrics，并按 PORT_NOTES 避免 test 泄漏。

## 运行前置与未验证事项

- 本地 GPT：`${BCMLF_MODEL_ROOT}/gpt2-chinese-cluecorpussmall`，默认解析为 `<repo>/models/gpt2-chinese-cluecorpussmall`。项目 D 盘的对应模型目录在本次审查时不存在；配置名表明使用 UER `gpt2-chinese-cluecorpussmall`，需人工按许可准备 tokenizer 和权重，并确认真实 hidden size、层数与 `mmgpt` 的 768 维设置相容。不会隐式下载。
- BI_ENC checkpoint：`${BCMLF_CHECKPOINT_ROOT}/bcmlf-bienc-simsv2/bienc-simsv2-1990.pth`，默认解析为 `<repo>/checkpoints/...`。检查时没有此文件；先按官方 split 训练 vendored `BI_ENC` 并将兼容 state dict 保存到该路径，或同步覆盖 resolver roots/config。
- 特征与 tokenized text：`${BCMLF_DATA_ROOT}/simsv2/Processed/unaligned_39.pkl` 和 `.../mms2s`，默认根为 `<repo>/data/mmsa`。这些 v2 MMSA 特征在本次适配中没有生成或检查；准备后需重新核对 split 数、维度和文件 hash。
- Runtime 需要 PyTorch、Transformers 和 NumPy。使用现有 `D:/MYL/JevAffect/work/train-env/Scripts/python.exe` 将 adapter 与 strict-loader 单元测试固定在 CPU 上运行；11 项通过，包括五类 CE/CLM/contrast loss 的 CPU backward、LM shift/mask、BI_ENC vendor import，以及完整/缺失/shape 错误 checkpoint 加载。未实例化完整 MSALM 或 BI_ENC、未运行真实权重 forward/backward、未训练。
- 当前 bundled Python 静态编译和 JSON 解析通过。语法和 CPU helper 测试通过不说明真实 GPT/BI_ENC 权重兼容、完整训练可运行或显存合适；真实 checkpoint 尚不存在。
- 16 GB GPU 仍需小批量真实 preflight；source config 的 batch 32 保留用于忠实配置，不代表已经验证可运行。

## Integrated drivers (2026-10-12)

`jevaffect.bcmlf_training` and `jevaffect.bcmlf_stage2` provide Stage 1/2 training from the canonical cache. The source AdamW Stage 1 effective default weight decay 0.01 is explicit. Accumulation is normalized by sample count and short final groups are stepped; Stage 2 also clips AV gradients. These are documented trainer corrections. Stage 1 code identity contains only versioned project files. Stage 2 requires a completed Stage 1 checkpoint with matching seed/source/run identity; it preserves native AV maxlen 55 independently of text length 39. Integrated stage tests pass, including exact resume. Full real BC-MLF training remains unverified pending native data; a cloud pilot v1 failed before training due to a corrected file-ID typo.
