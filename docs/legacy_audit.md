# JevAffect 旧代码静态审计

## 范围与结论

只读检查了 `D:\MYL\JevAffect-backup\legacy-source` 下的 `MSAbypkl`、`MSAbyvideo`、`video2pkl`、README、requirements 和 `test_script.py`。未修改备份、未安装依赖、未导入项目代码、未训练或调用 GPU。文件清单中没有 `.pt`、`.pth` 或 `.pkl` 权重/数据文件；相邻镜像未纳入本次代码审查。

这份快照可以作为旧代码和归属的证据保存，但现在还不能作为可复现的训练基线或可信的结果来源：PKL 训练入口取不到训练集，两个脚本式入口与 DataLoader/Trainer 接口不匹配；视频版把五类编号当连续回归目标；所有主要特征都已按片段池化，代码中的 Transformer 与 cross-attention 实际只处理一个 token。

## 主要发现

### P0：当前训练和验证管线不能闭环

1. `MSAbypkl/main.py:136-142` 在训练流程中调用 `get_dataloaders(..., force_test_mode=True)`。`MSAbypkl/src/data/dataset.py:343-369` 只有在该值为 `False` 时才创建 `train` 和 `valid`；随后 `main.py:193-197` 却读取 `dataloaders["train"]` 和 `dataloaders["valid"]`，训练启动即因缺少键失败。最小修复：训练入口传 `False`；仅评估入口采用 test-only 模式。

2. `MSAbypkl/scripts/train_multimodal.py:142-145` 默认传 `language,acoustic,visual`，而 `MSAbypkl/src/data/dataset.py:62-64` 只接受 `text,audio,vision`。即使改正模态名，DataLoader 返回键是 `valid`（`dataset.py:357-365`），脚本却读取 `val`（`train_multimodal.py:230-240`）。此外该脚本传入 `optimizer=`，但 `MSAbypkl/src/training/trainer.py:47-62` 的构造函数没有此参数。逐项修正这些接口后才能运行脚本式训练。

3. `MSAbypkl/main.py:193-202` 创建 `Trainer` 时没有传语言专用模型路径，`MSAbypkl/src/training/trainer.py:53-54,91-94` 因而默认尝试加载 `best_models/zh.pt` 和 `best_models/en.pt`。这会让从头训练依赖外部评估权重。即使权重齐全，`trainer.py:199-218` 的验证仍按 `language == "zh"/"en"` 选择这两个外部模型而不是当前训练的 `self.model`；默认 DataLoader 会把语言字段整理成一批字符串（通常是 list/tuple），与单个字符串比较不匹配，`outputs` 也可能未赋值。最小修复：常规训练验证统一用 `self.model(inputs)`；如确需按语言路由，单独设计并测试逐样本路由，不能将外部模型混进训练模型的验证。

4. `MSAbyvideo/scripts/train_multimodal.py:176-179` 以 `modalities=` 调用 `get_dataloaders`，但该目录当前启用的函数签名是 `get_dataloaders(video_dir, csv_path, batch_size, ...)`（`MSAbyvideo/src/data/dataset.py:318-345`），缺少必需路径；脚本还读取 `val`（`train_multimodal.py:230-240`），而函数返回 `valid`，并传了 Trainer 不支持的 `optimizer=`（`MSAbyvideo/src/training/trainer.py:40-53`）。`MSAbyvideo/scripts/evaluate.py:79-83` 也沿用不匹配的 `modalities=` 调用。修复前，这些脚本不是可用的训练/评估入口；应选定一个权威入口，避免两套 API 并存。

### P1：特征和注意力没有保留时间结构

`video2pkl/video2pkl/video2pkl.py:138-159` 虽请求 Whisper word timestamps，却只拼成一段文本，再用固定长度 BERT 输入的 CLS 向量作为单个 768 维样本特征。音频帧特征在 `:177-190` 对全片求均值，视觉帧特征在 `:307-328` 对全片求均值。视频版也把整段转录压成 BERT CLS（`MSAbyvideo/src/data/preprocess.py:181-202`），音频做全局 MFCC 均值（`:205-223`），视觉做帧特征均值（`:226-250`）。词/帧顺序和跨模态时间对齐没有进入模型输入。

随后各模态 encoder 对 `[B,D]` 输入加一个序列维度，再取第一个 token：文本 `MSAbypkl/src/models/text.py:127-149`，音频 `audio.py:112-135`，视觉 `visual.py:128-150`。cross-attention 把每个 `[B,D]` 当成长度 1（`MSAbypkl/src/models/attention.py:52-74`），softmax 只有一个 key，权重恒为 1；因此 attention 分支不可能在多个时间步/词之间选择上下文，编码器的 self-attention 也只有单 token。残差和投影仍可学习模态组合，但不能据此声称做了时间对齐的 cross-attention。

最小修复取决于目标：若要研究时间融合，保存词/窗口/帧级特征、各自时间戳和 padding mask，并让模型接收 `[B,T,D]` 序列及位置编码；若只保留片段级向量，则应采用并如实描述为向量级 MLP/融合基线。已池化的旧 PKL 无法恢复丢失的顺序和对齐，需从原始媒体重提特征。

### P1：五分类标签和回归目标混在一起

- `MSAbyvideo/src/data/preprocess.py:31-37,87-100` 把五个字符串标签映射成 `0..4`，并写入 `class_labels`；活动的 `MultimodalDataset.__getitem__` 把它作为 `label` 返回（`MSAbyvideo/src/data/dataset.py:23-43`）。但 `MSAbyvideo/config.py:65-66` 设置 `NUM_CLASSES=1`，Trainer 因此用 MSE（`MSAbyvideo/src/training/trainer.py:70-75,121-137`）。结果是单输出回归标签编号，而非五分类，也不是原始连续情感分数。选定一个任务后统一数据、输出头、损失和指标：五分类用 `0..4` + CrossEntropy；连续分数则保留连续目标 + 单输出回归。
- `MSAbypkl/src/data/dataset.py:38-57` 对字符串五分类的映射是明确的；但对 `regression_labels` 的五分类映射只覆盖精确的 `-1..1`、步长 `0.2`（`:217-226`），而注释中的 `map_to_label3`（`:136-163`）没有被调用。其他值会被映射为 `-1`，而 `CrossEntropyLoss` 不接受这个目标。`video2pkl/video2pkl/1pkl_to_4pkl.py:31-40` 又按回归标签生成 `num_classes=1` 元数据，但运行配置默认 `NUM_CLASSES=5`（`MSAbypkl/config.py:80-82`）。最小修复：明确并版本化标签 schema/分箱边界；样本标签不合法时在数据加载阶段报错，不要把 `-1` 送进损失函数。
- `MSAbyvideo/src/training/metrics.py:207-215` 分类分支只返回 `accuracy` 与 `f1`，但 `log_metrics` 的分类分支要求 `precision`、`recall`、`batch_accuracy`（`:236-247`）；若切换到五分类，日志会因缺字段失败。指标返回键需与记录器保持一致。

### P1：验证集和指标不能支撑可靠结果

`MSAbyvideo/src/data/dataset.py:318-345` 把 CSV 全部样本重新随机分成 80/15/5，未读取官方 train/valid/test 列，也未固定 `random_split` 的生成器；`MSAbyvideo/main.py:91-97` 和 `:218-225` 还把用户输入的 `all_test` 硬编码成 `False`。同一输入重复运行会得到不同切分，且不保证按官方视频/说话人边界切分。最小修复：以明确的 split manifest 为准，保存样本 ID 与 split；验证/测试不重新随机切分，测试集只在模型选择完成后评估。

`MSAbypkl/src/training/metrics.py:206-223` 实际报告 weighted F1；README 声称 Macro-F1（`README.md:119`）。`calc_multiclass_metrics` 文档写 7 类（`metrics.py:85-111`），却不对应默认五类输出，也没有被 `evaluate_mosei` 调用。`MSAbypkl/scripts/evaluate.py:120-121` 和 `MSAbyvideo/scripts/evaluate.py:120-121` 导入并调用模块中不存在的 `compute_metrics`。最小修复：按任务定好指标和平均方法，统一训练、验证、测试的同一实现，并移除/实现不存在的函数调用；发布时注明指标定义、split 和类别映射。

### P1：声称的原始视频推理流程不完整

`MSAbyvideo/src/data/preprocess.py:26-49,67-100` 要求 CSV 中有已知类别标签，遇到未知标签直接跳过（`:87-91`）；当前 DataLoader 再随机切分数据（`MSAbyvideo/src/data/dataset.py:318-345`）。因此它没有一个可直接接收无标签视频并返回预测的推理路径。`MSAbyvideo/main.py:184-263` 的“评估”仍要求 CSV 与 checkpoint，并评估随机划出的 test 子集；`all_test` 输入被忽略。

顶层 `test_script.py:31-54` 会把测试 CSV 标签都写成占位 `NEUT`（`:12-17`），再启动交互式 `MSAbypkl/main.py`（`:42-53`）；变量 `model_checkpoint` 定义后没有传入任何调用（`:44,51-53`），所以它不是可靠的一键推理脚本。最小修复：提供明确的单视频/批量预测 API，只需要媒体和模型，不要求真值标签；让 CLI 直接接收 checkpoint 和输入路径，并返回/保存预测；推理样本不得进入训练或指标计算。

两份 extractor 都把检测到的语言写回实例字段并沿用到后续视频：`MSAbyvideo/src/data/preprocess.py:177-182`、`video2pkl/video2pkl/video2pkl.py:133-138`。一个数据集实例处理混合语言视频时，第一段视频可能决定后续所有样本的 Whisper 语言和 BERT 模型。最小修复：每个视频使用局部 `language` 变量独立检测、转录、选 tokenizer，并将该语言与样本一起保存。

### P2：依赖、权重和发布归属缺少可复现清单

`requirements.txt:14-86` 没有 `transformers` 或 `tqdm`，但预处理直接导入它们（`MSAbyvideo/src/data/preprocess.py:5-17`）且 Trainer 导入 `tqdm`（两个训练器顶部）；单独按该文件安装会缺包。最小修复：补齐直接运行依赖并锁定兼容版本；把 PyTorch/CUDA 组合、FFmpeg 和外部模型下载说明写成可验证步骤。

权重来源：`MSAbypkl/main.py:264-265` 硬编码 `best_models/zh.pt`、`en.pt`；README 只给共享 Google Drive 链接（`README.md:189-198`），没有快照内权重、校验和、训练配置或模型卡。原始视频版还要求项目根目录本地 BERT 目录和 `resnet18/resnet18-f37072fd.pth`（`MSAbyvideo/src/data/preprocess.py:109-132`）；这些文件不在审查快照中。Whisper 使用在线模型名 `base`（视频版 `preprocess.py:112-122`；PKL 版 `video2pkl.py:84-93`），下载版本也未记录。发布修复基线前应冻结文件 SHA-256、来源 URL/发布者、模型与数据许可、预处理版本、标签/split 配置及测试指标；旧共享盘权重的来源和可再分发权限不能仅凭这份代码确认。

归属方面，快照带有 MIT `LICENSE`；README 列出原团队成员及联系信息（`README.md:220-221`）。新仓库发布应保留原许可证和必要署名，并将外部数据、BERT/Whisper/ResNet 权重分别列明来源与许可。当前审计没有证据可证明共享盘 checkpoint 的训练主体、权重散列或再分发权。

## 建议分开记录两个基线

**原始基线（仅作为历史快照）**：原样封存 `legacy-source` 与相邻完整镜像，记录来源提交/文件哈希、归属、数据和外部权重位置。现有 README 的 `0.4350` 榜单结果（`README.md:202-206`）本次无法由代码、数据、权重或日志复核，应标为“旧文档声明，未验证”，不能作为新训练结果。

**修复基线（用于 MaYangle/JevAffect 真训练和发布）**：在新仓库中先固定任务（五分类或连续回归）、数据许可和官方切分；重提带时间戳的序列特征，或明确采用片段向量基线；统一标签、模型输出、损失、指标和推理 API；修复 DataLoader/Trainer 契约；最后记录环境锁文件、数据/split 清单、checkpoint 校验和与独立测试结果。只有修复基线跑通并完成发布归档后，旧库删除才不会丢失来源、权重或署名证据。
