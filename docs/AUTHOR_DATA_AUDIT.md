# 作者数据与本地资产审计

审计日期：2026-10-12。只读检查本地清单、说明、下载脚本和可读资产；未下载数据、启动训练、访问凭据或占用 GPU。本报告只记录本次核对范围内的证据。

## 结论

CH-SIMS v2 的作者连续标注和官方划分已在本地，可以马上以连续回归目标开始 train/valid 研究循环。完整的本地多模态特征缓存也已存在，但它是项目从媒体提取的 MacBERT/WavLM/DINOv2 特征，不是作者提供的原生特征文件。回归训练必须按样本 ID 从作者 CSV 重新 join 浮点 `label`，不能使用缓存中的五分类 `labels.npy`。

CMU-MOSEI 的作者划分、原始连续分数和完整 train/valid 文本嵌入已在本地，可开始文本回归基线。作者完整多模态特征尚未落盘；本地 MOSEI 音视频和现有多模态 pilot 属于另一套竞赛 ID cohort。要做作者 cohort 的多模态复现，仍需从 CMU SDK 获取官方 CSD 序列，或从 MMSA 作者分享目录取得并校验已记录哈希的 pickle 特征。

## 作者来源与标注

| 数据集 | 作者发布内容 | 本地连续标注与划分 |
|---|---|---|
| CH-SIMS v2.0 | [作者主页](https://thuiar.github.io/sims.github.io/chsims)说明监督集提供原视频和预提取特征，CSV 给出视频 ID、划分与标签；连续回归范围为 −1 至 1。作者[代码仓库](https://github.com/thuiar/ch-sims-v2)直接列出 v2(s) 的 Google Drive/Baiduyun 下载入口。 | `data/author/simsv2/meta.csv` 与 `data/simsv2/meta.csv` 字节一致，SHA-256 `be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150`，440,396 字节、4,403 行；`label` 全部可解析为数值，范围 −1 至 1，共 12 个观测分值。划分为 train/valid/test = 2,722/647/1,034。`data/author/simsv2/provenance.json` 记录作者文件 ID 和来源。 |
| CMU-MOSEI | [CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)的 MOSEI 说明将情感分数定义为 −3 至 3；官方 [SDK recipe](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py)列出文字、COVAREP、OpenFace2、FACET 和标签 CSD 序列。[SDK README](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/README.md)说明 sentiment 位于 All Labels 的第一个分量。SDK 采用带区间和完整性元数据的 CSD；其 README 明确不分享原始 YouTube 视频，只分享匿名化处理特征。 | `data/mosei/mmsa_label.csv` 来自 MMSA 链接的 CMU-MOSEI 分享目录，SHA-256 `52d885bd1e8b29a4c06a69e2b1dd67dd02f7bcbdfa38ad918144fd5673a83521`，3,512,157 字节、22,856 行；原始 `label` 全部为数值，范围 −3 至 3，共 28 个观测分值。作者 `mode` 划分为 16,326/1,871/4,659；`data/mosei/author_manifests/` 保存独立作者 cohort 清单和冻结 ID。 |

CH-SIMS v2 的作者主页写监督集为 4,402 条，但本地作者 CSV 的划分相加为 4,403 条；作者论文表中的 split 数也合计 4,403。保留当前 CSV 的字节、哈希和真实行数，并在对外报告样本量时注明此一行来源差异。不要为了凑 4,402 删除唯一连续值 −0.1 的 test 标签：旧 `simsv2_audit_summary.json` 将其标为 unresolved，是因为旧的五分类映射不接受该数值；它仍在合法回归区间内。

MOSEI 作者 CSV 与竞赛 CSV 是不同 cohort。`data/mosei/author_manifests/mosei_author_cohort_summary.json` 记录 17,460 个字面 ID 重合中有 13,258 个原始分数不同。文件里的组内顺序 crosswalk 仍明确是未验证的媒体身份假设，不能据此把竞赛 MP4/NPZ join 到作者标签。MOSEI 的连续分数范围也与 CH-SIMS 不同，训练和评价中分别保留原量纲。

## 本地特征、媒体与完整性

**CH-SIMS v2：** `data/feature_cache/simsv2_full_macbert_raw_av/` 记录 2,722 train、647 valid 特征，文本 768 维、音频 768 维、视觉 384 维，最大序列长分别为 64/400/55；variant report 记录 `strict_loader_passed=true` 且没有 test 特征。元数据将它标为完整 train/valid raw-video cache，视觉/音频由项目编码器生成，文本是 MacBERT 替换特征。因此它可作为现在的本地多模态回归起点，前提是按 `video_id/clip_id` 对齐 CSV 的原始浮点标签，并另行记录特征截断策略；它不能被称为作者原生特征复现。该缓存仍带旧五分类 `class_order`，只复用特征、ID 和 mask，回归目标从作者 CSV 建立。

完整作者原始媒体尚未核实：`data/simsv2/Raw/` 只有 2 个 MP4；`data/raw_tail_media/CH-SIMSv2/ch-simsv2s/Raw/` 下 169 个 MP4 的下载清单标明来自 Kaggle competition API，而非作者 Drive。`data/simsv2/unaligned.pklca8a19ph.part` 是 371,720,192 字节的未完成 `.part`，没有完整文件校验记录。`data/simsv2_drive_listing.json` 虽列出 `Processed/unaligned.pkl`，但只是文件目录清单，不证明本地部分文件完整。可读资产中未发现一份已校验的作者原生 CH-SIMS 特征包。

**CMU-MOSEI：** `data/feature_cache/mosei_author_mbert/` 的 meta/progress 记录 `complete=true`，覆盖作者 train/valid 16,326/1,871 条，但只有 768 维文本 CLS 特征；它明确不兼容多模态训练。回归文本基线可复用这些 embedding，目标仍从作者 CSV 的原始 `label` 读取。可读的 `data/mosei/` 树中没有 `.csd`、HDF5 或 pickle 文件。`data/mosei_processed_listing.json` 只有 MMSA Drive 文件 ID 和文件名，没有对应本地数据。`docs/protocol_research.md` 已给 MMSA `unaligned_50.pkl` 与 `aligned_50.pkl` 记录期望 SHA-256，分别为 `ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f` 和 `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`；本次没有文件可供实 hash。

`data/mosei/full_media/` 当前有 128 个 MP4、130 个 WAV，下载索引的路径是竞赛 cohort。官方 CMU SDK 本来也不分享原始 YouTube 视频，所以这些文件不能代表 CMU 作者媒体齐备。当前可见 MOSEI 多模态 pilot 以及 `data/mosei/manifests/` 的 16,327/1,871 竞赛划分均不能替代 `mmsa_label.csv` 的 16,326/1,871 作者划分。

## 最短可复现路径

1. **先跑 CH-SIMS v2 连续回归。** 使用作者 CSV 原始 `video_id`、`clip_id`、`mode`、`label` 生成独立回归清单；把字符串 ID 精确 join 到 `simsv2_full_macbert_raw_av` 的 train/valid ID，核对 2,722/647 唯一 ID、标签和 split 一致后启动 train/valid。loss 和指标直接以 −1 至 1 的原始分数计算。保留 1,034 个 test ID 的冻结哈希，调参与模型选择期间不读 test 标签。旧分类 `simsv2_manifest.jsonl`、`labels.npy` 和 `invalid_label_rows=1` 不作为回归标签源。
2. **把本地结果与作者特征结果分开。** 若目标是作者原生特征基线，从 CH-SIMS v2(s) 作者 Drive/Baiduyun 完成下载 `Processed/unaligned.pkl` 和需要的原视频；记录实际文件 ID、字节数与 SHA-256，再检查其样本 ID、split 和原始连续分数能否与作者 CSV 精确对应。当前 `.part` 不能进入训练。
3. **MOSEI 先做文本连续回归，再补官方多模态序列。** 从 `mosei_author_mbert` 的 ID 取 16,326/1,871 train/valid 特征，并按作者 ID 从 `mmsa_label.csv` join 数值 `label`。多模态复现时优先按官方 recipe 下载标签 CSD 与选定模态 CSD，用 CSD 元数据完整性校验并核对 segment IDs；将其与 [SDK 标准 folds](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py)逐 ID 对照本地作者 manifest。也可使用 MMSA 作者 Drive 的 pickle，但需先验证上述预期哈希及 ID 映射。不得使用未验证的竞赛顺序 crosswalk。
4. **统一数据协议，分开标签量纲。** 两个数据集分别保存 source manifest、标签/划分哈希和回归指标；至少记录 MAE、RMSE、Pearson 相关与 CCC，按固定作者 validation 选模型。任何从连续分数导出的分类结果都作为次要分析单独定义映射，不覆盖或改写原始标注。

## 脚本与文档状态

`scripts/download_author_metadata.py` 只负责作者 CSV；`scripts/prepare_simsv2_manifest.py` 和 `scripts/prepare_mosei_author_manifest.py` 当前会构造旧五分类字段，虽然 MOSEI 记录也保留 `raw_label`/`label_numeric`。`scripts/download_mosei_full_media.py` 读取竞赛媒体 inventory，不是 CMU SDK 下载器。当前没有完成且校验过的作者原生特征下载记录。

当前范围以 `docs/ACTIVE_SCOPE.md` 为准。`docs/PROGRESS.md`、`docs/EVALUATION_PROTOCOL.md` 和部分历史研究文档仍把五分类或竞赛 cohort 写成主任务；后续更新应把这些数字标为历史对照，并以作者连续回归结果为主。`docs/protocol_research.md` 里的 CMU CSD URL 是官方 recipe 中列出的地址，但直接 HTTP 下载可用性仍未在本审计中测试。
