# JevAffect 数据与评测协议核查

只读核查；未下载数据、未调用 Kaggle API，也未修改项目文件。当前 Kaggle token 由主 agent 处理，本报告不重复调用。

## 竞赛规则：权威标签、格式与计分

竞赛页的 Description 对 MOSEI 边界的原话是 “one possible mapping”；因此下表是**本项目采用的竞赛建议映射**，不应表述为赛事强制的唯一转换方案。CH-SIMS v2 的映射则写为按 `meta.csv` 的 `label` 值分箱：

| 源标签 | 竞赛输出 |
|---|---|
| MOSEI `[-3.0, -2.0]` | `SNEG` |
| MOSEI `(-2.0, -0.5]` | `WNEG` |
| MOSEI `(-0.5, 0.5)` | `NEUT` |
| MOSEI `[0.5, 2.0)` | `WPOS` |
| MOSEI `[2.0, 3.0]` | `SPOS` |
| CH-SIMS v2 `meta.csv` label `{−1, −0.8}` | `SNEG` |
| CH-SIMS v2 label `{−0.6, −0.4, −0.2}` | `WNEG` |
| CH-SIMS v2 label `{0}` | `NEUT` |
| CH-SIMS v2 label `{0.2, 0.4, 0.6}` | `WPOS` |
| CH-SIMS v2 label `{0.8, 1}` | `SPOS` |

来源：[Kaggle 竞赛 Overview/Description 与 Evaluation](https://www.kaggle.com/competitions/msa-challenge-at-the-4th-pazhou-ai-competition)。网页同时称 Kaggle 公共测试集来自两个有标签数据集，因此其 accuracy 不是最终分数；最终由主办方手工建立的 250 条中英 MP4 私测集评估。测试脚本输出到 `Test_Results/label_prediction.csv`，列顺序为 `ID, Label`，ID 必须等于 `Test_Data` 里的文件名并保留 `.mp4` 后缀；标签示例包含五类。

**协议文字存在两处冲突，需主办方确认：**

- 映射表和预测 CSV 示例写 `NEUT`，同页 Private Test Data 的标签清单写 `NEU`。不要假定二者可互换，输出时需要主办方明确拼写。
- 同页 Final Leaderboard 表的权重为 Accuracy 40%、Cross-lingual generalization 15%、Explainability 15%、Privacy/security 10%、Technical innovation 10%、Practicality 5%、Technical coherence 5%；后面的 Evaluation Criteria 却列 Accuracy 40%、Cross-lingual 20%、Explainability 20%、Privacy/security 10%、Innovation/practicality 10%。最终榜单表是已公布实际排名的表格，但文字版细则的差异仍未解释。公开 Kaggle 分数只明确提到 accuracy；交叉语言与解释性评分没有可复现的公式。

网页没有展示 Kaggle `Data` 页的训练/测试文件列表或 Kaggle 阶段 sample submission 内容；最终模型 CSV 的上述字段是能从公开描述确认的格式。此处等主 agent 用已配置凭据读取竞赛 metadata/files 后补齐。竞赛页面称模型上限 16 GB RAM、16 GB VRAM、8 GB 文件，建议平均每段推理含解释低于 5 秒。

## CH-SIMS v2.0：官方版本、划分与链接

作者[数据主页](https://thuiar.github.io/sims.github.io/chsims)的 Download 部分说提供原视频与预提取特征，ID、split、label 在 CSV；数据分为 4,402 条有监督 v2(s) 和 10,161 条无监督 v2(u)，有监督 train/valid/test 约 9:2:3，测试说话人与 train/valid 无关。作者[论文](https://arxiv.org/abs/2209.02604) Table 1 给出 v2(s) 划分：train 2,722、valid 647、test 1,034；其和为 **4,403**，比主页/论文摘要所报 4,402 多 1，属于来源中的计数不一致，应按最终下载的 CSV 实数确认。论文实验指标是 Acc2、F1、弱情绪 Acc2、MAE、Pearson Corr、R²；这些是作者基线指标，不是竞赛五类公开评分公式。

作者发布的[官方代码/下载说明](https://github.com/thuiar/ch-sims-v2)提供如下直链：

- v2(s) supervised： [Google Drive 文件夹](https://drive.google.com/drive/folders/1wFvGS0ebKRvT3q6Xolot-sDtCNfz7HRA?usp=sharing)；[Baidu 网盘](https://pan.baidu.com/s/13Ds2_XDIGUqMHt4lXNLQSQ)，提取码 `icmi`。
- v2(u) unsupervised： [Google Drive 文件夹](https://drive.google.com/drive/folders/1llIbm3gwyJRwwk58RUQHWBNKjHI9vGGB?usp=sharing)；[Baidu 网盘](https://pan.baidu.com/s/1tezEDR3Y23hJ6Mp5fmcp-w)，作者页面给提取码 `icmi`。

Google Drive 链接可解析到共享文件夹页面，但本次浏览器文本视图未列出文件名/大小，也未下载或核对包内 `meta.csv`。作者主页及仓库 README 没找到数据集许可证或商业/再分发条款；“链接可公开访问”不能视作许可。若使用 Kaggle 竞赛提供的文件，另须遵守竞赛 [Rules](https://www.kaggle.com/competitions/msa-challenge-at-the-4th-pazhou-ai-competition/rules)；其中规定仅限非商业科研/教育用途，并禁止向未接受规则者或非参赛者再分发竞赛数据。作者直链包的许可仍需以包内条款或作者答复确认。

## CMU-MOSEI：作者下载路径、划分与权利限制

CMU [官方数据主页](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)称数据经 CMU Multimodal Data SDK 提供；SDK README 说明数据序列可通过 SDK 获取，并明确**不会分享原始 YouTube 视频**，只分享匿名化处理特征。SDK README 对已对齐 MOSI/MOSEI 数据另指向 [MultiBench](https://github.com/pliang279/MultiBench)。官方 SDK 的 [MOSEI recipe](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py)列出可直接访问的 CSD 序列 URL：

- 文字：<http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedWords.csd>
- GloVe：<http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedGloveVectors.csd>
- COVAREP 音频：<http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/CMU_MOSEI_COVAREP.csd>
- OpenFace2 视觉：<http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd>
- OpenSMILE 音频：<http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/CMU_MOSEI_openSMILE_IS09.csd>
- FACET 4.2 视觉：<http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/CMU_MOSEI_VisualFacet42.csd>
- sentiment labels：<http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_LabelsSentiment.csd>
- emotion labels：<http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_LabelsEmotions.csd>

这些是 SDK recipe 中的上游直接序列地址；本次未验证 HTTP 下载是否仍可用。MMSA 作者分享的[特征与标签 Drive 文件夹](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk?usp=sharing)提供其预处理 pickle：`MOSEI/Processed/unaligned_50.pkl` SHA-256 `ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`；`MOSEI/Processed/aligned_50.pkl` SHA-256 `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`。MMSA README 说明其中 MOSEI/MOSI 文本特征是 BERT，而非 SDK 原始 GloVe。MMSA 配置确认标准训练数 16,326；SDK 另有标准 train/valid/test fold 定义。竞赛页本身未说明是否直接沿用原始划分或如何抽取 Kaggle 公共测试项。

权利方面：CMU SDK 仓库的 MIT 许可是 SDK 代码许可；不能据此推导数据也为 MIT。CMU SDK 文档的边界是不会分享原始 YouTube 视频、仅分享匿名化处理特征；但主 agent 通过竞赛文件 metadata 核实 Kaggle 竞赛包另有 MOSEI `Raw` Drive 文件，Drive ID 为 [`10N5J_8N6ajecYvvIUXPsH8XoOF9OF1Ra`](https://drive.google.com/open?id=10N5J_8N6ajecYvvIUXPsH8XoOF9OF1Ra)，Kaggle 文件 API 也显示包含 Raw MP4。该竞赛包应按上述 Kaggle Rules 的非商业科研/教育用途和禁止再分发条款处理，CMU SDK 文档不能替代这套竞赛数据许可。CMU 原始/处理版数据的单独授权文本仍未从 CMU 数据主页确认。

## MMSA 的 v2 / MOSEI 特征是否可用

MMSA 仓库当前 [config_regression.json](https://github.com/thuiar/MMSA/blob/master/src/MMSA/config/config_regression.json)包含 `simsv2`：路径 `SIMSv2/Processed/sims_unaligned.pkl`、特征维数 `[768, 33, 709]`、序列长度 `[39, 400, 55]`、训练样本数 2,722；[data_loader.py](https://github.com/thuiar/MMSA/blob/master/src/MMSA/data_loader.py)也有 `simsv2` loader。因此代码/配置支持 CH-SIMS v2。可是 MMSA README 的公开下载说明/校验和清单只列 `SIMS/Processed/unaligned_39.pkl`（即不带 v2 的旧 SIMS）、没有列上述 `SIMSv2/Processed/sims_unaligned.pkl`；本次不能证实 v2 pickle 是否在共享 Drive 文件夹内。可靠替代获取路由是 CH-SIMS v2 作者自己提供的 v2(s) Drive/百度链接，包内包含预提取特征。

MOSEI 的 MMSA aligned/unaligned 特征则在 README 所指 Drive 文件夹中，并附上面两个 SHA-256，链接/文件路径可作为可验证获取方案；本次没有下载或检验文件。

## 尚待主 agent / 赛方补齐

1. 通过已有 Kaggle 凭据确认竞赛 Data 页完整文件名、格式、许可字段、Kaggle public submission 模板和评分配置；本报告未调 Kaggle API。
2. 明确比赛实际 package 的数据版本/抽样清单；不要假定所有 train/valid/test 遵循数据集作者 split。
3. 向组织方确认 `NEUT` vs `NEU`、两个权重方案的适用阶段，以及 class accuracy 是否为 macro/overall（页面未给公式）。
4. 直接从作者下载的 CH-SIMS v2 包及 CMU SDK 特征的单独授权文本：作者/CMU 页面没提供足够条款作结论。Kaggle 竞赛副本已有明确的非商业科研教育用途及再分发限制；不要把 Kaggle 条款外推成数据集作者给所有版本的通用许可证。
