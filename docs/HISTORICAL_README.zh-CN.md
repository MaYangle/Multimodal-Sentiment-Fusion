# Historical README snapshot

Status statements below are historical; see [current README](../README.md).

# Multimodal Sentiment Fusion（中文说明）

[English](../README.md) · [中文说明](../README.zh-CN.md)

GitHub 首页：[MaYangle/Multimodal-Sentiment-Fusion](https://github.com/MaYangle/Multimodal-Sentiment-Fusion)。
实验运行期间保留本地 JevAffect 工作目录和包名。GitHub 当前仅为明确标注研究状态的初始首页，完整源码及结果版本仍待发布。按用户最新明确指令，旧仓库已在备份验证后删除。[迁移与恢复证据](../docs/GITHUB_MIGRATION_20261014.md)。

JevAffect 是**实验阶段的多模态情感研究项目**，按 CH-SIMS v2.0 的 [-1,1] 和 CMU-MOSEI 的 [-3,3] 原生连续标签分别训练、评估。两个数据集均已完成文本、简单融合和 ALMT 适配的三个随机种子及审计；原创机制验证、该机制的跨数据集复验及独立测试尚未完成。Jev 提供结构化决策与概率校准思想，本项目不使用 Jev 权重，也不声称复现其训练配方。

此前统一五分类及失败的片段效用监督结果作为**历史竞赛实验**保存，不能替代原生任务结果。本地尚无 Git 远端或发布权重。参阅[当前范围](../docs/ACTIVE_SCOPE.md)、[原生协议](../docs/NATIVE_EVALUATION_PROTOCOL.md)、[作者数据审计](../docs/AUTHOR_DATA_AUDIT.md)和[贡献指南](../CONTRIBUTING.md)。

## 当前原生回归结果

同预算的时间统计对照已完成12次正式训练。配对协方差MAE0.40924±0.00254未通过与序列ALMT0.40868及静态乘积／方差／错位对照的种子门槛，停止晋升。[成绩与复现命令](../docs/NATIVE_TIMED_MOMENT_FINDINGS_V1.md)保留负结果；[时间来源审计](../docs/NATIVE_AV_TIME_AUDIT_20261014.md)仍未证明全语料物理同步。[仅TRAIN标注诊断](../docs/NATIVE_ANNOTATION_DIAGNOSTIC_20261014.md)发现30.8%的单模态情感符号冲突，作为下一项研究判断的描述性证据。

新取得的两条作者 TRAIN 原视频还已跑通完整 CPU 编码和原生融合推理，与旧缓存预测差异不超过1.669e-6。[作者媒体演示与命令](../docs/NATIVE_AUTHOR_MEDIA_DEMO_20261014.md)验证这两条样本身份，不代表全语料复现。

原生标量 CRPS 对照已完成36次留源概率头训练，候选的误差命中和区间评分未通过预登记条件，概率形状路线已结束。[独立审计负结果](../docs/NATIVE_CDF_CRPS_FINDINGS_V1.md)及[36次精确源码复跑](../docs/NATIVE_CDF_CRPS_REPRODUCTION_20261014.md)保留全部证据。补齐的同统计序列 ALMT 基线 MAE 为0.4087±0.0067，优于原始输入的0.4271；与强池化融合0.4115的来源区间仍含零，二分类准确率也较低。[完整成绩、输入一致性和CPU重载](../docs/NATIVE_SEQUENCE_CONDITIONING_FINDINGS_V1.md)。

最新概率研究完成九个全过程按来源隔离的点模型及90次概率形状拟合，校准来源不参与点模型拟合、选模或标准化。六阶候选在两个种子改善分箱NLL，第三个种子变差；误差命中概率也未通过预登记条件，因此不晋升。[完整负结果、协议边界和命令](../docs/NATIVE_NESTED_SHAPE_FINDINGS_V1.md)保留全部对照，并区分中文作者裁剪指标与原始点输出的误差事件。本轮尚未验证原创方法。

最新中文已有方法基线将同一编码器池化向量按TRAIN统计标准化：融合MAE0.4115在三个种子均优于原始0.4221，文本却变差。官方VALID按源视频配对区间包含零，不代表独立测试显著提升或原创贡献。[完整比较和命令](../docs/NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md)包括真实MP4推理与无需训练数组的描述符部署。独立源码副本精确复现完整缓存准备和六次正式训练，见[复现范围](../docs/NATIVE_STANDARDIZED_REPRODUCTION_20261013.md)。54个已有GroupDRO／DANN来源对照未通过收益门槛，见[来源对照与九次标准化训练](../docs/NATIVE_SOURCE_ROBUST_FINDINGS_V1.md)。

已取得的作者推荐 MOSEI aligned50 特征队列现已完成原生三种子基线审计，TRAIN/VALID为16,265/1,869条时间戳ID，保留四条当前SDK划分例外。其GloVe／音频／视觉输入与中文编码缓存、MMSA序号CSV分别记录：

| MOSEI模型／输入 | 验证 MAE ↓ | Corr ↑ | Non0 Acc2 ↑ |
|---|---:|---:|---:|
| 原始文本有效均值 | 0.6697 ± 0.0001 | 0.5325 | 78.64% |
| 原始简单融合 | 0.6709 ± 0.0013 | 0.5209 | 77.16% |
| 原始ALMT标量适配 | **0.6115 ± 0.0051** | 0.6044 | 76.79% |
| TRAIN标准化文本 | 0.6516 ± 0.0005 | 0.5646 | 77.18% |
| TRAIN标准化简单融合 | 0.6450 ± 0.0017 | 0.5735 | 77.74% |
| 原始ALMT有效步修正 | 0.6127 ± 0.0040 | 0.6056 | 77.30% |

ALMT回归更好，但二分类准确率低于文本。45项冻结序列测试显示有限的顺序敏感性和较强的padding位置敏感性，独立打乱AV平均不变差；这些是特征诊断，不是原视频鲁棒性或因果证据。参阅[完整成绩、预算和局限](../docs/NATIVE_MOSEI_BASELINE_FINDINGS_V1.md)。
新增mask／位置对照将padding平移差异消除到4.8e-7以内，但MAE没有稳定收益。见[三种子修正及15项诊断](../docs/NATIVE_MOSEI_MASK_BASELINE_FINDINGS_V1.md)。

最新中文标注参考残差假设也完成12次训练。直接联合预测加辅助监督的MAE0.4205优于同监督T+D解码的0.4218，T+D跨种子未稳定胜出，**未通过结构晋升门槛**。[负结果](../docs/NATIVE_REFERENCE_RESIDUAL_FINDINGS_V1.md)区分了辅助监督与解码收益，不把它称为已验证新方法。

六次原生文本适配对照也已完成：MacBERT顶部四层微调MAE0.4272±0.0022优于同结构冻结标量头的0.4574±0.0067，仍落后于简单融合0.4221。它改善强情感，却加重弱情感／中性／标注冲突样本误差。这是已有骨干适配对照，见[完整结果与命令](../docs/NATIVE_TEXT_TUNING_FINDINGS_V1.md)。

随后固定点预测的概率研究完成48个留源教师和21次校准拟合。分歧候选分箱对数损失1.326491在三个种子均差于特征混合对照1.326072，按源视频配对区间包含零；该机制不晋升，点MAE保持不变。[完整负结果、对照和命令](../docs/NATIVE_ANCHOR_CDF_FINDINGS_V1.md)包含检查点重载及所有概率／区间分数。另九个留源文本模型将参数量匹配融合到.16%以内，平均MAE0.3965仍低于融合0.4132。这是来源诊断线索，尚非因果干扰或原创结论，见[预注册容量对照](../docs/NATIVE_SOURCE_CAPACITY_CONTROL_PLAN_V1.md)。

来源误差分解显示，融合相对宽文本的额外MSE只有26.6%来自来源均值偏移，主要差异在同源片段内部；[来源诊断与完整命令](../docs/NATIVE_SOURCE_CAPACITY_FINDINGS_V1.md)限定了这个解释。独立510文件源码副本在隔离依赖环境精确复跑全部21次校准，复用已核验OOF表；[复现范围](../docs/ANCHOR_CALIBRATION_REPRODUCTION_20261013.md)不包括重新提取编码器或重训教师。

拥有完整可信作者包及核验中文缓存、安装 `train,uncertainty` 依赖后，可复现如下；数据使用条件仍适用，准备阶段不导出TEST标签或特征：

```powershell
python scripts/prepare_mosei_author_features.py --source data/author/mosei/multibench/mosei_senti_data.pkl --receipt data/author/mosei/multibench/mosei_senti_data.pkl.provenance.json --output-dir data/feature_cache/mosei_multibench_native_v1 --trust-author-pickle
python scripts/train_author_baselines.py --config configs/native_mosei_author_baselines_v1.json --output docs/native_mosei_author_baselines_v1.json
python scripts/audit_author_baselines.py --suite docs/native_mosei_author_baselines_v1.json --output docs/native_mosei_author_baseline_audit_v1.json
python scripts/prepare_standardized_author_features.py --cache-dir data/feature_cache/mosei_multibench_native_v1 --output-dir data/feature_cache/mosei_multibench_standardized_v1
python scripts/train_author_baselines.py --config configs/native_mosei_standardized_baselines_v1.json --models text early_fusion --output docs/native_mosei_standardized_baselines_v1.json
python scripts/audit_author_baselines.py --suite docs/native_mosei_standardized_baselines_v1.json --models text early_fusion --output docs/native_mosei_standardized_audit_v1.json
python scripts/diagnose_author_sequences.py --suite docs/native_mosei_author_baselines_v1.json --output docs/native_mosei_sequence_diagnostic_v1.json
python scripts/train_reference_residual.py --config configs/native_reference_residual_v1.json --output docs/native_reference_residual_formal_v1.json
python scripts/audit_reference_residual.py --suite docs/native_reference_residual_formal_v1.json --output docs/native_reference_residual_audit_v1.json
```

已有训练用 `--resume`；原始准备入口解析调用者路径并保留V1数值配方和hash，标准化采用独立目录、拒绝覆盖。新增[原生中文 MP4 加文本推理](../docs/NATIVE_MEDIA_INFERENCE.md)已在两个真实 TRAIN 视频实跑，与旧缓存输入的预测差异不超过1.7e-6；完整作者媒体及全队列重提取、全新环境完整复现和原创方法验证仍待完成。

后续联合专家 MoNIG 适配完成六条件 × 三种子和198项已核验的池化特征压力测试。联合动态融合验证 MAE 为0.4346 ±0.0035，名义90%区间覆盖仅61.67%，仍落后于简单融合。参阅[联合专家结果](../docs/NATIVE_CONTEXT_NIG_FINDINGS_V1.md)。

进一步训练了九个仅用 TRAIN 标签、按源视频隔离的交叉拟合模型，为六种固定均值的不确定性对照拟合参数。残差相关协方差未稳定优于单一全局方差缩放：验证密度 NLL 为0.8002对0.8006，按源视频配对 bootstrap 的差异95%区间为[-0.0106,0.0096]。缩放的独立协方差为0.7892；缩放的正确 Student-t 概率混合覆盖达88.97%，同时区间更宽。点预测 MAE 不变。这是已有方法对照与被否决的应用假设，尚非验证成立的原创贡献；完整结果与局限见[相关性研究结论](../docs/NATIVE_DEPENDENCE_FINDINGS_V1.md)。

取得相同核验缓存并安装下方 `train,uncertainty` 依赖后，可复现：

```powershell
python scripts/train_context_nig.py --config configs/native_context_nig_v1.json --output docs/native_context_nig_formal_v1.json
python scripts/audit_context_nig.py --suite docs/native_context_nig_formal_v1.json --output docs/native_context_nig_audit_v1.json
python scripts/evaluate_context_nig_stress.py --suite docs/native_context_nig_formal_v1.json --output docs/native_context_nig_stress_v1.json
python scripts/summarize_context_stress.py --stress docs/native_context_nig_stress_v1.json --suite docs/native_context_nig_formal_v1.json --output docs/native_context_nig_stress_audit_v1.json
python scripts/train_source_crossfit.py --config configs/native_dependence_v1.json --output docs/native_dependence_oof_v1.json
python scripts/evaluate_dependence.py --config configs/native_dependence_v1.json --oof-suite docs/native_dependence_oof_v1.json --final-suite docs/native_context_nig_formal_v1.json --output docs/native_dependence_results_v1.json
python scripts/summarize_dependence.py --results docs/native_dependence_results_v1.json --output docs/native_dependence_summary_v1.json
python scripts/bootstrap_dependence.py --results docs/native_dependence_results_v1.json --output docs/native_dependence_bootstrap_v1.json
```

已有训练目录需加 `--resume`；评估使用冻结检查点和预测。[预注册协议](../docs/NATIVE_DEPENDENCE_PLAN_V1.md)明确区分训练集内部源视频选模与官方验证集。

随后注册的可靠性实验已完成四条件 × 三种子。动态 NIG 融合加单模态辅助监督的 MAE 为 0.4360 ± 0.0035，仍差于下表的简单融合；其名义90%区间实际覆盖仅82.33%。辅助监督与动态融合未表现出稳定交互效应。每个检查点都评估了七种模态子集和四种共享噪声条件。该候选**不晋升为最终方法**，参阅[负结果及下一基线决策](../docs/NATIVE_RELIABILITY_FINDINGS_V1.md)。

CH-SIMS v2.0 官方验证集选模成绩，种子13/42/2026，均值 ± 样本标准差。各行来自同一冻结MacBERT／WavLM／DINOv2缓存，标为标准化的行额外使用TRAIN统计缩放，均按VALID MAE最小值选模。这是共同编码器上的方法适配，尚非作者原生预提取特征复现。

| 方法 | MAE ↓ | Pearson Corr ↑ | Acc2 ↑ |
|---|---:|---:|---:|
| 文本 CLS + MLP | 0.4373 ± 0.0024 | 0.5820 ± 0.0003 | 75.06% ± 0.78 pp |
| 带有效长度池化的简单融合 | 0.4221 ± 0.0007 | 0.6234 ± 0.0060 | 77.13% ± 0.00 pp |
| TRAIN标准化文本MLP | 0.4425 ± 0.0014 | 0.5736 ± 0.0032 | 73.98% ± 1.33 pp |
| TRAIN标准化简单融合 | **0.4115 ± 0.0039** | 0.6358 ± 0.0058 | 76.82% ± 0.56 pp |
| ALMT，恢复标量回归头 | 0.4271 ± 0.0065 | 0.5994 ± 0.0091 | 74.19% ± 1.17 pp |
| TRAIN标准化序列ALMT | 0.4087 ± 0.0067 | 0.6157 ± 0.0118 | 73.62% ± 1.20 pp |

[逐样本独立审计](../docs/native_simsv2_baseline_audit.json)核对了九个运行的原生标签、预测、哈希及选模轮次。简单融合首轮 MAE 最低。117 个单模态标注正负冲突的验证样本上，简单融合 MAE 为 0.4718，ALMT 为 0.5651；[错误分析](../docs/native_simsv2_error_analysis.json)为可靠性研究提供动机，不证明新方法有效。文本、融合、ALMT 三种子的训练与验证循环总耗时分别为 23.1 / 16.1 / 469.8 秒，不含缓存核验、启动和文件保存。

重新下载的作者 CSV 经 SHA 核验实际包含 **4,403** 条（2,722/647/1,034），与网页的 4,402 不一致。原生回归保留旧分类映射未能处理的 -0.1 测试分数，未将测试标签用于建模或评分。除此前盲解码的作者MP4外，2026-10-14又完整取得、解码两条作者TRAIN视频，SHA与对应历史编码来源记录完全相同；这只证明两条样本身份，全语料覆盖和旧缓存等价仍未证实。[实际复查收据](../docs/AUTHOR_RAW_RECHECK_20261014.md)记录边界。中文作者特征包仍不完整。SDK推荐的 **MOSEI MultiBench特征包已完整取得并审计**，含16,265/1,869/4,643条时间戳ID样本及对齐的GloVe／音频／视觉特征；四条训练记录不在当前SDK folds，文本有前端padding，音频有-inf，独立适配及三种子基线训练已完成。CMU SDK CSD端点仍出现HTTP500或HTTPS错误。详见[作者获取状态](../docs/AUTHOR_ACQUISITION_STATUS.md)与[实际MOSEI队列](../docs/MOSEI_MULTIBENCH_COHORT.md)。

已有核验缓存 `data/feature_cache/simsv2_full_macbert_raw_av` 时，可执行：

```powershell
python -m pip install -e ".[train,data]"
python scripts/download_author_metadata.py --dataset simsv2 --output-dir data/author/simsv2
python scripts/prepare_native_manifest.py --source data/author/simsv2/meta.csv --dataset simsv2 --output-dir data/author/simsv2/manifests
python scripts/train_native.py --config configs/native_simsv2_text.json --seed 13 --smoke 32 --stop-after 1
python scripts/train_native.py --config configs/native_simsv2_text.json --seed 13 --smoke 32 --stop-after 1 --resume
python scripts/run_native_baselines.py --configs configs/native_simsv2_text.json configs/native_simsv2_early_fusion.json configs/native_simsv2_almt.json --output docs/native_simsv2_baselines_v1.json
python scripts/audit_native_results.py --suite docs/native_simsv2_baselines_v1.json --output docs/native_simsv2_baseline_audit.json
```

已有套件用 `--resume` 继续。检查点绑定作者 CSV、特征哈希、代码、配置及随机状态。[原生指标一致性记录](../docs/native_metric_conformance.json)核对固定提交的作者代码及 float32/float64 数据；可选复核依赖为 `pip install -e ".[train,verify]"`，命令为 `python scripts/verify_native_metrics.py`。这些入口已在本地实跑；全新环境从数据准备到训练的完整复现仍待完成。下文保留历史五分类流程，待原生数据准备与推理入口逐步替换。

## 安装与最小检查

使用本地核验缓存和检查点运行原生连续分数演示，参阅[无标签特征推理](../docs/NATIVE_FEATURE_INFERENCE.md)。两条真实验证输入已在CPU通过简单融合、ALMT及联合NIG模型；这是特征输入演示，原生原视频推理仍待完成。

同一核验缓存上的因子实验复现命令如下，分布评估需要 SciPy：

```powershell
python -m pip install -e ".[train,uncertainty]"
python scripts/train_reliability.py --config configs/native_reliability_v1_smoke.json --seed 13 --smoke 32 --stop-after 1
python scripts/train_reliability.py --config configs/native_reliability_v1.json --output docs/native_reliability_formal_v1.json
python scripts/audit_reliability_factors.py --suite docs/native_reliability_formal_v1.json --output docs/native_reliability_factor_audit_v1.json
python scripts/evaluate_reliability_stress.py --suite docs/native_reliability_formal_v1.json --output docs/native_reliability_stress_v1.json
python scripts/summarize_reliability_stress.py --stress docs/native_reliability_stress_v1.json --suite docs/native_reliability_formal_v1.json --output docs/native_reliability_stress_audit_v1.json
```

已有训练调用加 `--resume` 可继续。损失使用原始分数，训练器不解析验证集辅助标注；公式、梯度和精确恢复测试包含在常规测试套件中。

需要 Python 3.11 以上。实测环境为 Windows 11、Python 3.12.10、PyTorch 2.11.0+cu128、16 GB 显存的 RTX 5070 Ti。隔离环境安装、完整 MOSEI 特征重新准备、真实训练恢复演练及特征推理已通过；完整正式训练复现及其他平台仍待核验。GPU 训练前需自行安装与驱动匹配的 CUDA 版 PyTorch。见[实际全新环境命令与证据](../docs/FRESH_NATIVE_REPRODUCTION_20261013.md)。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[train,encoders,uncertainty]"
python -m unittest discover -s tests
```

当前项目在隔离依赖环境通过325项测试；历史633文件独立源码副本通过288项，复用已审计点表精确复跑36个CRPS概率头。此前副本通过235项测试，精确复现18个标准化缓存／元数据文件和六个完整中文点模型。更早副本复跑21次校准及九次MOSEI基线。原创方法与完整原视频编码器复现仍待完成。测试使用小型样例，不会下载数据，也不是模型成绩。`data` 可选依赖包含 Kaggle 和 Google Drive 下载客户端；`uncertainty` 提供回归分布评估所需的 SciPy。

## 数据与文本基线

按照[琶洲比赛](https://www.kaggle.com/competitions/msa-challenge-at-the-4th-pazhou-ai-competition)条款取得 CH-SIMS v2.0，放置作者版 `meta.csv` 于 `data/simsv2/meta.csv`。预期 SHA-256 为 `be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150`。以下流程只用官方训练 2,722 条和验证 647 条，不用测试标签调参。

```powershell
python scripts/prepare_simsv2_manifest.py --meta-csv data/simsv2/meta.csv --output-dir data/simsv2/manifests --allow-unresolved-test-labels
python scripts/download_encoders.py --model mbert --output-root data/models
python scripts/cache_simsv2_text.py --meta-csv data/simsv2/meta.csv --model-dir data/models/mbert --output data/feature_cache/simsv2_mbert_local --device cuda --batch-size 32 --max-length 64
python scripts/train_features.py --config configs/simsv2_text_cls_local.json
```

下载脚本固定 mBERT 版本并校验权重哈希。这里是**提供转录文本、冻结 mBERT 编码器**的单模态对照，不是原始视频完整模型。若仅有 CPU，可将编码命令的 `--device` 改为 `cpu`，并修改训练配置里的设备；CPU 路径尚未做计时复现。

MOSEI 作者队列由 `scripts/prepare_mosei_author_manifest.py` 和 `scripts/cache_mosei_text.py` 单独处理。作者版的训练/验证/测试数为 16,326/1,871/4,659，比赛版为 16,327/1,871/4,662；同名片段可能有不同标签，不能按文件名直接合并。见[队列核对](../docs/MOSEI_AUTHOR_COHORT.md)。

## 实测结果及适用范围

以下均为**验证集选模成绩**，不是冻结测试集成绩，也不代表领先结果。± 为种子 13、42、2026 的样本标准差；Macro-F1 固定平均五类。

| 队列与条件 | 模型 | 验证准确率 | Macro-F1 |
| --- | --- | ---: | ---: |
| CH-SIMS v2.0，完整 2,722/647，提供文本 | 冻结 mBERT CLS + MLP | 39.16% ± 0.39 百分点 | 33.82% ± 2.09 百分点 |
| CH-SIMS v2.0，完整 2,722/647，提供文本 + 原视频音视频特征 | ALMT 五分类同骨干适配 | 44.87% ± 0.50 百分点 | 41.21% ± 0.82 百分点 |
| CH-SIMS v2.0，完整 2,722/647，MacBERT 文本 + 原视频音视频特征 | ALMT，更强同骨干适配 | 47.04% ± 0.64 百分点 | 44.69% ± 1.19 百分点 |
| CH-SIMS v2.0，同一 MacBERT/音视频特征 | Jev-inspired 掩码时序模型 | 49.72% ± 0.24 百分点 | 47.13% ± 1.00 百分点 |
| CH-SIMS v2.0，仅从 TRAIN 构建、源视频隔离 2,182/540 | ALMT，MacBERT/音视频特征 | 48.95% ± 0.47 百分点 | 43.48% ± 3.18 百分点 |
| CH-SIMS v2.0，同一源视频隔离划分 | Jev-inspired 时序模型 | 49.75% ± 0.47 百分点 | 44.24% ± 2.53 百分点 |
| CH-SIMS v2.0，同一源视频隔离划分 | 128 维投影逐 token 交叉注意力 | 49.94% ± 0.47 百分点 | 43.04% ± 3.15 百分点 |
| CMU-MOSEI 作者版，完整 16,326/1,871，提供文本 | 冻结 mBERT CLS + MLP | 53.47% ± 0.43 百分点 | 38.70% ± 1.17 百分点 |

预测已按每条选模 ID、标签重新核算。[原骨干 ALMT](../docs/almt_full_validation_results.json)、[MacBERT 同骨干比较](../docs/macbert_full_model_paired_audit.json)、[源视频隔离比较](../docs/group_holdout_model_paired_audit.json)、[英文文本指标](../docs/mosei_text_validation_results.json)和[错误分析](../docs/TEXT_BASELINE_ERROR_ANALYSIS.md)记录分类证据。**中文官方验证集 647 条中有 646 条的源视频也出现在官方训练集**；额外的 2,182/540 划分只取官方 TRAIN，源视频零重叠，但仍是选模集。作者 [MMSA 结果表](https://github.com/thuiar/MMSA/blob/master/results/result-stat.md)的 SIMSv2 原生特征五分类分数更高，不能与本项目不同特征协议的适配版直接作领先比较。逐样本预测与检查点保存在不入 Git 的 `runs/`。完整数据 mBERT 文本微调试验仅有 38.49% / 27.66%，没有被选用。

另有一个**有偏的 339 条训练、43 条验证原始视频前缀试验**，实际跑通视频解码、Whisper 转写、mBERT/WavLM/DINOv2 特征、三模态 ALMT 训练及五类概率输出。种子 13 的验证成绩：ALMT 53.49% / 44.97%，简短训练的 CMC pseudo 适配版 41.86% / 29.05%，未加入效用监督的新时序模型 44.19% / 36.79%（前者为准确率，后者为 Macro-F1）。该新模型三种子平均为准确率 44.19% ± 2.33 百分点、Macro-F1 35.19% ± 2.22 百分点。它出现过拟合：种子 13 末轮训练准确率达到 100%，选中轮次验证 NLL 为 2.76。样本太少、来源不具代表性，训练日程也不同，**不可据此给方法排正式名次**。使用提供文本时，从 MP4 重新推理的概率与训练保存的验证概率最大差异小于 `8e-8`。见[原始视频试验](../docs/raw_video_pipeline_pilot_results.json)、[CMC 试验](../docs/cmc_raw_pilot_results.json)和[新模型三种子试验](../docs/jev_temporal_three_seed_pilot_results.json)。

小样本先导试验中，效用监督未改善准确率。更大的源视频隔离实验用训练子集内三折 OOF 教师生成了 30,377 个有效片段目标：JevAffect V2 加入这些目标后，种子 13 为 **49.26%**，同架构不加效用监督为 **50.37%**；门控与真实遮蔽损失变化的相关系数仍仅 **0.032**，对照为 0.021。因此[新机制当前是负结果](../docs/architecture_reassessment.md)，不能宣称原创性或可信片段解释。直接逐 token 注意力的种子 13 去注意力消融也从 49.44% 升到 49.81%。完整中文特征缓存前 3,200 条由 Kaggle T4 提取，后 169 条由本机 RTX 5070 Ti 用相同固定权重提取；硬件数值差异仍是局限。原缓存有 69 条音频超过 400 步、1 条视频超过 55 帧。BC-MLF 原生特征和冻结测试成绩仍未完成。

## 方法、相关工作与复现边界

研究模型保留冻结的 mBERT 或 MacBERT token、WavLM 语音序列和 DINOv2 画面序列；音视频按实际时间戳建模，文本仅有 token 顺序，不假定词与音视频帧已经对齐。新试验将三模态投影到 128 维，加入模态与位置编码，并按真实掩码进行逐 token 注意力。JevAffect V2 还加入受限幅度的跨模态残差，用**仅来自训练集源视频分组 OOF 的片段遮蔽收益**引导注意力。当前对照否定了效用收益假设。普通交叉注意力、有序融合、门控和温度校准都是已有方法，不能单独作为原创贡献。见[近期文献与架构重评](../docs/architecture_reassessment.md)、[Jev 关系](../docs/JEV_SCOPE.md)和[评测协议](../docs/EVALUATION_PROTOCOL.md)。

仓库已保留有来源说明的 [BC-MLF 适配](../vendor/bcmlf/README.md)及 [CMC pseudo 适配](../vendor/cmc/README.md)代码。CMC 的两阶段五分类适配版在完整中文验证集有一颗 mBERT 同骨干种子：准确率 40.80%、Macro-F1 37.13%；这不是论文原生模型的成绩。BC-MLF 原生特征文件因公共下载配额仍不可得。论文的二分类或不同分箱五分类数值不能直接与本协议比较。

原始 MP4 的 ALMT 试验推理入口为 `scripts/predict_raw.py`，需要自行取得比赛数据、四个经校验的编码器，以及本地训练产生的 `best.pt` 和精确缓存元数据；这些数据和试验权重尚未分发。输入 JSONL 需有 `id`、`video_path`，可选 `transcript` 与 `asr_language`。每条成功输入写出完整五类概率与标签，低置信度提示另存。当前 ALMT 路径要求三种模态均可解码。`scripts/export_competition_csv.py` 会按带 `.mp4` 扩展名的文件名严格输出 `ID,Label`，低置信度记录仍留在主 CSV；[两个验证视频的格式演练](../docs/competition_format_rehearsal.json)已通过。面向比赛执行的 `test_script.py`、最终训练模型、完整缺失模态处理及全新环境推理仍未完成。

英文原始媒体链路已逐文件验哈希并提取 76 条真实 MOSEI 训练/验证记录，其中 6 条没有 MP4，但有 WAV 和文本，正确输出空视觉掩码。54/22 条真实数据的训练/断点恢复演练选中 9/22 条正确，同时末轮训练达到 100%，显示小样本严重过拟合。见[媒体流程审计](../docs/mosei_pilot_feature_flow.json)和[训练流程审计](../docs/mosei_pilot_training_flow.json)；**这不是英文完整队列成绩**。历史比赛归档下载尝试不能证明作者数据齐备，也不能证明当前有传输任务在运行。比赛最终脚本及中英完整推理仍待完成。

## 权利、局限与后续工作

比赛数据和衍生特征不进入 Git；再分发需单独核查比赛条款。项目自有代码采用 [MIT 许可证](../LICENSE)；第三方代码保留各自许可及[来源说明](../THIRD_PARTY_NOTICES.md)。编码器和最终权重的发布条件还需分别核验，当前没有公开权重。

下一阶段需验证原创机制相对于竞争性基线的贡献，在两个原生任务上完成三种子复验及公平消融、独立最终测试、原生数据／编码器／推理入口，并从全新环境完成完整复现后发布。实验取舍将依据实际错误和所选假设。旧Git历史已经实际bundle恢复及fsck检查，en／zh权重归档通过SHA与CRC；额外预训练归档和最终备份／发布门槛尚未完成。旧仓库保留未删，[备份证据及删除条件](../docs/LEGACY_BACKUP_STATUS.md)已明确。见[进度记录](../docs/PROGRESS.md)。

相关公开工作：[ALMT](https://github.com/Haoyu-ha/ALMT)、[CMC](https://github.com/gw-zhong/CMC)、[BC-MLF](https://github.com/sunyulin0421/BC-MLF)与 TypeSafe 的[Jev 介绍](https://typesafe.ai/blog/introducing-system-one-models-and-jev)。JevAffect 是独立研究项目，并非 TypeSafe Jev 的发布版本。
