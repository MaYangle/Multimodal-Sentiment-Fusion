# MOSEI fold 与可下载特征来源核查

核查日期：2026-10-12。范围限于本地作者清单、固定 SDK source、SDK/MultiBench 官方仓库说明和轻量网页查阅；没有下载数据包或启动训练。本次只新增此报告。

## Fold 差异：单个源视频不在 SDK 任一 fold

本地 `docs/author_channel_probe.json` 固定了 SDK commit `4f2eadcd7e7b9e20e83b868cdad385b41830285c` 的 `cmu_mosei_std_folds.py`，记录源文件 SHA-256 `4e13a2f538cf02ab56144eb2a91ca48a4024ce64faaae790ab9ff1c3f399b473`。我通过 GitHub 文件接口再次读取该 commit，并解析三个标准 fold 的 ID 列表：train 2,249、valid 300、test 678；三组之间没有重叠。

将这三组源视频 ID 与本地作者 `data/mosei/mmsa_label.csv` 对照，结果如下：

| 检查项 | SDK folds | MMSA 作者 CSV 的源视频组 |
|---|---:|---:|
| train | 2,249 | 2,249 |
| valid | 300 | 300 |
| test | 678 | 676 |
| 总数 | 3,227 | 3,225 |

`-NFrJFQijFE` 不在 SDK 的 train、valid 或 test 任一列表，也没有去掉首字符后的别名。作者 CSV 中该视频恰有 4 条记录，ID 后缀分别为 `_1`、`_8`、`_2`、`_0`，四条 `mode` 全为 `train`；它们正好对应 probe 记录的 4 个 membership errors。作者 cohort 其余源视频均在 SDK 标注的同名 split 中，没有发现被 SDK 放进另一个 fold 的情况。

差异不只是一条 train 视频的缺席：SDK folds 另有三个本地作者 CSV 没有的源视频，分别是 train `HuIKyKkEL0Q`，test `7l3BNtSE0xc` 和 `dZFV0lyedX4`。所以该证据表明两份来源的源视频集合不完全相同；它不足以判断哪份清洗时删除或新增了哪些内容，也不能仅凭它断言 `-NFrJFQijFE` 是作者笔误。ID 是 11 字符、仅含常见 YouTube ID 字符的合法形式；固定 SDK fold 列表中也有其他以连字符开头的 ID，因此首位连字符本身不能解释差异。由于 CMU CSD 端点当前不可读，这次不能用标签 CSD 或媒体核实该 ID 是否存在于底层序列。

**处理建议：** 保留作者 CSV 的 4 条记录及其 train 划分，不静默删样本或改分割。当前作者 cohort 以 `mmsa_label.csv` 的 `mode` 作为切分来源，并在结果中记录这 4 条与 SDK folds 不一致。若要复现 SDK `standard_folds`，建立独立的 SDK-fold cohort：从 SDK labels/CSD 的实际 ID 集合生成 manifest，核对全部 source IDs、标签和序列覆盖，再单独报告与作者 CSV 的差异。不要把两个清单合并成一个“官方 split”。

本地 `mmsa_label.csv` 已有完整可用副本：SHA-256 `52d885bd1e8b29a4c06a69e2b1dd67dd02f7bcbdfa38ad918144fd5673a83521`，22,856 行。新一轮 metadata `gdown` 超时不改变这份文件的完整性证据；无需为了这次 fold 差异重复下载。

## MultiBench 与 CMU 官方处理输出

CMU SDK README 对已对齐、可直接实验的数据明确链接到 [pliang279/MultiBench](https://github.com/pliang279/MultiBench)；MultiBench README 又列出作者维护的 [MOSEI Google Drive 包](https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv)。这是存在的可下载处理版备选，证据等级高于第三方镜像。不过其 Drive 文件列表在本次网页环境中无法读取，仓库 README 和 [MultiBench 数据加载器](https://github.com/pliang279/MultiBench/blob/main/datasets/affect/get_data.py)都没有提供包级 SHA-256/MD5 或固定 source ID manifest；所以目前只能确认发布入口，不能确认包内文件名、具体哈希或它与当前 `mmsa_label.csv` 的逐 ID 覆盖一致。

MultiBench 加载器从 pickle 读 `train`、`valid`、`test` 三个独立分区，取 `labels` 字段；当 `task != "classification"` 时标签作为浮点数返回，说明格式可承载回归分数。它在加载时还会从每个 split 删除全零文本样本，因此最终样本数未必原样等于输入包。公开代码没有证明 Drive 包保留的是 CMU 原始连续 sentiment 分量、固定 SDK fold IDs 还是 `mmsa_label.csv` 的 utterance IDs。故该包可作为下载核验的首选备份，不能在尚未检查包内哈希、ID、原始标签和 split 前称为已验证的原生标签/官方 fold 备份。

更可审计的规范来源仍是 CMU SDK 的 CSD recipe：固定 commit 的 [MOSEI recipe](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/4f2eadcd7e7b9e20e83b868cdad385b41830285c/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py)把 `All Labels`、文字、GloVe word vectors、COVAREP 和 OpenFace2 等序列映射到可下载地址。SDK CSD 带特征/时间区间及完整性元数据；[MOSEI README](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/4f2eadcd7e7b9e20e83b868cdad385b41830285c/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/README.md)说明 `All Labels` 第一个分量是范围 `[-3,3]` 的原生 sentiment。

SDK 自己的 [process_mosei.py](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/4f2eadcd7e7b9e20e83b868cdad385b41830285c/examples/mmdatasdk_examples/full_examples/process_mosei.py)给出精确直链 `http://immortal.multicomp.cs.cmu.edu/ACL20Challenge/`，称其可下载该处理流程的全部输出。该脚本从 SDK recipe 下载 `raw`、`highlevel`、`labels` 三组序列；将 GloVe、COVAREP、OpenFace2 等 highlevel 序列先按 `glove_vectors` 对齐并填补缺失，再附加 `All Labels`，按标签区间对齐并执行 `hard_unify`。最后它以 `seq_len=50`、`All Labels` 为非序列输入，显式传入 SDK 的 train/valid/test `standard_folds`，生成三 split tensors。这个脚本为 `ACL20Challenge` 直链提供了官方处理内容和标签/划分依据；但该目录在本次网页查阅中不可读，文件清单、包哈希和当前 HTTP 状态仍未知。它是比未验证的第三方副本更好的候选，但需先取得目录索引并核验打包内容。

`docs/author_channel_probe.json` 当前包含从固定 SDK recipe 自动解析的五个正确 CSD URL；五条 HTTP 请求均返回 500 HTML，五条 HTTPS 请求均报 SSL 错误。正确 GloVe endpoint 是 `CMU_MOSEI_TimestampedWordVectors.csd`，该路径现已在 probe 中验证为同样的 HTTP 500/HTTPS SSL。probe 的 `rejected_previous_paths` 还记录并排除了旧脚本拼出的 `CMU_MOSEI_TimestampedGloveVectors.csd`，所以其旧错误不再与正确地址混淆。

`docs/protocol_research.md` 还记录了 MMSA Drive 的 `unaligned_50.pkl`、`aligned_50.pkl` 预期哈希，但两个文件当前本地缺失；它们是 MMSA 作者预处理特征版本，不等同于 CMU SDK CSD，也不代表 SDK fold 复验已完成。本地 `data/mosei_processed_listing.json` 只有 Drive IDs。推荐顺序是：先检查 MultiBench Drive 的轻量目录/文件元数据并记录实际哈希；如果它不能证明原生标签与标准 folds，就核实 CMU 的 `ACL20Challenge` 处理包；待 CSD endpoint 恢复时以 CSD recipe 做最终规范复验。MMSA pickle 仅作为单独特征版本对照。

## 复核记录

- 固定 SDK fold source：`docs/author_channel_probe.json` 中的 commit、URL、SHA-256 和 22,856 行检查结果；GitHub 固定 commit 的三组 fold 列表及逐源视频集合差异。
- 作者 split 与内容：`data/mosei/mmsa_label.csv`、`data/mosei/mmsa_label_provenance.json`、`data/mosei/author_manifests/mosei_author_cohort_summary.json`。
- 下载源说明：CMU SDK [README](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)、[MultiBench README](https://github.com/pliang279/MultiBench/blob/main/README.md)、MultiBench [get_data.py](https://github.com/pliang279/MultiBench/blob/main/datasets/affect/get_data.py)。
- 没有把 SDK membership errors 当作移除标注或重排 split 的依据；没有下载 MultiBench/CMU 大文件，也没有修改实现或数据。
