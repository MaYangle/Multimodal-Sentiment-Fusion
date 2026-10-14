# 作者数据下载诊断

审计时间：2026-10-12 11:49 UTC。检查了作者下载脚本、已保存的失败记录和官方公开下载说明，并对 Google Drive 端点发送了受限 Range 请求。没有读取凭据、下载完整文件、启动训练或使用 GPU；探测只读了约 133 KiB 响应数据。现有 MOSEI 和 CH-SIMS v2 `.part` 文件均未修改。

## 结论

MOSEI MultiBench 作者文件仍可通过公开 Drive 直链续传。失败点在 `drive.google.com` 的中转响应超时；对 `drive.usercontent.google.com` 的分段读取已验证可从文件开头和现有 2 MiB 偏移准确返回字节区间。CH-SIMS v2 作者特征包的 371,720,192 字节部分文件也通过了同类断点探测。CMU SDK 的 CSD 直链当前仍按本地探测记录返回 TLS 错误或 HTTP 500，未发现已验证的可用替代端点。

MultiBench 仓库将 MOSI/MOSEI 处理数据链接到公开 Google Drive，并将原始数据指向 CMU MultimodalSDK；[gdown](https://github.com/wkentaro/gdown) 支持普通 HTTP URL、`--continue`、Range 续传和重试。

## MOSEI MultiBench Drive 实测

清单 `data/author/mosei/multibench_listing.json` 将 `mosei_senti_data.pkl` 关联到 Drive 文件 ID `180l4pN6XAv8-OAYQ6OrMheFUMwtqUWbz`。保存的失败回执 `data/author/mosei/multibench/mosei_senti_data.pkl.provenance.json.failure.json` 记录 `DownloadError`；唯一部分文件是 `mosei_senti_data.pklpu0ap5_m.part`，大小 2,097,152 字节。完整目标文件和成功 provenance receipt 尚不存在。

下载日志 `work/multibench_author_download.err.log` 显示 gdown 6.4.2 用 40 秒 read timeout 在 `drive.google.com` 读取超时，重试一次后结束；进度日志曾记录 524,288 字节。它不是文件 ID 无效或权限错误的证据。

两个端点的响应不同：

| 请求 | 结果 |
|---|---|
| `https://drive.google.com/uc?export=download&id=…`，Range 从 0 和 2 MiB 起始 | 两次均为 HTTP 200、`text/html`、2,468 字节 Virus Scan Warning 页面；没有 `Content-Range`，不能用于直接拼接数据。 |
| `https://drive.usercontent.google.com/download?id=…&export=download&confirm=t`，Range `bytes=0-65535` | HTTP 206，`Content-Range: bytes 0-65535/3730509179`，`Accept-Ranges: bytes`，`application/octet-stream`；返回内容以 pickle 二进制头开头。 |
| 同一 usercontent URL，Range `bytes=2097152-2162687` | HTTP 206，精确返回 `bytes 2097152-2162687/3730509179`。 |
| 同一 URL，Range 从 2 MiB 起并带 `If-Range: Thu, 01 Jul 2021 20:17:21 GMT` | HTTP 206，精确返回请求区间；响应的 `Last-Modified` 与验证器一致。 |

总长度由服务器响应确认是 3,730,509,179 字节。这个文件 ID 的响应没有 ETag，但提供 `Last-Modified`；gdown 6.4.2 会在续传时发 `Range`，有 ETag 或 Last-Modified 时再带 `If-Range`，并校验返回区间与总长。该版本的 `_prepare_partial_file()` 会按目标文件 basename 搜索唯一的 `*.part`，用追加方式复用它；因此必须保留原输出名和目标路径。

可直接把同一个输出路径传给 gdown，让它续接现有部分文件：

```python
import gdown

file_id = "180l4pN6XAv8-OAYQ6OrMheFUMwtqUWbz"
url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
output = r"data/author/mosei/multibench/mosei_senti_data.pkl"
gdown.download(
    url=url,
    output=output,
    use_cookies=False,
    resume=True,
    timeout=(15, 120),
    retries=5,
)
```

实测中 gdown 会把 `drive.usercontent.google.com` 识别为普通 HTTP URL，不会将其改写回超时的 `drive.google.com/uc` 路径。若把这条路由接入 `scripts/download_author_payload.py`，只需将当前 `id=file["id"]` 下载入口改为由清单 ID 构造的上述 URL，保留现有成功后的 provenance 写入逻辑和缓存检查。成功后应核对字节数为 3,730,509,179、计算本地 SHA-256，并运行 pickle schema/样本 ID 检查；作者清单没有发布此文件的校验和，所以本地 SHA 只用作缓存身份校验，不能写成作者 checksum。脚本已有 receipt 后会先核对文件 ID 和 SHA-256，再命中缓存且不访问网络。单独运行 gdown 的临时方案不会生成该 receipt，不能把它当作已建立的可复用缓存记录。

## CH-SIMS v2 与 CMU SDK 状态

CH-SIMS v2 作者 Drive 清单 `data/simsv2_drive_listing.json` 将 `Processed/unaligned.pkl` 关联到文件 ID `13JdO6GbPHOGZ8yLBFUrNR8c2FvHN-E_O`。本地 `data/simsv2/unaligned.pklca8a19ph.part` 仍为 371,720,192 字节。直链 Range `bytes=0-0` 返回 `206 bytes 0-0/3620390413`；对现有部分文件的下一个偏移 `bytes=371720192-371720192` 加上相同响应的 `If-Range` 后，也返回精确 206。响应提供 `Last-Modified: Tue, 19 Sep 2023 09:07:03 GMT`。这验证了该部分文件也可按同一模式续接，没有读取剩余载荷。

CMU SDK recipe 列出的 CSD URL 与之前保存在 `docs/author_channel_probe.json` 中的探测结果一致：五个 HTTPS CSD 地址均为 `SSLError`；对应 HTTP Range 探测均返回状态 500、744 字节 HTML，没有 `Content-Range` 或 HDF5 header。此记录支持“当前地址不可用”的判断，不证明服务器永久不可用。探测覆盖标签、文字、文字向量、COVAREP 和 OpenFace2 文件；详见[官方 SDK recipe](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py)。

本次诊断未重复上述 CSD 请求，也未改变已有下载日志或探测记录。

## 来源

- [MultiBench 作者仓库](https://github.com/pliang279/MultiBench)：列出处理后的 MOSI/MOSEI Drive 数据及 CMU MultimodalSDK 来源。
- [gdown 文档与源码](https://github.com/wkentaro/gdown)：记录任意公开 HTTP URL、断点续传与重试行为。
- [CH-SIMS v2 作者仓库](https://github.com/thuiar/ch-sims-v2)：作者数据下载入口。
- 本地清单与记录：`data/author/mosei/multibench_listing.json`、`data/simsv2_drive_listing.json`、`docs/author_channel_probe.json`、`work/multibench_author_download.log`、`work/multibench_author_download.err.log`。
