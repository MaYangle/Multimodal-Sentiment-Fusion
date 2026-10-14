# Author acquisition status — 2026-10-12

2026-10-14 update: after more than27hours, one fresh strict prefix andwhole-object
request for each previously blockedTRAIN target succeeds. `video_0003/0002`
1,755,107bytes SHA40cddabf922ef30c67e2be2fb49fb5023a89c57b9c03674ba51fcca87c9f3e1e,
and `video_0006/0006`2,119,371bytes SHAb37619116720abfc82ffae00355db07b0a38260123db9733a5ce8d8f6ab27933.
Both decode onCPU andmatch theexact respective competition-source cacheSHA.
This establishes twoTRAIN identities, notfullcorpus equivalence. The oldblind
TESTmedia sample remains unpredicted. All historical failedreceipts are preserved.
[Fresh receipts](author_train_raw_recheck_20261014.json) and
[precise scope](AUTHOR_RAW_RECHECK_20261014.md). A resumable fullTRAIN/VALID
rawcohort acquisition is being prepared; its live handle and progress mustbe
checked before claiming an active transfer. NoTEST scores are used.

2026-10-13 update: two new official TRAIN targets from different videos,
`video_0003/0002` and `video_0006/0006`, returned200 quota HTML on one-byte
author Drive preflight. No payload/retry/account change followed; actual media
bytes0, expected size/version/author SHA unknown. Existing competition SHA
cannot establish byte identity without author payloads.
[Missing receipts](author_train_raw_acquisition_20261013.json). The original
TEST integrity sample remains unpredicted; acquisition status is not upgraded.

This current-state supplement preserves the earlier read-only
[source audit](AUTHOR_DATA_AUDIT.md) and failed download attempts as historical
evidence. Partial objects are not data available for training. Download session
records are snapshots; only current tool/process checks establish a live job.

## CH-SIMS v2.0

- Author repository: https://github.com/thuiar/ch-sims-v2; author data page:
  https://thuiar.github.io/sims.github.io/chsims.
- Author Drive CSV ID `1OINbBnCat988BNPV0RnS5iphRlHd6SW2`, re-downloaded into
  `data/author/simsv2/meta.csv`; SHA256
  `be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150`.
  Actual4,403 rows:2,722/647/1,034. Author homepage says4,402; disclosed conflict.
- Raw directory listing contains6,245 MP4 entries and exact IDs for all4,403
  supervised rows; [coverage audit](author_simsv2_raw_id_coverage.json).
  A listing is not proof of downloaded bytes.
- One full original MP4 `video_0001/0003`, Drive ID
  `1nfFDOayCU78s6N22AfpEB31yNhNJTfFo`, acquired in three finite ranges:
  2,940,297 bytes, SHA256
  `66363c7c4a8d420dfcb61f2cc923f908ca98aa209e951ac8c6b797d2f1a8b2d9`.
  Adjacent receipt/source record are in ignored `data/author/simsv2/raw`.
  ffmpeg decoded3.28 seconds,52,565 audio samples at16kHz and7 frames at2fps.
  Both streams present, no truncation. This test-source clip was used only for
  blinded media integrity, not model fitting/prediction/test scoring. It does
  not establish identity of the previously competition-derived training cache.
- Official `Processed/unaligned.pkl`, Drive ID
  `13JdO6GbPHOGZ8yLBFUrNR8c2FvHN-E_O`, expected3,620,390,413 bytes. Preserved
  prefix371,720,192 bytes and independently checksummed range blocks. Earlier
  quota HTML stopped transfer; the fresh [finite probe](author_finite_probe_20261012.json)
  returned exact64KiB206 and identical Last-Modified on2026-10-12. The new
  bounded resume added1,207,959,552 bytes, then stopped on a200 response lacking
  Content-Range/Last-Modified. Received prefix/blocks/partials total1,588,068,352
  bytes; this includes incomplete blocks. A later first-missing-offset probe
  returned fresh206; bounded resume added16,777,216 bytes then stopped on wrong200
  at1596456960. Total received prefix/blocks/partials is now1,604,845,568 bytes.
  Whole package/schema remain unverified. Partial bytes never enter training;
  no unchanged retry is running.
  On2026-10-13 the same missing offset returned exact64KiB206 with/without
  If-Range. A serial8MiB attempt nevertheless returned200 HTML with a download
  quota phrase and added0 bytes. This response is now directly classified;
  earlier unsampled200 responses remain unclassified. See the
  [bounded probe](CH_SIMS_FRESH_RANGE_DIAGNOSIS_20261013.json) and
  [serial response audit](simsv2_serial_range_diagnosis_20261013.json).
- Author Baidu alternative browser opened the extraction-code page, but DOM,
  accessibility and screenshot observation timed out. No login/CAPTCHA/payment
  interaction occurred. This is an automation observation failure; it does
  not establish that the public Baidu payload is unavailable.
- Two exact author VALID raw IDs were freshly probed on2026-10-13. Prefix
  requests returned206, but finite3.8/4.2MB payload ranges returned quota HTML;
  neither MP4 completed. [Actual records](author_valid_raw_acquisition_20261013.json).
  The native CPU media demo instead reuses two documented competition-acquired
  TRAIN files, whose SHA matches the prior cache, and makes no author-original
  media equivalence claim.

## CMU-MOSEI

- Author SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK, recipe and
  folds pinned at `4f2eadcd7e7b9e20e83b868cdad385b41830285c`. Official CSD probes
  returned HTTP500/HTTPS errors; [recipe-derived probe](author_channel_probe.json).
  The SDK does not publish the original YouTube videos. No complete author
  raw-video collection is available locally.
- MMSA CSV preserved at `data/author/mosei/meta.csv`, SHA256
  `52d885bd1e8b29a4c06a69e2b1dd67dd02f7bcbdfa38ad918144fd5673a83521`,
  actual16,326/1,871/4,659. Four training records are outside current SDK folds;
  [discrepancy](MOSEI_FOLD_DISCREPANCY.md). Neither silent dropping/reassignment
  nor filename joins with differently labeled competition media are allowed.
- SDK-recommended MultiBench official Drive folder
  `1A_hTmifi824gypelGobgl2M-5Rw9VWHv`, recorded inventory in ignored data.
  `mosei_senti_data.pkl` ID `180l4pN6XAv8-OAYQ6OrMheFUMwtqUWbz`, expected
  3,730,509,179 bytes. Original2MiB prefix and SHA-verified8MiB blocks are
  retained. Earlier wrong200/transport responses ended previous handles.
  The fresh64KiB206 response matches Last-Modified, enabling a new bounded
  resume. It completed at3,730,509,179 bytes, SHA256
  `31007381c01dd346945ed8a64c7321239de5aaf1ef05bf112e952848a823186d`.
  The complete receipt is adjacent to the ignored data payload. Actual cohort
  is16,265/1,869/4,643, temporal IDs, aligned50 steps with300/74/35 features.
  IDs/source videos have no cross-split overlaps, but four train rows are
  outside current SDK folds and audio contains -inf. See the
  [actual cohort/adapter boundary](MOSEI_MULTIBENCH_COHORT.md). This is not the
  full ordinal MMSA cohort and cannot be joined by guessed segment order.

## Reuse, integrity and research limits

Native Chinese experiments currently use complete train/valid frozen
MacBERT/WavLM/DINOv2 features derived from previously acquired competition
media, joined to author IDs and original scores and verified by hashes.
They are explicitly common-encoder adaptations, not author-feature results.
The existing MOSEI MMSA-cohort mBERT cache is text only. MultiBench is a
different author-linked multimodal feature cohort; its exact-label/mask adapter,
nine raw and six standardized native baseline runs are complete and audited.
These are published-input structural adaptations, not original-raw/CSD paper
reproductions or cross-dataset original-method replication.

Finite transfers bind exact Content-Range/total bytes/Last-Modified, keep
immutable prefix hashes and per-block checksums, and assemble only verified
blocks. Local SHA detects corruption/reuse changes; publishers supplied no
whole-file checksum for these Drive objects, so no independent publisher
checksum claim is made. Decode/schema/ID/score/split checks are separate gates.
All media/features remain outside Git and require their published usage terms.
