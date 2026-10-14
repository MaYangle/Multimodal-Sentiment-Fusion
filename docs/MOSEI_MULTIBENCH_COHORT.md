# Acquired MOSEI feature cohort and adapter boundary

2026-10-12. Complete SDK-recommended MultiBench Drive object acquired and
independently re-hashed:3,730,509,179 bytes, SHA256
`31007381c01dd346945ed8a64c7321239de5aaf1ef05bf112e952848a823186d`.
This is author-linked pre-extracted aligned50-step features, not original
videos/transcripts or the project's MacBERT/WavLM/DINOv2 cache.

[Actual schema](mosei_multibench_schema.json) and
[cohort audit](mosei_multibench_cohort_audit.json) preserve every package row;
no test predictions or test label scoring occurred.

| Split | Samples | Source videos | Current SDK fold errors | Audio -inf values |
|---|---:|---:|---:|---:|
| train | 16,265 | 2,247 | 4 | 1,249 |
| valid | 1,869 | 300 | 0 | 191 |
| test | 4,643 | 675 | 0 | 425 |

Each ID is `(video_id,start_timestamp,end_timestamp)`, with original timestamp
strings retained. IDs are unique within and across splits; source-video sets
also have no cross-split overlap. The four non-member training records belong
to `-NFrJFQijFE`, the same discrepancy in the MMSA CSV. Preserve the published
split and disclose this variant; do not silently drop/reassign these examples
or claim exact conformance to the current SDK source-video membership list.

Feature shapes per row: text50×300 (GloVe), audio50×74 (acoustic features),
vision50×35. Train/valid sentiment labels have one scalar per instance and
range[-3,3]. There is no five-class remapping. Package counts differ from the
ordinal MMSA CSV16,326/1,871/4,659; their literal IDs cannot be joined or inferred
from within-video order. This acquired package supports a separately named
processed-feature native regression study, not a claim of the complete MMSA
or original-CSD cohort. Causes of all count differences remain unverified.

## Source-informed preprocessing requirements

Public loader pinned at MultiBench revision
`49f9be224f342de005b7f001281f33df4301eb22`:
[datasets/affect/get_data.py](https://github.com/pliang279/MultiBench/blob/49f9be224f342de005b7f001281f33df4301eb22/datasets/affect/get_data.py),
SHA256`032531dd949c032c6e6ac7e4368474bc0d7c2c2955e22c9b04332ab48aee960f`.
The source was read/hash-pinned, never imported/executed with its dependencies.
It removes zero-sum text records, replaces audio -inf with0, slices all aligned
modalities from the first nonzero text step, and preserves regression labels
unless classification is explicitly requested. z-normalization is optional.

Actual zero-sum text drop counts are0 for every split. Most text sequences have
leading padding (15,092 training records);93 training sequences have interior
zero steps after the first nonzero step. Keep the aligned suffix including such
interior zeros, rather than treating every all-zero word vector as padding.
Text lacks a BERT CLS token: the existing CLS pooling path would mostly pool
front padding. New author-feature preparation needs explicit GloVe valid-mean
or sequence modeling, suffix shifting/right-padding, declared masks and audio
-inf handling with original-value counters. No learned preprocessing statistic
may fit on validation/test. Do not edit the old cache or frozen implementation
hashes to make this input look compatible.

There are163/15/46 timestamp interval anomalies under the strict nonnegative
media-start check. Examples have a slightly negative start. Exact original
strings remain identifiers; the prepared sequence already exists. These do
not justify deleting feature instances, inventing ordinal clip IDs, or silently
clamping raw-video extraction intervals. The audit separates negative starts
from nonpositive durations. Their clock convention requires source-specific
investigation before an original-media crosswalk is claimed.

Update 2026-10-13: the separate version-bound adapter, independent full-cache
audit, real smoke/backprop/resume and three-seed text/fusion/ALMT baselines are
complete. See [native baseline findings](NATIVE_MOSEI_BASELINE_FINDINGS_V1.md).
Acquisition and these public adaptations do not establish a verified original
method or raw-media reproduction. Cross-dataset method replication and final
test evaluation remain pending.
