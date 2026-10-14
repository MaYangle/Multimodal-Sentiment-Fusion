# JevAffect execution ledger

Started 2026-10-11 (Asia/Shanghai). Full user objective remains active.

## Acceptance gates (all required)
- [ ] Legacy Git history, available weights and provenance backed up; restore verified.
- [ ] Official competition mapping, data versions, split/ID rules and metrics verified.
- [ ] Legacy code audit and reproducible necessary fixes.
- [ ] Real text, legacy and competitive public multimodal baselines under one protocol.
- [ ] Independent Jev-inspired method; closest-work comparison and falsifiable hypothesis.
- [ ] Raw-video-to-prediction pipeline and complete legal five-label CSV.
- [ ] Real final training; at least three seeds; matched-backbone mechanism, time and calibration ablations.
- [ ] Accuracy, macro-F1, language/class results, calibration, robustness, latency/memory and intervention evidence.
- [ ] Frozen test protocol; validation-only selection; sample-level predictions and resumable checkpoints.
- [ ] English README, Chinese guide, configs, demo, contribution and issue/PR templates.
- [ ] Data/weight/code license audit; new-environment install and inference reproduced.
- [ ] GitHub MaYangle/JevAffect published with eligible weights.
- [ ] Only then: final backup verification and old repository deletion.

## Verified state
- RTX 5070 Ti: 16303 MiB total; 2858 MiB occupied at initial inspection.
- Intel Core Ultra 7 270K Plus, 24 logical CPUs; 68,053,336,064 bytes RAM.
- D: 1,207,932,018,688 bytes free at initial inspection; C: 33,900,593,152 bytes.
- Project: D:/MYL/JevAffect; backup: D:/MYL/JevAffect-backup.
- Python 3.12.14 isolated environment created. CUDA PyTorch installation in progress; GPU computation NOT yet verified.
- GitHub CLI authenticated as MaYangle; repo and delete_repo scopes present.
- Old main f6f9aaf6d80f596e3600fca0aa0d452cfcca0414; Memor 22a53e35864b281eb65360bc86a27f7c5e0d4a80.
- Complete mirror + bundle created; git fsck and bundle verify pass. Independent restore still pending.
- GitHub releases endpoint returns []; external Google Drive weights require retrieval.
- Existing Kaggle credentials detected, contents never logged; access not yet verified.

## Current experiment budget
Single local GPU; no paid compute. First cache and pilot stages before wider sweeps.
Do not estimate full training time from hardware alone: measure sample/epoch throughput first.
No data or training metrics yet. No claim of originality or superiority yet.

## Coordination
Main owns environment, architecture, experiments, integration and GPU scheduling.
Luna Max agents own separate reports: legacy audit, protocol research, closest-method research.
They do not edit implementation or use the GPU.
## 2026-10-11 data access update
Kaggle authentication succeeded using existing local credentials; official pages saved with SHA256 snapshot. Full file inventory pagination is in progress. Kaggle exposes CH-SIMS v2 raw MP4 files and the data page links a CMU-MOSEI raw archive. The competition rules explicitly permit non-commercial academic research and forbid redistributing competition data to parties outside the stated access conditions. Keep data local and publish retrieval instructions; code and weights require separate license decisions.
Historical checkpoint archive is available (126 MB) and actively downloading to the backup. No inference or training performance has been measured yet.
## 2026-10-11 first real-data run
- Downloaded author CH-SIMS v2(s) meta.csv; trained a supplemental hashed-character TF-IDF softmax on 2,722 official training rows and evaluated on 647 official validation rows.
- First smoke run used 256 train / 128 validation rows. Then full train/validation runs used seeds 13, 42, 2026.
- Validation accuracy: 0.3647604, 0.3632148, 0.3647604; mean 0.3642452, sample std 0.0008923. Macro-F1 mean 0.3036603, sample std 0.0093991. Train-majority validation accuracy 0.2812983.
- Per-sample predictions independently recomputed through tested shared metrics; accuracy and macro-F1 match. This is a lexical sanity baseline, not the planned pretrained-text or multimodal baseline and not a test result.
- Runtime evidence uses Python 3.12.14 / NumPy 2.3.5 in the bundled CPU runtime. Recorded optimization timings exclude input feature encoding; no end-to-end speed claim.
- Protocol + metric suite: 26 unittest tests pass in an independent project-root run.
- Legacy and cached-feature ALMT adapters integrated with source attribution; statically compile, but tensor behavior/optimization is NOT yet verified because CUDA PyTorch is still downloading.
- Full metadata audit found one test label -0.1 outside the suggested discrete competition mapping. Do not coerce it or silently omit it from reported tests. Compare the author feature cache before deciding how to report it. Train/validation labels used above passed strict mapping.
## 2026-10-11 manifest audit
Downloaded CSV contains 4,403 records, train/valid/test = 2,722/647/1,034, matching the sum of the paper split table rather than its 4,402 headline. Frozen test-ID SHA256: 70e28fb4e53db9ea8102afb518aeced349db7ae18cce2cffd5a09f112b4cfc01.
129 source-video groups overlap splits; 4 normalized-transcript groups repeat across splits, with 2 groups differing in raw scores. This does not establish speaker leakage because metadata has no speaker ID, but it is a source-overlap risk to report and examine with grouped sensitivity analysis.
One test target is explicitly unresolved and stored as JSON null. All IDs are retained. Scoring must reject unresolved targets instead of silently dropping or counting them. Two official training MP4 samples downloaded for future raw-input verification. No raw-video inference yet.
34 protocol/metrics/manifest tests pass after integration. Active downloads remain in work/active_sessions.json; poll actual sessions before resuming or replacing any job.

## 2026-10-11 full data inventory
Authenticated Kaggle pagination completed: 75,384 unique files, 32,126,490,749 uncompressed source bytes. Inventory includes MOSEI raw/audio/text, CH-SIMS raw, public test and validation material. Full official archive download has started; do not extract or train against an incomplete archive. Only dataset-author train/valid/test splits are used for research; Kaggle public/validation samples can overlap them and must not be mixed in as new independent data.
## User scope revision (2026-10-11)
The user explicitly authorized skipping the old method because its model cannot train and the scheme is simplistic. Stop repairing/training the old baseline. Existing historical backups/audits remain preserved, but no research time should be spent proving an old-model score.

Revised main comparison: identify and reproduce a recent strong publicly available method for the target data, then develop an independent Jev-inspired model that exceeds it under the same five-class protocol. ALMT may remain as an auxiliary older reference; it must not be presented as the latest public strong method. Re-check 2025-2026 literature and usable author code before choosing the primary comparator.

Unchanged: real data/training, validation-only selection, three final seeds, matched-backbone/budget ablations, calibration/robustness/evidence checks, complete raw-video predictions, English/Chinese docs, clean-environment reproduction, GitHub release and conditional deletion of the old repository after verified backup and successful new release. No superiority claim before measured evidence.
