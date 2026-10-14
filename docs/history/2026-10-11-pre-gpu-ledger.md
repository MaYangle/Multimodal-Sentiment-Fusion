# JevAffect current progress

Updated 2026-10-11 (Asia/Shanghai). Goal active; not complete.

## Current scope
The user waived repairing/training the old method. Preserve its backup/audit, then prioritize a recent public strong baseline and a Jev-inspired model that exceeds it under the same five-class protocol. Original final experimental, reproduction, publication and conditional-deletion gates remain. See ACTIVE_SCOPE.md.

## Verified
- D:/MYL/JevAffect exists, with versioned source and experiment records. RTX 5070 Ti 16 GB, 64 GB system RAM, D: initially >1 TB free.
- Full historical Git mirror/bundle verified and independently restored. Historical en.pt/zh.pt ZIP downloaded and CRC/SHA256 verified; old repository remains untouched on GitHub.
- Official Kaggle inventory: 75,384 unique files, 32.13 GB uncompressed. Full media archive transfer is deferred with its resume marker intact so training prerequisites get bandwidth first.
- SIMSv2 CSV: 2,722 train / 647 valid / 1,034 test. Test-ID hash frozen. One test raw label -0.1 is unresolved; no test scores produced. 129 source-video groups overlap splits; speaker metadata unavailable.
- MOSEI competition CSV: 16,327 / 1,871 / 4,662. No source-group overlap across splits. Inventory lacks 237/12/76 video files; audio/text cover all rows. Test IDs frozen. This differs from the common MMSA feature-cache cohort and must be reconciled explicitly.
- Real supplemental lexical validation baseline: 36.42% +/- 0.09 percentage points accuracy across seeds 13/42/2026. Not the pretrained-text or final multimodal baseline; no test evaluation.
- Two real training MP4s decoded successfully into audio, frames and timestamps. Measured decoding only: 0.465s and 0.141s. This is not end-to-end prediction.
- 56 non-Torch protocol/manifest/cache tests pass in the project venv. Torch training/resume/model tests have not run.

## Recent public comparison
BC-MLF (Sep 2026, MIT) is the recent primary candidate; CMC pseudo is an additional strong reference with native v2 support. BC-MLF source and five-class adaptation preserve its actual GPT/AV architecture. BI_ENC initialization, exact losses (including CLM), verified GPT weights, v2 data adaptation and runtime checks are still required. See latest_baseline_research.md and bcmlf_implementation_audit.md. ALMT is older auxiliary code, not the latest comparator.

## Current resource state
Network transfers repeatedly reset or time out. Completed ranges are retained. The CUDA PyTorch wheel has not completed SHA256 verification and is not installed. NumPy, Transformers and FFmpeg dependencies are installed. Model/features downloads are incomplete. No GPU training is currently proven live.
A private Kaggle pilot package is prepared, and the account quota snapshot showed 6 free GPU hours. Authentication/preflight attempts failed before kernels_push; no cloud job or competition submission has been created. Do not assume the pilot is running.
Tool handles and exact failed/deferred jobs are in work/active_sessions.json; always revalidate them before any restart.

## Next actions
1. Finish/hash-check CUDA PyTorch and model assets; execute CUDA probe and real tensor/resume tests.
2. Verify complete feature files against frozen manifests; inspect actual schemas and unresolved label/cohort differences.
3. Run pretrained text and faithful recent strong baselines, recording all phases/budgets; then use validation errors to decide the proposed mechanism.
4. Complete model experiments, three seeds, ablations, robustness/calibration/interventions and raw-video inference; fresh-environment reproduction before publishing.
5. Delete the old repository only after the full requested new release and final backup audit succeed.
