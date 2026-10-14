# Evaluation protocol (pre-results registration, v0)

**Historical competition protocol, superseded 2026-10-12.** The current
research uses [NATIVE_EVALUATION_PROTOCOL.md](NATIVE_EVALUATION_PROTOCOL.md).
The five-class definitions below explain earlier experiments only.

Date: 2026-10-11. No model has been trained or scored when this document was written.

## Data and selection
Use CMU-MOSEI and CH-SIMS v2(s), not SIMS v1. Preserve author splits and create hashed manifests with dataset, language, exact clip ID, original source/video ID, speaker when known, split, raw label, mapped label, and content hash when raw bytes exist. Check exact and near-duplicate clips across splits; record discrepancies before training. If official splits have source overlap, publish that evidence and keep a separate grouped sensitivity protocol rather than silently relabeling official results. Never imply unknown speaker/pretraining overlap is absent.

Training consumes train only. Hyperparameter and checkpoint selection uses validation only. Calibration uses a fixed documented portion of validation or validation out-of-fold predictions; test labels never enter calibration. Final test manifest is frozen and hashed before any final inference. Train/validation label counts may be inspected for debugging; do not print repeated test scores during development. Failed runs and pilot subsets are recorded. Pilot scores are not final results.

## Comparisons
Models: text-only; repaired old fusion on the same cached backbones; public competitive multimodal method; candidate new method. Keep identical feature inputs, splits, maximum epochs, optimizer search budget, early stopping policy and comparable trainable capacity where feasible. Record unavoidable differences and each model's parameters, updates and GPU time. Separate historical-weight inference (if recovered) from retraining repaired legacy code.

Final seeds: 13, 42, 2026, identical across compared models. Select architecture from validation before final test use. Minimum ablations: proposed new mechanism removed; temporal structure removed; calibration removed. Add intervention checks using segment occlusion and modality removal. Calibration-only changes must not be credited as accuracy gains if they preserve argmax.

Primary research metric: pooled five-class accuracy with class order SNEG,WNEG,NEUT,WPOS,SPOS. Also report equal-language average accuracy, per-language accuracy/Macro-F1, per-class precision/recall/F1/support and confusion matrices. Macro-F1 always averages all five classes with zero division set to zero. Report per-seed and mean/std; use paired bootstrap grouped by source video for uncertainty when group IDs exist. Do not compare five-class accuracy numerically against published binary/seven-class results.

Additional metrics: multiclass NLL, Brier score (sum over five classes then mean), top-label ECE with 15 fixed equal-width bins, reliability bins. Missing-modality and fixed-seed noise probes use the same examples. Record all modalities missing separately and still emit a valid prediction with explicit low-information status. Measure raw video latency including decoding/ASR/features/fusion, cache-only latency separately, warm/cold setup and CUDA synchronization, peak VRAM and process RAM. No speed numbers before measurement.

Competition integration: test_script.py, Test_Results/label_prediction.csv, exact ID including .mp4; low confidence never drops a row. NEUT follows the user requirement and competition mapping/example, with the NEU prose discrepancy documented. Kaggle phase submission template must be separately verified. Competition comprehensive human-scored criteria are not equivalent to five-class accuracy or a locally reproducible leaderboard score. Historical private set may be inaccessible; report that scope explicitly.

## Promotion
An experimental method is not called original or superior until closest-work comparison and controlled results support it. If accuracy does not improve, retain negative results, use validation error analysis for another justified experiment, and do not repeatedly tune on final test results. New development after test disclosure requires a fresh untouched holdout or explicit exploratory-only designation.
