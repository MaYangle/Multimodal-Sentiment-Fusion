# Historical README snapshot

Status statements below are historical; see [current README](../README.md).

# Multimodal Sentiment Fusion

[English](../README.md) · [中文说明](../README.zh-CN.md)

GitHub home: [MaYangle/Multimodal-Sentiment-Fusion](https://github.com/MaYangle/Multimodal-Sentiment-Fusion).
The local working name/package remains JevAffect while experiments run. The
GitHub repository currently contains an explicit research-status scaffold;
the complete source/results release is pending. The legacy repository was
deleted at the user's newer explicit request after backup verification.
[Migration and restore evidence](../docs/GITHUB_MIGRATION_20261014.md).

JevAffect is an **experimental multimodal sentiment research project** using each dataset's native continuous scores and author evaluation rules. It studies speech, video and transcripts on CH-SIMS v2.0 ([-1,1]) and CMU-MOSEI ([-3,3]). Both datasets now have audited text, simple fusion and ALMT adaptations with three seeds. A verified original method, its cross-dataset replication and final-test results remain pending. Jev inspires structured and calibrated decisions; this project does not use Jev weights or reproduce its training recipe.

Earlier common five-class runs are preserved as **historical competition experiments**, including a failed intervention-utility mechanism. Their scores are not native benchmark results. This checkout has no configured Git remote or release weights. See the [active scope](../docs/ACTIVE_SCOPE.md), [native protocol](../docs/NATIVE_EVALUATION_PROTOCOL.md), [author-data audit](../docs/AUTHOR_DATA_AUDIT.md), and [contribution guide](../CONTRIBUTING.md).

## Native regression: current results

Twelve matched-budget decoder-time moment fits complete the latest falsification
study. Paired covariance MAE0.40924±0.00254 fails the registered seed gates against
sequence ALMT0.40868 and static/marginal/broken-pair controls; it is not promoted.
[Results and reproducible commands](../docs/NATIVE_TIMED_MOMENT_FINDINGS_V1.md)
retain the negative result. [Time-origin audit](../docs/NATIVE_AV_TIME_AUDIT_20261014.md)
does not establish corpus-wide physical synchrony. A TRAIN-only
[annotation diagnostic](../docs/NATIVE_ANNOTATION_DIAGNOSTIC_20261014.md) finds30.8%
unimodal sign conflicts; it is descriptive evidence for the next research decision.

Two newly downloaded author TRAIN videos also pass the complete CPU encoder and
native fusion inference path, agreeing with prior cached predictions within
1.669e-6. [Author-media demo and command](../docs/NATIVE_AUTHOR_MEDIA_DEMO_20261014.md)
verify the two byte identities without claiming full-corpus reproduction.

Native scalar-score CRPS training also completed36 source-held head fits.
The candidate's error-hit and interval scores fail the registered gate; the
shape line is stopped. [Audited negative study](../docs/NATIVE_CDF_CRPS_FINDINGS_V1.md)
and [exact36-fit source-copy repeat](../docs/NATIVE_CDF_CRPS_REPRODUCTION_20261014.md)
preserve this evidence. A fair standardized-sequence ALMT baseline now scores
MAE0.4087±0.0067 versus raw0.4271; its difference from pooled fusion0.4115
has a zero-crossing source interval and its binary accuracy is lower.
[All results, input identity and CPU replay](../docs/NATIVE_SEQUENCE_CONDITIONING_FINDINGS_V1.md).

The latest probability study uses nine source-nested point teachers and90
shape fits, excluding calibration sources from point fitting/selection/scalers.
Degree6 improves bin NLL in two seeds and worsens in the third; error-hit
probability scores also fail the registered gate. It is rejected.
[All controls, protocol limits and commands](../docs/NATIVE_NESTED_SHAPE_FINDINGS_V1.md)
preserve the negative result and distinguish native CH clipping from raw-point
error events. No original method is verified by this study.

The latest known Chinese baseline uses TRAIN-only coordinate standardization
of the same encoder-derived pooled vectors: fusion MAE0.4115 improves allthree
seed means over raw0.4221, while text worsens. Its conditional official-VALID
source interval includes zero, so no final-test significance or original-method
claim follows. [Full comparison and commands](../docs/NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md)
include a real MP4 inference path and descriptor-only deployment. Complete cache
preparation and six retrainings reproduce exactly from an independent source
copy; [reproduction scope](../docs/NATIVE_STANDARDIZED_REPRODUCTION_20261013.md).
The54 known source GroupDRO/DANN controls do not pass their improvement gates;
[source controls and nine normalization runs](../docs/NATIVE_SOURCE_ROBUST_FINDINGS_V1.md).

The acquired author-linked MOSEI aligned50 cohort now has audited native
three-seed baselines (16,265/1,869 TRAIN/VALID temporal IDs; four current-SDK
train-membership exceptions retained). Its GloVe/acoustic/visual inputs are
separate from the Chinese encoder cache and the ordinal MMSA CSV:

| MOSEI model/input | Valid MAE ↓ | Corr ↑ | Non0 Acc2 ↑ |
|---|---:|---:|---:|
| Raw text valid-mean | 0.6697 ± 0.0001 | 0.5325 | 78.64% |
| Raw early fusion | 0.6709 ± 0.0013 | 0.5209 | 77.16% |
| Raw ALMT scalar adaptation | **0.6115 ± 0.0051** | 0.6044 | 76.79% |
| TRAIN-standardized text | 0.6516 ± 0.0005 | 0.5646 | 77.18% |
| TRAIN-standardized early fusion | 0.6450 ± 0.0017 | 0.5735 | 77.74% |
| Raw ALMT with valid-step correction | 0.6127 ± 0.0040 | 0.6056 | 77.30% |

ALMT improves regression but loses binary accuracy to text. Forty-five frozen
sequence tests find modest order sensitivity and strong padding-position
sensitivity; AV permutation does not hurt on average. These are feature
diagnostics, not raw robustness or causal evidence. [Full results, audits,
budgets and limits](../docs/NATIVE_MOSEI_BASELINE_FINDINGS_V1.md) report all metrics.
The additional mask/position control eliminates padding-layout changes within
4.8e-7 but does not stably improve MAE. See the
[three-seed correction and15 diagnostics](../docs/NATIVE_MOSEI_MASK_BASELINE_FINDINGS_V1.md).

The newest Chinese annotation-reference hypothesis also completed12 runs.
Direct+auxiliary prediction MAE0.4205 beats the equally supervised T+D decoder
0.4218; T+D is not consistently better across seeds. This **fails its structure
promotion gate**. [Negative result](../docs/NATIVE_REFERENCE_RESIDUAL_FINDINGS_V1.md)
separates auxiliary supervision from decoding rather than claiming a new method.

Six native text adaptation controls also completed. Top-four MacBERT tuning
scores0.4272±0.0022 MAE versus its matched frozen scalar head0.4574±0.0067,
but remains worse than simple fusion0.4221. It helps strong sentiment while
hurting weak/neutral/conflicting cases. These are known backbone controls;
[results and complete commands](../docs/NATIVE_TEXT_TUNING_FINDINGS_V1.md).

The subsequent frozen-point probability experiment completed48 source-CV
teachers and21 calibration fits. Disagreement bin log score1.326491 is worse
than the feature-only Mixture1.326072 in all three seeds; the paired source
interval includes zero. The mechanism is rejected, with point MAE unchanged.
[Full negative result, controls and commands](../docs/NATIVE_ANCHOR_CDF_FINDINGS_V1.md)
include independently reloaded checkpoints and all probability/interval scores.
Nine additional source-OOF text models match fusion capacity within.16%:
their mean MAE0.3965 remains below fusion0.4132. This motivates source
diagnostics, not a causal nuisance or original-method claim; see the
[registered capacity control](../docs/NATIVE_SOURCE_CAPACITY_CONTROL_PLAN_V1.md).
The source-error decomposition attributes only26.6% of fusion's extra MSE
against wide text to source-mean offsets; most is within-source variation.
[Findings and audited commands](../docs/NATIVE_SOURCE_CAPACITY_FINDINGS_V1.md)
limit that interpretation. An independent510-file source copy reproduced all21
calibration fits exactly in isolated dependencies; this reuses OOF tables,
[rather than re-extracting encoders or retraining point teachers](../docs/ANCHOR_CALIBRATION_REPRODUCTION_20261013.md).

Reproduce the new native studies with the complete trusted publisher package
and verified Chinese cache. `train,uncertainty` extras are needed; package usage
conditions still apply. No TEST labels/features are exported by preparation:

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

Existing training runs require `--resume`. The raw preparation entry resolves
caller paths and preserves the numerical V1 recipe/hash; standardized output
uses a separate directory and rejects overwriting. Full native MOSEI author-feature
preparation/training/audit/inference has reproduced all nine models in an
isolated environment. Original method and full raw-encoder reproduction remain pending.

The subsequent joint-context MoNIG adaptation completed six cases × three
seeds and198 audited pooled-feature stress evaluations. Joint dynamic fusion
reached0.4346 ±0.0035 valid MAE with61.67% nominal90 coverage, still worse than
simple fusion. See [joint-context findings](../docs/NATIVE_CONTEXT_NIG_FINDINGS_V1.md).

Nine further source-video-disjoint TRAIN OOF models fit six frozen-mean
uncertainty controls without fitting on official validation errors. Correlated
covariance did not consistently improve on a single global variance scale:
valid density NLL0.8002 vs0.8006, paired source-bootstrap95% interval for the
difference[-0.0106,0.0096]. Scaled independent covariance scored0.7892; a
scaled proper Student-t mixture reached88.97% coverage with wider intervals.
Point MAE is unchanged. These are known controls and a rejected application
hypothesis, not a verified original method; [dependence findings](../docs/NATIVE_DEPENDENCE_FINDINGS_V1.md)
record the complete three-seed results and limitations.

Reproduce these studies after obtaining the same verified cache:

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

Install the `train,uncertainty` extras shown below first. Existing training
directories require `--resume`; evaluation uses frozen checkpoints/predictions.
The [registered TRAIN-only plan](../docs/NATIVE_DEPENDENCE_PLAN_V1.md) distinguishes
inner source-group selection from the official validation set.

The next registered reliability experiment completed four conditions × three
seeds. Best dynamic NIG + unimodal auxiliary supervision scored 0.4360 ±
0.0035 MAE, worse than simple fusion below; its nominal90% intervals covered
only82.33%. Auxiliary supervision did not show a stable interaction with
dynamic fusion. Seven modality subsets plus four shared noise conditions were
evaluated for every checkpoint. This candidate is **not promoted**; see the
[negative result and next baseline decision](../docs/NATIVE_RELIABILITY_FINDINGS_V1.md).

Reproduce the registered factor study with the same verified cache (SciPy is
required for distribution evaluation):

```powershell
python -m pip install -e ".[train,uncertainty]"
python scripts/train_reliability.py --config configs/native_reliability_v1_smoke.json --seed 13 --smoke 32 --stop-after 1
python scripts/train_reliability.py --config configs/native_reliability_v1.json --output docs/native_reliability_formal_v1.json
python scripts/audit_reliability_factors.py --suite docs/native_reliability_formal_v1.json --output docs/native_reliability_factor_audit_v1.json
python scripts/evaluate_reliability_stress.py --suite docs/native_reliability_formal_v1.json --output docs/native_reliability_stress_v1.json
python scripts/summarize_reliability_stress.py --stress docs/native_reliability_stress_v1.json --suite docs/native_reliability_formal_v1.json --output docs/native_reliability_stress_audit_v1.json
```

Add `--resume` to an existing training invocation. Losses use original scores,
and validation auxiliary annotations are never parsed by this trainer. Formula,
gradient and precise-resume tests are in the regular test suite.

Official CH-SIMS v2.0 validation selection scores, seeds13/42/2026, mean ± sample SD. All derive from the same frozen MacBERT/WavLM/DINOv2 cache and use minimum VALID MAE selection; the marked rows apply TRAIN-only scaling. These are common-encoder adaptations, not paper reproductions using author pre-extracted features.

| Model | MAE ↓ | Pearson Corr ↑ | Acc2 ↑ |
|---|---:|---:|---:|
| Text CLS + MLP | 0.4373 ± 0.0024 | 0.5820 ± 0.0003 | 75.06% ± 0.78 pp |
| Masked-mean early fusion | 0.4221 ± 0.0007 | 0.6234 ± 0.0060 | 77.13% ± 0.00 pp |
| TRAIN-standardized text MLP | 0.4425 ± 0.0014 | 0.5736 ± 0.0032 | 73.98% ± 1.33 pp |
| TRAIN-standardized early fusion | **0.4115 ± 0.0039** | 0.6358 ± 0.0058 | 76.82% ± 0.56 pp |
| ALMT, scalar regression head | 0.4271 ± 0.0065 | 0.5994 ± 0.0091 | 74.19% ± 1.17 pp |
| TRAIN-standardized sequence ALMT | 0.4087 ± 0.0067 | 0.6157 ± 0.0118 | 73.62% ± 1.20 pp |

[Per-ID audit and full native metrics](../docs/native_simsv2_baseline_audit.json) verify all nine runs against author labels, artifact hashes and the selection rule. Simple fusion has the lowest first-round MAE. On 117 validation examples with opposite-sign unimodal annotations, MAE is 0.4718 for fusion and 0.5651 for ALMT; [error analysis](../docs/native_simsv2_error_analysis.json) motivates reliability experiments without establishing a new contribution. Three-seed training/validation loops totaled 23.1 / 16.1 / 469.8 seconds for text / fusion / ALMT, excluding cache verification, startup and artifact I/O.

The author CSV was re-downloaded and SHA-verified: it contains **4,403** records (2,722/647/1,034), while the homepage says 4,402. Native regression retains the -0.1 test score excluded by the old discrete mapping. Final test labels have not been used for modeling or scoring. In addition to the earlier blinded original MP4, two author TRAIN videos were acquired and decoded on2026-10-14; their SHAs exactly match their historical encoder-source records. This proves those two identities only; full author media coverage/cache equivalence remains unverified. [Fresh original-media receipts](../docs/AUTHOR_RAW_RECHECK_20261014.md) preserve the scope. The Chinese author feature package remains partial. The SDK-recommended **MOSEI MultiBench package is fully acquired and audited**: 16,265/1,869/4,643 temporal-ID samples, aligned GloVe/acoustic/visual features. It has four training records outside current SDK folds, leading text padding and audio -inf; the separate adapter and three-seed baseline study are complete. CMU SDK CSD endpoints returned HTTP 500 or HTTPS errors. See [author acquisition](../docs/AUTHOR_ACQUISITION_STATUS.md) and [actual MOSEI cohort](../docs/MOSEI_MULTIBENCH_COHORT.md).

With the existing verified raw-derived cache at `data/feature_cache/simsv2_full_macbert_raw_av`, run:

```powershell
python -m pip install -e ".[train,data]"
python scripts/download_author_metadata.py --dataset simsv2 --output-dir data/author/simsv2
python scripts/prepare_native_manifest.py --source data/author/simsv2/meta.csv --dataset simsv2 --output-dir data/author/simsv2/manifests
python scripts/train_native.py --config configs/native_simsv2_text.json --seed 13 --smoke 32 --stop-after 1
python scripts/train_native.py --config configs/native_simsv2_text.json --seed 13 --smoke 32 --stop-after 1 --resume
python scripts/run_native_baselines.py --configs configs/native_simsv2_text.json configs/native_simsv2_early_fusion.json configs/native_simsv2_almt.json --output docs/native_simsv2_baselines_v1.json
python scripts/audit_native_results.py --suite docs/native_simsv2_baselines_v1.json --output docs/native_simsv2_baseline_audit.json
```

Append `--resume` to continue an existing suite. Checkpoints bind the author CSV, exact feature hashes, implementation, configuration and random states. [Native metric conformance](../docs/native_metric_conformance.json) compares pinned author evaluators on float32/float64 inputs. Optional verification requires `pip install -e ".[train,verify]"` and `python scripts/verify_native_metrics.py`. These commands have run locally; a complete preparation-to-training reproduction from a fresh environment is still pending. The sections below retain the historical five-class pipeline until native data preparation and inference replace it.

## Start with the tests

For a native continuous-score demo on your locally verified cache/checkpoint,
see [label-free feature inference](../docs/NATIVE_FEATURE_INFERENCE.md). Two real
validation inputs ran through early fusion, ALMT and context NIG on CPU.
This feature demo now has a separate [native CH-SIMS MP4 plus transcript entry](../docs/NATIVE_MEDIA_INFERENCE.md), actually run on two genuine TRAIN videos. Recomputed scores match cached-input scores within1.7e-6; full author-media acquisition and full-cohort encoder reproduction remain pending.

Use Python 3.11 or newer. A CUDA-capable PyTorch installation is needed for the recorded GPU training runs; install a build compatible with your GPU and driver before attempting them. The measured local setup used Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, and an RTX 5070 Ti (16 GB). A fresh isolated install, complete MOSEI feature re-preparation, real training/resume smokes and feature inference have passed; full formal reproduction and other platforms remain unverified. See the [actual fresh reproduction commands and evidence](../docs/FRESH_NATIVE_REPRODUCTION_20261013.md).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[train,encoders,uncertainty]"
python -m unittest discover -s tests
```

The test suite uses small fixtures; it does not download data or produce benchmark scores. The current root passes 325 tests in isolated dependencies. The historical 633-file source copy passed 288 tests and reproduced all 36 CRPS heads exactly with audited point tables reused. The earlier source-only copy passed 235 tests and reproduced all 18 normalized cache/metadata files and six complete Chinese point models exactly. Earlier snapshots reproduced 21 calibration fits and all nine MOSEI baselines. Original-method/full raw-encoder reproduction is pending. The optional `data` extra adds the Kaggle and Google Drive download clients; `uncertainty` adds SciPy for regression distribution evaluation.

## Data and a reproducible text baseline

Obtain CH-SIMS v2.0 through the [Pazhou competition](https://www.kaggle.com/competitions/msa-challenge-at-the-4th-pazhou-ai-competition) under its terms. Place the author's `meta.csv` at `data/simsv2/meta.csv`. Its expected SHA-256 is `be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150`. These commands use only the official 2,722 training and 647 validation rows; they do not train on or score test labels.

```powershell
python scripts/prepare_simsv2_manifest.py --meta-csv data/simsv2/meta.csv --output-dir data/simsv2/manifests --allow-unresolved-test-labels
python scripts/download_encoders.py --model mbert --output-root data/models
python scripts/cache_simsv2_text.py --meta-csv data/simsv2/meta.csv --model-dir data/models/mbert --output data/feature_cache/simsv2_mbert_local --device cuda --batch-size 32 --max-length 64
python scripts/train_features.py --config configs/simsv2_text_cls_local.json
```

The download script pins the mBERT revision and verifies its weight hash. This run uses provided transcripts and frozen mBERT features. It is a **text-only control**, with three seeds selected by validation accuracy. On a CPU-only machine, set the encoder's `--device cpu` and change the training config's device from `cuda` to `cpu`; that path has not been timed as a reproduction.

For the MOSEI author cohort, the separate manifest and text-cache tools are `scripts/prepare_mosei_author_manifest.py` and `scripts/cache_mosei_text.py`. The author CSV has 16,326/1,871/4,659 train/valid/test rows. The competition CSV has 16,327/1,871/4,662; same-looking segment IDs can carry different labels, so the two cohorts must not be joined by literal filename. See [the cohort audit](../docs/MOSEI_AUTHOR_COHORT.md).

## Measured results and scope

All numbers below are **validation selection scores**, not frozen-test results or claims of state of the art. Means and sample standard deviations use seeds 13, 42, and 2026. Macro-F1 always averages all five classes.

| Cohort and condition | Model | Validation accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| CH-SIMS v2.0, full 2,722/647, provided text | frozen mBERT CLS + MLP | 39.16% ± 0.39 pp | 33.82% ± 2.09 pp |
| CH-SIMS v2.0, full 2,722/647, provided text + raw-derived A/V | ALMT five-class common-backbone adaptation | 44.87% ± 0.50 pp | 41.21% ± 0.82 pp |
| CH-SIMS v2.0, full 2,722/647, MacBERT text + raw-derived A/V | ALMT, matched stronger backbone | 47.04% ± 0.64 pp | 44.69% ± 1.19 pp |
| CH-SIMS v2.0, full 2,722/647, same MacBERT/A/V | Jev-inspired mask-aware temporal model | 49.72% ± 0.24 pp | 47.13% ± 1.00 pp |
| CH-SIMS v2.0, TRAIN-only source-video-disjoint 2,182/540 | ALMT, MacBERT/A/V | 48.95% ± 0.47 pp | 43.48% ± 3.18 pp |
| CH-SIMS v2.0, same source-disjoint selection set | Jev-inspired temporal model | 49.75% ± 0.47 pp | 44.24% ± 2.53 pp |
| CH-SIMS v2.0, same source-disjoint selection set | 128-D projected token cross-attention | 49.94% ± 0.47 pp | 43.04% ± 3.15 pp |
| CMU-MOSEI author, full 16,326/1,871, provided text | frozen mBERT CLS + MLP | 53.47% ± 0.43 pp | 38.70% ± 1.17 pp |

Predictions were rechecked against exact selection IDs and labels. [Original-backbone results](../docs/almt_full_validation_results.json), [matched MacBERT comparisons](../docs/macbert_full_model_paired_audit.json), [source-disjoint comparisons](../docs/group_holdout_model_paired_audit.json), [English text results](../docs/mosei_text_validation_results.json), and [error analysis](../docs/TEXT_BASELINE_ERROR_ANALYSIS.md) include per-class evidence. **646/647 official Chinese validation clips share a source video with official training**; the separate 2,182/540 split uses only official TRAIN, has zero shared source videos, and is also a model-selection set. The [author MMSA result table](https://github.com/thuiar/MMSA/blob/master/results/result-stat.md) reports higher native-feature SIMSv2 five-class numbers under a different backbone/protocol; our adaptations are not a state-of-the-art claim. Per-example predictions and checkpoints remain local ignored `runs/` artifacts. Changing CLS to masked-mean pooling yielded small accuracy changes and lower Macro-F1; a full-data mBERT text fine-tuning trial selected only 38.49% / 27.66% and was not promoted.

An additional **biased 339-train/43-valid raw-video prefix pilot** has exercised actual video decoding, Whisper transcription, pretrained mBERT/WavLM/DINOv2 features, a trained three-modality ALMT head, and five legal output probabilities. Its seed-13 validation scores were 53.49% accuracy / 44.97% Macro-F1 for ALMT; 41.86% / 29.05% for a short CMC pseudo adaptation; and 44.19% / 36.79% for an early Jev-inspired temporal model **without its proposed utility supervision**. Across three pilot seeds, that early model averaged 44.19% ± 2.33 pp accuracy and 35.19% ± 2.22 pp Macro-F1. It overfit: seed 13 reached 100% last-epoch training accuracy while its selected validation NLL was 2.76. These pilot scores use a tiny, nonrepresentative subset and differing training schedules; they do not rank the methods on the official full cohort. A provided-transcript raw-video inference check matched the saved ALMT validation probabilities within `8e-8`. See [ALMT pipeline evidence](../docs/raw_video_pipeline_pilot_results.json), [CMC evidence](../docs/cmc_raw_pilot_results.json), and [candidate pilot runs](../docs/jev_temporal_three_seed_pilot_results.json).

In a matched seed-13 pilot, train-only group-held-out utility targets left accuracy at 44.19% and lowered Macro-F1. On the larger source-disjoint cohort, three-fold source-group OOF training produced 30,377 finite window-effect targets. JevAffect V2 with those targets scored **49.26%**, versus **50.37%** for the same architecture without that supervision (one seed); gate-to-masking-effect correlation remained only **0.032**, versus 0.021 for the control. This [negative mechanism result](../docs/architecture_reassessment.md) prevents a novelty/interpretability claim. Direct token cross-attention also failed its seed-13 removal control: 49.44% with attention versus 49.81% without it on the 540-record source-disjoint selection set. The full Chinese cache combines 3,200 Kaggle T4 rows with 169 locally extracted rows using the same pinned encoder bytes; hardware-specific numeric differences remain. In the first cache, 69 audio clips exceeded the 400-step cap and one vision clip exceeded 55 frames. Native-feature BC-MLF and frozen-test results remain unfinished.

## Method and reproducibility boundary

The research models use frozen mBERT or MacBERT token features and WavLM/DINOv2 sequences. Audio and video use recorded seconds; text has token order but no verified word-to-frame alignment. The newer 128-D token-cross-attention variant projects each modality, adds modality/position encodings, and respects missing/padded masks. JevAffect V2 adds a bounded cross-token correction biased by predicted eight-window utility, with optional **out-of-fold, train-only intervention targets**. The current controls reject the utility-benefit hypothesis. Generic attention, ordinal fusion, gating and temperature scaling are established prior work. See the [literature and architecture reassessment](../docs/architecture_reassessment.md), [Jev relationship](../docs/JEV_SCOPE.md), and [evaluation protocol](../docs/EVALUATION_PROTOCOL.md).

The source-attributed [BC-MLF adaptation](../vendor/bcmlf/README.md) and [CMC pseudo adaptation](../vendor/cmc/README.md) are present as research code. The CMC two-stage adaptation has one full-cohort mBERT-backbone seed at 40.80% accuracy / 37.13% Macro-F1; it is not its published native-model result. BC-MLF's native-feature route still lacks the author feature file due a public download quota. Published Acc2 or differently binned Acc5 figures cannot be compared directly to these scores.

Raw MP4 inference with the pilot ALMT checkpoint is implemented in `scripts/predict_raw.py`. Its pilot checkpoint and licensed data are not distributed. The script accepts a JSONL input with `id`, `video_path`, and optional `transcript`/`asr_language`, plus the run directory, exact cache metadata, and four pinned local encoders. It writes all five probabilities and a label for each successful record, with low-confidence entries also in a separate file. Its current ALMT path requires all three decoded modalities. `scripts/export_competition_csv.py` enforces one `ID,Label` row per MP4 filename, including its extension and low-confidence records; a [two-validation-video format rehearsal](../docs/competition_format_rehearsal.json) passed. The organizer-facing `test_script.py`, final trained model, complete missing-modality handling, and clean-environment inference verification are still pending.

The English raw pipeline separately decoded and SHA-checked 76 genuine MOSEI train/valid clips, including six with no MP4 but valid WAV and transcript; all produced correctly masked mBERT/WavLM/DINOv2 features. A 54/22 real-data training/resume exercise selected 9/22 correct predictions while reaching 100% training accuracy, showing severe small-sample overfitting. The [media-flow audit](../docs/mosei_pilot_feature_flow.json) and [training-flow audit](../docs/mosei_pilot_training_flow.json) are **not** English full-cohort model results. The historical competition archive attempt does not establish author-data coverage or a currently running transfer. The final competition script and full bilingual inference remain pending.

## Rights, limitations, and next work

Competition data and derived features stay outside Git. Its terms must be checked before any redistribution. Original JevAffect code uses the [MIT License](../LICENSE); bundled ports retain their own licenses and attribution in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Encoder and trained-weight redistribution conditions need a separate final review. No weights are offered for download.

The next gates are a verified original mechanism against competitive baselines, three-seed replication and matched ablations on both native tasks, independent final-test evaluation, native data/encoder/inference entries, and full reproduction from a fresh environment before release. Experiment choices will follow observed errors and the selected hypothesis. Legacy Git history passed actual bundle restore/fsck, and en/zh checkpoint archives passed SHA/CRC; the extra pretrained archive and final backup/release gates are pending. The old repository remains intact; [backup evidence and deletion conditions](../docs/LEGACY_BACKUP_STATUS.md) are explicit. Current status is kept in [PROGRESS.md](../docs/PROGRESS.md).

Closest public work: [ALMT](https://github.com/Haoyu-ha/ALMT), [CMC](https://github.com/gw-zhong/CMC), [BC-MLF](https://github.com/sunyulin0421/BC-MLF), and TypeSafe's [Jev introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev). JevAffect is an independent research project, not a release of Jev.
