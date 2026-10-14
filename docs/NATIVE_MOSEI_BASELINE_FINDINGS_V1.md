# MOSEI native publisher-feature baselines — completed

Nine raw and six standardized runs completed on2026-10-12/13. All preserve
MultiBench16,265 TRAIN/1,869 VALID temporal IDs and original[-3,3] sentiment
scores. This author-linked aligned50 GloVe300/acoustic74/vision35 cohort is a
separate release variant with four current-SDK train membership discrepancies;
not the ordinal MMSA CSV or original-video encoder pipeline. Final TEST remains
independent. [Registered raw plan](NATIVE_MOSEI_BASELINE_PLAN_V1.md),
[known normalization plan](NATIVE_MOSEI_NORMALIZATION_PLAN_V1.md).

| Model/input | Native valid MAE ↓ | Corr ↑ | Non0 Acc2 ↑ | Parameters |
|---|---:|---:|---:|---:|
| Text valid-mean, raw | 0.6697 ± 0.0001 | 0.5325 ± 0.0019 | 78.64% ± 0.16 pp | 77,313 |
| Early fusion, raw | 0.6709 ± 0.0013 | 0.5209 ± 0.0053 | 77.16% ± 0.39 pp | 105,217 |
| ALMT scalar structural adaptation, raw | **0.6115 ± 0.0051** | **0.6044 ± 0.0119** | 76.79% ± 1.38 pp | 3,047,937 |
| Text, TRAIN coordinate standardization | 0.6516 ± 0.0005 | 0.5646 ± 0.0014 | 77.18% ± 0.42 pp | 77,313 |
| Early fusion, same standardization | 0.6450 ± 0.0017 | 0.5735 ± 0.0034 | 77.74% ± 0.89 pp | 105,217 |

Mean±sample SD, seeds13/42/2026. Native raw MAE is not clipped; Non0 excludes
true neutral then uses score>0. Has0 includes neutral with >=0, Acc5/7 use
author clipping and ties-even rounding; all metrics are in the audits.
ALMT improves MAE/Corr but has worse binary accuracy than text. No universal
metric dominance, paper reproduction score or state-of-the-art claim.

All use original-score L1, the same fixed optimizer/80-epoch/patience12 and
validation-MAE selection policy. They have different capacity and executed
updates. Three-seed loop totals, excluding hashing/startup/checkpoint I/O:
raw text42.0s, fusion111.5s, ALMT2,455.0s; standardized text50.6s/fusion49.5s.
ALMT executed25/32/36 epochs; hardware shared with desktop apps and per-epoch
time varied materially. These are measured training durations, not controlled
inference-speed or encoder acquisition benchmarks.

## Integrity and error evidence

[Independent complete adapter audit](mosei_author_adapter_independent_audit.json)
matches every TRAIN/VALID label/ID and all six split/modality float32 sequences
against public-loader slicing/audio rules. All19 raw cache artifacts pass hash
checks; no TEST labels/features accessed by that audit. Root CPU fixtures cover
masking, coordinate normalization without validation fitting and exact resume;
three real32/32 smoke runs plus actual GPU resume preceded training.

[Raw per-ID/hash/epoch audit](native_mosei_author_baseline_audit_v1.json) and
[six standardized audits](native_mosei_standardized_audit_v1.json) independently
recompute every native metric. Standardization uses TRAIN clip-pooled coordinate
mean/population STD, scale1 for STD<1e-6, no label fitting and no test access.
It improves cheap fusion from0.6709 to0.6450; this is a known preprocessing
effect, not an original fusion mechanism. Teacher/correction studies must give
all compared methods the same input policy and account for extra models.

[Source-cluster paired errors](native_mosei_author_errors_v1.json) show all
three ALMT-minus-raw-fusion MAE intervals below0, conditional on the selected
models. However, the pattern is heterogeneous: neutral MAE0.3704→0.2702,
strong |y|>=2 MAE1.7556→1.3672, weak nonzero |y|<=1 MAE0.4669→0.5208 versus
raw text. True intensity strata are retrospective analysis, not deployable
router inputs. VALID has433 neutral,929 weak-nonzero and188 strong records.

## Frozen feature sensitivity, not causal/raw robustness

[Registered diagnostic](NATIVE_MOSEI_SEQUENCE_DIAGNOSTIC_PLAN.md) evaluates
all nine raw models under five frozen feature conditions (45 evaluations).
[Saved metrics/prediction hashes](native_mosei_sequence_diagnostic_v1.json):

| ALMT condition | Three-seed valid MAE |
|---|---:|
| Identity | 0.6115 |
| Joint valid-step permutation (pairing retained) | 0.6223 |
| Independent audio/vision permutation | 0.6108 |
| Joint reversal | 0.6190 |
| Front padding with same valid vectors/order | 0.6699 |

Pooled controls preserve predictions within1e-5 under every multiset-preserving
transform; identity matches frozen outputs. ALMT has modest joint-order
sensitivity and substantial padding-position sensitivity. Independent AV
permutation does not hurt on average, so fine-grained AV alignment is not a
supported explanation for all of its advantage. Padding change is outside
the fixed training convention; this does not show benchmark corruption or
justify a causal explanation. The features are transformed representations,
not newly labeled real utterances. A new temporal mechanism must isolate order,
normalization, capacity and masking effects, rather than infer them from a
single architecture gap.

## Next research gate

Both native tasks now have real three-seed baseline evidence, but a competitive
original mechanism and cross-dataset method replication remain unproven. The
[annotation-reference experiment](NATIVE_REFERENCE_RESIDUAL_FINDINGS_V1.md)
failed its structure-specific gate. Broad acquisition and AV residual concepts
also have close prior work. Retain these baselines and negative controls while
choosing the next narrow mechanism; final test, source-disjoint checks, complete
native raw inference, fresh reproduction, publication and cleanup remain pending.
