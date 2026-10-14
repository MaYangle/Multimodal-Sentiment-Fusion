# Native Chinese pooled standardization: stronger known fusion baseline

2026-10-13. The [plan](NATIVE_CH_STANDARDIZATION_PLAN_V1.md) was saved before
preparation/fitting. Same source encoder cache, official2722/647 IDs, original
M[-1,1], frozen point MLP, seeds and minimum VALID MAE selection. Only
TRAIN-fitted coordinate scaling changes. No TEST arrays/labels export or scoring.

| Official VALID development model | MAE ↓, mean ± sample SD | Corr ↑ | Acc2 ↑ |
|---|---:|---:|---:|
| Original text CLS MLP | .437309 ± .002435 | .5820 |75.06% |
| Standardized text CLS MLP | .442491 ± .001405 | .5736 |73.98% |
| Original fusion MLP | .422070 ± .000736 | .6234 |77.13% |
| Standardized fusion MLP | **.411524 ± .003855** | .6358 |76.82% |

Fusion MAEs .408055/.415674/.410844 improve the original baseline in all
three seeds. Text worsens in all seeds; standardization is not uniformly
beneficial to every encoder/task. Acc2 also does not improve with regression.
[Six-run independent audit](native_simsv2_standardized_audit_20261013.json)
checks native labels/IDs, code/cache/selected checkpoints, actual budgets,
and CPU-recomputed647 forecasts. Loops total5.92s(text)/8.00s(fusion), excluding
prep/startup/hash/I/O. Text/fusion parameter counts remain197,121/492,033.

The [paired comparison](native_simsv2_standardized_comparison_20261013.json)
gives normalized-minus-raw fusion mean MAE−.010546 and conditional95% source
interval[-.023699,.001989]. It includes zero. These selected models were
developed on VALID; neither three consistent seed directions nor this bootstrap
establishes unbiased final-test significance. Source-OOF normalization separately
improved all seeds with a zero-excluding interval, but the cohorts/selection
rules differ and cannot establish a causal source effect.

Freeze standardized fusion as a stronger **known** research baseline while
retaining raw text/fusion and mature public adaptations. This preprocessing
result is not promoted as JevAffect's original algorithm.

## Data and representation

`data/feature_cache/simsv2_native_pooled_standardized_v1` has17 bound artifacts
and metadata SHA `50c067883c3e52d6130eb8ed80f2df9d8a29bbc0c0a1dd9ac6a9e7d2847d1e7d`.
It stores MacBERT CLS, WavLM length-masked mean and DINOv2 length-masked mean as
one-step vectors, standardized by full TRAIN mean/population STD. Tiny STD
below1e-6 becomes1; no label normalization occurs. Missing modalities keep
length0/zero vectors. Raw source bytes are unchanged; transformations and
source/code hashes are explicit. This is the same competition-acquired,
author-ID-matched cache, not the unavailable author feature payload.

## Actual media inference and deployment

Two genuine competition-acquired TRAIN MP4s plus provided transcripts execute
the full CPU encoder path. Original truncation happens before CLS/masked mean,
then the bound TRAIN scaler is applied. [Numerical audit](native_standardized_raw_inference_audit_20261013.json)
reproduces scores on cached trained vectors within3.815e-6. This is a flow check,
not held-out accuracy, complete re-extraction, latency or author-original
byte-equivalence. New author TRAIN retrievals still returned quota HTML;
[missing-source receipts](author_train_raw_acquisition_20261013.json) record0
media bytes for each of two previously unattempted targets.

The inference loader can read normalized dimensions from the bound metadata;
training arrays are unnecessary for deployment. An actual local descriptor-only
export validates checkpoint and encoder contracts without copying labels,
training arrays or weights. It contains TRAIN-derived mean/STD; redistribution
conditions still require final review. Pretrained encoder weights remain separate.

```powershell
python scripts/prepare_native_pooled_standardization.py --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --author-csv data/author/simsv2/meta.csv --output-dir data/feature_cache/simsv2_native_pooled_standardized_v1
python scripts/run_native_baselines.py --configs configs/native_simsv2_standardized_text.json configs/native_simsv2_standardized_fusion.json --output docs/native_simsv2_standardized_baselines_20261013.json
python scripts/audit_native_results.py --suite docs/native_simsv2_standardized_baselines_20261013.json --run-root runs/native_simsv2_standardized_v1 --output docs/native_simsv2_standardized_audit_20261013.json
python scripts/compare_native_standardization.py --raw-suite docs/native_simsv2_baselines_v1.json --normalized-suite docs/native_simsv2_standardized_baselines_20261013.json --raw-run-root runs/native_simsv2_v1 --normalized-run-root runs/native_simsv2_standardized_v1 --output docs/native_simsv2_standardized_comparison_20261013.json
python scripts/export_native_input_descriptor.py --checkpoint runs/native_simsv2_standardized_v1/early_fusion/seed_13/best.pt --cache-dir data/feature_cache/simsv2_native_pooled_standardized_v1 --raw-source-cache data/feature_cache/simsv2_full_macbert_raw_av --output-dir work/native_standardized_inference_descriptor_20261013
python scripts/predict_native_media.py --checkpoint runs/native_simsv2_standardized_v1/early_fusion/seed_13/best.pt --cache-dir work/native_standardized_inference_descriptor_20261013 --raw-source-cache work/native_standardized_inference_descriptor_20261013/raw_encoder_descriptor --macbert-dir data/pretrained/chinese-macbert-base --model-root data/models --input-jsonl work/native_raw_demo_20261013/cached_train_input.jsonl --output-dir work/native_raw_demo_20261013/standardized_descriptor_cpu --device cpu
```

Use new preparation/export/inference destinations; existing training uses
`--resume`. The result files and complete output descriptors remain local
ignored artifacts. A569-file independent source snapshot passes235 tests,
reproduces all18 data/metadata files and allsix models exactly;
[reproduction scope](NATIVE_STANDARDIZED_REPRODUCTION_20261013.md).
Original-method and release gates are not redefined as passed.
