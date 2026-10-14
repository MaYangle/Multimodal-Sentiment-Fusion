# Stronger conditioned sequence baseline

2026-10-14. Allthree native ALMT seeds are trained and independently audited.
TRAIN pooled-coordinate conditioning improves raw ALMT MAE in every seed,
but does not establish an advantage over the strong pooled fusion comparator.
This is a known preprocessing control, not an original method or final-test gain.

| Input / model | VALID MAE mean ± SD | Corr mean ± SD | Acc2 mean ± SD |
|---|---:|---:|---:|
| Raw sequence ALMT | .427086 ± .006547 | .5994 ± .0091 | 74.19% ± 1.17pp |
| Standardized pooled fusion | .411524 ± .003855 | .6358 ± .0058 | 76.82% ± .56pp |
| Standardized sequence ALMT | .408680 ± .006690 | .615730 ± .011848 | 73.62% ± 1.20pp |

Selected MAEs13/42/2026: .416190/.403356/.406495; best epochs11/21/21.
The primary native CH MAE clips predictions to[-1,1], targets remain original
M scores. Binary zero belongs to the negative class under the author rule.
Lower MAE is accompanied by lower binary accuracy than pooled fusion.

## Fairness and exact inputs

Same author TRAIN2722/VALID647, CSV pin, raw-derived MacBERT/WavLM/DINOv2
bytes and encoder versions. Apply the existing TRAIN clip-pooled statistics
to each valid token/frame, not statistics refit across frames. Preserve
T64/A400/V55, valid lengths and original scalar labels; no new encoder/data
or auxiliary supervision. Alloriginal PAD entries arezero, so no extra PAD
cleanup intervention occurs. There are21 hashed artifacts plus metadata;
four A/V time arrays are copied byte-for-byte. Their identity does not prove
cross-stream synchrony. Maximum pooled equivalence differences: text0,
audio1.216e-5, vision9.537e-7, under1e-4. No TEST is exported or scored.

Same ALMT/native training code,3,288,577 parameters and max80/patience12,
lr1e-4/WD1e-4, batch32, unbounded L1 training/first-min native VALID MAE.
Upstream unmasked padding participation remains. Three formal GPU loops
total362.454s, excluding preparation, verification, startup and serialization.
Pooled fusion has492,033 parameters and a much shorter training budget;
the architecture comparison does not isolate temporal information by itself.

[Native metric/checkpoint audit](native_sequence_almt_audit_20261014.json)
recomputes the647-row target/score protocol. A separate
[CPU replay](native_sequence_almt_cpu_replay_20261014.json) reconstructs all
three models and full sequence input shapes; max CPU/GPU prediction difference
9.239e-7, within1e-5. Shape loading was corrected in the untrained inference
helper: a shared pooled-statistics kind does not imply a one-step input;
trusted sequence_shapes determine the ALMT projection lengths. The fixture
now exercises actual training/loading/prediction and closes memmaps on Windows.

## Paired source comparison

5000 paired VALID video-source draws (120 groups), seed20261013; differences
mean candidate minus reference MAE. Against raw ALMT, three seed differences
−.015206/−.016197/−.023815; pooled−.018406 with95% interval
[-.029832,−.007161]. Against pooled fusion, differences+.008135/−.012318/
−.004349; pooled−.002844 with interval[-.021147,.015315]. These intervals
condition on selected development models, do not adjust repeated VALID use,
and are not final-test significance. Preserve both comparators:
[raw comparison](native_sequence_almt_vs_raw_20261014.json),
[fusion comparison](native_sequence_almt_vs_fusion_20261014.json).

## Commands and next hypothesis

```powershell
python scripts/prepare_native_sequence_standardization.py --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --pooled-standardized-cache data/feature_cache/simsv2_native_pooled_standardized_v1 --output-dir data/feature_cache/simsv2_native_sequence_standardized_v1
python scripts/run_native_baselines.py --configs configs/native_simsv2_sequence_standardized_almt.json --output docs/native_sequence_almt_formal_20261014.json
python scripts/audit_native_results.py --suite docs/native_sequence_almt_formal_20261014.json --run-root runs/native_simsv2_sequence_standardized_v1 --output docs/native_sequence_almt_audit_20261014.json
python scripts/reload_native_sequence_almt.py --suite docs/native_sequence_almt_formal_20261014.json --output docs/native_sequence_almt_cpu_replay_20261014.json
```

Existing training needs `--resume`; data preparation reuses only a fully
matching completed output. Real32-row smoke/resume preceded formal training.
Original author media and feature acquisition remain incomplete; this is a
common-encoder adaptation, not author processed-feature reproduction.

Both standardized baselines must be competitors in the next point-fusion
test. The [timed-covariation review](NATIVE_TIMED_COVARIATION_REVIEW_20261013.md)
allows one conditional test after time-origin verification, with static-product,
marginal-variance and unpaired controls. Covariance is established prior art;
even an empirical gain does not establish formula originality. The CDF-shape
line is stopped. Original mechanism, MOSEI method repeat, frozen TEST, final
fresh pipeline reproduction, release and legacy cleanup remain open.
