# Nested native probability shapes: negative result

2026-10-13. **Do not promote degree6.** The registered all-seed gate fails:
bin NLL improves over the feature-only mixture in13/42 and worsens in2026.
Raw-point hit Brier worsens in13/2026. No original algorithm, final-test or
cross-dataset gain is established. The [pre-fit plan](NATIVE_NESTED_SHAPE_PLAN_V1.md)
is preserved unchanged; mathematical ingredients have prior art.

## Execution and verification

Nine standardized fusion points use five source-disjoint roles per outer
fold, followed by ten cases xthree seeds xthree folds (90 fits). Only2722
author TRAIN IDs are used. No official VALID/TEST prediction or score is made.
Point teachers exclude calibration sources from fitting, selection and
scalers. Every outer source/ID appears once; combined point-export holdouts
repeat across folds and are not global once-only OOF.

Real eight-source CUDA point smoke/resume and allten CPU shape smoke/resume
passed before formal fitting. Point loops total109.246s; shape loops49.822s,
excluding loading, hashing, startup, serialization and readout. Allnine point
checkpoints and27 exports pass646 independent checks, including CPU replay
of raw pooling, fit-only statistics, contexts and points:
[point audit](native_nested_shape_point_audit_20261013.json). Maximum context
difference5.007e-6 and point difference7.153e-7 are under1e-5.

Root integrated267 tests pass after six probability-audit checks and four
native-support prototype tests (`work/integrated_tests_native_support_prototype_20261013.log`),
distinct from the earlier569-file source copy with235 tests.
[Independent probability audit](native_nested_shape_probability_audit_20261013.json)
passes1711 checks on all90 models/81660 outer rows and30 canonical merges.
Separate NumPy equations recover PMFs, hit probabilities, quantiles, CRPS,
native metrics and checkpoint selection within2e-6. The table below is
independently confirmed; no official held split is used.

## All registered cases

Each seed covers2722 outer rows. NLL scores the author five-bin outcome.
Hit means |target-raw_point|<=.1. CRPS uses1001-point trapezoid integration
on[-1,1]. All scores are lower-better. Parameters count only the shape model.

| Case | Parameters | Bin NLL13 /42 /2026 | Mean hit Brier | Mean hit log score | Mean CRPS |
|---|---:|---|---:|---:|---:|
| scale | 8289 | 1.309374 /1.310716 /1.318055 | .129776 | .421368 | .277498 |
| mixture | 8322 | 1.311071 /1.311670 /1.318897 | .129648 | .421223 | .277827 |
| bern_uniform | 8322 | 1.455791 /1.449752 /1.457078 | .131335 | .435013 | .301521 |
| bern_tied | 8553 | 1.360308 /1.347224 /1.392335 | .132603 | .430687 | .291097 |
| bern_d3 | 8586 | 1.321071 /1.314236 /1.335510 | .131522 | .428305 | .279019 |
| **bern_d6, primary** | 8784 | 1.307789 /1.307807 /1.330469 | .130692 | .423547 | .277281 |
| bern_d12 | 9180 | 1.322804 /1.309258 /1.361253 | .133104 | .432774 | .281552 |
| median_mdn | 8784 | 1.337620 /1.326488 /1.337727 | .130793 | .425397 | .280663 |
| mdn | 8784 | 1.335846 /1.335312 /1.336851 | .131986 | .438603 | .283697 |
| spline | 8553 | 1.317827 /1.314410 /1.319902 | .130063 | .422548 | .277975 |

Independent sides improve tied/uniform restrictions but fail the required
comparison with simpler mixture. Degree12 does not recover stable gains.
Degree6 improves NLL over median-MDN in allseeds, insufficient when the
mixture and event gates fail. Point forecasts are identical across allcases;
this study makes no point-accuracy improvement claim.

Across5000 paired129-source bootstrap draws (seed20261013), define gain as
control score minus primary score, positive favoring degree6. Against Mixture,
bin-NLL gain is−.001476,95% interval[-.014306,.011128]; Hit-Brier gain
−.001044,interval[-.002529,.000593]; hit-log gain−.002324,interval
[-.007558,.003265]. Against median-MDN, bin gain+.018590 has interval
[.008248,.028739], but hit Brier/log intervals still includezero and seed13
worsens. The complete gate fails. These intervals condition on three fitted
seeds and sample source-video clusters; they are not final-test evidence.

## Native clipping and probability interpretation

Point outputs are stored unchanged; probabilistic anchors clip to[-1,1].
CH author MAE/Corr themselves clip predictions (unlike MOSEI). The producer
key `native_raw_point` is a misleading alias of the clipped author-protocol
metric, not an unclipped MAE. No trained code or pre-fit gate changed after
readout. Native MAEs are .402430/.399843/.403212; clipped counts163/173/125.
These nested models fit fewer point sources than earlier OOF models, so their
point scores are not direct improvements over those earlier runs.

Independently computed unclipped raw-point MAEs are
.413946/.411094/.410452, reported separately from the native values above.

The registered hit event stays centered on raw points. An additional
clipped-point hit diagnostic matches CH author-scored forecasts; it is
separate and cannot replace the failed primary event gate. Fixed medians
refer to bounded anchors, never out-of-range raw outputs. Endpoint atoms
represent censoring, not inferred human-rater states.

## Why the next experiment changes

[Post-study diagnostic](native_nested_shape_objective_diagnostic_20261013.json)
replays selected-checkpoint fit/selection scores without refitting.
Unweighted fold/seed mean degree6 fit NLL is .945 versus selection1.389 and
outer1.315; mixture is1.207/1.367/1.314. Flexible shapes have a larger fitting
advantage that does not transfer. This describes overfitting, not a causal
explanation of every error.

Author TRAIN has11 distinct native scores on the .2 grid. Five bins merge
2/3/1/3/2 scores;318 neutral-score rows are exactly0. A central logistic
kernel is not an observed-score atom. Bin NLL is proper for its coarsened
outcome but does not identify every within-bin regression-error probability.
Review native-score supervision before another shape family or larger degree.
This does not prove a changed objective will help or constitute an original method.
The amended [native-score review](NATIVE_SCORE_RESOLUTION_REVIEW_20261013.md)
rechecked an existing off-grid author-metadata annotation without opening
held labels. The [eleven-score PMF plan](NATIVE_SUPPORT_PROBABILITY_PLAN_V1.md)
is deferred before fitting and its prototype is only a TRAIN diagnostic.
The [next complete-protocol plan](NATIVE_CDF_CRPS_PLAN_V1.md) retains bounded
CDFs and original scalar-score CRPS; no CRPS fits or originality claim yet.

## Commands and remaining scope

Requires the same verified raw-derived TRAIN cache and `train,uncertainty`
extras. This study does not solve author media acquisition or redistribution.

```powershell
python scripts/prepare_nested_shape_points.py --config configs/native_nested_shape_points_v1.json --output runs/native_nested_shape_points_v1/suite.json
python scripts/audit_nested_shape_points.py --suite runs/native_nested_shape_points_v1/suite.json --report docs/native_nested_shape_point_audit_20261013.json
python scripts/train_nested_shapes.py --config configs/native_nested_shapes_v1.json --output runs/native_nested_shapes_v1/suite.json
python scripts/audit_nested_shapes.py --suite runs/native_nested_shapes_v1/suite.json --point-suite runs/native_nested_shape_points_v1/suite.json --report docs/native_nested_shape_probability_audit_20261013.json
python scripts/diagnose_nested_shape_objectives.py --suite runs/native_nested_shapes_v1/suite.json --points runs/native_nested_shape_points_v1/suite.json --output docs/native_nested_shape_objective_diagnostic_20261013.json
```

Existing runs need `--resume`; `--seed`, `--fold`, `--stop-after` support
interruption/recovery. Point smoke uses `--smoke-groups 8`; shape smoke needs
a separate config pointing to that suite and `--allow-smoke`. Smoke/subsets
cannot establish formal completion. Checkpoints bind code/data/config/roles,
IDs/scalers and RNG state. A new full-source/environment reproduction of
this negative study has not yet run. Original contribution, MOSEI method
replication, final TEST and release remain open.
