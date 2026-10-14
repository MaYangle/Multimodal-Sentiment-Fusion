# Native scalar CRPS: stop the shape line

Study began2026-10-13; reviewed2026-10-14. **Reject promotion and stop the
registered probability-shape line.** Degree6 does not stably beat the simpler
mixture, and its native error-hit forecasts and90% interval score deteriorate.
No original algorithm, final-test benefit or MOSEI replication is established.
The [pre-fit plan](NATIVE_CDF_CRPS_PLAN_V1.md) remains frozen.

## Actual execution

Four cases xthree seeds xthree source folds complete36 CPU fits, reusing
the independently audited nine point teachers/27 tables. Four real eight-source
smoke heads completed backward/checkpoint/epoch resume before formal fits.
Formal loops total27.906s, excluding initialization, hashing, serialization
and readout. Original2722 TRAIN scalar targets are retained, without rounding
or excluding off-grid scores. No official VALID/TEST is evaluated.

The [independent audit](native_cdf_crps_audit_20261013.json) replays36 models,
32664 outer predictions and12 canonical2722-ID merges;987 checks pass.
NumPy equations independently reproduce CDFs, bins, native/raw events,
quantiles, CRPS, first-min checkpoint selection and fit-only statistics.
The primary event is |target-clipped_native_point|<=.1, matching CH author
point scoring; raw-point events are explicitly auxiliary. Point predictions
and native point metrics remain identical across head cases.

48/96/192-node quadrature passes for allselected models. Maximum per-model
mean absolute row discrepancy is1.412e-5, below the registered1e-4 gate.
The pre-fit assay on36 older NLL models had maximum mean discrepancy5.042e-8.
Uniform/point-atom analytic limits, independent adaptive integration and
actual median-MDN gradient tests pass. CRPS is a numerical integral here,
not an exact closed-form claim;1001-point trapezoid is separately recorded.

## Complete registered readout

Scores average all2722 outer rows perseed. Lower is better. Counts are head
parameters; each case shares the same frozen point/encoder inputs.

| Case | Parameters | CRPS13 /42 /2026 | Mean native-hit Brier | Mean native-hit log score |
|---|---:|---|---:|---:|
| scale | 8289 | .276109 /.276370 /.278836 | .138524 | .445762 |
| mixture | 8322 | .276101 /.276705 /.279504 | .138418 | .445569 |
| median_mdn | 8784 | .276886 /.278165 /.279134 | .141859 | .455557 |
| **bern_d6, primary** | 8784 | .276291 /.273159 /.275990 | .141013 | .450968 |

Use5000 paired source/video draws over129 groups,seed20261013, with shared
draws across cases/seeds. Gain means control score minus primary score.

| Reference | Pooled CRPS gain [95% CI] | Native-hit Brier gain [95% CI] | Native-hit log gain [95% CI] |
|---|---|---|---|
| mixture | +.002290 [-.000494,.005081] | −.002596 [-.004309,−.000901] | −.005399 [-.010528,−.000118] |
| median_mdn | +.002915 [.000902,.004897] | +.000845 [-.000847,.002563] | +.004588 [-.001720,.011087] |

CRPS improves against median-MDN in allthree seeds, but not against Mixture
in seed13. Native-hit Brier/log worsens against Mixture in13/2026 and against
median-MDN in13. The pooled90% interval score also deteriorates by.04648
against Mixture and.02699 against median-MDN. A favorable global distribution
score does not establish calibrated small-error probability or useful intervals.
All failed required comparisons remain failed; no post-hoc winner is substituted.
Source intervals condition on these fitted seeds and are not final-test evidence.

## Numerical and provenance limits

Readout only permits CDF range roundoff within1e-12 and bin row-sum error
within1e-10. Significant negative mass raises an error. Maximum corrections
are recorded; NumPy/Torch can differ in the count of repaired rows. The audit
checks probability-valued minima with forecast tolerance, correction bounds
with strict roundoff tolerances, and counts by range. Interleaved quantile
levels are sorted before monotonicity checks. Interval-score penalties amplify
edge roundoff, so scalar-score tolerance is2e-5 while probability/edge checks
remain separate. These were auditor fixes; trained model outputs were unchanged.

Checkpoint identities bind the trainer/CDF/data modules, configuration,
plan, tables/IDs and normalization. The runner script hash is **not embedded
in the checkpoints**; its pre-fit handoff SHA
`4d50b46dd23b99c4d667f9171856a047777db22f5c25ce99676cfb7a1ac5e36c`
matches the inspected script after training. Preserve that distinction rather
than claiming every runner byte was checkpoint-bound. All36 full head fits
[reproduce exactly from independent source](NATIVE_CDF_CRPS_REPRODUCTION_20261014.md);
encoder/point/raw-media re-extraction and final-method full-environment
reproduction remain incomplete.

The root and independent source integrated288-test passes include the CRPS
audit, exact-resume, off-grid scalar, sequence preparation/timestamp and
actual normalized-sequence inference checks.
Intermediate audit-helper failures were fixed and are not successful-test
receipts; use the latest final log and study receipt.

## Reproduce and next decision

With the same audited point suite and `train,uncertainty` dependencies:

```powershell
python scripts/train_native_crps.py --config configs/native_cdf_crps_v1.json --output runs/native_cdf_crps_v1/suite.json
python scripts/audit_native_crps.py --suite runs/native_cdf_crps_v1/suite.json --point-suite runs/native_nested_shape_points_v1/suite.json --report docs/native_cdf_crps_audit_20261013.json
```

Use `--resume` for existing runs; case/seed/fold and stop-after support partial
work without falsely marking full completion. Do not extend this rejected
line with another CDF family or degree. The next point-fusion review recommends
a [standardized-sequence ALMT baseline](NATIVE_TIMED_COVARIATION_REVIEW_20261013.md)
before any paired temporal interaction. This is a known conditioning control,
not an original method. Verify time origins before claiming A/V synchrony;
saved time arrays do not alone prove it. Author media/feature acquisition,
verified original contribution, MOSEI method repeat, final TEST, release and
conditional legacy deletion remain open.
