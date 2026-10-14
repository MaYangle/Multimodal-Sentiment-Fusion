# Dependence correction: no verified advantage beyond calibration

2026-10-12. [The pre-training plan](NATIVE_DEPENDENCE_PLAN_V1.md) tests a known
covariance readout across experts of one scalar target, not a new formula.
Three seeds × three source-disjoint fold models completed. Every one of the
2,722 official TRAIN records has exactly one OOF prediction per seed. Its
129 source videos are grouped without labels; fit, inner selection and OOF
sources are disjoint within each fold. The OOF trainer never parses official
validation/test targets. Exact CPU resume and group leakage rejection passed.
Three real source-group smoke runs preceded formal training. Nine OOF loops
totaled122.3 seconds, excluding hashing, startup and checkpoint I/O. All
comparators share these models/predictions and the same frozen final checkpoint.

Correlation is estimated on standardized TRAIN OOF residuals, then shrunk
by the fixed0.1 coefficient. Each calibrated comparator fits exactly one
variance scale q on the same OOF records by raw Student-t density NLL. Official
validation errors fit neither q nor correlation. All fits are inside their
predeclared bounds. The official 647 validation examples evaluate the resulting
distributions; they are still development/selection evidence, not final test.

| Distribution/readout | Valid density NLL ± sample SD ↓ | Coverage90 | Width90 |
|---|---:|---:|---:|
| Original summed NIG | 1.2600 ± 0.1900 | 61.67% | 0.9633 |
| NIG, global variance scale | 0.8006 ± 0.0324 | 85.27% | 1.5378 |
| Independent-expert covariance, scaled | **0.7892 ± 0.0171** | 86.76% | 1.6159 |
| Residual-correlated covariance, scaled | 0.8002 ± 0.0353 | 85.78% | 1.5600 |
| Proper Student-t linear pool | 0.8260 ± 0.0360 | 85.11% | 1.5820 |
| Proper pool, component variance scale | 0.8094 ± 0.0228 | 88.97% | 1.7038 |

All native MAEs are0.434643±0.003459. Covariance/scale controls preserve stored
point means and degrees of freedom exactly. The proper pool's recomputed
float64 mean differs from stored float32 fused means by at most1.563e-7;
the native float32 MAE remains identical. No accuracy improvement is claimed.

The correlated-minus-global NLL differences are+0.002375,+0.002900,-0.006338
for seeds13/42/2026. The gain is inconsistent and its mean difference is only
-0.000354. Independent covariance beats correlated covariance in two seeds
and on average. The correlation candidate **fails the registered stability
gate** and is not promoted. Scalar calibration itself accounts for most of
the original-NIG improvement; achieving coverage closer to90% also expands
intervals. No comparator is claimed calibrated at all levels or under shift.

[Paired source-video bootstrap](native_dependence_bootstrap_v1.json) resamples
whole validation sources2,000 times, retaining clip-weighted scoring. Averaging
the paired NLL difference over the three frozen seeds gives-0.000354 with
percentile95% interval[-0.010632,0.009622] against global scaling. It includes
zero, consistent with the rejection. This conditional interval does not account
for model training, fitting or validation selection, and is not an independent
final-test significance claim. Per-seed and other comparator intervals remain
available in the artifact.

## Error analysis and limits

Mean raw off-diagonal residual correlation is0.6658. Maximum correlation entry
variation across seeds is0.0397; within a seed across OOF folds it reaches0.2418.
Global correlation is not negligible, but is insufficient evidence that adding
it improves scalar predictive uncertainty. For this same-target setup, the
shared y enters every expert residual, unlike different-target joint-density
estimation in [Residual-ICM](https://arxiv.org/html/2609.30085v1). Marginal
miscalibration, common prediction bias, input-dependent dependence and the
teacher/final training-size difference can all limit this moment approximation.
These are plausible limitations, not experimentally isolated explanations.

The correlated/independent readouts use a Student-t with the original fused
degrees of freedom and replaced variance; they are not an exact correlated
Student-t sum. The proper pool evaluates the actual mixture density and
numerically inverts its CDF. Sentiment scores are quantized and bounded;
continuous density NLL is not a discrete probability or classification loss.
OOF fit NLL is an in-fit diagnostic; held-out results are the validation table.
Multiple development rounds have used validation, so final test must stay
independent and results cannot be called unbiased benchmark performance.

Evidence: [OOF manifests/results](native_dependence_oof_v1.json),
[per-seed fitted matrices/scalars/metrics](native_dependence_results_v1.json),
[paired summary and gate](native_dependence_summary_v1.json). Checkpoints,
histories and per-example readout parameters remain in ignored runs. Tests
also cover covariance arithmetic, mask behavior, duplicated-source variance
invariance with replicated R, moment matching and scalar fit recovery.

Next research decision: stop promoting NIG evidence-counting/correlation on
this weaker point backbone. Retain these known calibration controls, obtain
author features, and prioritize competitive point/representation or a focused
robustness mechanism before designing another dependence correction. Original
contribution, MOSEI replication, final test, fresh reproduction and publication
are still incomplete.
