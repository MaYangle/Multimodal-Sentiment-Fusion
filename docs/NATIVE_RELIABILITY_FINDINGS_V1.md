# Reliability v1: measured negative result

2026-10-12. Four registered conditions × seeds 13/42/2026 finished on local
RTX 5070 Ti. Author continuous labels, exact 2,722/647 train/validation IDs,
encoder bytes and 496,143 trainable parameters were identical across factors.
No test evaluation or calibration fit occurred. The 12 per-ID audits passed.
Smoke runs and formula/gradient/resume tests are separate from research results.

## Clean validation selection results

| Auxiliary supervision | Fusion | MAE ↓ (mean ± sample SD) | Corr ↑ | 90% interval coverage | 90% interval width |
|---|---|---:|---:|---:|---:|
| off | equal means | 0.4397 ± 0.0021 | 0.6161 | 56.67% | 0.9129 |
| off | dynamic NIG | 0.4363 ± 0.0036 | 0.6075 | 80.32% | 1.4045 |
| on | equal means | 0.4393 ± 0.0026 | 0.6164 | 56.21% | 0.9046 |
| on | dynamic NIG | 0.4360 ± 0.0035 | 0.6059 | 82.33% | 1.4641 |

The existing simple-fusion control is 0.4221 ± 0.0007 MAE, with 492,033
parameters on the same inputs. All four new conditions are worse on the
registered primary metric. Dynamic NIG improves density NLL and coverage
relative to equal-mean controls, but also gives wider intervals; neither reaches
its nominal 90% coverage. These raw Student-t densities are uncalibrated, and
the bounded/quantized sentiment score makes their density assumption imperfect.

The supervision × dynamic-fusion interaction in MAE is −0.000824, +0.000525,
+0.000829 in the three seeds; negative means a helpful interaction. The signs
do not support a stable interaction. Auxiliary supervision alone changes MAE
by less than .001 per seed. For dynamic models, mean uncertainty/error Spearman
falls from 0.1885 without auxiliary supervision to 0.1539 with it. Thus the
registered mechanism is not supported as a competitive or better-calibrated
contribution. No novelty follows from implementing established NIG operators.

## Stress results and gate behavior

All seven nonempty modality subsets and four train-STD-scaled noise conditions
used identical validation IDs and corruption seed105. Corruption is on pooled
features, not raw audio/video. The selected checkpoints were fixed beforehand.

At 1 STD visual noise, S1 dynamic vs S1 uniform MAE is 0.4461 vs 0.4722: a
useful stress difference within this architecture. However, without text,
audio+vision MAE is 0.5463 vs 0.5094, so dynamic gating is not uniformly more
robust. S1 uniform with text+audio is 0.4224, versus 0.4393 with all modalities;
the added visual expert can damage its prediction. This is a diagnostic
selection-set observation, not a basis for choosing a final-test subset.

With visual noise, the S1 dynamic visual mean weight rises in seeds13/42 and
falls in2026. Only 34%/45%/57% of examples lower the corrupted visual weight.
Consequently the weights cannot be described as validated modality reliability.
Audio weight changes are small and approximately half go in either direction.

## Research decision

Do not promote S1G1 or tune this four-condition suite further to hide its loss
to simple fusion. Preserve all checkpoints, configurations, per-example NIG
parameters/weights and stress predictions. This falsifies the current proposal
in the tested implementation; it does not rule out every multi-task or
uncertainty model.

The next controlled baseline should include the **joint/pseudo-modality
expert present in the original MoNIG method**. The current model omitted that
cross-modal predictor and has only independent pooled experts. Its failure
relative to a joint MLP is evidence that joint context may matter, but this is
an inference requiring a capacity/budget-matched comparison. Establish that
stronger public-method adaptation before designing an original mechanism.
Separately investigate whether summing correlated expert evidence produces
the observed undercoverage; any proposed correction must preserve point
prediction controls and compare against proper predictive-mixture/calibration
baselines. This is a research question, not an accepted explanation.

## Reproducible evidence

- [Registered plan](NATIVE_RELIABILITY_PLAN_V1.md)
- [Mathematical implementation boundary](NIG_IMPLEMENTATION.md)
- [All 12 training results](native_reliability_formal_v1.json)
- [Per-ID factor audit](native_reliability_factor_audit_v1.json)
- [132 validation stress evaluations](native_reliability_stress_v1.json)
- [Paired stress effects and actual gate responses](native_reliability_stress_audit_v1.json)

Total training/validation loops: 326.3 seconds across all 12 runs, excluding
source/cache hashing, startup, artifact I/O and subsequent stress evaluation.
Executed epochs and updates differ under the same early-stopping policy;
the factor audit reports them explicitly. Current integrated tests: 149 passed.
