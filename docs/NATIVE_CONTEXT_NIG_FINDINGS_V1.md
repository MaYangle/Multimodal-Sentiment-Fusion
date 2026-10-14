# Joint-context NIG findings — validation only

All six cases × seeds 13/42/2026 completed on 2026-10-12; exact native targets,
647 IDs, feature/checkpoint hashes, selected epochs and native metrics pass
[the independent audit](native_context_nig_audit_v1.json). No test evaluation.
The [registered plan](NATIVE_CONTEXT_NIG_PLAN.md) records source revision,
source-summed branch loss, architecture and budget before training.

| Common-encoder adaptation | Valid MAE ± sample SD | Nominal90 coverage | Width90 |
|---|---:|---:|---:|
| Three experts, uniform | 0.4523 ± 0.0006 | 59.61% | 0.9975 |
| Three experts, dynamic | 0.4412 ± 0.0032 | 77.18% | 1.3222 |
| Joint expert, uniform | 0.4389 ± 0.0038 | 57.55% | 0.9329 |
| Joint expert, dynamic | 0.4346 ± 0.0035 | 61.67% | 0.9633 |
| Joint disabled, uniform | 0.4519 ± 0.0037 | 62.91% | 1.0532 |
| Joint disabled, dynamic | 0.4411 ± 0.0018 | 77.69% | 1.3281 |
| Simple fusion, native L1 | **0.4221 ± 0.0007** | not probabilistic | not probabilistic |

The MoNIG joint expert improves point accuracy against both its directly
disabled counterpart and the capacity-matched wider three-expert reference.
It remains worse than simple fusion, and its dynamic intervals are more
undercovered than the three-expert dynamic model. This is a negative result
for competitive calibrated prediction, not an original contribution.

Three-expert requires-grad capacity is 495,372; joint is 494,736 (within0.2%).
Joint-off has the same stored architecture/initialization but only371,532
first-batch nonzero-gradient parameters. This direct ablation removes effective
capacity. Reallocating widths for the wider reference is another change;
neither comparison alone perfectly isolates a pure architectural causal effect.
All use shared cache bytes and one fixed hyperparameter config, source-summed
NIG losses and the same validation-MAE selection/early-stop policy. Actual
epochs, updates and times are in the audit, not an equal-step claim. Total
training/validation loop time for the18 runs was about2,010 seconds, excluding
startup, cache hashing and artifacts, with GPU also used by desktop apps.

This is a pooled MLP adaptation of pinned public MoNIG. Its Transformers and
author feature representations are replaced; it is not a full reproduction.
V1 reliability used a mean rather than sum of branch losses. The difference
prevents treating cross-study score changes as an isolated joint-expert effect.
[Formula conformance](monig_formula_conformance.json) checks active utility
fusion/loss arithmetic in float64: max fusion difference1.78e-15, loss6.66e-16.

## Missing/noisy pooled features

All18 checkpoints ×11 conditions produce198 saved prediction files.
[The stress audit](native_context_nig_stress_audit_v1.json) verifies exact IDs,
targets and native metrics for every file. Joint-dynamic mean MAE is0.4526 for
text-only,0.5455 for audio+vision, and0.5682 for audio-only. Training had clean
inputs; these are extrapolation diagnostics, not a missing-modality trained
system. Noise is added to pooled features, scaled by training feature STD;
this does not establish raw recording robustness.

## Redundant evidence diagnostic

[The frozen diagnostic](expert_redundancy_diagnostic.json) duplicates every
expert without retraining: mean changes are below1e-10 when recomputed in
float64, but predictive variance falls to0.8450 of its original value and
90% coverage falls from61.67% to56.31%. Identical duplicated evidence contains
no new observation. This diagnoses sensitivity to evidence counting, not
proof of the precise real-world source of undercoverage.

A proper gamma-weighted linear mixture of component Student-t distributions
is a known comparator. It retains the recomputed mean and attains85.11%
coverage, with width1.5820 instead of0.9633. Wider intervals accompany the
improvement; neither original calibration nor a new method is established.
The subsequent [train-only dependence study](NATIVE_DEPENDENCE_FINDINGS_V1.md)
tests equally fitted covariance and scale controls. Stronger point prediction,
author packages, MOSEI, final test and release gates remain outstanding.
