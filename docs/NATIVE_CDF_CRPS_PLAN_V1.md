# Native scalar-score CRPS control study

2026-10-13. Registered before any CRPS-head fitting. The90-fit five-bin
shape study is rejected and remains unchanged. The eleven-TRAIN-score PMF
plan was deferred before fitting after review of an existing off-grid
author-metadata annotation. This study preserves original scalar scores and
the complete native range, with no target rounding, support-based exclusion
or test-driven changes. It is a known proper-score adaptation, not a new
algorithm claim. [Research decision](NATIVE_SCORE_RESOLUTION_REVIEW_20261013.md).

## Question and shared inputs

Does replacing coarsened-bin supervision with native scalar-score CRPS
improve honest source-held distribution and error-confidence predictions?
Keep the existing distribution families and frozen points. Reuse the audited
nine point teachers and27 tables from `runs/native_nested_shape_points_v1/suite.json`.
TRAIN only2722 IDs/129 sources, seeds13/42/2026 and the same three outer folds
with point_fit, point_select, shape_fit, shape_select, outer source isolation.
Both calibration roles remain excluded from point fitting/selection/scalers.
All point outputs and author-native MAEs remain unchanged.

Four fixed cases: **scale**, **mixture**, **median_mdn**, **bern_d6**. Degree6
remains the shape candidate; no additional degree or architecture search.
All four use the unchanged257-coordinate context+anchor input, fit-only
normalization,32-unit GELU/dropout.1 heads, identical common encoder bytes
and preprocessing. Shapes retain the bounded probabilistic anchor as a
generalized median and censor to[-1,1]. No unimodal labels or additional
teacher inputs. Mathematical constructions and CRPS have established precedents.

## Objective, training and numerical contract

Fit and select the **final censored CDF** using

CRPS(F,y) = integral[-1,y] F(x)^2 dx + integral[y,1] (1-F(x))^2 dx.

This accepts every original finite scalar y in[-1,1], including off-grid
scores. Do not evaluate an unshifted MDN surrogate or uncensored density.
Use differentiable float64 Gauss-Legendre quadrature,48 nodes on each of the
three intervals partitioned by{-1,min(anchor,y),max(anchor,y),1}. Zero-length
intervals contribute0. Anchors are detached; original labels enter the
integration limits. Verify against analytic limiting cases and an independent
high-resolution numerical integral before real-source smoke.

CPU, batch64, AdamW lr1e-3, weight_decay1e-4, norm1, max100epochs, patience12,
first minimum shape_select CRPS, no scheduler or search. Four cases xthree
seeds xthree folds =36 neural fits. Real-source smoke/backward/save/resume
precedes fitting. Checkpoint identities bind all code/config/plan/point-table
and ID hashes, fit-only scalers and RNG state. Completed selected models
must independently compare48/96/192-node scores on held-source readout;
report numerical discrepancies. If mean differences exceed1e-4, treat the
quadrature as unverified and repair uniformly before repeating the study.
Do not selectively raise nodes only for a losing case. The older1001-point
trapezoid score remains a separately labeled common readout, not exact CRPS.

Five-bin readout permits only float64 roundoff: raw CDF values must stay
within[-1e-12,1+1e-12] before clamping to[0,1]; negative bin mass below−1e-12
or row-sum discrepancy above1e-10 raises an error. Report maximum CDF/bin
corrections; tiny negative masses may be clamped and rows renormalized.
This applies equally to allcases and does not modify the CRPS loss/CDF family.

## Event and evaluation alignment

CH author point MAE/Corr clip predictions to[-1,1]. Therefore the **primary
confidence event here is |Y-clipped_point|<=.1**, with original float32 target
and clipped-point subtraction for truth. Include censoring atoms for the
predicted event. Raw-point hit probability/error is an explicitly auxiliary
readout. This is a pre-fit correction to match the author-scored regression
decision, not a replacement of the failed earlier raw-point gate. Old results
remain negative. MOSEI's unbounded native point protocol must be separate if
later investigated; do not transfer this clipping convention.

Report CRPS, primary native-hit Brier/log score/reliability, raw-hit readouts,
five-bin NLL/Brier, author-native point scores, raw unclipped MAE/Corr, anchor
clipping counts, central50/80/90/95% coverage/width and proper interval scores.
No claim about individual-rater uncertainty follows. No discrete support
lookup is allowed to reject a valid native scalar target in this CDF study.

Merge every case/seed to canonical2722 IDs once. Independent CPU audit must
recompute final CDF/bin/event/quantile/CRPS forecasts, numerical accuracy,
normalization, source roles, first-min checkpoint selection and stop budget.
Use5000 paired129-source bootstrap draws,seed20261013, shared across cases
and seeds. Intervals condition on these fitted models. Compare corresponding
older NLL-fitted checkpoints only as an objective-change diagnostic; every
primary architecture comparison uses CRPS fitting and selection.

## Gate and remaining scope

Degree6 must improve CRPS against **both mixture and median_mdn** in allthree
seeds with pooled paired95% intervals excludingzero. It must also improve
primary native-hit Brier and log score against both in allthree seeds with
paired intervals excludingzero. Report90% interval score as a required
additional proper-score check: no positive pooled deterioration against either
reference. Report scale and all auxiliary comparisons without post-hoc case
promotion. If any required comparison fails, stop this probability-shape
line and retain the negative evidence. A win supports an empirical adaptation
only, not originality. Original mechanism, MOSEI method replication, final
official TEST, final-method fresh reproduction, GitHub release and conditional
legacy cleanup remain separate unmet requirements. No official VALID/TEST
labels are read for this development decision and no paid compute is needed.
