# Native-support probability baseline plan

**Status: deferred before any fits.** The subsequent review rechecked an
existing author-metadata off-grid annotation recorded in the earlier audit,
without opening held labels. The11-level PMF is a TRAIN-only diagnostic;
do not promote it as a complete native-score observation model or round/drop
off-grid targets. Its numerical prototype supports exact CRPS on any bounded
scalar target via the energy identity, but its discrete NLL rejects off-support
targets explicitly. The next complete-protocol study should train the bounded
CDF references using native-score CRPS, as described in the amended review.

2026-10-13. Registered after the negative90-fit shape study and before any
native-support probability fit. This is a **known-method baseline repair**,
not an original-algorithm proposal. The
[primary-source review](NATIVE_SCORE_RESOLUTION_REVIEW_20261013.md) recommends
stopping arbitrary shape-family search: five-bin likelihood coarsens the
released native scores and does not identify within-bin regression events.

## Single question

Does proper supervision of the actual released score improve source-held
regression-error probabilities compared with the rich same-input reference?
Keep the frozen point unchanged. Do not turn a baseline win into a claim
of novelty, human-rater uncertainty or better action utility. No action policy
has been specified. An original mechanism remains a separate unresolved gate.

## Data and reuse

CH-SIMS v2 author TRAIN only:2722 IDs/129 sources, CSV
`be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150`.
Native support is float32[-1,-.8,-.6,-.4,-.2,0,.2,.4,.6,.8,1], exactly equal
to the11 unique TRAIN target values. Fit on original M scores; support indices
are an internal likelihood lookup, not a relabeling of the native regression
task. Preserve author5-bin and clipped CH point metrics. MOSEI is excluded
from this baseline study; its native label semantics are different.

Reuse the completed/audited nine point models and27 tables from
`runs/native_nested_shape_points_v1/suite.json`, with unchanged five source
roles, scalers, seeds13/42/2026 and canonical IDs. Outer sources enter no
point/probability optimization, normalization, calibration or epoch selection.
Reuse is allowed because this is a fixed TRAIN development diagnostic, not
a new untouched final test. Official VALID/TEST remain unused. No new encoder
data, teacher compute or backbone tuning is introduced.

## Fixed cases and exact final forecasts

All neural cases read the same257 context+clipped-anchor coordinates and use
shape_fit-only coordinate normalization, 32-unit GELU trunk and dropout.1.

1. **ordinal**: classical proportional-odds PMF over11 ordered scores.
   A conditional scalar location and10 globally ordered cutpoints define
   cumulative probabilities through the logistic link. Represent9 positive
   cutpoint gaps with softplus+1e-4 and center the cutpoints to mean0, removing
   location/cutpoint offset nonidentifiability. Initialize gaps at.2.
2. **ordinal_projected**, primary: the same PMF with per-row KL I-projection
   enforcing F(anchor-)<=.5<=F(anchor). Score the final projected PMF during
   fitting and selection, not its unconstrained surrogate.
3. **median_mdn_cells**: refit the existing16-output median-recentered MDN
   with masses integrated over10 midpoint cells of the11 native scores.
   Include endpoint censoring mass in the two edge cells. Train/select on
   exact native-score PMF NLL, not continuous density or five-bin NLL.
4. **median_mdn_cells_projected**: the same integrated reference with the
   same discrete KL constraint, scored after projection. Quantization of a
   continuous median can otherwise move the discrete median; this is the
   rich fair same-median comparison.
5. **frequency_prior**: shape_fit class counts with fixed.5 pseudocount per
   score, no feature/label reweighting or selection. Three unique fold priors
   produce nine seed/fold readouts because frozen points and hit events vary.
6. **frequency_prior_projected**: the same prior projected using each anchor,
   a cheap same-median reference with no learned fusion-context parameters.

All PMFs use a1e-12 numerical floor followed by normalization before optional
projection; record this convention. For a positive PMF, split scores below,
equal to and above the anchor. If lower mass>.5, rescale lower to.5 and its
complement to.5 proportionally. Otherwise, if upper mass>.5, rescale upper
and its complement similarly; otherwise retain the PMF. This is a standard
convex information projection, not a new mechanism. At off-grid anchors,
half-mass on each side makes the anchor a possibly nonunique generalized
median. Discrete inverse-CDF quantiles must stay on the native support;
do not force the reported.5 quantile to an off-grid anchor.

Initialize neural trunks identically within a seed and report all parameters
and actual budgets. Four neural cases xthree seeds xthree folds =36 fits;
six cases xthree seeds xthree folds =54 forecasts, plus three unique fold
frequency estimates. Use CPU/batch64/AdamW lr1e-3/weight_decay1e-4, norm1,
max100epochs/patience12, first minimum shape_select11-score NLL; no scheduler
or hyperparameter sweep. Save best/last/history/predictions, normalization,
code/config/plan/table/ID hashes and RNG state. Real-source smoke/resume must
precede formal fitting. Any numerical repair precedes formal freeze.

## Metrics and gates

Primary: exact11-score NLL and discrete CRPS, computed as the sum of support
gap times squared cumulative-probability error over the10 inter-score gaps.
Also report author5-bin NLL/Brier by summing PMF categories, native point
metrics, and P(|Y-raw_point|<=.1) using the same float32 support-distance
comparison as hit truth. Report clipped-point hit probabilities separately.
Return discrete50/80/90/95% equal-tail coverage/width and interval score;
interval endpoints must be support values. PMF likelihood predicts the
released aggregate clip score, not an individual-rater distribution.

Merge each case/seed to canonical2722 IDs once. Independent CPU recomputation
checks final PMFs/events/quantiles/scores, point invariance, source isolation,
first-min checkpoint selection and stopping budget. Use5000 paired source
draws with seed20261013 and common draws across allcases/seeds. Confidence
intervals are conditional on these three fitted models, not uncertainty over
all possible training seeds. Report complete cases, including projection cost.

The primary must improve11-score NLL over frequency_prior_projected and
median_mdn_cells_projected in allthree seeds and pooled paired95% intervals
must exclude zero. It must improve both raw-point hit Brier/log score over
those controls; CRPS and coverage/width must remain useful. If any required
comparison fails, stop this shape/probability line rather than select another
case post hoc. A win establishes an empirical baseline advantage only.
Original algorithm validation, independent final TEST, MOSEI method replication,
full fresh-environment reproduction and publication remain separate open
requirements. Do not delete the old repository before release/backup gates.
