# AnchorCDF: bounded investigation of conditional probability shape

Registered2026-10-13 before implementation/training. This is a falsifiable
candidate, not a verified original contribution. New text tuning helps strong
scores but hurts neutral/weak/annotation-conflict cases; pooled NIG uncertainty
was poorly calibrated. The project now has competitive point references and
exact full feature-baseline reproduction. The next question concerns confidence
shape while retaining those original-score predictions.

## Question and prior boundary

Can inference-available disagreement between common-target unimodal predictions
improve a predictive CDF beyond feature-only heteroscedastic/mixture calibration,
without changing the native point predictor? Jev inspires inspectable typed
score/probability outputs; no Jev weights or training recipe are used.

This differs from the rejected latent-neutral-atom proposal: no human neutral
state, individual-rater distribution or exact annotation-generation mechanism
is claimed. Compare probabilities of the author's existing evaluation bins,
not incompatible continuous-density and atomic NLLs. The
[independent review](NATIVE_ANNOTATION_OBSERVATION_RESEARCH_20261013.md)
documents conditional quantile mixtures, regression calibration, PDL/DFMU and
the remaining originality risk. The algebra below is not claimed as new.

## Median constraint

Let m be the frozen native point anchor. On CH-SIMS it is the author's clipped
score. Let G be a logistic CDF centered at0 with fixed scale0.05(native units),
s>0 and u=G(m). Set pi_max=0.5/max(u,1-u), pi=(1-1e-4)*pi_max*sigmoid(z),
q=(0.5-pi*u)/(1-pi), mu=m-s*logit(q). Then

F(x)=pi*G(x)+(1-pi)*logistic((x-mu)/s)

has F(m)=0.5 in the continuous interior. Clip/censor the predictive variable
to[-1,1], retaining endpoint probability mass; verify the generalized median
at either endpoint. Numerical tests must cover all signs, near-boundary/zero
anchors, pi extremes, monotonicity, probability sums and finite gradients.
Do not report pi as the probability that a person is neutral: report total
native-bin probability or native regression-error-band probability.

## Matched cases

All cases share the same point models, source-CV predictions/features and
fixed point outputs. Same small32-wide calibration context, initial weights,
configuration and optimization budget where learned; disabled paths and actual
gradient counts must be reported.

|Case|Calibration shape|Purpose|
|---|---|---|
|Global|One fitted logistic scale|Known scalar calibration reference|
|Scale|Feature-conditioned scale only|Conditional capacity reference|
|Mixture|Median-preserving center mixture; feature-only scale/weight|Shape reference|
|Disagreement|Same mixture plus predicted view differences in weight path|Only candidate mechanism|
|Direct-hit|Same feature/disagreement context predicts error-band hit event|Strong event-calibration reference; no full CDF claim|

Disagreement input is the three pairwise absolute and signed differences of
T/A/V view predictions of the common original M target. It is model disagreement,
not true annotation disagreement. Each view is independently trained under
the same declared L1 policy. No real T/A/V annotation inputs/extra labels are
used, keeping the semantics reproducible on MOSEI later.

## Source/data and fitting gates

CH pilot first, raw frozen author-ID-matched MacBERT/WavLM/DINOv2 features,
original M scores and seeds13/42/2026. Rebuild source-group OOF for both the
fusion anchor and all three views. Three TRAIN folds; optimization, inner
selection and held-source OOF must be disjoint. Every TRAIN ID is predicted
once. Existing in-sample residuals or the old MoNIG point model cannot replace
this stronger-anchor OOF table. Source selection parses TRAIN labels only.

Calibration uses OOF TRAIN records only, same source-based inner selection for
all learned cases, fixed max100 epochs/patience12/batch64/AdamW1e-3/WD1e-4,
gradient norm1. Fit any feature normalization on optimization rows only.
Select by common-bin log score on the declared inner selection, first tie.
Freeze the configuration before official VALID comparison. Those scores remain
development evidence because the full point models used VALID selection.
An honest source-transfer result requires an outer held-source pipeline that
rebuilds both point/view OOF and calibration inside its optimization sources;
do not label reuse of an outer source in any lower-stage model as source-disjoint.

Use author CH boundaries[-0.7,-0.1,0.1,0.7] for common-bin probabilities and
the original clipped point/score rules for MAE/Corr. Keep float32 target boundary
semantics. Report bin log score/Brier, P(abs(error)<=0.1) Brier/reliability and
risk-coverage, and50/80/90/95% intervals with widths and actual coverage.
All CDF cases must preserve identical native point predictions/MAE within1e-6.
No TEST tuning, fitting or scoring. This is not a claimed annotation-cell law.

## Promotion/rejection

Disagreement must beat feature-only Mixture in common-bin log score in all
three seeds and in a paired source-cluster interval, improve error-band
calibration versus Scale/Mixture and remain competitive with Direct-hit on
that event. Better than Global alone is insufficient. No improvement in
point MAE is attributed to calibration; point invariance is an explicit check.
Require honest outer-source sensitivity before promotion and then separately
replicate on MOSEI native[-3,3] input/metrics. Check unbounded MOSEI anchors
before choosing a bounded CDF; any clipping control must be shared and disclosed,
with the original un-clipped MAE reference preserved. Never unify label scales.

Reject on inconsistent seed directions, shape/gate gain explainable by added
capacity or teachers/budget, invalid medians/probabilities, latent-atom claims,
or failure against conditional/simple calibration. Only after positive evidence
and a tighter prior-work comparison can a narrow algorithm contribution be
claimed. Current original-method, final-test and release gates remain open.

## Implementation details fixed before calibration fitting

The learned context uses32 units/GELU/dropout0.1. Feature and auxiliary-vector
mean/std fit only on calibration optimization rows. Initialize scale at0.35
and mixture logit bias−2. Direct-hit selects on inner Bernoulli log score,
since it has no full-bin CDF; the CDF families select on inner common-bin log
score. No source-transfer independence is claimed for this inner head split:
other OOF teachers may have seen those sources while generating optimization
rows. An outer pipeline rebuild remains mandatory before promotion.

Two stronger controls are added before any fitting: **Views** reads the same
teachers' three common-M predictions and their absolute values through the
same six-input weight path, isolating extra supervised teacher information
from the pairwise disagreement transformation. **MDN** uses the same context,
one scale and15 learned mixture weights on fixed latent centers−1.4:0.2:1.4,
with censored logistic kernels. This is a known mixture-density reference,
not a new contribution; kernels give continuous support for all legal scores.
The four locations beyond the observed bounds permit endpoint probabilities
above0.5, avoiding an artificially weak boundary-mass reference.
Its distribution median may differ from the frozen primary point. All methods
still report the same original point MAE and error band around that point.
Exact active parameter/gradient counts disclose the small head-size difference.
Disagreement must also beat Views consistently and be competitive with MDN.
Any gain only over a weaker control does not satisfy the mechanism gate.
