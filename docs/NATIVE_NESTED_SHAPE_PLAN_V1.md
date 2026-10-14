# Registered nested probability-shape study

2026-10-13. Written before any real nested point or shape fit. This is a
limited empirical adaptation study, not an original-method claim. The
[prior-art review](NATIVE_ANCHORED_BASIS_REVIEW_20261013.md) establishes that
Bernstein/Beta density bases, censoring and frozen regression calibration
have precedents. Jev motivates structured probability forecasts and efficient
local inference; no Jev weights, training recipe or architecture are reproduced.

## Question and decision

Can independent lower/upper conditional distribution shapes improve honest
source-held native sentiment probabilities while preserving a strong frozen
point prediction? Degree 6 is the sole primary candidate. All cases use the
same frozen standardized fusion representation and point, without modality
teachers or unimodal annotation labels. Native scores and author bins remain
unchanged. The primary objective is proper log score for the five author bins,
not a continuous density likelihood or human-rater latent interpretation.

## Cohort and nested isolation

CH-SIMS v2 author TRAIN: 2722 IDs, 129 source videos, CSV SHA256
`be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150`.
Raw-derived MacBERT/WavLM/DINOv2 feature cache is reused with its existing
provenance limitations. Author raw acquisition is still incomplete. No
official VALID or TEST is read in this study.

Reuse the three outer source folds (seed 20261012). Each TRAIN ID is outer-held
once. Within each remaining-source cohort, shuffle with seed
20261013+1000*fold; reserve ceil(15%) calibration sources; reserve ceil(10%)
of the remainder for point selection; fit points on the rest. Shuffle reserved
calibration sources using the same RNG, reserve ceil(30%) for shape selection,
and fit shapes on the rest. All five roles are source-disjoint within a fold.
Only point_fit fits point feature normalization. Calibration sources enter
neither point fitting nor point selection. Outer sources enter no normalization,
optimization or selection. Separate outer point teachers avoid the older
global-OOF head-only isolation problem. The point export holdout combines
shape_fit, shape_select and outer; these combined holdouts repeat across folds
and must never be described as global once-only OOF.

Point training: 256-unit fusion MLP, GELU, dropout .1, batch32, AdamW lr1e-4,
weight decay1e-4, gradient norm1, max80 epochs, patience12; ReduceLROnPlateau
patience3/factor.5. First minimum point_select MAE selects a checkpoint.
Seeds 13/42/2026. Export frozen 256-wide trunk context, raw native prediction,
clipped probabilistic anchor, original M target and canonical ID per role.

## Cases fixed before fitting

All conditional cases use context+anchor (257 coordinates), fit-only coordinate
normalization, 32-unit GELU trunk, dropout .1 and a case-specific output head.
Different output dimensions and trainable parameter counts will be reported.
The common native bounds are [-1,1]; censor latent distributions to include
endpoint atoms. Median preservation refers to the clipped probabilistic anchor;
native MAE/Corr use the unchanged raw point. Report anchor clipping frequency.

| Case | Definition | Outputs |
|---|---|---:|
| scale | Existing anchored censored logistic | 1 |
| mixture | Existing median-constrained central logistic kernel mixture | 2 |
| bern_uniform | Fixed uniform Bernstein weights, independent margins | 2 |
| bern_tied | Degree6, shared standardized-side simplex, separate margins | 9 |
| bern_d3 | Independent degree3 side simplexes, separate margins | 10 |
| bern_d6 | **Primary:** independent degree6 side simplexes, separate margins | 16 |
| bern_d12 | Independent degree12 side simplexes, separate margins | 28 |
| median_mdn | 15 fixed-center logistic kernels + scale, implicit differentiable median recentering | 16 |
| mdn | Same 15-kernel reference without median recentering | 16 |
| spline | Conditional anchored monotone piecewise-linear CDF map of logistic, knots 0,.125,...,1 | 9 |

MDN centers are -1.4,-1.2,...,1.4 in native CH units. Score the actual
recentered-and-censored distribution, with implicit derivative of its latent
median, never an unshifted surrogate. Spline has four positive increments
on each half, each half summing to .5; G(0)=0,G(.5)=.5,G(1)=1.
It is a bin-aware monotone spline reference trained on common bin likelihood,
not an exact reproduction of Kuleshov isotonic/PIT calibration.

Shape training: CPU, batch64, AdamW lr1e-3, weight decay1e-4, gradient norm1,
max100 epochs, patience12, first minimum shape_select bin NLL; no scheduler.
Same seeds as point teachers, all 10 cases x3 seeds x3 folds =90 fits.
No extra hyperparameter search. Small real-source smoke/resume precedes these.
Checkpoints bind table/ID/code/config hashes, normalization and all RNG state.

## Readout and gate

After selection, export each outer prediction. Merge each case/seed in canonical
TRAIN order, assert once-only source-held coverage. Report five-bin NLL/Brier;
P(|Y-raw_point|<=.1) Brier/log score and reliability; native MAE/Corr; central
50/80/90/95% coverage/width and interval score. Compute CRPS via a fixed
1001-point trapezoid integral on [-1,1], explicitly a numerical approximation.
Native bin boundaries use author float32 [-.7,-.1,.1,.7] and right-closed bins.
Hit truth uses the same float32 absolute-error comparison as earlier studies;
event probability includes censoring atoms at endpoints.

Use 5000 paired source/video bootstrap draws (seed20261013), sharing sampled
sources across cases/seeds. Primary bern_d6 must improve bin NLL over both
mixture and median_mdn in every seed and pooled paired95% intervals must
exclude zero. It must improve both hit Brier/log score without worsening useful
coverage/width, and its shape benefit must survive tied/uniform controls and
degree sensitivity. Report all comparisons; no post-hoc degree selection or
selective winner. If the rich same-median control matches it, seed directions
disagree, or intervals include zero, reject promotion and retain the negative
study. An empirical win alone still does not establish algorithm originality.

Only a surviving mechanism warrants native MOSEI replication, official-native
development runs and frozen final TEST evaluation. MOSEI keeps its native raw
point MAE and [-3,3] bounded CDF anchor control; hit events stay centered on
raw points and anchor clipping must be disclosed. Full project publication and
legacy deletion remain conditional on the original release requirements.
