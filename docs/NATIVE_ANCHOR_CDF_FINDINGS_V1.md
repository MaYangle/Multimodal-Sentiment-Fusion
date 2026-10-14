# AnchorCDF: disagreement does not improve the registered probability score

2026-10-13. The [pre-fitting plan](NATIVE_ANCHOR_CDF_PLAN_V1.md) tests one
candidate: whether differences between common-M text/audio/vision forecasts
improve the probability shape around a frozen strong fusion point. The median
constraint is ordinary mixture algebra, not a novelty claim. No test labels
are fitted, selected or evaluated.

## Real data and training

The author CSV pins 2,722 TRAIN clips from 129 sources and 647 VALID clips.
All cases use the same author-ID-matched, competition-acquired
MacBERT/WavLM/DINOv2 cache. Complete original-author Chinese media and the
official Chinese feature payload remain unavailable; this does not establish
equivalence to the author's own media/features or reproduce published scores.

Four point views × three seeds × three source folds plus four full teachers
× three seeds produced 48 selected models. Each TRAIN clip is predicted once
out of source for each view/seed. All views learn the original M score using
L1; no true unimodal annotations are used. Full teachers use 116 optimization
sources and 13 selection sources. Their label-free VALID forecasts provide
the T/A/V inputs; the original stronger official fusion model supplies the
frozen VALID anchor. Full-teacher TRAIN forecasts and full-fusion VALID
forecasts are audited but excluded from calibration inputs/anchor respectively.

The preparation audit verifies source splits, IDs, original targets, features,
code, checkpoints, histories and prediction hashes. A separate CPU reload of
all 48 checkpoints reproduces saved GPU predictions within 5.663e-7 and
checks inner selection scores. [Teacher prediction evidence](native_anchor_teacher_prediction_audit_20261013.json).
Teacher loops total 112.69 seconds, 1,270 epochs and 73,872 optimizer updates;
these totals exclude preparation, startup, hashing and artifact I/O.

Calibration normalization fits only 2,395 optimization rows; 327 clips from
13 TRAIN sources select heads. This head split is source-disjoint, but it is
**not a nested source-transfer pipeline**: teachers in other folds may have
trained on calibration selection sources when generating fit-row OOF outputs.
Official VALID is development evidence because its anchor used VALID selection.
Promotion would require rebuilding every stage inside honest outer source folds.

Seven real-sample smoke cases and epoch restoration preceded 21 formal
CPU fits. All learned families use a 32-unit context, dropout .1, AdamW,
maximum 100 epochs, patience 12 and the same batch/gradient limits. CDF heads
select the first minimum TRAIN-inner native-bin log score; Direct-hit selects
its Bernoulli log score. Early stopping gives 242 actual epochs and 9,196
gradient updates in total. Reported times total17.00 seconds: learned epoch
fit/selection loops plus Global fitting/VALID readout, excluding startup/I/O.
Selected learned heads stop at epochs 1–3, revealing weak generalization.

## Official VALID development results

Mean ± sample SD over seeds 13/42/2026. Lower is better for all displayed
scores. Bin probabilities use original author five-class boundaries
[-.7,-.1,.1,.7] with float32 boundary semantics. The hit event is the native
regression event `abs(Y - frozen_point) <= .1`, not classifier accuracy.

| Probability family | Bin log score ↓ | Hit Brier ↓ | Hit log score ↓ |
|---|---:|---:|---:|
| Global logistic scale | 1.373643 ± .000644 | .122591 ± .008940 | .408145 ± .022732 |
| Feature-conditioned Scale | 1.350875 ± .005926 | .122462 ± .007835 | .402368 ± .018268 |
| Feature-only Mixture | 1.326072 ± .002439 | .122007 ± .007787 | .400564 ± .018262 |
| Same teachers, direct Views | 1.327085 ± .002511 | .122089 ± .007760 | .400791 ± .018269 |
| Disagreement candidate | 1.326491 ± .002251 | .121906 ± .007883 | .400270 ± .018461 |
| Direct-hit classifier | — | .139608 ± .009387 | .462150 ± .032721 |
| Known mixture-density reference | 1.309652 ± .019168 | .122580 ± .007371 | .419079 ± .021563 |

The candidate's bin-log-score differences against feature-only Mixture are
positive in every seed: approximately +.000211, +.000687, +.000357.
It improves Hit Brier slightly, but that does not satisfy the predeclared
full-distribution gate. MDN has a better mean bin score; merely beating Global,
Scale, Views or Direct-hit is insufficient. **Reject promotion of this
disagreement mechanism and retain all negative results.**

The [independent21-model audit](native_anchor_calibration_audit_v2_20261013.json)
recomputes every prediction, probability, interval, selected objective and metric.
5,000 paired draws of the120 VALID source-video clusters give candidate-minus-
Mixture bin score +.0004185, percentile95% interval [-.0008396,.0017330].
Hit-Brier difference is −.0001005 with interval [-.0003191,.0001007]. Both
include zero; the gate also requires improvement in every seed. These intervals
condition on the trained/selected models and do not account for training,
selection or repeated development use. MDN competitiveness is **not established**:
no noninferiority margin was registered, so a contrast interval containing zero
cannot prove equivalence. V2 supersedes the first audit's weaker descriptive
competitiveness flag; training and numerical scores are unchanged.

All primary point predictions are frozen, so native MAE stays
.422070 ± .000736 across cases. The known MDN is permitted a different forecast
median; its primary regression point and error-band center remain identical.
For anchored CDFs, numerical inverses must recover the frozen point within1e-6.
The mixture coefficient is not the probability of neutral sentiment or human
annotation conflict.

Candidate nominal90% coverage is 95.98%/94.44%/91.81% with widths
1.6846/1.6382/1.5249. Thus it is not nominally calibrated at all levels.
Full 50/80/90/95% coverage/width, native metrics, ECE, reliability and
risk-coverage outputs remain in the [21-run suite](native_anchor_calibration_formal_20261013.json).
Small ECE alone does not establish conditional or shifted-data calibration.

## Error diagnostics and next research decision

[Diagnostics](native_anchor_calibration_diagnostics_20261013.json) show weak
spread/error Spearman correlations: −.054/−.038/−.042 in TRAIN OOF and
.052/−.014/.009 in VALID. Simple view disagreement is not a validated
reliability estimate. The Mixture head's optimization score falls from about
1.3 to .8, while its source-inner selection score rises from about1.3 to1.7–2.0.
This indicates overfitting; it does not identify its cause or prove that a
lower-dimensional head would rescue the gate.

The same held-source TRAIN rows reveal a stronger separate question. Text
point teachers have MAE .39477/.39431/.39171 versus fusion
.41941/.41489/.40540. Text-minus-fusion paired mean difference is −.01964;
a 2,000-draw whole-source conditional bootstrap gives95% interval
[-.02828,-.01081]. Input dimensions and parameter counts differ, and all models
were selected on their own source-inner rows, so this is diagnostic evidence,
not a causal AV-shortcut result. It cannot be directly compared with the
different official VALID cohort to claim a transfer penalty.

Stop extending the rejected probability gate to MOSEI. Next examine whether
AV helps sentiment prediction within sources but impairs new-source prediction,
using matched-capacity controls and closest public debiasing baselines before
proposing a new source-invariance mechanism. All final-test, original-method,
cross-dataset method replication, release and legacy-deletion gates stay open.

## Reproduction entries

Requires the identical verified Chinese cache and `train,uncertainty` extras.
Existing training runs use `--resume`; table export and audit reject overwrites.

```powershell
python scripts/train_anchor_crossfit.py --config configs/native_anchor_crossfit_v1.json --output docs/native_anchor_crossfit_formal_20261013.json
python scripts/prepare_anchor_calibration_tables.py --suite docs/native_anchor_crossfit_formal_20261013.json --config configs/native_anchor_crossfit_v1.json --output-dir data/calibration/native_anchor_v1
python scripts/audit_anchor_teacher_predictions.py --suite docs/native_anchor_crossfit_formal_20261013.json --config configs/native_anchor_crossfit_v1.json --output docs/native_anchor_teacher_prediction_audit_20261013.json
python scripts/train_anchor_calibration.py --config configs/native_anchor_calibration_v1.json --output docs/native_anchor_calibration_formal_20261013.json
python scripts/audit_anchor_calibration.py --suite docs/native_anchor_calibration_formal_20261013.json --config configs/native_anchor_calibration_v1.json --output docs/native_anchor_calibration_audit_v2_20261013.json
python scripts/analyze_anchor_calibration.py --suite docs/native_anchor_calibration_formal_20261013.json --table-dir data/calibration/native_anchor_v1 --output docs/native_anchor_calibration_diagnostics_20261013.json
```

Independent calibration audit and source bootstrap are complete. An independent
510-file source snapshot passes214 tests and reproduces all21 calibration fits
exactly; [scope and commands](ANCHOR_CALIBRATION_REPRODUCTION_20261013.md)
separate this from point-teacher/encoder retraining. Current root source adds
two source-error diagnostic tests. These are audited development results, not a final test.
