# Source-CV capacity control and error decomposition

2026-10-13. Known-baseline diagnostic, not an original algorithm. The
[plan](NATIVE_SOURCE_CAPACITY_CONTROL_PLAN_V1.md) was saved before smoke/formal
training. Original M labels, cached encoder features, source folds, inner
selection and seeds remain identical to the original point-teacher suite.
Official VALID and TEST are not used in these nine runs or diagnostics.

| TRAIN source-OOF model | Parameters | MAE, seeds13/42/2026 |
|---|---:|---|
| Text MLP, hidden256 | 197,121 | .394769 / .394309 / .391710 |
| Text MLP, hidden640 | 492,801 | .396386 / .396074 / .397116 |
| T/A/V fusion MLP, hidden256 | 492,033 | .419411 / .414890 / .405405 |

Each TRAIN clip has one source-heldout forecast in every seed. Width640 text
matches fusion parameters within.16%. It is worse than narrow text in every
seed, yet still better than fusion. The paired wide-text-minus-fusion MAE
average is −.016710, with5,000 whole-source draws giving95% interval
[-.025688,-.007441]. One per-seed interval (2026) includes zero; the preregistered
diagnostic gate required all seed mean directions and the pooled interval, not
significance in each seed. No final-test inference follows from this conditional
bootstrap of development models.

Nine selected checkpoints, source identities, hashes, exact heldout IDs/targets,
best epochs, actual budgets and CPU/GPU predictions passed
[independent audit](native_source_text_capacity_audit_20261013.json).
Real eight-source smoke and epoch resume preceded221 formal epochs/11,412
updates. Reported teacher loop time is17.84s, excluding hashes/startup/I/O.
The larger text layer changes width as well as capacity; this is sensitivity
evidence rather than a pure causal identification of the AV source effect.

## What the source errors actually show

The [TRAIN-only decomposition](native_source_regression_error_diagnostic_20261013.json)
uses the exact identity `MSE = clip-weighted source-mean error squared +
clip-weighted within-source error variance`. MAE has no such decomposition.
Source IDs identify videos, not independently verified speakers.

Averaging the paired fusion-minus-wide-text components over seeds gives
extra MSE .011707: .003112 from source mean offsets and .008595 from within-source
variation. Source offsets explain26.6% of that increase. Relative to narrow
text, they explain34.0%. Fusion's source-equal pair-difference error is also
higher in all seeds than both text controls. Two tests distinguish constant
source offsets, within-source variation, pair cancellation and clip weighting.

This fails the simple explanation that fusion's extra error is primarily a
fixed source intercept. Do not promote source-mean subtraction or the proposed
same-source pair loss as an established remedy. Pairwise regression and
source debiasing already have strong precedents, as the
[primary-source review](NATIVE_SOURCE_GENERALIZATION_REVIEW_20261013.md) documents.

Next establish a mature debiasing baseline (GEAR or source GroupDRO/DANN with
explicitly disclosed adaptation) on the same source folds and inspect whether
AV representations capture within-source sentiment variation at all. A new
mechanism needs a more specific failure explanation and matched controls;
neither a wider MLP nor generic disagreement gating supplied it here.
The full project goal, author Chinese acquisition, independent final test,
cross-dataset method replication, release and legacy deletion remain incomplete.

## Commands

Requires the verified Chinese feature cache; training is serial on GPU.
Each fold selector intentionally reports an incomplete48-model AnchorCDF
matrix. The nine-model audit proves completion of this separate control stage.

```powershell
python scripts/train_anchor_crossfit.py --config configs/native_source_text_capacity_v1.json --output docs/native_source_text_capacity_fold0_20261013.json --view text --fold 0
python scripts/train_anchor_crossfit.py --config configs/native_source_text_capacity_v1.json --output docs/native_source_text_capacity_fold1_20261013.json --view text --fold 1
python scripts/train_anchor_crossfit.py --config configs/native_source_text_capacity_v1.json --output docs/native_source_text_capacity_fold2_20261013.json --view text --fold 2
python scripts/audit_source_text_capacity.py --config configs/native_source_text_capacity_v1.json --baseline-suite docs/native_anchor_crossfit_formal_20261013.json --output docs/native_source_text_capacity_audit_20261013.json
python scripts/diagnose_source_regression_errors.py --table-dir data/calibration/native_anchor_v1 --capacity-config configs/native_source_text_capacity_v1.json --capacity-audit docs/native_source_text_capacity_audit_20261013.json --output docs/native_source_regression_error_diagnostic_20261013.json
```

Training resume uses `--resume`. Checkpoints and per-sample predictions remain
local ignored artifacts; the project does not redistribute restricted media.
