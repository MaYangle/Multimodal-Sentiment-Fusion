# Known normalization control — registered before preparation/training

The raw author-feature baseline suite remains unchanged and is running serially.
This additional control addresses feature-scale imbalance in the cheap pooled
MLP: GloVe/acoustic/visual coordinates have different units, while ALMT includes
normalization/projection layers. A gain from standardized input cannot be called
a new temporal/fusion mechanism. It strengthens the reference before novelty.

Preserve the same16,265/1,869 IDs, original scores,50-step aligned suffixes,
source object SHA, frozen test IDs and all quality flags. Fit a per-coordinate
mean and population STD of **TRAIN clip pooled means only**, separately per
modality. STD<1e-6 uses scale1. No label is read to fit these statistics. Transform
valid sequence prefixes using these constants, leave padding zero; averaging
the transformed valid prefix yields the standardized pooled mean, up to float32
roundoff. VALID fits no statistic; no TEST feature/label is opened.

Train text and early fusion, seeds13/42/2026, using the unchanged author trainer
and raw-study optimizer/80-epoch/patience12/validation-MAE selection budget.
Same hidden256/dropout.1 and parameter count as corresponding raw MLPs.
Run only after the nine raw baseline runs finish; do not compete for the GPU.
New cache/run roots identify this known preprocessing control separately.
Report native metrics, actual updates/time, and normalization vs raw paired
differences. All current ALMT results remain raw-input adaptations. No original
contribution or end-to-end acquisition-speed claim follows from normalization.
