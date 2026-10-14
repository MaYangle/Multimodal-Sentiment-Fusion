# Native evaluation protocol — 2026-10-12

This protocol replaces the earlier competition-first common five-class task.
Author-version continuous scores are training targets. Historical classifier
results remain documented but are not native benchmark results.

## Cohorts and inputs

CH-SIMS v2.0 author CSV: SHA-256
`be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150`;
2,722 train / 647 valid / 1,034 test, 4,403 actual records despite 4,402 in the
homepage. Original scores remain on [-1,1], including the valid -0.1 test
annotation that the old classifier adapter could not map.

First baseline inputs: pinned frozen MacBERT text tokens, WavLM audio sequences
and DINOv2 frame features from the existing verified train/valid raw-derived
cache. This is a common-encoder ALMT adaptation, not native-author-feature
reproduction. The CSV was downloaded anew through the author's Drive link and
is identical to the former competition CSV. Media identity/provenance and
hardware/truncation caveats are in cache metadata. No test features are present.

CMU-MOSEI retains [-3,3] scores. Existing MMSA metadata has 16,326 / 1,871 /
4,659 rows. Current SDK-fold membership inspection flags four training rows
for `-NFrJFQijFE`; resolve and document this cohort discrepancy before claiming
strict CMU-protocol reproduction. Competition segment renumbering is never
used as a media join. Original SDK media is not publicly supplied by the SDK.

## Optimization and selection

Real smoke training and checkpoint reload/resume precede full training.
Baseline suite seeds: 13, 42, 2026. All Chinese baselines use AdamW, learning
rate 1e-4, weight decay 1e-4, batch size 32, max 80 epochs, patience 12,
gradient norm cap 1, and a validation-MAE plateau scheduler. Train loss is L1
on original unbounded model outputs. Best epoch minimizes **native validation
MAE**, with the first checkpoint winning a tie. This is our documented
single-task selection criterion, not AV-MC's combined multi-task loss.

Text baseline: CLS plus 256-wide MLP. Simple fusion: CLS plus masked acoustic
and visual means, concatenated into the same 256-wide MLP. ALMT: author's
projection/hyper-modality/cross-transformer structure with restored scalar
regression head; frozen contextual text replaces internal BERT fine-tuning.
ALMT preserves upstream padding participation. All models share input bytes,
splits and tuning policy; capacity and actual executed epochs/time differ and
are reported. No architecture-specific tuning was done in this first suite.

Only train/valid loaders exist in native training. Test IDs are frozen with a
hash in native manifests; test labels are neither exported nor parsed by the
training loader. A final evaluation tool will be added after architecture,
hyperparameters, uncertainty protocol and calibration are frozen. The official
validation set is a selection set, not independent evidence of final gains.
Existing source-video-disjoint diagnostics remain sensitivity experiments,
never replacements for official splits or pristine tests.

## Metrics and uncertainty

Definitions are independently checked against pinned author code in
[native_metric_conformance.json](native_metric_conformance.json).
[NATIVE_PROTOCOL_AUDIT.md](NATIVE_PROTOCOL_AUDIT.md) gives source links and
boundary details. All errors use each dataset's own label scale.

| Metric | CH-SIMS v2.0 | CMU-MOSEI |
|---|---|---|
| MAE (lower) | clip prediction and truth to [-1,1] | original un-clipped scores |
| Pearson Corr (higher) | after native clipping | original scores |
| Binary accuracy/F1 (higher) | zero is negative, threshold >0; weighted F1 | Has0: >=0; Non0: remove true zeros, then >0; weighted F1 |
| Weak binary accuracy | true score within [-.4,.4] | not transferred |
| Acc3/Acc5 | author left-open/right-closed boundaries at -.1/.1 and -.7/.7 | Acc5 clips [-2,2], then NumPy ties-to-even rounding |
| Acc7 | not transferred | clips [-3,3], then NumPy rounding |
| R² | author's R_squre field | not in this evaluator |

Full precision is retained in stored artifacts. Undefined correlation or
empty-support metrics are JSON null with support counts. A five-class softmax
is not regression uncertainty. Proposed distributional methods must report
predictive coverage and width at declared nominal levels, held-out residual
ranking/risk-coverage, and distribution-appropriate NLL. Fit calibration only
using a predeclared held-out calibration part or training cross-fitting; final
test data is untouched. Calibration cannot be credited as new fusion accuracy.

## Controlled research gates

Choose one hypothesis from real baseline errors and closest prior work.
Current Chinese selection errors identify opposite-sign unimodal annotations
as a difficult subgroup (117/647); their labels are analysis targets, never
inference inputs. This motivates, but does not prove, reliability fusion.

Mechanism/control comparisons must share encoder bytes, inputs, optimizer
search and comparable budget, include a mechanism-removal control, and isolate
extra unimodal supervision/teacher pretraining. Count additional teacher compute.
Noise and missing-modality conditions share deterministic transformations and
train-derived feature statistics. Preserve negative results; three-seed
replication, final test, MOSEI replication and a fresh-environment run are
required before release claims.
