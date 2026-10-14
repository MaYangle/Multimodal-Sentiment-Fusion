# MOSEI author-linked feature baselines — registered before training

2026-10-12. Prior goal turn made progress: complete MultiBench package acquired,
actual cohort audited, train-only dependence correction rejected, legacy Git
and en/zh checkpoint backups verified. No original mechanism has earned promotion.

Use the complete SDK-recommended MultiBench object, SHA256
`31007381c01dd346945ed8a64c7321239de5aaf1ef05bf112e952848a823186d`.
Keep all published16,265 TRAIN/1,869 VALID and frozen4,643 TEST IDs. Preserve
temporal ID strings, original[-3,3] scores and split membership. Four train rows
are outside current SDK folds; this release variant is disclosed rather than
silently altered. Compare only within this feature/cohort version. No joins to
MMSA ordinal IDs or competition media. Test targets/features are not exported
to the training cache and no test scoring occurs in this round.

## Fixed preparation

Independent implementation follows pinned public loader's deterministic aligned
preprocessing: audio -inf to0; all modalities sliced from first nonzero text
step; regression scores retained. Zero-sum text counts are0 in the audited
package. Unexpected zero-sum text, NaN/+inf, duplicates or changed input version
cause rejection instead of a silent cohort change. Negative start timestamps
are retained as identity metadata; no raw-media crosswalk is invented.

Cast source float64 features to float32 as the public PyTorch loader does;
shift the complete aligned suffix to the front and zero-pad back to50 steps.
All three sequence lengths equal50-start. Interior zero text vectors remain
valid steps. No z-normalization or learned preparation statistic. Text is
GloVe300-D, audio74-D, vision35-D; this is author-linked pre-extracted input.
All-zero visual sequences are recorded as a quality flag, not silently dropped
or reinterpreted as absent without author metadata. Pooled controls mean every
valid step in every modality, including text; there is no CLS token.

## Models and budgets

Text valid-mean +256-wide MLP; concatenated three valid-means +256-wide MLP;
scalar-head ALMT structural adaptation (128-wide projection,8 hyper tokens).
The ALMT adapter uses the shared GloVe inputs rather than fine-tuned internal
BERT and preserves upstream unmasked padding attention. These are common-input
adaptations, not a paper score reproduction or identical-parameter comparison.
Report parameters, actual epochs, updates, loop time and original input form.

All use original-score L1, AdamW lr1e-4/weight decay1e-4, batch32, max80 epochs,
patience12, gradient norm1, validation-MAE plateau scheduler (patience3/factor.5).
One fixed config per model, seeds13/42/2026. Select first minimum VALID native
MAE checkpoint. No additional tuning budget. Smoke32/32 real samples and exact
CPU resume/data/mask checks precede formal serial GPU runs; user apps stay running.

## Evaluation and decision

MOSEI raw MAE/Corr; Acc5/7 use author clipping and NumPy ties-even rounding;
Has0 binary uses >=0 including neutral; Non0 excludes true0 and uses >0. Preserve
native scores; no CH-SIMS rescaling. Audit exact validation IDs/scores, config,
feature/checkpoint hashes, selected epoch and saved per-example predictions.
Keep final test independent. Error analysis determines whether temporal/fusion
representation is competitive enough for the next original hypothesis.

Three-seed results on this acquired cohort do not complete original contribution,
raw-media coverage, cross-dataset method replication, final test, fresh environment,
publication or conditional old-repository cleanup. Those gates remain active.
