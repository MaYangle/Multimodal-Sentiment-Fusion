# Candidate hypothesis (not a novelty or performance claim)

Primary objective remains higher five-class accuracy under identical data/backbones/training budgets. Robustness and calibration are additional endpoints.

## Working hypothesis
Temporal disagreement is not uniformly harmful: a confident text branch can disagree with informative prosody/visual evidence. A gate trained to estimate the validation-like benefit of retaining a modality may distinguish useful disagreement from corrupted evidence better than a gate based only on entropy or global consensus.

Candidate mechanism: retain sequence representations; produce per-modality five-class posteriors in temporal windows; represent ordinal disagreement by distances between cumulative class distributions. Supervise a small utility head with leave-modality-out loss differences estimated from training-only held-out folds. Use this learned utility to modulate a residual multimodal correction. Keep the text-only path available. Do not use test labels or validation labels to train the utility head. Cross-fitting cost counts in the budget.

This is provisional until real baseline error analysis: it adds both modeling and supervision cost, and may lose useful signal when time windows are not aligned. Native unaligned feature caches do not provide guaranteed physical timestamp alignment; relative sequence positions must not be described as exact synchronized segments. Real timestamped raw-video caches are needed to support that claim.

## Closest-work distinctions to test
MulT/ALMT: temporal interaction and language-guided filtering are prior work.
SELF-MM/CMC: modality-specific supervision and consensus routing are prior work.
TMSON and other uncertainty/ordinal methods: ordinal uncertainty is prior work.
Thus a reliability gate, a five-way softmax, temporal attention, or temperature scaling alone is not claimed as original. Potential contribution is narrowly the extra predictive value of training-only intervention utility for local ordinal disagreement. Search and ablations must show whether that distinction is substantive.

## Pre-registered controls
- Identical cached backbone features and five-class split protocol for text, legacy, ALMT, candidate.
- Candidate without utility supervision, with global rather than local disagreement, and with no temporal information.
- Compare against entropy/global-JSD gates, not just no fusion.
- Keep parameter count and tuning attempts comparable; count extra cross-fit passes.
- Inspect validation clean accuracy and corruption NLL first. Only promote if clean five-class accuracy improves over the strongest reproducible baseline; report all failures.
- After selection, three fixed seeds and held-out test evaluation; calibration separately fitted from a fixed validation calibration partition.

Do not implement expensive variants before the cached baselines establish the actual bottleneck.

## Current evidence boundary (2026-10-12)

`jevaffect.jev_temporal` implements the candidate's masked time windows, five-class modality posteriors, ordinal-CDF disagreement, and gate modes for later controls. It now has train-only, source-group-held-out intervention targets on the 339-row pilot; full-cohort utility targets and a fair budget-matched comparison remain pending. On the 339/43 biased pilot, its unsupervised-gate version averaged 44.19% validation accuracy across three seeds and overfit; ALMT on the same subset reached 53.49%, with different schedules. This is negative preliminary evidence for the unguided gate. The claimed mechanism and originality remain unverified until full-cohort, matched-backbone/budget comparisons and the planned utility/temporal/JSD/entropy/calibration ablations are complete.

The seed-13 pilot compared the same candidate architecture and training protocol with and without these targets. Both selected validation accuracies were 44.19% on 43 rows. Utility supervision lowered NLL from 2.76 to 1.75 but also lowered Macro-F1 from 36.79% to 26.86%. Three extra teacher fits are not budget-matched to the controls. See `docs/jev_utility_pilot_ablation.json`; the method is not validated as an accuracy improvement.
