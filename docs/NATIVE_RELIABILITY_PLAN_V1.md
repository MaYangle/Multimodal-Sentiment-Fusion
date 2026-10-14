# Native reliability factor experiment — v1

Registered before candidate training, 2026-10-12. Previous native baselines are
validation selection results: early fusion MAE 0.4221 ± 0.0007 is the main
point-estimate comparator, and ALMT MAE 0.4271 ± 0.0065 is the public-method
structural comparator. Opposite-sign unimodal annotations identify 117/647
validation records with elevated errors; these are analysis labels, never
inference inputs.

## Hypothesis and nearest work

Test whether real unimodal auxiliary supervision improves the error relevance
of per-modality predictive distributions, and whether learned confidence
fusion has benefits beyond the auxiliary supervision itself. NIG distribution
heads and summation come from MoNIG (NeurIPS 2021), auxiliary sentiment heads
from the established multi-task idea used by AV-MC/SELF-MM. This is a controlled
adaptation hypothesis, not a new claimed MoNIG operator or Jev reproduction.
Jev's relevance is typed numerical decisions with explicitly evaluated
confidence, and efficient discriminative inference; normalized weights do
not themselves prove calibration or interpretation.

## Four conditions and fixed choices

| Condition | Real T/A/V auxiliary targets | Fusion |
|---|---|---|
| S0 G0 | none | equal means, no per-example confidence weighting |
| S0 G1 | none | MoNIG confidence-weighted summation |
| S1 G0 | train only | same equal-mean control |
| S1 G1 | train only | same MoNIG summation |

All heads exist in all four conditions. Every modality distribution predicts
the same multimodal target M; separate scalar auxiliary heads predict the
corresponding T/A/V target when enabled. No auxiliary label enters forward
inputs, and no validation auxiliary loss selects a checkpoint. Auxiliary
heads are not interpretations of modality reliability.

Inputs are exactly the existing hashed MacBERT CLS and length-masked means of
WavLM/DINOv2 train/valid sequences. No ASR substitution, new encoders, external
data, augmentation or teacher pretraining. Three per-modality experts use
hidden width 256/dropout .1 (comparable capacity to the existing early-fusion
head); no pseudo fourth modality is included. This is a three-expert MoNIG
adaptation, not the author's full task-specific backbone reproduction.

Loss: fused NIG NLL plus .01 times evidence regularizer; plus mean valid-branch
NIG loss with the same regularizer; plus .5 times auxiliary L1 only in S1.
The exact loss and equal-mean control math must be verified against the author
paper before any formal run. Report loss magnitudes; a negative density NLL
is possible and does not imply an invalid loss.

AdamW lr 1e-4, weight decay 1e-4, batch32, max80 epochs, patience12, gradient
norm1, ReduceLROnPlateau on native valid MAE. Fixed seeds 13/42/2026, identical
initialization per seed across factors. Selection minimizes native valid MAE,
first checkpoint on a tie. No tuning after seeing the initial four runs;
retain all four and all three seeds. Measure executed updates, epochs, loop
time and parameter count. Smoke runs are prefix32/32, purely flow checks.

## Assessment and decision gates

For this screening round all evaluations use **validation**, which already
serves model selection. Do not open the official final test. Report native
MAE/Corr/Acc2 and author's auxiliary metrics from raw predictions. Compute
Student-t predictive NLL, interval coverage/width at 50/80/90/95%, predictive
variance versus absolute-error Spearman correlation, and selective MAE at
25/50/75/100% retained coverage. These are regression statistics, not softmax
calibration. Intervals and NLL initially use the raw unbounded Student-t
distribution; any range intersection is separately labeled. CH-SIMS scores
are quantized and this density assumption is a limitation.

Evaluate all seven nonempty modality subsets, then clean audio/video pooled
features corrupted by deterministic train-STD-scaled Gaussian noise at .5
and 1.0 STD. Shared seed105 transformations and source IDs across conditions;
no labels determine the corruption. No training augmentation in this round:
this is a stress test of learned reliability, not a noise-trained comparison.
All-missing input must be explicitly rejected rather than assigned confidence.
Feature corruption is not a raw-media acoustic/visual robustness claim.

Use same-supervision G1 vs G0 comparisons to attribute a mechanism benefit,
and same-fusion S1 vs S0 comparisons to attribute auxiliary supervision. Clean
MAE may not materially degrade merely to obtain wider intervals. A mechanism
that beats ALMT but loses to early fusion remains uncompetitive in this setup.
Unsupported benefits or poor calibration are negative results. Only promote
a promising mechanism after stronger matched controls and train-only
cross-fitting/calibration; freeze choices before final test and MOSEI use.

MOSEI has no real T/A/V sentiment annotation counterpart. Any replication of
the S1 effect needs an explicitly independent strategy; S0 distribution fusion
can be tested without pretending to have such labels. Official data/cohort
access remains an independent required gate and must not be silently replaced
with the competition cohort.
