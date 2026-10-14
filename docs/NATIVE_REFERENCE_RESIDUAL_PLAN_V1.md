# Annotation-reference residual: narrow controlled CH-SIMS test

Registered 2026-10-13 before implementation training. Broader modality-routing
and scalar AV residual claims are covered by prior work; see
[primary-source comparison](NATIVE_RESIDUAL_REFERENCE_RESEARCH.md). This study
does not claim an original contribution in advance. It tests a specific target
and decoding interaction before deciding whether this direction merits more work.

## Core hypothesis and boundary

Use the real TRAIN text annotation t*=label_T and derived d*=label_M-label_T.
Predict t from text only and d from all available features. Structured joint
score is t_hat+d_hat. The hypothesis is that anchoring these two outputs to
annotation-defined roles improves native regression versus equally supervised
independent joint prediction, especially when annotations differ. The difference
is descriptive annotation contrast, not causal AV contribution or reliability.
AV-MC already uses real T/M labels, and recent V-A compensation learns model
error residuals. Only a controlled additional benefit would justify this target
choice. MOSEI lacks label_T; no fabricated annotation or equivalent-variant claim.

## Four matched cases

All four contain the same text projection, shared context projection and three
scalar heads (reference T, correction D, direct joint M), same initialization,
frozen MacBERT/WavLM/DINOv2 pooled feature bytes and2,722/647 author IDs/scores.
Text latent feeds the context trunk in every case, preventing a large unused
text pathway in the flat control. Compute all heads in all cases.

| Case | Main decoder | Real T / M-T auxiliary loss |
|---|---|---|
| flat0 | M head | off |
| flat1 | M head | on |
| sum0 | T head + D head | off |
| sum1 | T head + D head | on |

Text width128; context width304. Total492,691 parameters, within0.2% of the
existing492,033-parameter simple-fusion reference. All cases have same parameters;
only tiny scalar heads lose gradients in disabled conditions. Report actual
nonzero-gradient counts rather than concealing this difference. Shared LayerNorm
and text bottleneck apply to all four; comparisons with old fusion include
these architectural changes, so the flat1 comparison isolates decoding better.

Unclipped L1 on original M scores, plus when enabled0.3*T L1 +0.3*(M-T) L1.
Only TRAIN T labels are parsed; no validation auxiliary loss/input. Validation
uses M only and selects first minimum native MAE. Score clipping/author bins
occur only in CH-SIMS evaluation. No five-class targets or label-scale unification.
Same AdamW1e-4, weight decay1e-4, batch32, max80, patience12, gradient norm1,
MAE plateau scheduler as native reference baselines. Seeds13/42/2026, one config.
Real32/32 smoke, parameter/gradient/coupled-target and exact-resume checks first.

## Gates and next decisions

Evaluate12 formal runs; audit exact M targets/IDs/hashes/selected epoch and
per-example T/D/M outputs. Report MAE/Corr/Acc2, structure and auxiliary effects
paired by seed, and annotation-difference error strata only after training.
Final test remains independent. A gain only over sum0 or flat0 is not enough:
sum1 must consistently improve over equally supervised flat1 and competitive
simple fusion0.4221. A gain from auxiliary labels alone is labeled that way.

If this gate fails, retain the negative result and abandon core promotion.
If it passes, reproduce source-disjoint sensitivity, AV-MC-style multi-task and
OOF base-error residual / recent residual compensation controls before originality
claims. Any later MOSEI teacher-proxy experiment has different reference semantics
and must include matched teacher budget. This test alone does not complete cross-
dataset replication, final test, fresh reproduction, raw inference or publication.
