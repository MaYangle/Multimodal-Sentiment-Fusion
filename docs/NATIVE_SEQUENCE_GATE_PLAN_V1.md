# Native sequence policy and gate placement: plan v1

Registered 2026-10-14 before any fit of this model. The last goal turn made
verified progress (12 negative moment fits, original media and audits). This
turn implements a genuinely multi-key sequence model and a target-free deployed
policy. The core hypothesis is that an observable, train/inference-consistent
policy transfers TRAIN score-error guidance into useful modality routing.

This is a research candidate, not a verified original method. LayerNorm scale
invariance, scalar gating, learned queries and teacher/router distillation are
established. The [primary-source review](NATIVE_SEQUENCE_GATE_PRIOR_REVIEW_20261014.md)
identifies close alternatives including KuDA, D2R, CMAD, TGR-MoE and MASER.
Only real controlled gains can justify further task-specific contribution work.
The user explicitly requires that contribution and competitive effect before
final publication; negative studies and baseline adaptations do not satisfy it.

## Inputs and task

Pinned CH-SIMS v2 author CSV SHA256
be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150.
Keep all2722 TRAIN/647 VALID IDs, native M=CSV `label`, [-1,1], and author
metrics. T/A/V labels are `label_T/A/V`, available for all TRAIN records.
Stage1 parses only TRAIN scores. Stage2 VALID exposes M only. TEST labels,
arrays, predictions and model-selection statistics are never instantiated.

Use the same pinned MacBERT/WavLM/DINOv2 cached contextual sequences T64/A400/V55
and their valid lengths as the conditioned ALMT reference. Frozen upstream
encoders, provided transcripts, no extra media/weights/augmentation. Raw input
source is `simsv2_full_macbert_raw_av`; full refit/fusion use
`simsv2_native_sequence_standardized_v1`. Coordinate stats are the saved official
TRAIN clip-CLS/audio-mean/vision-mean scaler from
`simsv2_native_pooled_standardized_v1`. Padding remains zero after normalization.
Prefix truncation and incomplete author-media equivalence remain disclosed;
17 author TRAIN/VALID bytes match their source records, not the complete corpus.
No physical synchronization, word/frame alignment or new temporal-order claim.

## Model

Per modality: token Linear(input,128)/LayerNorm creates H; a LayerNorm/
Linear128-to64/GELU/Linear64-to128 residual adapter creates K. Concatenate
[H,K] (256) and use a masked mean plus decoder256-to64-to32-to1 to predict
the modality's native sentiment score. Each branch owns separate parameters.

Fusion aligns these 256D tokens to128D. Eight learned latent queries start
with the sum of the three valid token means. Three blocks each use separate
four-head cross-attention to all retained valid keys in each modality,
residual/LayerNorm/FFN256, then branch routing, normalized mixing, latent
self-attention and another FFN. The final scalar head uses the query mean.
Missing modalities use a numerical dummy key, then their complete branch is
zeroed; this prevents all-masked NaNs and excludes projection biases. Explicit
nonprefix masks are supported for future native MOSEI inputs.

This differs from KuDA's trainable sequence-length alignment, larger BERT-based
encoders and source inference path. It is a common-encoder structural adaptation,
not a paper reproduction. Actual stage1 parameters353091; total2084871,
stage2 trainable1978692 after adapters/decoders are frozen. Projectors, aligned
features, fusion and policy remain trainable. Cases have identical architecture
and initialization; policy head25027 parameters is dormant in the first three
reference arms and must be enumerated rather than hidden in nominal counts.

The policy takes the three masked means of aligned features and detached
unimodal scores (387), Linear-to64/GELU/Linear-to3, masked softmax. It accesses
features only at inference. TRAIN teacher ratio is
`R*=softmax(-0.3*(predicted_channel_score - native_M)^2)` over present modalities.
The distilled target detaches both the scores and M. It is a soft score-error
preference, not calibrated correctness probability or causal marginal utility.
All-absent rows have zero ratios and zero distillation loss.

## Exact stages and budgets

Seeds13/42/2026. Stage1 selection: shuffle sorted129 TRAIN source video IDs with
NumPy seed20261014, hold the lastceil(0.1*129)=13 sources. Fit coordinate scalers
only on optimization-source raw clip vectors, apply them to valid sequence steps,
then restore padding. Held sources never fit stats. Train three independent
heads jointly for50 fixed epochs, batch32, AdamW1e-4/WD1e-4, summed channel L1,
separate gradient cap1 per branch, no scheduler/early stop. First minimum held
native-clipped MAE selects each channel's epoch count. No official VALID score
participates. Refit from the same initial weights on all official TRAIN, using
the saved full TRAIN scaler for each channel's selected count. Stop updates and
weight decay for a completed branch. Save full joint checkpoint with the three
selected heads. All seven fusion arms reuse that exact checkpoint per seed.

Stage2: freeze adapters/decoders, clone identical fusion initialization, data
order and RNG per seed. Fifty fixed epochs, batch32, 86updates/epoch=4300,
AdamW3e-5/WD1e-5, gradient cap1, no scheduler/early stop. L1 on raw nativeM plus
0.01 asymmetric unit-cosine CPC/InfoNCE (three modalities, same loss in all arms).
Distilled arms add0.1 KL(R*||policy). First minimum official VALID native-clipped
MAE selects a checkpoint, but all arms finish the same4300 updates.

| Case | TRAIN route | Extra KL | VALID/inference route |
|---|---|---|---|
| static | equal ratio, pre-LN | none | present-modality R=1 |
| pre_norm | teacher, LN((1+R)z) | none | present-modality R=1 |
| post_norm | teacher, (1+R)LN(z) | none | present-modality R=1 |
| distilled_static | equal ratio, pre-LN | policy trained identically | present-modality R=1; policy not executed |
| distilled_pre_norm | learned policy, pre-LN | policy | same learned policy, no target |
| distilled_post_norm | learned policy, post-LN | policy | same learned policy, no target |
| learned_post_norm | learned policy, post-LN | none | same learned policy, no target |

The last three learned-route arms are genuine inference-conditioned networks.
Static-distilled isolates executing a policy from its extra supervision; the
distilled pre/post pair isolates placement; learned-post without KL isolates
the teacher distillation contribution. Pure teacher/static arms are published
mechanism controls, with dormant policy params disclosed. Total formal matrix:
3 selection+3 refit+21 fusion runs. New raw/encoder effects and added annotations
are not attributed to the candidate since all primary arms share them.

## Checks, monitoring and promotion

First six model fixtures (mask/padding, nonzero Q/K, equal params, frozen heads,
detached targets, target-free policy, same-KL control) and exact stage resume
fixtures. Then real32-row CUDA select/refit/fusion smoke and saved-checkpoint
resume before formal fits. Use deterministic algorithms, highest FP32 precision,
math SDPA and disabled MHA fastpath; bind these choices to identity.

Bind source/config/plan/model/trainer/CLI/reader/metric/scaler/author hashes,
source split/fit IDs, stage initialization and pretrained checkpoints. Save
best/last optimizer/RNG/history and647 selected predictions, parameter usage,
updates/epochs/peakGPU/time. Track ratio spread, KL, branch variance and frozen
parameter equality. Independently replay selected predictions on CPU and check
source hashes/selection/native metrics and train/eval target boundaries.
Evaluate target-free arbitrary-ratio perturbations and the pinned code's
`None` inference branch (omitted branchLN) as sensitivity diagnostics, not
alternative checkpoint selection rules. Do not interpret tiny pre-LN effects
as a claim about already-trained official KuDA weights.

Candidate=distilled_post_norm. Proceed to source-held/MOSEI work only if its
native MAE is lower in ALLthree seeds than distilled_static, distilled_pre_norm,
learned_post_norm and the strongest teacher-placement arm, plus the existing
conditioned fusion(.411524) and ALMT(.408680). Require candidate-minus-reference
upper95% interval<0 for5000 paired whole-source VALID draws, seed20261014,
averaging seed errors with the same647 unique clips. These are conditional
development intervals, not selection-adjusted/final TEST evidence. Report
all author auxiliary metrics and tradeoffs, not just the best seed.

If a primary gate fails, retain the negative result and stop this registered
candidate; do not change KL/rank/querycount/seeds/temperature on these VALID
outcomes. If it passes, preregister source-held evaluation with fit-only stats
and matched M-only pretraining controls (MOSEI has no true T/A/V labels).
M-only native replication, final frozen TEST, full fresh workflow, source/data
terms, GitHub publication and backup-gated legacy deletion remain required.
No current implementation, proxy or design note claims those gates are complete.
