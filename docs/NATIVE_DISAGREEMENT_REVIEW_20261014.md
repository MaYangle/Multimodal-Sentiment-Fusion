# Native channel disagreement and modality utility review

2026-10-14. Independent literature review for the next CH-SIMS v2 native-
regression decision. No code, features, cache payloads, or TEST scores were
read or changed. The project scope file `.kaggle-companion/project.json` is
absent, so this is generic multimodal research, not Kaggle work.

## Decision

**No-go on a new disagreement, reliability, or dynamic-routing claim; hold the
pooled KuDA fit.** KuDA already combines unimodal sentiment supervision with a
modality score-gap ratio that guides dynamic fusion. More importantly, the
pinned implementation multiplies a scalar ratio into each cross-attended
branch immediately before LayerNorm. For positive scaling this is exactly
invariant when epsilon is zero and nearly invariant for typical variance with
finite epsilon. A one-token pooled adaptation also makes each attention
softmax constant. The matched pooled KuDA plan is therefore superseded.

The root-owned [TRAIN-only mechanism probe](native_scalar_gate_normalization_20261014.json)
uses 2,722 standardized TRAIN vectors with seed-20261014 normalized random
projections of width 128, not trained KuDA states. Swapping ratio vectors
`[0.1,0.2,0.7]` and `[0.7,0.2,0.1]` changed inside-LayerNorm outputs by maximum
absolute `1.33e-15` at epsilon zero and median relative L2 `1.945e-6` at
epsilon `1e-5`; applying the gate after normalization gave median relative L2
`0.35769`. Singleton-key attention had zero Q/K gradient; two keys produced
nonzero Q/K gradients (`2.18/3.25`). The probe accessed no VALID features or
TEST data and fit no models. It confirms the operation-level concern for this
probe, not a performance bug in the paper or its trained states.

If this line continues, the next design should preserve the existing full
multi-key sequence inputs and compare static ratio, the published pre-LayerNorm
ratio, and a matched normalize-then-gate placement. That last condition is a
placement adaptation with no originality claim. Multi-key inputs restore
nonconstant attention scores but do not remove pre-LayerNorm scale
cancellation; the source's attention masks must also be implemented explicitly
because the pinned cross-attention path ignores its mask argument. The revised
[design note](NATIVE_KUDA_ADAPTATION_DESIGN_20261014.md) records this hold and
the next architecture decision.

## What the current evidence says

The root-owned TRAIN-only annotation diagnostic reports all 2,722 TRAIN
records have the multimodal `M` target in CSV column `label` and the unimodal
targets in `label_T`, `label_A`, and `label_V`. It reports 838
records (30.79%) with a T/A/V sign conflict and 446 (16.385%) with a T-versus-M
sign conflict. Across the three source-held-out seeds, standardized fusion
MAE is 0.376294 on the agreement group and 0.432424 on the T/A/V-conflict
group; the T/M-conflict group is 0.500613 versus 0.393575 overall. The same
diagnostic reports mean absolute label gaps from M of 0.2103 for A, 0.2447 for
V, and 0.2919 for T. These are error strata and annotation relationships; they
do not establish which modality improves a model's prediction on an individual
clip. See the root-owned [TRAIN diagnostic](native_annotation_disagreement_20261014.json).

The current native development anchors remain the standardized fusion model
(VALID MAE 0.411524) and standardized sequence ALMT (0.40868 in the current
timed-covariance comparison). The registered timed covariance candidate at
0.40924 failed its promotion gates, with controls giving inconsistent
directions; retain that stop decision in
[the timed-moment plan](NATIVE_TIMED_MOMENT_PLAN_V1.md). Earlier T/M
annotation-reference decoding also did not establish a structural benefit in
[its findings](NATIVE_REFERENCE_RESIDUAL_FINDINGS_V1.md). These results leave
room for a strong published-baseline comparison, not a positive originality
claim.

## Prior-art boundary

| Work | What it already covers | Consequence here |
|---|---|---|
| [AV-MC / CH-SIMS v2.0](https://arxiv.org/html/2209.02604v1), with [author code](https://github.com/thuiar/ch-sims-v2) | CH-SIMS v2 supplies independently annotated unimodal and multimodal scores. AV-MC has separate M/T/A/V regression heads and applies L1 to each head's matching label; its full method also mixes audio and visual representations and labels. | Using true T/A/V targets as auxiliary losses, or claiming that they newly reveal nonverbal cues, is already covered. Keep AV mixup out of the core comparison unless it is applied to every arm, since it adds another factor. |
| [Self-MM](https://arxiv.org/html/2102.04830v1), with [MIT-licensed author code](https://github.com/thuiar/Self-MM/blob/main/LICENSE) | It generates modality-specific pseudo-labels, trains unimodal tasks beside the M task, and weights those tasks by `tanh(|y_s - y_M|)`. | Difference-weighted auxiliary supervision is established. Replacing pseudo-labels with the available true channel targets changes the supervision source, not the broad idea. |
| [KuDA, Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.865/), with its [MIT-licensed author repository](https://github.com/MKMaS-GUET/KuDA/blob/main/LICENSE) | KuDA pretrains modality sentiment predictors using the dataset's unimodal labels, then forms ratios proportional to `exp(-k |ŷ_m - y_M|²)` and feeds them into dynamic attention. The paper reports setting the ratio to one at inference after learning the weighting behavior. Its appendix explicitly says it uses the CH-SIMS/CH-SIMSv2 unimodal labels for this pretraining. | This is the closest match to “use T/A/V supervision, then route based on disagreement with M.” That formula and the general sentiment-gap-to-modality-weight mechanism are prior art. Its paper reports MAE, but it does not establish that the ratio equals per-example marginal MAE utility for the project's MacBERT/WavLM/DINOv2 frozen inputs. |
| [CMC, 2025 preprint](https://arxiv.org/html/2510.20256v1), with [MIT-licensed author code](https://github.com/gw-zhong/CMC) | CMC learns a per-example consensus router over modality predictions. Its CMC-GT experiment replaces pseudo unimodal pretraining targets with the dataset's ground-truth modality labels and reports separate results on semantic-consistent and semantic-inconsistent subsets. | A learned modality router combined with true T/A/V supervision has also been tested on CH-SIMS v2. The paper's endpoint is classification, so its published scores do not answer the native-regression question; the project's [baseline review](latest_baseline_research.md) already recommends it as a distinct published reference. |
| [ALMT, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.49/), with [author implementation](https://github.com/Haoyu-ha/ALMT) | Language-guided hyper-modality learning suppresses irrelevant or conflicting audio/visual information before fusion. It is an established conflict-handling mechanism and is already represented by the project's sequence baseline. | A generic claim that a new gate resolves modality conflict must beat this stronger native-input reference. |

The project's earlier [residual-reference review](NATIVE_RESIDUAL_REFERENCE_RESEARCH.md)
also identifies AV-MC, CMC-GT, and MTCL as prior examples of real unimodal-label
supervision or mismatch-conditioned branch objectives. The annotation
`label_T/A/V` values are scalar modality-specific sentiment targets. They are
not sensor-quality labels, physical reliability readings, or per-example
annotator-variance distributions; do not interpret closeness to the M score
in `label` as ground-truth reliability.

## Conditional multi-key follow-up

There is no formal fit recommendation until the root completes the next study
decision. If work continues, preserve the existing standardized sequence
cache (`T64/A400/V55`) so modality attention has multiple keys. Use the same
sequence backbone, T/A/V Stage 1 supervision, M output head, optimizer, seeds,
and update count in each arm. Compare three placements with matched parameters:

| Arm | Ratio placement | Purpose |
|---|---|---|
| Static | Fixed equal `R_m=1/3` before each branch LayerNorm. | Reference for channel supervision and branch layout. |
| KuDA pre-LN | Published score-gap ratio before each branch LayerNorm. | Measures the source formula under multi-key attention. |
| KuDA post-LN | The same ratio multiplies normalized branch features: `(1+R_m)*LN(z_m)`. | Tests whether ratio placement preserves branch contribution. |

Multi-key attention restores nonconstant Q/K scores, but the published
pre-LayerNorm scalar still has the scale-cancellation issue. The post-LN
operation is a placement adaptation that needs its own prior-art review; it is
not an original formula. Preserve valid lengths and apply key-padding masks:
the pinned KuDA branch receives a mask argument but does not use it in its
cross-attention calls, so its padding behavior cannot be imported silently.

Use the same TRAIN-only source-held selection for Stage 1 channel heads, then
official VALID `label` (M) only for fusion checkpoint choice. Never use VALID
`label_T/A/V` or TEST. At inference, follow the paper's `R_m=1` target-free
boundary for every arm. Recompute parameter, memory, and update budgets for
the multi-key model, retaining the 492,033-parameter standardized fusion and
3,288,577-parameter sequence ALMT as external anchors. The superseded
one-token 2.05M estimate does not apply.

This three-arm study is only justified if the root's feature-level mechanism
check supports proceeding. If the post-LN placement is selected, describe the
result as a matched KuDA gate-placement adaptation and make no originality
claim. If no stable native-M gain appears over the static arm, stop this line.

## Why this does not yet test modality utility

An annotation gap such as `|label_A - label|` describes a difference between
two score targets. KuDA converts its predicted score gaps into a contribution
ratio. Neither quantity measures the counterfactual change in a particular
model's native-M error when audio, vision, or text is removed. The root's OOF
conflict-stratum result shows that disagreement is associated with higher
fusion error in this cohort; it does not say whether text-only, A/V-only, or a
particular subset would have been more accurate on those same clips. The
LayerNorm cancellation and singleton-attention concern are additional reasons
to check the operation before interpreting a static-versus-dynamic fit.

If later work is considered, first test whether channel gaps predict
source-held-out, cross-fitted leave-one-modality-out changes in native-M loss.
That is the narrow disagreement-versus-utility question. Learning a router
from those risk differences falls within established mixture-of-experts,
conditional fusion, and learning-to-defer families; it would need a more
specific, independently verified contribution before being presented as a new
method. Jev remains design inspiration under [the project scope](JEV_SCOPE.md),
not a locally reproduced algorithm or source of model weights.

## Primary sources and local evidence

- [CH-SIMS v2.0 and AV-MC paper](https://arxiv.org/html/2209.02604v1); [author repository](https://github.com/thuiar/ch-sims-v2)
- [Self-MM paper](https://arxiv.org/html/2102.04830v1); [MIT license](https://github.com/thuiar/Self-MM/blob/main/LICENSE)
- [KuDA paper](https://aclanthology.org/2024.findings-emnlp.865/); [repository](https://github.com/MKMaS-GUET/KuDA); [MIT license](https://github.com/MKMaS-GUET/KuDA/blob/main/LICENSE)
- [CMC preprint](https://arxiv.org/html/2510.20256v1); [repository](https://github.com/gw-zhong/CMC); [MIT license](https://github.com/gw-zhong/CMC/blob/main/LICENSE)
- [ALMT paper](https://aclanthology.org/2023.emnlp-main.49/); [author repository](https://github.com/Haoyu-ha/ALMT)
- KuDA source audit and design-only adaptation plan: [NATIVE_KUDA_ADAPTATION_DESIGN_20261014.md](NATIVE_KUDA_ADAPTATION_DESIGN_20261014.md).
- Local context: [standardized native findings](NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md), [sequence conditioning findings](NATIVE_SEQUENCE_CONDITIONING_FINDINGS_V1.md), [annotation-reference findings](NATIVE_REFERENCE_RESIDUAL_FINDINGS_V1.md), and [source-generalization review](NATIVE_SOURCE_GENERALIZATION_REVIEW_20261013.md).
