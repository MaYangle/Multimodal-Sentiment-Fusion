# Native sequence gate prior-art and model review

2026-10-14. This is a read-only review of `jevaffect/sequence_gates.py`,
`NATIVE_SCALAR_GATE_DIAGNOSTIC_20261014.md`,
`NATIVE_KUDA_ADAPTATION_DESIGN_20261014.md`, and the existing native sequence,
standardized-fusion, and MOSEI findings. `.kaggle-companion/project.json` is
absent, so this is generic multimodal sentiment work. No code, data, labels,
weights, or model fits were accessed or changed; no TEST data were read.

## Recommendation

The original three-arm placement study by itself does not meet the user's
stated novelty bar. The added six-case candidate is a more defensible
**task-specific architecture hypothesis**: learn a target-free modality policy
that distills the TRAIN-only score-gap oracle, then apply its learned weights
before or after branch normalization. This combines known ingredients, and the
search below did not establish novelty. Proceed only as a gated candidate;
describe the result as a CH-SIMS v2 oracle-guided policy adaptation unless a
broader prior-art review and strong matched evidence justify more.

The six cases can test whether using the learned route helps beyond the
distilled-static control and whether placement matters within the distilled
policy. They do **not** isolate the effect of KL supervision on a learned route:
there is no learned-route/no-KL arm. If the six-by-three budget is fixed, report
the policy-plus-KL package rather than claiming that KL itself caused the gain;
otherwise add a learned-route/no-KL condition before making that causal claim.
The new `distilled_static` policy head is trained by KL and can send gradients
through its non-detached pooled-feature context, even though its route does not
weight the fusion branches. It is unused for fusion/inference, not unused in
training or parameter accounting. State this distinction and each case's
parameter/trainable counts explicitly.

Keep a target-free inference sensitivity evaluation on each selected
checkpoint using the pinned author code's `None` path, which omits per-branch
LayerNorm; report it separately from the paper's `R=1` path. Define the
all-modalities-absent policy row explicitly: a masked softmax over all `-inf`
logits is NaN, so use zero weights and omit/zero its KL term. State whether
`distilled_static` uses `R=1/3` at both train and evaluation. For cross-dataset
MOSEI work, add a matched M-only pretraining control because MOSEI does not
provide the true T/A/V channel sentiment labels needed by the CH-SIMS v2
Stage 1 recipe.

## Prior-art judgment

For a vector `z`, ordinary LayerNorm is

\[
\operatorname{LN}_{\epsilon}(cz)=\gamma\,\frac{c(z-\bar z)}
 {\sqrt{c^2\operatorname{Var}(z)+\epsilon}}+\beta.
\]

For positive `c`, this is exactly independent of `c` when `epsilon=0` and
approximately independent when branch variance is comfortably above epsilon.
The learned affine `gamma` and `beta` are applied after normalization and do
not restore the removed sample-specific scalar. At nonzero epsilon, low
variance branches can retain a measurable effect. This is a direct consequence
of the standard LayerNorm definition, not a new observation about a learned
KuDA checkpoint. [Ba et al., *Layer Normalization* (2016)](https://arxiv.org/abs/1607.06450)
gives the normalization and learned post-normalization gain/bias; the current
near-zero-homogeneity account is also explicit in [TaperNorm (2026)](https://arxiv.org/abs/2602.10408).

KuDA already publishes the score-gap softmax ratio and the
`LN((1+R_m) z_m)` placement for sentiment routing. The present `pre_norm`
case is that known operation in a different, frozen-feature sequence model.
The `post_norm` case is the natural placement ablation that lets branch
amplitude survive its own LayerNorm. It is not a new formula. Conditioning
normalized features and gating residual outputs are established Transformer
designs as well; [DiT (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)
uses adaptive scale/shift after normalization and gates residual updates.

Recent multimodal sentiment work further narrows any originality claim.
[Tri-MCA (Scientific Reports, 2026)](https://www.nature.com/articles/s41598-026-56224-y)
uses per-dimension sigmoid gates before branch LayerNorm and then sample-wise
softmax modality weights over the normalized features. [MISA-GMC (2026)](https://www.mdpi.com/2227-7390/14/1/115)
normalizes modality codes before gating and describes a post-gate LayerNorm
as optional. These are not the same score-error ratio or the same sequence
stack, but they already use both broad placement patterns in multimodal
sentiment fusion. I found no primary source in this narrow review reporting
the exact KuDA error-derived `R_m` with the exact `(1+R_m) * LN(z_m)` operation
on this architecture. That bounded search result is not evidence of novelty;
the operation is best presented as an empirical control or adaptation.

The new policy candidate is more specific than moving the gate, but its
components also have close primary precedents. [D2R (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.207/)
learns sample-dependent soft routing paths for multimodal sentiment detection;
it routes image-text interaction operations and does not distill a
target-conditioned regression-error ratio. [CMAD (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf)
uses teacher/student distillation and modality-combination difficulty weights
for incomplete-modality sentiment, rather than an input-conditioned router
that predicts a per-example continuous fusion weight. [MASER (CVPR FMEA
Workshop 2026)](https://arxiv.org/abs/2606.02463) trains an MLP policy from
oracle per-modality adapter outcomes, but chooses one specialized adapter for
3D VQA rather than softly fusing all three sentiment streams. The closest
general routing-distillation precedent is [TGR-MoE (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Kada_Teacher-Guided_Routing_for_Sparse_Vision_Mixture-of-Experts_CVPR_2026_paper.pdf):
it trains a teacher router, distills its distribution to a student router with
KL plus task loss, and removes the teacher at inference. Its teacher route is
learned from dense vision features and router regularizers, not from a
label-conditioned multimodal regression oracle. Finally, [QAF-NET (Neurocomputing,
28 June 2026)](https://www.sciencedirect.com/science/article/pii/S092523122600891X)
already uses learned queries to distill long sentiment sequences before
asymmetric fusion. The current latent-query stack therefore cannot itself be
presented as a new architecture contribution.

The closest defensible distinction is the **combination** of a native-M
score-error oracle, a target-free learned soft modality policy, and
normalization-aware sequence-branch execution for continuous sentiment
regression. This is a plausible narrow contribution to test, not a confirmed
novel method. A stable VALID gain is necessary but insufficient: the result
must beat its matched distilled-static and pre-normalized controls across the
three seeds, remain competitive with the saved native sequence and fusion
baselines, and survive source-held evidence. Do not claim that the policy
distillation itself is necessary without the learned-route/no-KL control.

## Code and comparison audit

The updated model fixes the two defects from the superseded pooled draft:
there are multiple valid sequence keys, so cross-attention has nonconstant
attention scores, and every modality branch masks padded tokens before and
after its biased projections. The dummy-key handling also returns an exact
zero branch when a modality has length zero, assuming the configured sequence
storage axis is nonempty (`T64/A400/V55`). I found no critical forward-path
bug for those configured shapes.

The three cases are matched in structure and parameter count (2,059,844
total; 1,953,665 trainable in Stage 2, per the registered implementation
brief). `static` supplies equal `R=1/3` during training; `pre_norm` and
`post_norm` use the same differentiable unimodal predictions, detached native
M targets, and `k=0.3`. At evaluation all cases set every present modality's
ratio to `1`, as the paper specifies, and the public predictor rejects a
target. The training ratios sum to one while evaluation ratios do not. This
is an intentional paper boundary but changes the scale regime. In the
post-normalized case, the evaluation multiplier is the same `2` for each
present branch, and the following `mix_norm` largely cancels that common scale;
the learned score-gap differences therefore affect training branch balance,
not inference-time dynamic routing. The current study should describe that
role accurately.

The author repository has a distinct target-free `None` path that skips the
per-branch normalizations. The planned implementation follows the paper's
explicit `R=1` rule, which is the right shared primary boundary for comparing
the three arms, but does not reproduce the repository's evaluation operation.
Run both target-free forward paths on the same selected checkpoints as an
inference-only sensitivity check. Do not mix that source-code path into just
one arm or call the primary table an exact repository replication.

For the proposed policy MLP, the stated `387→64→3` two-linear-layer head adds
25,027 parameters if both Linear layers use biases. If omitted from the three
oracle cases, the six-case table is not parameter-matched across policy/no-
policy conditions; if it is present everywhere, distinguish nominal counts
from active inference parameters. The learned policy's student distribution
must be target-free at VALID/inference, while the detached oracle target may
use only TRAIN M labels. The route head's feature-mean input is not described
as detached, so KL can also update its upstream aligned projectors; include
this in the interpretation of `distilled_static` and in gradient accounting.

This remains a material architecture adaptation. Learned latent queries pool
the aligned sequence streams; the source KuDA design aligns A/V to text
positions, fine-tunes upstream encoders, and retains different backbone
capacity. The branch projectors continue training here while the adapter and
score decoder are frozen. These choices are reasonable for the local cache,
but the result cannot establish equivalence to KuDA's method or reported
training regime.

For practical context, the saved native CH-SIMS v2 findings put standardized
sequence ALMT at VALID MAE `0.408680 ± 0.006690` and standardized pooled fusion
at `0.411524 ± 0.003855`; both must remain external comparators in any later
performance claim. The proposed gate-arm match alone only isolates placement
inside this new architecture. On native MOSEI, current baselines use a
different author-linked feature cohort and ALMT adaptation; true T/A/V
sentiment targets are unavailable, so the CH-SIMS v2 Stage 1 supervision
cannot be repeated. A matched M-only pretraining control is necessary before
interpreting a MOSEI result as support for the same method.

## Primary anchors

- [KuDA, Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.865.pdf)
  and its [pinned author implementation](https://github.com/MKMaS-GUET/KuDA/tree/e60ef6bf070f23855e186c9fd7563902e9a8c056).
- [Layer Normalization, Ba et al. (2016)](https://arxiv.org/abs/1607.06450).
- [Scalable Diffusion Models with Transformers, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf).
- [Tri-MCA fusion, Scientific Reports (2026)](https://www.nature.com/articles/s41598-026-56224-y).
- [MISA-GMC (2026)](https://www.mdpi.com/2227-7390/14/1/115).
- [D2R, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.207/),
  [CMAD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf),
  and [QAF-NET, Neurocomputing (2026)](https://www.sciencedirect.com/science/article/pii/S092523122600891X).
- [MASER, CVPR FMEA Workshop (2026)](https://arxiv.org/abs/2606.02463)
  and [TGR-MoE, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kada_Teacher-Guided_Routing_for_Sparse_Vision_Mixture-of-Experts_CVPR_2026_paper.pdf).
- Local evidence: [normalization diagnostic](NATIVE_SCALAR_GATE_DIAGNOSTIC_20261014.md),
  [adaptation design](NATIVE_KUDA_ADAPTATION_DESIGN_20261014.md),
  [CH-SIMS v2 sequence findings](NATIVE_SEQUENCE_CONDITIONING_FINDINGS_V1.md),
  [standardized fusion findings](NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md),
  and [MOSEI native findings](NATIVE_MOSEI_BASELINE_FINDINGS_V1.md).
