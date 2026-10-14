# KuDA adaptation review: hold after mechanism check

2026-10-14. **Status: HOLD after the TRAIN-only mechanism sanity check.** This
is a design correction, not a request to fit the published adaptation. No model
fit, code change, cache read, pretrained-weight download, or TEST access was
performed. Nothing here claims a new router or an exact reproduction of KuDA.

## Decision in scope

**HOLD formal fitting.** The root-owned TRAIN-only proxy check confirms that
the proposed one-token pooled ratio path is nearly cancelled by its
pre-LayerNorm scaling, and the singleton attention keys eliminate Q/K score
gradients. These proxy results do not establish KuDA's behavior on learned
states or a performance bug in the published sequence method.

If research continues, the next design should use the existing multi-key
standardized sequence inputs (`T64/A400/V55`) and compare matched static,
published pre-LayerNorm, and normalize-then-gate placements. Multi-key inputs
restore nontrivial attention-score gradients; they do **not** remove the
LayerNorm scale cancellation. The scalar-gate placement is an empirical
baseline question, not an originality claim. The one-token pooled design
below is retained as an audit trail only and must not be fitted.

The dataset uses `label` for the multimodal M target. `M` is the semantic
channel name used inside the model and loader. The modality-isolated targets
are CSV columns `label_T`, `label_A`, and `label_V`; the reader may map these to
internal keys `T`, `A`, and `V`. Do not implement a CSV column named
`label_M`.

## Source snapshot and reproducibility

The public author repository is `MKMaS-GUET/KuDA`, default branch `main`, with
current observed commit
[`e60ef6bf070f23855e186c9fd7563902e9a8c056`](https://github.com/MKMaS-GUET/KuDA/commit/e60ef6bf070f23855e186c9fd7563902e9a8c056),
dated 2024-11-27. This was the newest commit returned by the repository's
descending commit listing on 2026-10-14. The repository license at that
revision is MIT (Copyright 2024 Massive Knowledge Management & Service). The
file identifiers below are GitHub Git-blob SHA-1 values fetched at that exact
commit; they pin the small source files used for this review. No pretrained
weights were fetched. The paper's full method uses
`bert-base-chinese`, which is outside this pooled-feature adaptation and has
separate model terms.

| Pinned source file | Git blob SHA-1 |
|---|---|
| [`models/OverallModal.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/models/OverallModal.py) | `0a4d9407eda7bc5072569b0edaeb22168ecde936` |
| [`models/DyRoutFusion_CLS.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/models/DyRoutFusion_CLS.py) | `700a29219c05c8fb48eb8ab3de55595045e3ab5c` |
| [`models/Encoder_KIAdapter.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/models/Encoder_KIAdapter.py) | `1fccf3efdf83e092fca68c2805615caaa629bef3` |
| [`core/utils.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/core/utils.py) | `7bb42fce64cb6c372f9ee6d67e959d96d5903a9b` |
| [`core/dataset.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/core/dataset.py) | `451f1781519a2360901ca81814dacf7972965501` |
| [`pretrain.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/pretrain.py) | `7ac344e6538ce4809886f0c15f7105f65b87ed4e` |
| [`train.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/train.py) | `0c6d0b70439be3eae16d0377b5c1f8e58e211dbe` |
| [`opts.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/opts.py) | `75c707e42210d1ce0796ec869e7c2c4121d5dbe9` |
| [`LICENSE`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/LICENSE) | `9b7f8f3c6ee0dd1a3c6f43f36f44f3857a4e3504` |

## Paper mechanism and repository differences

The [Findings of EMNLP 2024 paper](https://aclanthology.org/2024.findings-emnlp.865.pdf)
defines the training-time sentiment ratio for modality `m` as

\[
D_m = \exp\{-k(\hat y_m-y_M)^2\},\qquad
R_m = D_m / \sum_{q\in\{T,A,V\}}D_q.
\]

The ratios are normalized across modalities and enter each dynamic-attention
block **after** cross-attention has updated the current fused query from one
modality's key/value representation. The paper applies
`LN(F̃_m + R_m F̃_m)` per modality, sums the routed branches with the previous
fused representation, then applies self-attention and an FFN. The final M
score comes from an MLP over the final fused feature; KuDA does not average
modality score predictions at the output. Its CH-SIMSv2 table specifies
`k=0.3`, `alpha=0.01`, three dynamic blocks, fusion width 256, batch 32,
learning rate `3e-5`, AdamW, and 50 epochs. The paper says inference sets each
`R_m=1`, so the M target is absent at inference.

The pinned repository implements the same normalized exponential formula but
does **not** use the CH-SIMSv2 paper setting: `OverallModal.forward` passes
`k=0.1`, while `core/utils.py` gives the helper a default `k=2.0`. In
`train.py`, `copy_label = label.clone().detach()` passes the training M target
to the ratio function. The predicted modality scores are not detached, so
gradients from the ratio can flow through the frozen decoder operations into
the still-trainable modality encoders. The target is detached and has no
gradient.

The paper/code inference boundary differs too. The paper sets all ratios to
one. The public `evaluate` and `test` functions call the model with `None`;
`DyRoutFusion_CLS.py` then takes a separate unweighted-sum branch instead of
the per-modality `LN((1+R_m)F̃_m)` path. That branch omits the per-modality
LayerNorms and is not the paper's explicit `R_m=1` calculation. The blueprint
below follows the paper boundary and uses `R_T=R_A=R_V=1` at inference for
every comparison arm.

## Mechanistic hold: normalization and singleton attention

The pinned implementation has no other use of `senti_ratio`: the three ratios
are consumed only in [`DyRout_block.forward`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/models/DyRoutFusion_CLS.py#L106-L118),
where each `cross_f_m` is replaced by
`LN_m(cross_f_m + R_m * cross_f_m)`. The helper returns a per-record ratio
normalized across T/A/V in [`core/utils.py`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/core/utils.py#L104-L114);
`R_m>0`, each `R_m<1`, and the three sum to one. Because each ratio is a
positive scalar broadcast across token and feature dimensions, the multiplier
before LayerNorm is `c=1+R_m`, between one and two.

For a LayerNorm input `z` with last-dimension variance `v`, PyTorch's
documented formula with `eps=1e-5` is

\[
\operatorname{LN}_{\epsilon}(c z)
= \gamma (z-\mu)\frac{c}{\sqrt{c^2 v+\epsilon}}+\beta
= \gamma (z-\mu)\frac{1}{\sqrt{v+\epsilon/c^2}}+\beta.
\]

With `eps=0`, the output is exactly invariant to every positive scalar `c`,
including `c=1+R_m`. Trainable affine `gamma,beta` do not remove that
invariance. With finite `eps`, the scalar path remains only through the
`epsilon/c²` term. Its scale derivative is
`d[c/sqrt(c²v+eps)]/dc = eps/(c²v+eps)^(3/2)`. When `v` is around one this is
on the order of `1e-5` before multiplication by the centered activation and
learned `gamma`; when `v` is near `eps/c²`, the effect can be material. A
large learned `gamma` can amplify the finite-epsilon difference but cannot
break the exact `eps=0` invariance. Exactly constant `z` still has no ratio
effect because `z-mean(z)=0`. The actual variance distribution, trained
`gamma` values, and resulting branch changes on TRAIN are not measured here.
The [official PyTorch LayerNorm documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.LayerNorm.html)
confirms the default `eps=1e-5` and last-dimension normalization.

This cancellation concerns the source's ratio placement before per-modality
LayerNorm. It is a mathematical condition on that operation, not a claim that
KuDA's published scores are invalid or that every input has unit variance. In
[`OverallModal.forward`](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/models/OverallModal.py#L20-L30),
`ŷ_m` remains attached and the M target is detached before ratio computation.
Thus a differentiable ratio may send a gradient through `ŷ_m` and `R_m`, but
when `d LN(cz)/dc` is small that route gradient is small too. The existing
correlation loss and downstream M loss still update other parameters; they do
not make the pre-normalization scale itself non-invariant.

The proposed one-token adaptation has an additional limitation. Its cross
attention has one key/value per modality, so each softmax is over a single
entry and is identically one. Query/key projection weights therefore have no
gradient through attention scores. The one-token self-attention in the fusion
state has the same property. Residual paths, value projections, output
projections, LayerNorms, and FFNs remain active, but this no longer performs
content-based attention selection. This is specific to the pooled adaptation;
the original sequence model aligns audio/visual keys to multiple text-length
positions.

In the proposed 128-dimensional pooled model, the nominal dynamic stack
contains about 393,216 Q/K projection weights across three blocks, each of
which has zero attention-score gradient at singleton length. Those weights
remain in the nominal count; they should not be used to argue that the
adaptation supplies 2.05M effective attention parameters. This zero-gradient
claim is specific to the proposed singleton pooled adaptation; the pinned
author sequence inputs have multiple aligned key/value positions.

A read-only public issue search of the pinned repository found no issue for
`ratio`; the `LayerNorm` search returned [issue #6](https://github.com/MKMaS-GUET/KuDA/issues/6),
which concerns a pretraining feature-dimension mismatch. This describes only
that issue tracker. It does not establish that the normalization question has
not been discussed elsewhere.

The root-owned [TRAIN-only mechanism probe](native_scalar_gate_normalization_20261014.json)
then tested a ratio swap from `[0.1,0.2,0.7]` to `[0.7,0.2,0.1]` on 2,722
standardized TRAIN vectors after seed-20261014 width-128 normalized random
projections. For the inside-LayerNorm placement, maximum absolute output
change was `1.33e-15` at `eps=0` and median relative L2 change was `1.945e-6`
at `eps=1e-5`. Applying the same coefficients after LayerNorm gave median
relative L2 change `0.35769`. With a singleton key, Q/K gradient norms were
exactly zero; with two keys, they were 2.18 and 3.25. This is an activation
and gradient probe on random projections, not trained KuDA states or evidence
of performance, novelty, or an author-paper bug. It accessed no VALID feature,
sentiment score, or TEST data and fit zero models. It makes the current pooled
formal fit a HOLD.

The two training stages also require a clean adaptation rather than direct
execution of the public scripts:

| Area | Paper | Pinned code | Superseded pooled draft (not to fit) |
|---|---|---|---|
| Stage 1 | Train modality encoders/adapters/decoders on unimodal labels; retain the best validation checkpoint. For CH-SIMS/CH-SIMSv2 the dataset's own unimodal labels are used. | `pretrain.py` trains all three modalities and computes validation, but also evaluates TEST each epoch; after the loop, `save_model` names the final checkpoint with TEST MAE/Corr. It uses MSE. | Select each modality's epoch count using source-held folds *inside TRAIN* and the corresponding T/A/V target. Refit on full TRAIN for those fixed counts. Use paper's L1 regression objective. Never instantiate or read TEST. |
| Stage 2 initialization/freeze | Load Stage 1 weights; freeze adapter and decoder parameters, update the remaining encoders and fusion model. | `OverallModal.preprocess_model` freezes parameters whose names include `adapter` or `decoder`; other encoder parameters remain trainable. `train.py` hardcodes `SIMS_*` (v1) checkpoint paths. | Load this adaptation's own CH-SIMSv2 TRAIN-pretrained heads; freeze the pooled-feature adapters and unimodal score decoders, fine-tune modality projections and the dynamic fusion stack. |
| Stage 2 objective | Native regression L1 plus `alpha * L_cor`, where `L_cor` is the sum of three contrastive correlation losses. | `train.py` uses MSE plus `0.1 * nce_loss`. | Use paper L1 plus `0.01 * L_cor` for the CH-SIMSv2 adaptation; keep this objective identical across primary arms. |
| Dataset/checkpoint wiring | Separate supervised target dataset and declared modality features. | `opts.py` defaults to `datasetName=sims`, v1 `unaligned_39.pkl`, and v1 sequence lengths; `train.py`'s hardcoded pretraining paths also point to `SIMS`, not `SIMSv2`. `core/dataset.py` maps the M target to internal `labels['M']` and loads T/A/V channel labels for every split. Its loader also sets `shuffle=True` on train, valid, and test. README instructs users to change stage parameters. | Read only the project's bound standardized pooled cache and official split IDs. Map CSV `label` to internal M and `label_T/A/V` to the channel targets. Load T/A/V targets only for TRAIN; VALID exposes only `label`; do not construct a TEST loader. Shuffle TRAIN only and keep eval predictions in fixed ID order. |
| TEST use | TEST is a final held-out evaluation. | `pretrain.py` and `train.py` score TEST during training epochs; the stage-2 code also calls a TEST statistic at epoch 11. | Do not reuse the scripts unchanged. Record no TEST rows, predictions, or scores in this study. |

The public README reports Python 3.8, PyTorch 1.9.0, one RTX 3090, and says
the code supports CH-SIMSv2. It also notes that the two stages need
dataset-specific parameter changes. These resource and environment notes are
for the original sequential model, not a runtime guarantee for this pooled
adaptation. The repository MIT license covers the repository code; dataset
assets and pretrained models have separate terms.

## Recommended next design direction after the HOLD

Do not formal-fit the singleton pooled attention candidate. If the project
continues this baseline line, preserve the existing standardized sequence
inputs (`T64/A400/V55`) and valid lengths from the sequence cache. The project
already has a standardized-sequence ALMT baseline on those inputs. A full
multi-key cross-attention block restores nonconstant attention scores; it does
not fix the pre-LayerNorm ratio cancellation derived above.

The next bounded comparison should use the same multi-key attention stack,
same true `label_T/A/V` Stage 1 supervision, same M head, parameters, update
budget, splits, and seeds in three conditions:

| Condition | Stage 2 branch operation |
|---|---|
| `static` | Fixed equal `R_m=1/3` in the published pre-LayerNorm position. |
| `KuDA-pre` | Published `R_m` from the detached M target and attached `ŷ_m`, before per-branch LayerNorm. |
| `KuDA-post` | Same `R_m`, applied to the normalized branch feature before summing modalities. |

For `KuDA-post`, write the operation explicitly as
`g_m=(1+R_m)*LN(z_m)`. This is a gate-placement adaptation, not KuDA's paper
formula or an originality claim. Every arm should use the paper's target-free
inference rule `R_m=1`; VALID selection uses CSV `label` (M) only. Stage 1
channel-head epoch selection uses source-video-held folds inside official
TRAIN, never official VALID T/A/V or TEST. Retain valid-length masks in
multi-key attention; the pinned code passes a mask argument that its cross-
attention implementation does not use, so do not reuse that unmasked behavior
silently. The exact length alignment and parameter budget need a separate
architecture decision before implementation.

This study should first verify the predicted cancellation and post-normalized
branch effect on actual TRAIN activations, then fix architecture, masks, and
parameter counts, and only then preregister the three-arm fit. Report any
positive result as a CH-SIMS v2 KuDA gate-placement adaptation. A downstream
benefit still requires the native-M three-seed matched comparison and a fresh
prior-art review.

## Superseded singleton pooled draft — do not fit

The following detailed draft records the earlier plan for audit. The root-owned
mechanism probe invalidated it as a useful attention comparator: its scalar
ratio is canceled by LayerNorm under ordinary-variance features, and its
single-key attention has no Q/K score gradients. Its parameter estimate and
static-versus-dynamic results must not be presented as evidence of a meaningful
attention mechanism.

Use one frozen standardized vector per modality: `x_T∈R^768`, `x_A∈R^768`,
`x_V∈R^384`. Do not run MacBERT, WavLM, or DINOv2 in this model. In the
author implementation, BERT and the audio/vision encoders remain trainable in
Stage 2; here only small pooled-vector input projections are trainable. Fit no
new normalizer on VALID. The official TRAIN fit uses the bound standardized cache;
source-held inner folds used for Stage 1 selection fit their coordinate
statistics on each inner-training source subset and apply them only to its
held source videos. The final Stage 1 refit uses the existing official TRAIN
statistics.

For each modality, form a global branch feature `H_m = LN(Linear(x_m,128))`.
Use a small bottleneck adapter
`K_m = H_m + Linear(GELU(Linear(LN(H_m),64)),128)`, and concatenate
`U_m=[H_m;K_m]∈R^256`. A modality decoder `D_m:256→64→32→1` predicts the
modality sentiment score. Stage 1 trains `D_T,D_A,D_V` against their matching
TRAIN labels with L1.

At Stage 2, freeze the three adapters and decoders, as the paper does; continue
training the modality input projections and all fusion layers. Project each
`U_m` to a shared 128-dimensional single-token feature `u_m`. Use three
dynamic blocks, each with four attention heads of dimension 32 and an FFN
width of 256. This halves the paper/repository's fusion width 256 and uses
four 32-dimensional heads instead of the source code's eight 64-dimensional
heads; it retains the paper's three dynamic blocks as a budget adaptation.
Start the fusion state with `F⁰=u_T+u_A+u_V`. Each block then:

1. Updates `F` with a separate cross-attention branch for each modality, using
   the current fused state as query and `u_m` as key/value.
2. Applies the score ratio inside each feature branch:
   `g_m = LN_m(z_m + R_m z_m)`, where `z_m` is the cross-attended feature.
3. Adds the normalized sum of the three branches to the prior fusion state,
   then applies self-attention and an FFN.

The scalar prediction is `MLP(mean(F^L))`; there is no weighted average of
`ŷ_T, ŷ_A, ŷ_V`. With one pooled vector per modality, each cross-attention key
set has length one. `softmax` over a single key equals one, so its Q/K
projections have zero score effect and zero attention-score gradient. The
within-fusion self-attention also has one token and the same limitation.
Residual paths, value projections, output projections, LayerNorms, and FFNs
remain active, but the pooled model does not perform content-based attention
selection. Describe it as a **frozen pooled-feature adaptation**, not full
sequence KuDA or a reproduction.

If the mechanism gate passes, the ratio arm computes during Stage 2 training
`R_m = softmax_m(-0.3 * (ŷ_m - label)^2)` using the current unimodal branch
predictions and the native M training target. This is algebraically the paper's
normalized exponential ratio; softmax is the numerically stable
implementation. Keep the prediction path attached to the trainable modality
projection, matching the repository's differentiable ratio. Treat `label` as
a constant target (`detach`); no T/A/V labels enter this calculation. At
validation and inference, set the three ratios to exactly one and do not read
`label` or any channel target. For the static-ratio control, use
`R_T=R_A=R_V=1/3` throughout training; at validation/inference it also uses
`R_m=1` to share the paper's target-free boundary. The static arm still runs
the same dynamic-attention stack. Compute the unimodal scores in that arm and
discard them so its forward/update budget matches the ratio arm. The paper
formula alone is insufficient to justify this fit: the source places the
scalar before LayerNorm, so observed branch variance must show a material
nonzero ratio effect first.

The shared Stage 2 loss is
`L = L1(pred_M, label) + 0.01 * sum_m InfoNCE(mean(u_m), mean(F^L))`.
Use the paper's CH-SIMSv2 learning setup where it applies: AdamW, batch 32,
learning rate `3e-5`, weight decay `1e-5` from the author options, and 50
epochs. Keep the schedule fixed by epoch, run the same number of updates in
every arm, save every epoch, then select the first minimum official VALID
native-M MAE. Report the project's native clipped MAE, Corr, and Acc2. The
project's `label` column is M; VALID `label_T/A/V` are never loaded or used.

## Pooled-draft arms and selection (historical only)

| Arm | Stage 1 | Stage 2 training ratio | Purpose |
|---|---|---|---|
| `TAV-static` | Three branch heads use their corresponding `label_T/A/V` targets. | Constant `R_m=1/3`. | Controls for added channel supervision and the full dynamic-attention architecture. |
| `TAV-KuDA` | The exact same Stage 1 checkpoints and channel labels as `TAV-static`. | Published KuDA ratio with fixed `k=0.3`. | Tests whether score-gap modulation helps beyond the same labels, parameters, and attention stack. |
| Optional `M-only-static` | Skip T/A/V Stage 1 losses. Initialize each residual adapter to identity and leave its T/A/V decoder head present but untrained and unused. | Constant `R_m=1/3`. | Estimates an M-only supervision reference; it is not evidence for or against the KuDA route. |

For each of seeds 13/42/2026, Stage 1 uses the same source-held selection and
one final full-TRAIN refit for both primary arms. Stage 2 clones those exact
weights and uses identical data order, optimizer initialization, schedule,
number of batches, epochs, attention heads, heads, and parameter count in the
two arms. The only primary difference is the ratio supplied during Stage 2
training. The optional `M-only-static` arm has the same declared total
parameter count, but its modality decoders are intentionally untrained and
unused; report those dormant parameters separately. It is only a supervision
comparison, not part of the primary static-versus-KuDA route test.

Do not use the public `pretrain.py` validation outputs to choose channel-head
epochs. Within official TRAIN, use the project's three source-video-held folds
(source-video-held, not speaker-held). For each modality and seed, use batch
32, AdamW, Stage 1 learning rate `1e-4`, weight decay `1e-3`, L1, and a fixed
50-epoch ceiling. Select the earliest minimum channel-specific MAE in each
fold, take the median of those three epochs, then refit on all official TRAIN
for that fixed count. These are explicit adaptation settings: the epoch cap
comes from the CH-SIMSv2 paper table, while the Stage 1 optimizer values match
the author's pretraining defaults. Stage 2 checkpoint selection uses official
VALID `label` (M) only. Train every arm for all 50 epochs, save every epoch,
and use the same fixed-by-epoch schedule so checkpoint selection does not
change update counts. With 2,722 rows, batch 32, and `drop_last=False`, Stage 2
has 86 updates per epoch and 4,300 updates per arm and seed. Preserve the
official split IDs and original label values. No TEST loader or TEST file is
part of data construction.

The analytical model estimate is 2,054,212 nominal parameters under the layer
widths above, with about 1,948,033 nominally trainable in Stage 2 after
freezing adapters and modality decoders. The score-ratio calculation has no
parameters. However, singleton attention makes approximately 393,216 Q/K
projection weights receive zero gradient through attention scores. The two
primary arms match in nominal counts, but that count is not evidence of useful
attention capacity. The total is about 4.2× the 492,033-parameter
standardized fusion MLP and 0.62× the 3,288,577-parameter standardized
sequence ALMT. These are architecture estimates to verify from an eventual
state dict, not measured throughput. The one-token draft reduces sequence
work but removes content-based attention selection, so it is not a fair
replacement for the full sequence comparator.

## Current hold and evidence before any formal fit

The root-owned [TRAIN-only mechanism probe](native_scalar_gate_normalization_20261014.json) has run. It uses all 2,722 standardized TRAIN vectors only through seed-20261014 width-128 normalized random projections; it has no trained KuDA states, no model fits, no VALID features or parsed scores, and no TEST access. With the gate vector changed from `[0.1,0.2,0.7]` to `[0.7,0.2,0.1]`, the inside-LayerNorm placement changed output by maximum absolute `1.33e-15` at `eps=0` and median relative L2 `1.945e-6` at `eps=1e-5`. Applying the same coefficients after LayerNorm changed the TRAIN-projected branches by median relative L2 `0.35769`. A singleton key produced exactly zero Q/K gradient norms; a two-key check produced Q/K norms `2.18/3.25`. These results confirm the operator analysis for the probe. They do not establish behavior in a trained KuDA model, a performance bug in the paper, or an accuracy gain.

The proposed pooled formal fit remains **on hold**. The next study direction should use the existing multi-key standardized sequence inputs `T64/A400/V55` with valid lengths and compare three matched branch placements: static equal ratio, published KuDA pre-LayerNorm ratio, and the same predicted ratio applied after branch LayerNorm as `(1+R_m)*LN(z_m)`. This post-normalization operation is a placement ablation, not a new formula or originality claim. Multi-key attention should make Q/K scores active; it does not remove the pre-LayerNorm scale cancellation. Any sequence adaptation must explicitly handle padding: the pinned code carries a mask argument but does not apply it in the cross-attention calls.

If this sequence study is approved after architecture and prior-art review, keep Stage 1 true `label_T/A/V` supervision identical across arms and select its epochs using source-video-held folds inside TRAIN only. Stage 2 uses native M loss and selects checkpoints on official VALID `label` only. No VALID T/A/V or TEST labels enter selection. Use `R_m=1` without labels at inference for all arms. Recompute the parameter and memory budget for the multi-key architecture; the superseded 2.05M singleton estimate does not apply. Before formal fitting, verify nonzero Q/K gradients on an actual multi-key TRAIN batch, the three ratio placements' branch effects, matching arm parameters/updates, resume, and the absence of TEST IDs or metrics. No formal fit is authorized by this design note.

## Primary sources and local anchors

- [KuDA paper PDF, Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.865.pdf)
- [Author repository at pinned commit](https://github.com/MKMaS-GUET/KuDA/tree/e60ef6bf070f23855e186c9fd7563902e9a8c056)
- [KuDA MIT license at pinned commit](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/LICENSE)
- Local baselines: [standardized native findings](NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md), [sequence conditioning findings](NATIVE_SEQUENCE_CONDITIONING_FINDINGS_V1.md), and [native evaluation protocol](NATIVE_EVALUATION_PROTOCOL.md).
