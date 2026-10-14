# ALMT adapter patch notes

## Source and scope

The read-only upstream checkout is `D:\MYL\JevAffect\work\ALMT`, origin
`https://github.com/Haoyu-ha/ALMT`, clean at commit
`da9461071f21e75eb8bf2bedff14ac23f5954605` (`Update`, 2025-04-09, Haoyu
Zhang). No upstream files were edited.

Only `models/almt_layer.py` and the repository `LICENSE` were copied into
`vendor/almt/`. The license is byte-identical to upstream MIT, copyright (c)
2024 Haoyu Zhang. `vendor/almt/UPSTREAM.md` records the repository, commit,
paper authors, and the source file's retained note about code referencing
`gupta-abhay/pytorch-vit`.

## Architecture kept

`jevaffect/almt.py` reconstructs the upstream `ALMT` module layout around the
vendored author layers: three `Linear + Transformer` projections,
language-guided `Transformer` with hidden-state collection,
`HhyperLearningEncoder`, and the final `CrossTransformer`. Forward ordering,
token slicing, residuals, attention dimensions, and the upstream
`h_hyper` repeat operation are preserved.

Defaults follow `configs/mosei.yaml`: feature dims text/audio/vision =
768/74/35, fixed sequence lengths = 50/500/500, shared projection/token dim =
128, hyper-token count = 8, projection depth = 1, projection heads = 8,
language encoder depth = `AHL_depth - 1` = 2, AHL depth = 3 with 8 heads and
16-dimensional heads, and fusion depth = 4 with 8 heads. `feature_dims` and
`lengths` can override the default mappings; sequence tensors must match the
configured fixed sizes.

## Explicit changes from upstream ALMT

1. **Text encoder input:** the upstream `ALMT` owns a `BertTextEncoder` with
   `use_finetune=True` and calls it inside `forward` (`models/almt.py:16,44`).
   This adapter removes that BERT module and expects cached contextual
   per-token text vectors of width `feature_dims["text"]`. It does not train or
   reproduce the internal BERT fine-tuning path. All model comparisons must
   use the same cached text features to make the feature backbone comparable.
2. **Task head:** upstream MOSEI ALMT has a one-output regression layer
   (`models/almt.py:35-37`). This adapter replaces it with a five-logit linear
   classifier. Its state dict is not compatible with an upstream regression
   checkpoint's BERT module or output head.
3. **Padding:** upstream `Attention`, `Transformer`, and `CrossTransformer`
   accept no padding mask (`models/almt_layer.py:91-105,181-212,237-250,267-284`).
   The adapter deliberately does not add one. All fixed-length positions,
   including padded positions, participate in downstream attention. Even if
   cached BERT outputs were computed with a BERT attention mask, their padded
   hidden states are still supplied to the ALMT projection Transformer and
   are not masked there. This preserves author-layer behavior but can make
   results depend on padding content/length. Comparisons with mask-aware models
   should disclose this difference; adding masks later would be an explicit
   architecture change and would require updating the shared protocol.

4. **Output labels:** the adapter emits logits `[B, 5]`; it does not encode a
   label mapping, loss, or metric policy. Those remain in the common training
   protocol.

This is a cached-feature ALMT structural baseline, not a reproduction of the
paper's fine-tuned-BERT model or published results. No upstream checkpoint is
loaded or claimed.

## Verification

No dependencies were installed; no model was imported, trained, or run on GPU.
The main agent is responsible for runtime checks after its environment is
ready.
