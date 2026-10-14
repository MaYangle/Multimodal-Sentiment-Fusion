# MOSEI ALMT valid-step baseline correction

Registered 2026-10-13 before implementation or training. The frozen upstream
adaptation ignores lengths. Its VALID MAE is0.6115; moving the same valid
vectors to front padding raises this to0.6699. This warrants a mask/position
control. It does not establish a new algorithm or raw-media robustness.

## Matched baseline

Add no learned parameters. Keep every existing projection, hyper-modality,
language and cross-modal weight and the scalar head. In each initial modality
projection, mask invalid keys and number valid feature positions by cumulative
valid rank after the eight learned adaptive tokens. Those eight tokens are
always valid. Subsequent language/hyper/cross-modal inputs are entirely learned
fixed-length tokens, requiring no feature-padding mask. Use the exact source
attention arithmetic. Fully valid inputs must match the previous model with
identical weights; arbitrary invalid feature values and padding position must
not change inference or valid-input gradients within float32 tolerance.

The input contract uses declared lengths for right padding, or explicit boolean
valid masks for transformed layouts. Published interior zero word vectors stay
valid; validity is never inferred from numeric nonzero values. Original features
and TRAIN/VALID temporal IDs/native labels stay byte-identical.

Same3,047,937 parameters, random initialization, raw features, L1, AdamW1e-4,
weight decay1e-4, batch32, max80, patience12, norm1 clipping, MAE scheduler
patience3/factor0.5, min VALID MAE first tie, seeds13/42/2026 as raw ALMT.
Keep frozen old implementation unchanged. Exact checkpoint resume and trusted
code/cache hashes are required. Test is not evaluated.

## Evidence and decision

First test all-valid parity, invalid-value/position invariance, valid gradients,
interior-zero retention and exact epoch resume. Run real32/32 smoke/backprop
and checkpoint reload before serial three-seed formal training. Save native
metrics, per-sample predictions, histories and checkpoints. Report whether
natural-input accuracy improves or worsens as a baseline adaptation, not a
contribution. Compare identity/front padding/joint shuffle/reverse/AV shuffle
on the selected checkpoints with declared masks and per-ID fixed transforms.

Padding-equivalent views are mathematically invariant in this corrected model.
Do not add an auxiliary consistency objective with identically zero target
discrepancy or promote it as novel. Once this correction is measured, reconsider
the research hypothesis on a competitive implementation and actual errors.
