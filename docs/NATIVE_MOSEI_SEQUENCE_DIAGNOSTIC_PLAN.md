# Frozen sequence sensitivity — registered before evaluation

Raw nine-run publisher-feature suite and six known standardization controls
are complete. ALMT lowers MAE but has lower Non0 accuracy than raw text; its
larger capacity, per-token nonlinearity/normalization and temporal attention
are confounded. Do not attribute all improvement to temporal order.

Freeze the raw text/fusion/ALMT checkpoints and all1,869 VALID IDs/targets.
Evaluate identity plus four transformations, seed105 deterministically per
temporal ID, same transform across models/training seeds:

- Joint permutation of valid aligned steps in T/A/V: preserves cross-modal
  pairing and marginal vectors, removes sequence order.
- Independently permute A and V valid steps: preserve text order/marginals,
  disrupt their correspondence to text and each other.
- Reverse every modality's valid prefix together: preserve pairing and
  vector multiset, reverse order.
- Move valid suffixes to right-aligned positions (front padding): keep valid
  vectors/order/length and identify their actual mask for pooled controls.
  ALMT retains its documented unmasked fixed-position attention.

These are frozen feature-sensitivity diagnostics, not new labeled utterances,
causal explanations or raw-media robustness proofs. No retraining, calibration,
test access, auxiliary label or hyperparameter search. Pooled controls should
be invariant within float32 reduction tolerance; check identity predictions
against saved checkpoints. Save all outputs and recompute native metrics.
Use evidence to choose the next mechanism, rather than presuming order explains
the baseline gap or calling the new regression routing hypothesis original.
