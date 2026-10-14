# Seed 13 teacher-routing mechanism diagnostic

This is a CPU-only diagnostic of the fixed, selected seed 13 checkpoints for `post_norm`, `distilled_post_norm`, and `distilled_static` on the same 647 VALID clips. Each registered deployment prediction came from `model(features)` in eval mode with no target argument. After those outputs were collected, external NumPy code used VALID native M to form a privileged oracle ratio, then passed that ratio to the explicit eval-mode `model.probe`. M was never passed as a target to `forward`; the oracle route is label-derived, privileged, nondeployable, and cannot be used for checkpoint selection or a performance claim.

| Fixed checkpoint | Registered deployment | Deploy MAE | Uniform present `R=1/3` MAE | Uniform present `R=1` MAE | Privileged VALID-M oracle MAE | Oracle ΔMAE vs deploy |
|---|---|---:|---:|---:|---:|---:|
| `post_norm` | `R=1` per present modality | 0.37981427 | 0.37981427 | 0.37981427 | 0.37890000 | -0.00091428 |
| `distilled_post_norm` | Learned target-free policy | 0.38025591 | 0.38026619 | 0.38026619 | 0.37987381 | -0.00038210 |
| `distilled_static` | `R=1` per present modality | 0.37961927 | 0.37961927 | 0.37961927 | 0.37961927 | 0.00000000 |

The candidate oracle route improves its fixed fuser by about `0.000382`, but remains about `0.000255` MAE above the `distilled_static` reference. Its learned policy’s mean route is near uniform, and fixed within-availability clip cycles changed MAE by only `-0.000009` to `+0.000027`. This suggests the policy-to-clip assignment has little aggregate effect in this selected checkpoint. The oracle result shows that score-error preferences can help the post-normalized fuser when privileged VALID labels supply the route; it does not make that route deployable.

For `distilled_static`, all tested probe routes reproduced deployment predictions within `3.6e-7`. Its branch normalization is pre-norm, so branch scale changes are removed by LayerNorm; the policy is an auxiliary distillation head and is not executed at inference. `post_norm` keeps an untrained dormant policy head, so its policy-cycle shifts are only a random-head negative control. Every VALID clip had all three modalities present (`111`). Synthetic all-absent inputs also produced finite predictions and zero route mass for all three checkpoints.

The selected checkpoints replayed their saved VALID predictions with maximum CPU differences below `4e-7`. The process used isolated Python `3.12.10`, NumPy `2.3.5`, and PyTorch `2.11.0+cu128` on CPU, with MHA fastpath disabled. No CUDA context, encoder, TEST record, or T/A/V VALID label was used.

The machine-readable report includes all native metrics, checkpoint identities, route summaries, and per-clip records for all 647 clips, including external M, unimodal scores, deployed/policy/oracle ratios, uniform probes, and availability-preserving policy cycles: [native_sequence_gate_teacher_diagnostic_20261014.json](native_sequence_gate_teacher_diagnostic_20261014.json).
