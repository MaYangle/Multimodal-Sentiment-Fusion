# Native sequence-gate final independent audit

All **27/27 formal cells** are complete. The CPU audit recorded 1,733 checks: 1,729 passed and four failed. Every failure is a registered candidate comparison gate; source, cache, checkpoint, training-budget, frozen-parameter, target-boundary, and numerical-replay checks passed. Full cache-array SHA-256 hashes were recomputed.

The audit independently reconstructed the 13-source holdout from 129 TRAIN sources, verified optimization-only scaler statistics, reproduced selected T/A/V held-source metrics, and checked the selected-count refits. Across all 21 fusion arms, it replayed all 647 selected VALID predictions on CPU with a maximum saved-device-to-CPU difference of `4.92e-7`. Independently recomputed native MAE and auxiliary metrics matched the saved results within `1e-5`.

The registered candidate, `distilled_post_norm`, fails the required comparison gate. Each confidence interval below is a 5,000-draw paired whole-source bootstrap with seed `20261014`, 120 VALID source clusters, and the same 647 unique clips. Deltas are candidate-minus-reference MAE averaged over the three seeds.

| Reference | Candidate lower in all seeds? | Seed deltas (13, 42, 2026) | 95% interval | Gate |
|---|---|---|---|---|
| Conditioned early fusion | Yes | -0.027799, -0.030759, -0.040523 | [-0.061711, -0.019393] | Pass |
| Conditioned ALMT | Yes | -0.035934, -0.018441, -0.036174 | [-0.046106, -0.013737] | Pass |
| `distilled_static` | No | +0.000637, -0.001426, +0.001522 | [-0.002472, +0.003195] | Fail |
| `distilled_pre_norm` | No | +0.000637, -0.001426, +0.001522 | [-0.002472, +0.003195] | Fail |
| `learned_post_norm` | Yes | -0.000073, -0.001547, -0.000798 | [-0.003323, +0.001850] | Fail |
| Strongest teacher placement | No | +0.000442, -0.000677, +0.002041 | [-0.001106, +0.002733] | Fail |

For the strongest teacher-placement comparison, the lower **native clipped** MAE `pre_norm` or `post_norm` arm was chosen separately per seed on that same VALID cohort: `post_norm` for seed 13, `pre_norm` for seed 42, and `post_norm` for seed 2026. The implementation clips predictions to the native CH-SIMS range before this ranking. Its interval is conditional on these choices. A regression fixture covers an out-of-range raw prediction case where raw-MAE ranking differs from clipped-native-MAE ranking.

The candidate fails the all-seed and interval requirements against the registered mechanism controls, so the registered gate does not pass despite passing both baseline comparisons. The full matrix remains a negative result; no TEST evaluation or promotion claim follows.

The audit ran in the isolated project environment: Python `3.12.10`, NumPy `2.3.5`, PyTorch `2.11.0+cu128`. Replay used CPU MHA with fastpath disabled and math SDP enabled; the auditor created no CUDA context. Checkpoint identities do not record the training Python or PyTorch versions, so exact training-runtime parity is not bound by the receipts. No TEST scores or features were read. In this formal audit, VALID M was used for registered selection and metric comparisons, and T/A/V labels came only from TRAIN. The separate oracle diagnostic uses VALID M externally to form a privileged probe ratio.

The separate teacher-routing diagnostic is [NATIVE_SEQUENCE_GATE_TEACHER_DIAGNOSTIC_20261014.md](NATIVE_SEQUENCE_GATE_TEACHER_DIAGNOSTIC_20261014.md). It uses fixed selected checkpoints and labels its VALID-M oracle as privileged and nondeployable. Machine-readable audit evidence, including hashes, per-cell checks, metrics, and bootstrap draws, is in [native_sequence_gate_audit_20261014.json](native_sequence_gate_audit_20261014.json).
