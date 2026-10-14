# Annotation-reference additive decoding: rejected core promotion

Registered on2026-10-13; four real32/32 smoke cases, target/isolation/gradient
and exact-resume tests preceded twelve full runs. Every case preserves original
CH-SIMS2722/647 M scores, the same frozen MacBERT/WavLM/DINOv2 feature bytes,
first minimum VALID MAE selection and seeds13/42/2026. Native TEST is not evaluated.
[Plan and gate](NATIVE_REFERENCE_RESIDUAL_PLAN_V1.md),
[complete target/decoder/hash/epoch audit](native_reference_residual_audit_v1.json).

| Main decoder | True T / M-T auxiliary supervision | Native valid MAE ± SD | Corr | Acc2 |
|---|---|---:|---:|---:|
| Direct M head | off | 0.4237 ± 0.0017 | 0.6145 | 77.69% |
| Direct M head | on | **0.4205 ± 0.0022** | 0.6198 | 78.16% |
| T+D | off | 0.4286 ± 0.0008 | 0.6117 | 78.10% |
| T+D | on | 0.4218 ± 0.0036 | 0.6230 | 78.05% |
| Previous simple fusion | no such supervision | 0.4221 ± 0.0007 | 0.6234 | 77.13% |

Additive-minus-equally-supervised-direct MAE differences are+0.005516,
-0.001714,+0.00000465. Additive-without-aux is worse than direct-without-aux
in every seed. Aux supervision improves both architectures in every seed,
with negative interaction, but does not establish an additive decoder benefit.
The mean additive MAE is only0.000319 below old simple fusion and has higher
seed variance. **The registered consistency gate fails; no original method
promotion.** Direct+aux is a useful multi-task baseline, not a validated new
residual mechanism or stable superiority claim from a tiny selection gap.

Total parameters492,691 in every case (within0.2% of old492,033 fusion).
Actual first-batch nonzero-gradient counts are492,257 for flat0, about492,691
for flat1, and492,386 for additive cases; unused scalar heads account for small
differences. The text representation remains active through the context trunk
in flat cases, avoiding a major dormant pathway. Shared LayerNorm/bottleneck
changes mean old-fusion comparisons include architectural changes; flat1 is
the cleaner structure comparator. Actual epochs/updates are in the audit;
twelve training/validation loops totaled106.3s, excluding hash/startup/artifacts.

Only TRAIN real label_T is used for auxiliary loss; correction target is the
annotation difference label_M-label_T. It is not causal AV contribution, not
uncertainty, and not a teacher residual. VALID T labels are read only by the
post-training diagnostic auditor for error strata, never by the trainer or
model. Without aux, output components lack annotation-role identification;
even with aux, this is a trained approximation rather than a guarantee that
the two heads recover the human decomposition.

Closest work already includes real-label AV-MC/MTCL multi-task learning and
model-error V-A scalar residual compensation. The precise annotation target
can be tested separately, but a changed target alone is not originality.
[Primary-source boundary](NATIVE_RESIDUAL_REFERENCE_RESEARCH.md). A proxy on
MOSEI would represent model prediction error, not unavailable real text labels.
Since the structure gate failed, no MOSEI proxy is promoted as equivalent
replication. Preserve this negative experiment and the competitive baselines;
choose a new question from measured failures before increasing model size.
