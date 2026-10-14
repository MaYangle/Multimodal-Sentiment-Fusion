# Experiments: evidence as of 2026-10-14

The completed registered CH sequence-policy study contains three source-held
branch selections, three full-TRAIN refits and 21 fusion fits: **27/27 complete**.
The registered candidate **did not pass promotion**. Its mean MAE is
0.378497, versus 0.378253 for matched distilled-static and 0.378011 for
teacher-post-LN. These are development VALID results, not final TEST scores.

## Three-seed matched architecture study

| Arm | Seed 13 MAE ↓ | Seed 42 MAE ↓ | Seed 2026 MAE ↓ | Mean ± sample SD ↓ | Mean Corr ↑ | Mean Acc2 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `static` | 0.379826 | 0.385593 | 0.371076 | **0.378832 ± 0.007309** | 0.6554 | 0.7743 |
| `pre_norm` | 0.379826 | 0.385593 | 0.371076 | **0.378832 ± 0.007309** | 0.6554 | 0.7743 |
| `post_norm` | 0.379814 | 0.385938 | 0.368280 | **0.378011 ± 0.008966** | 0.6596 | 0.7702 |
| `distilled_static` | 0.379619 | 0.386341 | 0.368799 | **0.378253 ± 0.008851** | 0.6550 | 0.7785 |
| `distilled_pre_norm` | 0.379619 | 0.386341 | 0.368799 | **0.378253 ± 0.008851** | 0.6550 | 0.7785 |
| `distilled_post_norm` | 0.380256 | 0.384915 | 0.370321 | **0.378497 ± 0.007455** | 0.6570 | 0.7764 |
| `learned_post_norm` | 0.380329 | 0.386462 | 0.371119 | **0.379303 ± 0.007723** | 0.6560 | 0.7713 |

[Machine-readable aggregate results](sequence_gate_results_20261014.json)
include all nine native metrics, selected epochs, checkpoint hashes, parameter
counts and runtime for every fit. Differences are candidate-minus-reference;
lower MAE is better. The candidate is worse than distilled-static in seeds 13
and 2026. It improves on learned-post-LN in all three seeds, but that paired
source-bootstrap interval still includes zero. The frozen requirement
was improvement for every seed against every primary matched control plus
source-paired uncertainty checks. This prerequisite already fails. No other
arm is promoted post hoc as an original-method success.

## Data, training and selection

CH author's CSV contains 4,403 clips (2,722 TRAIN / 647 VALID / 1,034 TEST),
contrasting with the homepage's 4,402. Continuous M/T/A/V are kept in their
native scale. Stage-one source holdout uses 13 of 129 TRAIN video sources
(260 held clips; 2,462 optimization clips), fixed NumPy seed 20261014. Each
channel selects its first minimum clipped held-channel MAE over 50 epochs.
Branches refit from the same initialization on all TRAIN for their selected
epoch counts. Stage two uses M only for the primary sentiment loss and teacher.
VALID T/A/V and TEST scores are not loaded by this study.

| Seed | Selected text/audio/vision refit epochs |
|---|---|
| 13 | 25 / 3 / 2 |
| 42 | 6 / 3 / 6 |
| 2026 | 8 / 5 / 5 |

Every fusion arm shares its seed's refit, features, initial fusion weights,
loader RNG and 50 epochs / 4,300 optimizer updates. AdamW is 3e-5, weight
decay 1e-5, batch size 32, gradient clipping 1. No early stopping or scheduler;
first minimum official VALID clipped M MAE chooses the checkpoint. Training
L1 uses raw unbounded predictions. Evaluation clips CH truth/prediction to
[-1,1]; binary positive is >0 (neutral zero negative), with the author's
three/five-bin and weak-sentiment rules. [Full native protocol](NATIVE_EVALUATION_PROTOCOL.md).

Models have 2,084,871 parameters; external encoders are frozen and additional.
Formal training used RTX 5070 Ti 16 GB on Windows/Python 3.12.10. The sum of
recorded select/refit/fusion elapsed times is **8000.4 s (2.22 h)**;
this excludes data download, encoder extraction, independent audits and user
interaction. Each fusion fit takes about 5.8–6.1 minutes. Peak allocated
fusion memory is about 0.39 GB, excluding cached external encoder extraction
and other applications. Runtime versions were observed separately rather
than launch-bound in checkpoint identities; preserve that limitation.

The frozen [plan](NATIVE_SEQUENCE_GATE_PLAN_V1.md), [config](../configs/native_sequence_gates_v1.json),
[architecture](ARCHITECTURE.md), [audit](NATIVE_SEQUENCE_GATE_AUDIT_20261014.md)
and [prior-art review](NATIVE_SEQUENCE_GATE_PRIOR_REVIEW_20261014.md) are published.
Real 32-row CUDA smokes exercised forward/backward, selection, refit, resume,
evaluation and checkpoint saving before the formal matrix. Fixtures also
guard source isolation, masks, metrics and resume identity.

## Failure analysis

Static and teacher-pre-LN predictions are numerically nearly identical across
three seeds. Distilled-static and distilled-pre-LN are likewise nearly equal.
This supports the LayerNorm cancellation mechanism, but does not prove the
post-LN policy is useful. The learned policy's contribution is small in the
selected seed-13 checkpoint: within-availability policy cycling changes MAE
by only -0.000009 to +0.000027. A **privileged VALID-M oracle** improves its
fixed fuser by about 0.000382 yet remains worse than distilled-static. This
is explicitly nondeployable diagnostic evidence, not a benchmark result or
a selection rule. [Teacher diagnostic](NATIVE_SEQUENCE_GATE_TEACHER_DIAGNOSTIC_20261014.md).

Potential explanation: score closeness to aggregate M may be a weak surrogate
for feature utility, and channel pretraining may account for most of the new
architecture family's advantage. These are hypotheses, not causal conclusions.
A next study must preregister a stronger utility target and isolate it with
the same pretraining and compute controls.

## Earlier baselines and stopped lines

These are separate input/supervision protocols, not apples-to-apples wins for
the new gate. All numbers below are native official VALID MAE, lower better.

| Dataset / input | Method | MAE | Evidence |
|---|---|---:|---|
| CH derived frozen features, M-only | Raw text / simple fusion | 0.437309 / 0.422070 | [baseline results](native_simsv2_baselines_v1.json) |
| CH derived frozen features, M-only | Raw sequence ALMT, 3 seeds | 0.427086 ± 0.006547 | [conditioning study](NATIVE_SEQUENCE_CONDITIONING_FINDINGS_V1.md) |
| CH TRAIN-standardized, M-only | Pooled fusion, 3 seeds | 0.411524 ± 0.003855 | [standardization](NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md) |
| CH TRAIN-standardized, M-only | Sequence ALMT, 3 seeds | 0.408680 ± 0.006690 | [sequence controls](NATIVE_SEQUENCE_CONDITIONING_FINDINGS_V1.md) |
| CH TRAIN-standardized, extra channel pretraining | Static sequence control, 3 seeds | 0.378832 ± 0.007309 | Current matched study |
| MOSEI author-SDK-linked MultiBench features | Raw ALMT, 3 seeds | 0.611524 ± 0.005063 | [MOSEI findings](NATIVE_MOSEI_BASELINE_FINDINGS_V1.md) |

MOSEI native [-3,3] regression MAE/Corr are unbounded; Has0/Non0 and rounded
Acc5/Acc7 follow its own protocol. It has no true T/A/V channel labels, so an
M-only matched pretraining design is needed before repeating the gate study.
Its cohort differences are [disclosed](MOSEI_MULTIBENCH_COHORT.md).

Preserved negative studies include [source-risk controls](NATIVE_SOURCE_ROBUST_FINDINGS_V1.md),
[reference residuals](NATIVE_REFERENCE_RESIDUAL_FINDINGS_V1.md),
[nested probability shapes](NATIVE_NESTED_SHAPE_FINDINGS_V1.md),
[native CRPS](NATIVE_CDF_CRPS_FINDINGS_V1.md), and
[timed moments](NATIVE_TIMED_MOMENT_FINDINGS_V1.md). Their frozen plans, code,
controls and audits remain available. Unsuccessful experiments are not removed.

## Remaining research gates

Stable gain against competitive matched controls, native MOSEI method
replication, full author-media identity, final TEST evaluation, independent
fresh repetition of this gate matrix and checkpoint distribution are pending.
The repository is a public research snapshot; it does not claim completion
of those gates. See [reproduction](REPRODUCING.md) and [contributing](../CONTRIBUTING.md).
