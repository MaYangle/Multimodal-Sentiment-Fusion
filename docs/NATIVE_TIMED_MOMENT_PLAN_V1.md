# Controlled decoder-time A/V moments: plan v1

Registered 2026-10-14 before any timed-moment model fit. The previous goal turn
was a user-requested pause; this resumed turn adds a timestamp audit and model
and cache fixtures. This study is one falsification step toward the full native
research goal, not a substitute for an original verified method or final release.

## Hypothesis and scope

Does a centered cross-moment of frozen A/V states add native scalar sentiment
information beyond clip means, static products, marginal variances and broken
pairing? Covariance/bilinear pooling is established prior art. This study cannot
establish mathematical originality, physical synchrony or chronological direction.
It could justify subsequent original-method research only if its controls pass.
Jev's structured, efficient decisions motivate the broader project; this operator
does not reproduce Jev, use its weights, or claim access to its training recipe.

The [independent time audit](NATIVE_AV_TIME_AUDIT_20261014.md) verifies the saved
per-stream grids, hashes and prefixes, with numeric overlap for 2715 TRAIN and
646 VALID records. Both locally inspected TRAIN videos begin A/V at PTS zero,
but first audio PTS was discarded during corpus decoding. Full-corpus physical
synchrony is UNVERIFIED. Features have clip-wide transformer context. Raw author
media byte equivalence is unverified for the competition-acquired derived cache.

## Frozen inputs and numerical pairing

Use complete official CH-SIMS v2 TRAIN 2722/VALID 647 IDs and original scalar M.
Author CSV SHA256 is be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150.
No TEST export, scores, predictions or selection. Reuse the pinned MacBERT CLS,
WavLM and DINOv2 cache; no new encoder, labels, data, teacher or fine-tuning.
Preserve global CLS/retained-prefix means exactly, using the same saved official
TRAIN clip-pooled mean/STD coordinate statistics as the conditioned baselines.
Normalize valid grid rows with those statistics; missing globals/invalid grids
are reset to zero after normalization. No new scaler is fitted.

For each record with at least two retained rows per stream, intersect the stored
seconds supports. Query four interior positions
`linspace(max(first A,V), min(last retained A,V), 6)[1:-1]` and linearly interpolate
each stream within its support. No extrapolation or index stretching. A zero or
negative overlap gives zero moment and `pair_valid=False`. Preserve global means
as fallback. Padding values never enter either mean or interpolation. Audio's
69 truncated records and vision's one truncated record cannot recover their tails.

This deliberately uses four interpolated numeric query positions rather than
the audit's proposed variable-length nearest-time candidate join. Fixed query
count enables exact feature-size/parameter/control matching; interpolation can
smooth signals and loses fine timing. Consequently this tests this representation
only and cannot reject every temporal-fusion mechanism if it fails. It assumes
zero relative decoder origins only as a disclosed diagnostic convention.

## Model and matched controls

All four cases have bias-free A768-to32 and V384-to32 projections and a shared
1952-to234 GELU/dropout0.1/234-to1 scalar head, with 494101 trainable parameters.
The head concatenates T768/A768/V384 global vectors and 32 moment coordinates.
All cases use identical pair eligibility and projection/head initialization for
each seed. The additional features are:

- `static_product`: projected full retained-clip A mean times V mean;
- `marginal_variance`: half the sum of centered population A/V grid variances;
- `paired_cov`: mean of centered projected A times centered projected V;
- `unpaired_cov`: the same covariance after a fixed one-row cyclic V shift.

This diagonal projected covariance is not the full outer-product matrix. The
four cases have equal parameter budgets. Existing fusion has 492033 parameters
(0.42% fewer) and standardized sequence ALMT has 3288577; report both compute
budgets. The shift preserves grid multisets, means and variances without random
noise or extra training RNG. Common paired permutations preserve paired moments;
constant offsets preserve only the interaction, not the retained global inputs.

## Training, checks and gate

First CPU fixtures and a real 32-row forward/backward/checkpoint/resume smoke.
Then 4 cases x seeds 13/42/2026 = 12 serial CUDA fits. Batch32, AdamW LR1e-4,
WD1e-4, L1 on unbounded predictions/raw M, gradient norm cap1, max80 epochs,
patience12, ReduceLROnPlateau VALID MAE patience3/factor0.5. First minimum native
VALID MAE selects the checkpoint. CH author MAE/Corr clip prediction and truth to
[-1,1]; Acc2 positive iff >0, including neutral on negative side. Preserve the
complete author metric implementation and disclose all metrics/tradeoffs.

Checkpoint identity binds config, seed, source and derived cache hashes, saved
scaler, author CSV, model/trainer/CLI/producer/reader/metric implementation.
Save best/last model, optimizer/scheduler/all RNG, logs and every VALID prediction.
Independent CPU and NumPy replay must reproduce all selected outputs within1e-5,
native metrics, source identities, parameter counts and first-minimum selection.

Promotion to a source-held diagnostic requires `paired_cov` lower native MAE in
ALL three seeds than each of conditioned pooled fusion, conditioned sequence
ALMT, static_product, marginal_variance and unpaired_cov. For each comparison,
5000 paired whole-source VALID bootstrap draws, fixed seed20261014, must give a
negative upper95% interval for candidate-minus-reference absolute error. These
are conditional development intervals and do not correct repeated VALID use.
On paired checkpoints, the fixed V shift at inference must worsen native MAE in
all three seeds with a positive lower95% source interval; common permutations
must preserve outputs numerically. Pair breaking is a selective mechanism check,
not a natural corruption robustness evaluation.

If any gate fails, retain the negative result and stop this covariance study;
do not tune rank, query count, shift, degree or seed on these VALID results.
If all pass, separately preregister TRAIN source-held diagnostic with optimization
source-only scalers/selection and then MOSEI native replication. Source-held and
official cohorts are distinct. Final TEST, original contribution, full fresh
reproduction, GitHub release and gated legacy deletion remain required.
