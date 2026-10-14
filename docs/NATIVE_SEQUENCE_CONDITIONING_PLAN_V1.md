# Standardized-sequence ALMT baseline

2026-10-14. Registered before model fitting. This is a known preprocessing
control, not an original algorithm. The CDF-shape line failed its registered
gate and is stopped. A new temporal mechanism must compare with a properly
conditioned mature sequence baseline, not only the weaker raw-input ALMT.

Reuse the same raw-derived MacBERT/WavLM/DINOv2 TRAIN2722/VALID647 inputs,
author CSV pin, original native M labels and splits. The saved TRAIN
clip-pooled coordinate statistics from the strong fusion baseline are applied
to each valid token/frame; no token-weighted statistics or new encoder/data
is fitted. Text CLS and A/V masked means match the already-trained pooled
vectors within1.216e-5; allraw PAD entries are alreadyzero, so no additional
padding-cleanup intervention occurs. Keep T64/A400/V55 shapes, lengths and
four A/V timestamp artifacts unchanged. Timestamp identity does not establish
cross-stream synchrony. No TEST features/labels are exported or evaluated.

Use unchanged native_training.py and vendor ALMT scalar adaptation. Preserve
its upstream unmasked padding participation and report that limit. Config
`native_simsv2_sequence_standardized_almt.json` changes only cache/runroot
relative to the raw ALMT reference. Seeds13/42/2026, AdamW lr1e-4/WD1e-4,
batch32/max80/patience12 and the same validation-MAE plateau schedule, L1
on original model outputs, first minimum native VALID MAE checkpoint. Real
32-row smoke/save/resume precedes allthree full GPU fits, run serially.

Report author-clipped CH MAE/Corr/Acc2 and allnative auxiliary metrics,
parameter counts, actual executed epochs/updates/time/memory. Independently
reload checkpoints/recompute predictions and compare raw ALMT versus scaled
ALMT perseed, with5000 paired VALID source/video bootstrap draws. Compare
strong pooled fusion as a separate architecture reference. Different parameter
counts and sequence versus pooled inputs must be explicit; a gain here is
not an original mechanism and not a final-test result. Official VALID is a
development/selection set with646/647 source overlap with TRAIN.

If conditioning improves ALMT consistently with supporting paired evidence,
use it as a stronger competitor. If not, retain the negative baseline control.
Do not claim temporal interaction from this comparison alone. The
[paired-covariation review](NATIVE_TIMED_COVARIATION_REVIEW_20261013.md) requires
time-origin verification plus capacity-matched static-product, marginal-variance
and pairing-break controls before any original-method discussion. Method
validation, MOSEI repetition, frozen TEST, final fresh reproduction/release
and conditional legacy deletion remain incomplete.
