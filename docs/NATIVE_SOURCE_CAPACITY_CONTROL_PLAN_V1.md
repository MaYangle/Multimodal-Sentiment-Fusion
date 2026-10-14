# Source-CV text capacity control, registered before training

2026-10-13. This is a known-baseline diagnostic, not a new method.
The [completed point-teacher study](native_anchor_calibration_diagnostics_20261013.json)
finds source-OOF text MAE .39477/.39431/.39171 versus fusion
.41941/.41489/.40540, on identical native TRAIN held-source IDs and original M
targets. Different input dimensions give different parameter counts even with
the same 256-unit MLP. A source nuisance interpretation is currently unproven.

Before designing a source-aware mechanism, train a text-only 640-unit MLP
(492,801 parameters) to match the 256-unit T/A/V fusion MLP (492,033) within
.16%. Retain the original 256-unit text model as an additional control.
The width change itself is not a causal separation of representation and
capacity; two text widths provide a basic sensitivity test.

Use the identical author CSV, cached MacBERT CLS text, original M L1 target,
three fixed source folds (seed20261012), source-inner .1 selection fraction,
seeds13/42/2026, AdamW1e-4/WD1e-4/dropout.1, batch32, max80/patience12,
gradient1 and plateau schedule. Select the first minimum source-inner MAE.
Run nine OOF teachers only; official VALID and TEST labels/features are not
needed. Smoke on eight complete source groups and verify epoch restoration
before full runs. Use `configs/native_source_text_capacity_v1.json` and serial
GPU execution. Do not modify frozen AnchorCDF trainers or its OOF tables.

Compare source-CV OOF MAE, Corr and author classifications on all2722 TRAIN
IDs with exact once-only coverage. Report every fold/seed, parameter count,
selected epoch, updates and actual time. Use whole-source paired bootstrap
for wide-text minus fusion, retaining clip-weighted losses. Negative differences
in all seeds plus an upper95% paired interval below0 support the **diagnostic**
that fusion's observed source-OOF disadvantage survives this capacity control.
Otherwise do not use the earlier gap to motivate AV-source suppression.
Neither outcome establishes causal nuisance, final-test performance, a new
algorithm, or cross-dataset validity. Inspect source/sentiment relations and
closest public debiasing baselines before any mechanism registration.

Run the existing trainer with `--view text --fold 0`, then1, then2. Each
selector report intentionally has `suite_complete=false` relative to the
original48-teacher matrix; a separate nine-run audit must prove this diagnostic
stage complete, rather than mislabel a subset report as a complete AnchorCDF
suite. Existing runs require `--resume`. No paid resources or concurrent GPU
training.
