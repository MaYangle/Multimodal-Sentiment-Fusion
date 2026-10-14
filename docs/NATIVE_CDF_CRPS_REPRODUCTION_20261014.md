# Independent source reproduction of CRPS heads

2026-10-14. All36 probability heads were retrained from an independent
633-file source copy, using the existing isolated Python3.12/Torch2.11/NumPy2.3.5
environment with `include-system-site-packages=false`. This is not a second
environment installation. Source/module paths and all633 file hashes are
[verified](source_native_cdf_crps_environment_20261014.json).

Original working tree and source copy each pass288 tests. The source snapshot
contains code/docs/configs/tests only; a133914-byte point-suite pointer manifest
was added separately. Audited original point checkpoints/tables are reused,
without copying raw data/encoder weights or re-extracting features.

[Exact comparison](native_cdf_crps_reproduction_comparison_20261014.json)
checks all36 runs and32664 prediction rows. Best/last model tensors, final
optimizer state, every CDF/bin/event/interval/score record, readout metrics,
selected epoch, updates and complete history excluding wall time are identical.
Only output runroot/config-file hash may differ. Full retraining loops24.0s;
the earlier run took27.9s. Checkpoint files need not match bytes because their
identities contain different output locations.

This reproduces the rejected probability-head study. It does not establish
a useful original mechanism, retrain its point/encoder stages, acquire all
author media, evaluate official TEST, or satisfy the final-method full-project
reproduction gate. The rejection remains unchanged.
