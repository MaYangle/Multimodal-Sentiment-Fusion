# Independent source-copy calibration retraining

2026-10-13. A510-file source-only snapshot was created in
`work/source-anchor-final-20261013`. It contains no data, pretrained weights,
training checkpoints, environment or Git metadata. Every source file hash and
all dependency/project import origins passed the
[environment audit](source_native_anchor_environment_20261013.json).

The runtime is the previously installed isolated Python3.12.10 venv
`work/fresh-env-20261013`, with `include-system-site-packages=false` and pinned
Torch2.11.0+cu128/NumPy2.3.5 dependencies. This round reuses that clean installed
environment; it is not a second independent dependency-installation attempt.
The older full nine-run MOSEI preparation/training reproduction remains
separately documented in [the fresh workflow](FRESH_NATIVE_REPRODUCTION_20261013.md).

The snapshot passes its complete214-test suite. An external configuration
changes only the absolute input table path and fresh run output directory,
keeping every architecture, fit/selection rule, hyperparameter and seed fixed.
All21 formal calibration fits are then retrained from initialization on CPU.
[Every-run comparison](native_anchor_calibration_reproduction_comparison_20261013.json)
proves exact best-model tensors, all647 prediction records (points, probabilities,
medians and intervals), selected objectives, epoch histories excluding timing,
native/probability metrics, selected epochs and actual update counts. Global
scalar parameters/calls/fit-row records also match exactly.

This verifies **calibration retraining from the same audited OOF tables**. It
does not re-extract encoders, reacquire author media or retrain48 point teachers
in the source copy. Later source-error diagnostic files/tests are outside this
510-file snapshot. The rejected mechanism remains rejected after reproduction;
this is not evidence that the full original-method/release goal is complete.

Commands executed in PowerShell from the source snapshot:

```powershell
$env:PYTHONPATH = 'D:\MYL\JevAffect\work\source-anchor-final-20261013'
& D:/MYL/JevAffect/work/fresh-env-20261013/Scripts/python.exe scripts/audit_reproduction_environment.py --expected-source D:/MYL/JevAffect/work/source-anchor-final-20261013 --output D:/MYL/JevAffect/docs/source_native_anchor_environment_20261013.json
& D:/MYL/JevAffect/work/fresh-env-20261013/Scripts/python.exe -m unittest discover -s tests -q
& D:/MYL/JevAffect/work/fresh-env-20261013/Scripts/python.exe scripts/train_anchor_calibration.py --config D:/MYL/JevAffect/work/native_anchor_reproduction_config_20261013.json --output D:/MYL/JevAffect/docs/native_anchor_calibration_reproduction_20261013.json
```

Root comparison command:

```powershell
python scripts/compare_anchor_reproduction.py --original docs/native_anchor_calibration_formal_20261013.json --reproduction docs/native_anchor_calibration_reproduction_20261013.json --original-base D:/MYL/JevAffect --reproduction-base D:/MYL/JevAffect/work/source-anchor-final-20261013 --output docs/native_anchor_calibration_reproduction_comparison_20261013.json
```

Commands target new output directories; existing training requires `--resume`.
