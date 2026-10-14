# Independent Chinese standardized preparation and retraining

2026-10-13. A569-file source-only snapshot in
`work/source-native-standardized-20261013` contains code/docs/configs/tests.
It contains no datasets, encoders, checkpoints, environment or Git history.
Every source SHA and dependency/project import origin passed the
[environment audit](source_native_standardized_environment_20261013.json).
Its235 tests pass in the previously installed isolated Python3.12.10 venv
(`include-system-site-packages=false`). This is reuse of the clean installation,
not a second dependency-installation attempt.

Preparation executes the snapshot script with the original workspace as working
directory, retaining identical authorized input path spellings. The script
imports project modules from the snapshot. A new cache gives all17 artifacts
plus metadata byte-identical, including native scores, IDs, mean/STD and vectors.
[Complete comparison](native_standardized_cache_reproduction_20261013.json).

External configs change only cache/output directories. Allsix text/fusion
models retrain from initialization in separate directories using the same
GPU/environment and native VALID selection. [Exact comparison](native_standardized_reproduction_comparison_20261013.json)
proves every tensor, all647 prediction records/metrics, selected epochs and
loss/metric histories (timing excluded). Actual update budgets match.

This is complete pooled-input preparation/training reproduction. It does not
re-extract all encoders, prove author-original media equivalence, verify an
original algorithm, evaluate final TEST or complete the release goal. The
two-MP4 raw inference check is a separate functional rehearsal.

Executed with the isolated interpreter from the original workspace:

```powershell
python scripts/snapshot_reproduction_source.py --output-dir work/source-native-standardized-20261013
python work/source-native-standardized-20261013/scripts/prepare_native_pooled_standardization.py --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --author-csv data/author/simsv2/meta.csv --output-dir data/feature_cache/simsv2_native_pooled_standardized_reproduction_v1
python work/source-native-standardized-20261013/scripts/run_native_baselines.py --configs work/native_standardized_reproduction_configs_20261013/native_simsv2_standardized_text.json work/native_standardized_reproduction_configs_20261013/native_simsv2_standardized_fusion.json --output docs/native_simsv2_standardized_reproduction_20261013.json
python scripts/compare_native_point_reproduction.py --reference docs/native_simsv2_standardized_baselines_20261013.json --reproduced docs/native_simsv2_standardized_reproduction_20261013.json --reference-root runs/native_simsv2_standardized_v1 --reproduced-root runs/native_simsv2_standardized_reproduction_v1 --output docs/native_standardized_reproduction_comparison_20261013.json
```

Environment/hash audit and tests run with the snapshot as working directory
and `PYTHONPATH` set to that snapshot. Later documentation changes do not rewrite
this569-file snapshot. Earlier full MOSEI and21-calibration repeats are preserved.
