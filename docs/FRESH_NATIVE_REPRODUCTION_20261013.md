# Fresh native baseline reproduction

Actual rehearsal on2026-10-13, Windows11 / Python3.12.10 / RTX5070Ti16GB.
All nine full native MOSEI baseline runs are reproduced. Original-method,
full raw-encoder, final-test and release reproduction remain pending.

## Verified now

Created a new venv without `--system-site-packages`, installed CUDA Torch2.11.0
from the official wheel index, then installed the README's
`train,encoders,uncertainty` extras with recorded numerical constraints.
No package directories were copied from the old environment. A separate
source-only checkout contains411 hash-bound files. All numerical dependencies
import from the new venv; JevAffect imports from that source checkout. The
[environment audit](fresh_environment_audit_20261013.json) records exact paths,
versions, manifest hash and GPU. The local complete dependency freeze is
`work/fresh_environment_freeze_20261013.txt`.

Reused the verified3,730,509,179-byte author-linked download and regenerated
the complete TRAIN/VALID cache in a new directory. All19 output artifacts and
the descriptor match the old preparation byte-for-byte, including scores,
IDs, feature arrays, lengths and frozen label-free TEST IDs. The
[cache comparison](fresh_mosei_cache_comparison_20261013.json) proves preparation
reproduction, not new downloading or complete raw-media coverage.

The independent checkout passed all176 tests. Three real32/32 CPU model
smokes (text/fusion/ALMT) ran one epoch, then resumed for another, including
checkpoint reload/score checks. Two full selected-model CPU inference paths
produced two label-free predictions each, differing from saved GPU VALID
predictions by at most1.8e-7. The
[rehearsal audit](fresh_native_rehearsal_20261013.json) binds logs and artifacts.
Smoke losses are not benchmark results or full formal training proof.

Subsequently, all three seeds of text and early fusion completed full training
in the new environment. For every run, all1,869 VALID predictions, native
metrics, selected epoch and executed budget are exactly equal to the old
experiment. [Six-run comparison](fresh_pooled_baseline_comparison_20261013.json).
The three ALMT runs also completed. Every prediction and MAE difference is0;
versions, selected epochs and executed budgets match for all nine runs.
[Full comparison](fresh_author_baseline_comparison_20261013.json) passed the
predeclared tolerances. The publisher package was re-prepared rather than
copying the old cache. This verifies the complete native author-feature
baseline workflow on this machine, not the whole project's release gate.

## Reproduction commands

From the project root, with the trusted publisher package and receipt acquired
through the recorded author channel:

```powershell
python scripts/snapshot_reproduction_source.py --output-dir work/fresh-source
python -m venv work/fresh-env
work/fresh-env/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
Set-Location work/fresh-source
../fresh-env/Scripts/python.exe -m pip install -c configs/reproduction_constraints_20261013.txt -e ".[train,encoders,uncertainty]"
../fresh-env/Scripts/python.exe -m unittest discover -s tests
../fresh-env/Scripts/python.exe scripts/prepare_mosei_author_features.py --source ../../data/author/mosei/multibench/mosei_senti_data.pkl --receipt ../../data/author/mosei/multibench/mosei_senti_data.pkl.provenance.json --output-dir work/reproduced_mosei_cache --trust-author-pickle
```

For full training, copy`configs/native_mosei_author_baselines_v1.json` to a new
config. Change only`cache_dir` to`work/reproduced_mosei_cache` and`run_root` to
`runs/reproduction_formal`; keep CUDA, seeds and every numerical setting.
Existing outputs require explicit`--resume`, never overwrite.

```powershell
../fresh-env/Scripts/python.exe scripts/train_author_baselines.py --config configs/reproduction_formal.json --output work/reproduction_formal.json
../fresh-env/Scripts/python.exe scripts/audit_author_baselines.py --suite work/reproduction_formal.json --output work/reproduction_formal_audit.json
../fresh-env/Scripts/python.exe scripts/make_native_feature_demo.py --cache-dir work/reproduced_mosei_cache --output-dir work/feature_demo
../fresh-env/Scripts/python.exe scripts/predict_native_features.py --checkpoint runs/reproduction_formal/formal/almt/seed_13/best.pt --cache-dir work/reproduced_mosei_cache --input-jsonl work/feature_demo/input.jsonl --output-jsonl work/feature_demo/almt_predictions.jsonl
```

The full nine-run job waited on the observed masked-ALMT process handle,
then trained serially on the same GPU, audited and compared with the reference
suite. It ended successfully. Its logs/configs/checkpoints stay under ignored`work`.
Comparison requires identical numerical config, implementation, versions,
selection and executed budget, exact VALID IDs/scores, maximum prediction
difference≤1e-5 and MAE difference≤1e-6. A failed comparison stays failed;
tolerances are not relaxed after seeing the result.

## Limits

Only native MOSEI features are covered. This does not reproduce original video
acquisition/encoders, CH-SIMS cache generation from author media, a successful
original algorithm, source-disjoint method validation or independent test.
Platform/CUDA pins describe this machine; portability is not proven. Data and
checkpoints remain local under source conditions.
