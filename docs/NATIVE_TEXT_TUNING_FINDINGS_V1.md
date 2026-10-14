# Native MacBERT adaptation controls

Six formal CH-SIMS TRAIN/VALID runs completed after real32/32 GPU smoke,
selected-checkpoint reload and actual epoch resume. Eight dedicated tests and
the current187-test suite passed. Test labels/text were not parsed by training.
The [registered plan](NATIVE_TEXT_TUNING_BASELINE_PLAN_V1.md) fixed both modes,
three seeds, raw L1, budget and minimum native VALID MAE/first-tie selection.

| Input/model | VALID MAE ± sample SD ↓ | Corr ↑ | Acc2 ↑ |
|---|---:|---:|---:|
| Frozen encoder + scalar CLS head |0.457426 ±0.006726|0.548471|74.137%|
| Top-four-layer tuning + same scalar head |0.427238 ±0.002180|0.603796|74.961%|
| Earlier frozen CLS +256-wide MLP |0.437309 ±0.002435|0.582005|75.064%|
| Earlier frozen-text/AV simple fusion |0.422070 ±0.000736|0.623408|77.125%|

The controlled top4-minus-frozen MAE is−0.027475/−0.035934/−0.027155 for
seeds13/42/2026. Top4-minus-simple-fusion is+0.006590/+0.006209/+0.002706;
it never beats the strongest simple fusion. Fine-tuning improves its matched
scalar reference; it does not establish an original algorithm or a better
multimodal backbone. Old MLP/AV comparisons have different heads and budgets.

Both new modes have102,268,417 total parameters. Frozen mode trains769 head
parameters; top4 trains28,351,488 encoder+769 head parameters. The encoder's
embeddings/bottom eight layers/pooler stay frozen. Both keep BERT train-mode
dropout0.1 and the same scalar head/dropout. Top4 executes10 epochs/860 updates
per seed; frozen executes12/1032. Recorded train/validation loops total114.10s
and96.34s, excluding setup and some checkpoint I/O; peak Torch-allocated CUDA
memory is1,234,907,648 and547,394,560 bytes. Shared-desktop times are not isolated
speed measurements. No paid compute or extra training data was used.

## Error tradeoff

Post-training strata use original VALID scores and T/A/V annotations only for
analysis, never as inference inputs or auxiliary training targets in this study.
[Exact per-ID errors](native_text_tuning_errors_20261013.json):

| Group/count | Frozen scalar | Top4 scalar | Simple fusion |
|---|---:|---:|---:|
|Exact zero/92|0.2842|0.3599|0.3039|
|Weak abs(M)≤0.4/236|0.3058|0.3654|0.3288|
|Strong abs(M)≥0.8/281|0.5988|0.4808|0.5068|
|Opposite T/A/V annotation signs/117|0.4883|0.5431|0.4718|

Text tuning helps strong sentiment but worsens weak/neutral/conflicting
examples relative to its frozen scalar reference. Global regression gain
does not prove conflict reliability or better calibration. True annotation
strata are retrospective; they cannot be used as deployment router features.
Extra text-tuning budget must be shared by all compared methods before any
later fusion effect is attributed to a new mechanism.

The author paper uses different BERT/OpenSMILE/OpenFace inputs and its reported
benchmark results are not this VALID suite. These public adaptations cannot
be presented as faithful paper reproductions or paper-level superiority.
[Author paper](https://arxiv.org/abs/2209.02604).

## Reproduction and evidence

Install the recorded`train,encoders` environment and obtain the pinned author
CSV/MacBERT locally. Native scalar targets retain their original[-1,1] scale.

```powershell
python scripts/train_native_text_tuning.py --config configs/native_simsv2_text_tuning_v1.json --seed 13 --smoke-limit 32 --stop-after 1 --output docs/native_text_tuning_smoke_stage1_20261013.json
python scripts/train_native_text_tuning.py --config configs/native_simsv2_text_tuning_v1.json --seed 13 --smoke-limit 32 --stop-after 1 --resume --output docs/native_text_tuning_smoke_stage2_20261013.json
python scripts/train_native_text_tuning.py --config configs/native_simsv2_text_tuning_v1.json --output docs/native_text_tuning_formal_20261013.json
python scripts/audit_native_text_tuning.py --suite docs/native_text_tuning_formal_20261013.json --output docs/native_text_tuning_audit_20261013.json
python scripts/analyze_native_text_tuning.py --suite docs/native_text_tuning_formal_20261013.json --baseline-suite docs/native_simsv2_baselines_v1.json --author-csv data/author/simsv2/meta.csv --output docs/native_text_tuning_errors_20261013.json
```

Formal existing runs require`--resume`. `--output` is the suite JSON;
`--run-root` can override the run directory. Checkpoints bind CSV/IDs,
pretrained/tokenizer files, current implementation/runtime, config/seed, full
RNG/optimizer/scheduler/cursor. A completed resume does not add updates.
Metrics stay float32 to preserve native boundary behavior. The
[six-run audit](native_text_tuning_audit_20261013.json) recomputes every score
and checks selected epoch, step budget, actual checkpoint identities and hashes.

Label-free provided-text inference is available for these trusted checkpoints:

```powershell
python scripts/predict_native_text.py --checkpoint runs/native_simsv2_text_tuning_v1/top4/seed_13/best.pt --pretrained-dir data/pretrained/chinese-macbert-base --input-jsonl work/native_text_demo_20261013/input.jsonl --output-jsonl work/native_text_demo_20261013/top4_predictions.jsonl
```

JSONL input contains unique`id` and nonempty`text`, without labels. Two official
VALID inputs actually ran on CPU; the
[checkpoint comparison](native_text_inference_audit_20261013.json) differs from
saved GPU outputs by at most1.1e-6. This is inference-flow verification, not a
new evaluation or final test.

Next: retain the measured strong-versus-neutral/conflict tradeoff. Choose one
specific uncertainty/fusion hypothesis with shared inputs/budgets and closest
prior-work controls. Source-disjoint method verification, MOSEI method
replication, final test, complete author media and release/deletion remain open.
