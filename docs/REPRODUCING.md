# Reproduction paths and current limits

Run all commands from the repository root. The public tree includes source,
configuration, tests, upstream notices, frozen plans and experiment summaries.
Media, transcripts, native labels, encoder weights, feature caches, checkpoints
and full per-sample predictions are local artifacts and are not distributed.
Published JSON reports may be explicitly redacted; see [publication scope](PUBLICATION_SCOPE.md).

## Installation and code-only demonstration

Python 3.12 is the measured runtime. The formal Windows environment used Python
3.12.10, PyTorch 2.11.0+cu128, NumPy 2.3.5, SciPy 1.18.0, Transformers 4.51.3,
einops 0.8.1, Pillow 11.3.0 and imageio-ffmpeg 0.6.0. The constraints file records
the numerical versions. Runtime versions were observed during the gate study;
the training identity itself binds runtime flags rather than version strings.
The Git attributes preserve source/config line endings because checkpoint
identities hash their exact bytes. Do not normalize a frozen implementation's
line endings and then attempt to resume its old checkpoint.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c configs/reproduction_constraints_20261013.txt -e ".[train,data,encoders,verify,uncertainty]"
python -m unittest discover -s tests
python scripts/demo_sequence_architecture.py
```

For CPU-only code inspection, install the CPU PyTorch 2.11.0 wheel and use
`configs/ci_cpu_constraints.txt` with `.[train,data,encoders,verify,uncertainty]`
instead of the CUDA constraint file. It is not the exact
formal numerical environment. The synthetic demo uses random weights and
reports shapes/routes, not trained sentiment accuracy. Linux uses
`source .venv/bin/activate` instead of the Windows activation command.

## Acquire and prepare data

Use [CH-SIMS v2 author's repository](https://github.com/thuiar/ch-sims-v2),
[author data page](https://thuiar.github.io/sims.github.io/chsims), and
[CMU author SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK).
Read [acquisition status](AUTHOR_ACQUISITION_STATUS.md) before downloading.
`scripts/download_author_metadata.py`, `scripts/list_author_downloads.py`,
`scripts/download_author_payload.py` and `scripts/download_ranged.py` expose
their arguments with `--help`; ranged downloads retain verified receipts and
partial files. The full CH author feature pickle is incomplete and unusable;
17 TRAIN/VALID author videos were acquired before a recorded quota stop. A
small byte-matched subset does not prove full media equivalence.

The completed CH study uses author-ID-matched, competition-acquired media with
project-derived frozen MacBERT/WavLM/DINOv2 features, not a complete verified
author raw-media cohort. [Historical preparation](HISTORICAL_README.md) preserves
encoder/cache commands, and [raw inference](NATIVE_MEDIA_INFERENCE.md) documents
the implemented media path. Dataset access and terms are prerequisites; the
public repository cannot currently rebuild the entire CH input automatically
from the available author media. Do not substitute different inputs and call
the result an exact reproduction.

For the exact CH study, populate these verified local paths from your acquired
data and cache preparation:

- `data/author/simsv2/meta.csv`: author's 4,403-row CSV, SHA in the frozen config.
- `data/feature_cache/simsv2_full_macbert_raw_av`: full sequence cache with IDs,
  lengths, provenance and revision-pinned encoders.
- `data/feature_cache/simsv2_native_pooled_standardized_v1`: TRAIN-fitted coordinate
  statistics produced by `scripts/prepare_native_pooled_standardization.py`.

Then preserve the sequence and masks while applying those TRAIN statistics:

```powershell
python scripts/download_author_metadata.py --dataset simsv2 --output-dir data/author/simsv2
python scripts/prepare_native_pooled_standardization.py --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --author-csv data/author/simsv2/meta.csv --output-dir data/feature_cache/simsv2_native_pooled_standardized_v1
python scripts/prepare_native_sequence_standardization.py --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --pooled-standardized-cache data/feature_cache/simsv2_native_pooled_standardized_v1 --output-dir data/feature_cache/simsv2_native_sequence_standardized_v1
```

The author-SDK-linked MOSEI MultiBench package is available as a complete,
verified feature pickle. [Cohort details](MOSEI_MULTIBENCH_COHORT.md) disclose
16,265/1,869/4,643 rows and four TRAIN IDs outside the current SDK fold list.
Use the verified receipt and `scripts/prepare_mosei_author_features.py --help`;
unpickling requires explicit `--trust-author-pickle`. Native MOSEI baselines use
their separate configs and evaluation protocol, not CH label rescaling.

## Train, resume and independently evaluate

The CPU fixtures test seven arms and exact resume before a GPU smoke. The
smoke uses 32 real samples per applicable split and separate output paths:

```powershell
python scripts/train_sequence_gates.py --config configs/native_sequence_gates_v1.json --smoke 32
python scripts/train_sequence_gates.py --config configs/native_sequence_gates_v1.json
python scripts/train_sequence_gates.py --config configs/native_sequence_gates_v1.json --resume
python scripts/audit_sequence_gates.py --root . --report work/sequence_gate_audit.json
```

The formal command runs source-held branch selection, full-TRAIN refitting,
then all seven fusion cases for seeds 13/42/2026 serially. `--stage`, `--seed`,
`--case`, `--stop-after` and `--output` provide bounded execution; inspect
`--help`. Existing runs require `--resume`; no concurrent GPU matrix is needed.
Checkpoints retain best/last states, optimizer and loader RNG states, identity,
history and selected VALID predictions. The independent audit hashes inputs,
checks splits/budgets/masks/metrics and replays the selected checkpoints on CPU.

Run the older M-only Chinese controls separately:

```powershell
python scripts/run_native_baselines.py --configs configs/native_simsv2_text.json configs/native_simsv2_early_fusion.json configs/native_simsv2_almt.json --output work/ch_native_baselines.json
python scripts/run_native_baselines.py --configs configs/native_simsv2_sequence_standardized_almt.json --output work/ch_sequence_almt.json
python scripts/train_author_baselines.py --config configs/native_mosei_author_baselines_v1.json --output work/mosei_native_baselines.json
```

## Trained inference

Existing native M-only pooled/ALMT checkpoints support a label-free NPZ demo:

```powershell
python scripts/make_native_feature_demo.py --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --output-dir work/native_feature_demo
python scripts/predict_native_features.py --checkpoint runs/native_simsv2_v1/early_fusion/seed_13/best.pt --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --input-jsonl work/native_feature_demo/input.jsonl --output-jsonl work/native_feature_demo/predictions.jsonl --device cpu
```

That command supports the baseline checkpoint schema; it does not load the new
sequence-gate `best_state` schema. The gate model's target-free `model(features)`
API and CPU audit replay are implemented, while a standalone gate checkpoint
inference CLI remains a contribution task. The two acquired author TRAIN videos
also completed the real CPU encoder-to-native-score path within 1.67e-6 of cached
scores; [details](NATIVE_AUTHOR_MEDIA_DEMO_20261014.md). No download link for
research checkpoints has been published. Train them locally using these configs.

## Reproduction already performed

[Fresh native baseline reproduction](FRESH_NATIVE_REPRODUCTION_20261013.md)
used a new isolated environment and repeated the nine-run author-feature
baseline workflow. Chinese pooled standardization and the negative CRPS head
study also have source-copy repeats. These are scoped checks, not a fresh
end-to-end repetition of the latest 27-cell gate study. That repetition,
full author-media preparation, MOSEI method replication and final TEST
evaluation remain outstanding. The existing final TEST split was not used to
select the gate method.
