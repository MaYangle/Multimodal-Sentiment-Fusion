# Feature-cache converter notes

## Source schema checked (code only)

No official pickle was opened. The converter contract was based on the
read-only loaders already in the workspace:

- `D:\MYL\JevAffect\work\ALMT\core\dataset.py:23-44` loads
  `data[split]["text_bert"]`, `audio`, `vision`, `raw_text`, `id`, and
  `regression_labels`; `valid` is an explicit split at `:70-81`.
- `D:\MYL\JevAffect\work\CMC\dataloader.py:40-47` loads the same
  `text_bert/audio/vision/id` arrays. It derives text length from
  `text_bert[:,1,:]` and reads `audio_lengths`/`vision_lengths` at `:70-73`;
  its `:48-68` classification labels are derived mappings. This cache instead
  uses the authoritative manifest's mapped `Label` for trainer targets and
  keeps the source pickle's raw score in a separate file.
- The current CH-SIMS v2 manifest builder uses string IDs `video_id/clip_id`,
  official `mode` values `train/valid/test`, and explicit
  `label_status`/`raw_label` fields. It permits unresolved test labels only;
  the converter requires a complete manifest and cross-checks every ID and
  split before writing.

The real pickle download is not complete, so its keys/shapes and ID encoding
remain unverified. When a verified source is available, the CLI summary reports
the observed `text` and `text_bert` shape/dtype for each split without printing
class distributions or raw labels.

## Text schema and readiness

The training contract is `text.npy: [N,T,D]` floating contextual features.
The converter selects source `text` only when every split has a numeric float
array `[N,T,expected_text_dim]` (default `expected_text_dim=768`) and it is not
the channel-first three-plane token bundle. It does not transpose or reshape
`text_bert` into `text.npy`.

If the source only has `text_bert: [N,3,T]`, the cache writes
`text_bert.npy`, `text_mask.npy`, and `text_lengths.npy`, sets
`text_features_ready=false`, and reports `embedding-stage-required`. The
default `load_cached_split(..., require_text_features=True)` raises
`CacheNotReadyError`, so this token-only cache cannot enter a trainer that
expects contextual sequences. The BERT embedding stage is intentionally
outside this NumPy-only tool and remains pending.

When `text.npy` is accepted, shape and dtype are verified. The cache records
that semantic provenance still depends on the source field definition; it
does not run BERT or infer context semantics from token IDs.

## Cache files and label policy

Each split directory contains float32 `audio.npy`/`vision.npy`, and either
ready `text.npy` or separately named `text_bert.npy`. It also contains int64
class-index `labels.npy`, `label_valid.npy`, raw-score `raw_labels.npy` and
`raw_labels.json`, `manifest_labels.json`, canonical `ids.json`, source ID
`id_mapping.json`, and true-means-valid masks plus length arrays for
text/audio/vision. NPY writes specify `allow_pickle=False`; reads use
`np.load(..., allow_pickle=False, mmap_mode="r")` by default.

If `raw_text` exists as `N` strings in every split, the exact strings are
stored in `raw_text.json`; `raw_text_hash_audit.json` records normalized-hash
matches/mismatches against the manifest without modifying the stored text. If
it is missing, `meta.json.raw_text_available` is false and
`load_cached_split(..., require_raw_text=True)` refuses the split. GPT/transcript
input therefore cannot be claimed from token IDs alone.

The trainer's `labels.npy` contract is int64 class indices from the manifest's
already mapped `Label` using `SNEG,WNEG,NEUT,WPOS,SPOS -> 0,1,2,3,4`. This
converter does not bin raw scores. Source continuous values are preserved in
`raw_labels.npy` and `raw_labels.json`. Unknown or unresolved labels in train or
valid fail conversion. An unresolved test label stays invalid: `labels.npy`
contains `-1`, `label_valid.npy` is false, and any original pickle value remains
in the raw-label files.

`ids.json` is a list of canonical manifest IDs in source order. Original
pickle IDs and the explicit normalization choice are kept in `id_mapping.json`;
the reader checks that their canonical list matches `ids.json`. IDs are never
stringified from integers. The caller must declare `--source-id-format` and may declare
`--manifest-id-format`. Supported grammars include verbatim strings,
`video/clip`, and the explicitly parsed CMC/SIMS form `video$_$clip`; there is
no automatic delimiter guessing. Duplicate, missing, extra, or split-mismatched
IDs fail conversion.

## Padding, lengths, non-finite features, hashes

Feature arrays retain their full source time dimension. Non-finite feature
elements are replaced with zero and counted per modality/split; labels are not
sanitized into zero. Masks use `True` for valid positions. Text mask comes
from a matching BERT attention-mask channel when available. Audio/vision masks
come from the source length fields when available. If lengths are absent, full
length is assumed and recorded as risky unless the caller opts into
`--infer-zero-padding-lengths`; that heuristic treats trailing all-zero rows
as padding and records its risk and row counts. Zero-valued feature frames can
be real observations, so heuristic lengths must be reviewed before training.

`meta.json` is the standard training contract: it exposes `source_sha256`,
`text_features_ready`, `raw_text_available`, `feature_dims`,
`sequence_lengths`, and explicit `split_paths` such as `splits/train`. The
dimensions and lengths come from the serialized NPY headers; train/valid/test
`[T,D]` axes must match or conversion fails. For token-only text,
`feature_dims.text` and `sequence_lengths.text` are null.

`metadata.json` keeps detailed provenance, raw schemas, split lengths, ID
hashes, transcript audit results, non-finite replacement counts, padding
risks, and hashes/sizes of output artifacts. `meta.json` includes the detailed
metadata SHA-256. `metadata.sha256` is a shared checksum index for both
`meta.json` and `metadata.json`; the reader verifies both without a circular
hash dependency.

## Trust boundary and use

Pickle is executable input. `prepare_feature_cache.py` requires both
`--trust-source-pickle` and a caller-supplied `--expected-sha256`; it checks the
digest before calling `pickle.load`. Only use a source whose origin and digest
you have verified. Conversion never downloads data or writes to the legacy or
project `D:` trees.

Example after the official source and manifest have been verified (not run):

```powershell
python work/cache_impl/scripts/prepare_feature_cache.py `
  --source-pickle <verified-source.pkl> `
  --expected-sha256 <64-hex-digest> `
  --source-reference <official-dataset-version-or-approved-source> `
  --manifest <author-manifest.jsonl> `
  --source-id-format video_dollar_clip `
  --manifest-id-format video_clip_path `
  --output-dir <new-cache-directory> `
  --trust-source-pickle
```

The converter has only been exercised with synthetic NumPy/pickle fixtures.
Those fixtures are protocol tests, not experimental data, and no model was
trained or scored.

