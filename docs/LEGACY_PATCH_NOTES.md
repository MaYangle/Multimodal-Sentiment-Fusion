# Legacy adapter patch notes

## Purpose and boundary

This work area contains an importable copy of the old **MSAbypkl** fusion
model, wrapped for the unified trainer's sequence batches. It is a fair
vector-level structural baseline against another model using the same
per-modality features. It does **not** reproduce historical training or
checkpoints and does not repair the model's temporal limitations.

The read-only source was `D:\MYL\JevAffect-backup\legacy-source`. Nothing in
that backup was edited. Only the five model files required by
`TransformerFusionModel` and the upstream MIT license were copied to
`vendor/legacy_msa/`.

## Model files and parameter compatibility

`vendor/legacy_msa/src/models/{fusion,attention,text,audio,visual}.py` retain
the original module definitions and learnable layers. The only edits inside
those copied model files are import-path changes:

- `from config import NUM_CLASSES` now imports the vendor package's minimal
  `NUM_CLASSES = 5` shim.
- `fusion.py` uses relative imports for its text/audio/vision encoders and
  attention module, avoiding collisions with the application's global
  `config` or `src` packages.

`LegacyMSAFusion` subclasses the original `TransformerFusionModel`, so it adds
no parameters and keeps the same state-dict key names. It fixes the output
width at five classes, matching the audited MSAbypkl configuration. The
default dimensions remain 768 text, 40 audio, 35 vision, hidden size 256, 8
heads, 4 text layers, and 0.3 dropout. Audio/vision encoder dimensions and
layer counts follow the original constructor logic.

## Sequence adapter behavior

`jevaffect/legacy.py` accepts a mapping with `text`, `audio`, and `vision`:

- `[B,D]` vectors pass through unchanged.
- `[B,T,D]` text keeps position zero, matching the old BERT `[CLS]` summary.
- `[B,T,D]` audio and vision are averaged over valid positions, matching the
  old clip-level temporal mean.
- Valid positions may be described with `lengths`/`masks` mappings or the
  `length`/`mask` aliases. A mask uses `True` for valid positions. Both forms
  can be combined; each valid frame must satisfy both.
- All metadata is consumed by the adapter and never enters the old model.

After reduction, the original encoders receive one vector per modality and
expand it to one token. Thus both self-attention and cross-attention still
operate on single-token sequences; the cross-attention softmax has only one
key. This adapter intentionally preserves that limitation. It must not be
reported as temporal fusion.

## Provenance and redistribution

`vendor/legacy_msa/LICENSE` is copied unchanged from the upstream snapshot.
`vendor/legacy_msa/UPSTREAM.md` preserves the source path and README credits
YanSong Hu, YangLe Ma, ZhouYang Wang, and RiJie Hao. Keep these files when
redistributing this vendor subset. The audited snapshot contained no model
checkpoint or feature pickle; this patch makes no claim about the external
Drive weights or the old README's reported score.

## Verification status

No dependency installation, training, GPU execution, or model test was
performed. A standard-library `compile()` pass succeeded for all 12 Python
files in this work area without importing project dependencies. The model
cannot be behavior-tested here; the main agent can run model checks after its
environment is ready.
