# Native CH-SIMS media inference

2026-10-14: two freshly downloaded author-original TRAIN MP4s now pass the full
CPU encoder/scaler/fusion inference path, matching cached scores within 1.669e-6.
[Actual author-media rehearsal](NATIVE_AUTHOR_MEDIA_DEMO_20261014.md) verifies
each source receipt and keeps transcripts/targets separate. This upgrades two
sample identities only; complete corpus and final-method reproduction remain open.

2026-10-13 addition: standardized pooled checkpoints are supported. The
original encoder/truncation recipe runs first, followed by CLS/masked mean and
the bound TRAIN-only scaler. Two genuine TRAIN MP4s match cached normalized
inputs within3.815e-6. Descriptor-only loading avoids training arrays;
[new baseline, export and inference commands](NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md)
record precise scope and redistribution limits. Use `--raw-source-cache` to
point to a relocated bound encoder descriptor. These paths remain label-free,
with no author-original whole-cohort or latency claim.

Implemented and actually run2026-10-13. Input is an MP4 plus a provided
transcript; output is the original scalar score, dataset range-clipped score,
checkpoint/feature hashes, and a separate feature/source/truncation audit.
The native checkpoint loader validates the exact trained architecture and
cache descriptor. No sentiment label, invented confidence or historical
five-class mapping enters this path.

## Encoder and input contract

This entry supports the frozen CH-SIMS MacBERT/WavLM/DINOv2 native checkpoints.
It reads the bound descriptor rather than choosing encoders from dimensions.
MacBERT revision/weights/all ancillary files must match the trained record;
WavLM and DINOv2 installations are checked against fixed published pins.
Every model load is local-only. It tokenizes provided text to64 positions,
decodes16kHz audio and224px frames at2fps(max120 seconds), uses the same
WavLM/DINOv2 processing, keeps the first400 audio /55 visual steps, and
zero-pads the invalid suffix. Valid lengths and actual truncation are recorded.

Provided nonempty transcripts and decodable required modalities are required.
There is no automatic ASR, verified missing-modality robustness or arbitrary
encoder substitution. The MOSEI author-linked GloVe/COVAREP/FACET feature
checkpoint has a different contract and is explicitly rejected by this loader.
Use its feature-input entry instead. Fine-tuned text checkpoints also need
their own backbone contract and are not silently substituted here.

```powershell
python scripts/predict_native_media.py --checkpoint runs/native_simsv2_v1/early_fusion/seed_13/best.pt --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --macbert-dir data/pretrained/chinese-macbert-base --model-root data/models --input-jsonl work/native_raw_demo_20261013/cached_train_input.jsonl --output-dir work/native_raw_demo_20261013/early_fusion_cpu --device cpu
```

Install`.[train,encoders]` first and obtain the exact trusted checkpoint and
source-bound encoder files. Each JSONL record contains`id`,`video_path` and
`transcript`; video paths may be absolute or relative to the input file.
Use a new output directory; existing predictions are not overwritten.
`features.jsonl`/NPZ can also be inspected and reused by the native feature CLI.
Data, transcripts, feature arrays and checkpoints remain outside Git.

## Actual verification and limits

Two genuine official TRAIN IDs,`video_0001/0001` and`video_0001/0002`, were
decoded and encoded on CPU in the new isolated dependency environment.
The media was previously downloaded through the competition channel.
Its SHA exactly matches each cached source record; this is not proof of
byte-equivalence to the unavailable author-original video distribution.
No TRAIN target was supplied to inference, and no TEST record was used.

The [source record](native_raw_demo_source_20261013.json) and
[numerical audit](native_raw_inference_audit_20261013.json) verify exact kept
lengths, source SHA, output feature/checkpoint SHA and actual encoder paths.
Scores are−0.928123 and0.290033. Recomputed predictions differ from the same
checkpoint on prior cached inputs by at most1.7e-6. Text/audio/vision maximum
feature differences are7.2e-6/1.9e-5/6.3e-5; feature bytes are not identical
across the prior GPU cache and new CPU extraction. This is a software/encoder
flow rehearsal on TRAIN examples, not held-out performance or full-cohort
re-extraction reproduction.

Two fresh author-original VALID retrievals were attempted through exact public
IDs/finite ranges but returned quota HTML before payload completion:
[acquisition result](author_valid_raw_acquisition_20261013.json). Their one-byte
prefixes are partial downloads, not usable media. The previously verified
author TEST MP4 remains a blinded integrity sample and was not predicted.
