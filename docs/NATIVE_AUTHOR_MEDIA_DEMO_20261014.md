# Native inference from verified author media

Two newly acquired author TRAIN videos now complete the CPU media-to-score
pipeline in the isolated Python environment. Their SHA-256 values match both
the official-download receipts and the corresponding historical encoder-source
records. This establishes identity for these two examples only.

The pipeline uses the same pinned MacBERT, WavLM and DINOv2 weights, original
decoder and retained-prefix recipe, saved TRAIN pooled scaler, and selected
standardized fusion checkpoint (seed 13). It reads supplied transcripts and
no sentiment target. It does not evaluate TEST or claim held-out performance.

| TRAIN ID | Re-encoded raw score | Cached-input score | Absolute difference |
|---|---:|---:|---:|
| video_0003/0002 | -.884096444 | -.884097278 | 8.345e-7 |
| video_0006/0006 | -.659703314 | -.659704983 | 1.669e-6 |

The maximum standardized feature differences are T 1.001e-5, A 2.399e-5 and
V 1.833e-5. CPU and earlier GPU features agree numerically, not byte-for-byte.
The unchanged comparison thresholds are 2e-4 for coordinates and 1e-5 for
predictions. [Full audit](native_author_media_demo_20261014.json) binds the
checkpoint, cache descriptor, input/source hashes and label-free inference.
The earlier [two-video author download receipt](AUTHOR_RAW_RECHECK_20261014.md)
and media files are independent from the historical competition acquisition.

After obtaining the exact encoders, checkpoint, saved scaler, author CSV and
the two author media files recorded in the receipt:

```powershell
python scripts/rehearse_author_media_inference.py --author-csv data/author/simsv2/meta.csv --transfer-report docs/author_train_raw_recheck_20261014.json --checkpoint runs/native_simsv2_standardized_v1/early_fusion/seed_13/best.pt --cache-dir data/feature_cache/simsv2_native_pooled_standardized_v1 --macbert-dir data/pretrained/chinese-macbert-base --model-root data/models --output-dir work/author_native_media_demo_20261014
```

Use a new output directory; existing examples are preserved. The helper verifies
author media SHA/size/ID and cached-reference hashes, selects TRAIN transcripts,
creates label-free inputs, invokes the existing CPU inference entry and compares
the outputs with cached vectors. JSONL, NPZ and model/media payloads remain
outside Git. This is a real original-media rehearsal of a known baseline,
not full-corpus encoding reproduction or a verified original algorithm.
