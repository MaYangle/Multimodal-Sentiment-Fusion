# Native A/V time-origin and retained-coverage audit

**Finding: full-corpus physical A/V synchrony is UNVERIFIED.** The frozen arrays have clean, reproducible per-stream timestamp grids and support a numerical paired-sequence diagnostic. They do not establish a shared physical time origin. The media decoder sends audio into a raw PCM pipe and saves no first audio PTS or sample-origin offset. Video timestamps come from FFmpeg's fps=2:start_time=0 output timeline. Two local TRAIN clips show both streams starting at PTS 0, but that sample check cannot establish origins or drift across all 3,369 TRAIN/VALID records.

## What was checked

The audit read raw and normalized cache metadata, canonical IDs, and only the small TRAIN/VALID audio_times, vision_times, audio_lengths, and vision_lengths arrays. It did not read labels or feature tensors, access TEST data, or run a GPU, learned encoder, or prediction model. It verified the metadata sidecar and SHA-256 values for the inspected time/length arrays.

The audio grid matches WavLM's pinned convolution geometry: a 400-sample receptive field at 16 kHz gives a first center at **0.01246875 s**, with a 320-sample (**20 ms**) stride. The video grid is **0, 0.5, 1.0, … s**. Every array value is finite; all valid prefixes are monotonic with no duplicates; all timestamp padding is zero. There are no deviations from either expected grid above 1e-9 seconds.

The normalized sequence cache records that it applies the saved TRAIN clip-pooled scaler to valid sequence steps and preserves the raw audio_times.npy and vision_times.npy byte-for-byte. Its source metadata hash matches the raw cache metadata hash.

## Retained coverage

The overlap below is the closed interval intersection of each record's first-to-last valid timestamp supports. Outside-anchor counts count valid timestamp rows beyond that interval. Outside seconds sum the start and end support tails beyond the interval; they are support-span durations, not reconstructed media durations. The candidate join uses a zero audio offset, a one-to-one nearest audio anchor for each video anchor, a 10 ms maximum difference, and requires both anchors to fall inside the numeric support intersection.

| Split | Records | Intersecting supports | No intersection | Median overlap | Outside anchors A/V | Outside support seconds A/V | Candidate pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 2,722 | 2,715 | 7 | 2.488 s | 61,548 / 2,930 | 1,210.817 / 111.197 s | 16,130 |
| VALID | 647 | 646 | 1 | 2.738 s | 15,159 / 697 | 298.393 / 26.713 s | 3,874 |

Candidate pair residuals are 7.531 ms on this cache's observed grids; the median is five candidate rows per record, with a maximum of 15. These are **numerical timestamp pairs**, not evidence that their rows represent physically simultaneous source events. Seven TRAIN clips and one VALID clip have no intersection between the observed timestamp supports; the machine report lists their IDs.

## Truncation

The cache keeps the first steps in original order without interpolation. Audio is capped at 400 rows: **69** records were truncated, dropping **6,723** rows in total (about **134.46 nominal 20 ms step-seconds**). Vision is capped at 55 frames: **one** record dropped 30 frames (15 nominal seconds at 2 fps). The dropped timestamps are absent, so the cache cannot recover those tails or quantify their per-record overlap. Do not pad, repeat, or stretch either retained prefix to match the other sequence.

## Representative source-media check

The two TRAIN clips named in native_raw_demo_source_20261013.json match their recorded SHA-256 hashes. With the bundled FFmpeg 7.1 build, both report container start 0, first audio packet/frame PTS 0, first video packet PTS 0, and filtered video output timestamps beginning at 0.0 seconds. Their first video packet DTS is −40 ms, consistent with H.264 decode reordering while its presentation timestamp remains 0. These observations apply only to those two clips. The audio decode path used for the cache does not retain the first audio PTS, and neither cache metadata file contains a corpus-wide per-record audio/video start-offset or drift record.

WavLM last_hidden_state rows also have transformer context that can span the clip. Their convolution-center timestamps are anchors, not proof that each representation contains only local audio.

## Pairing contract for a bounded diagnostic

Use matching canonical IDs and only rows before each modality's stored length. Establish and record the audio-to-video offset from source/decode records before describing pairs as physical-time synchronized. If that offset is unavailable, call the result a **paired-sequence diagnostic**. After any verified offset, restrict candidate rows to the retained support intersection and use a one-to-one nearest-time join with a declared tolerance (10 ms is compatible with the current 20 ms/500 ms grids). Report pair counts, timestamp residuals, support overlap, and outside-overlap rows. Drop unmatched rows; do not interpolate, duplicate, extrapolate, or stretch either sequence.

## Provenance and artifacts

The raw cache source SHA-256 is 11f8f242ee9f909b202bc0f98b5a122946bb07c842164c690102bb1aa7802d48; its TRAIN/VALID manifest SHA-256 is ca1290da3dfca63bd97ea946c0ea21ca2997476a99bd0f5f24e2b40b3dc986df. Raw meta.json SHA-256 is 1fae0633631fcde89bb0011e0f79953e9072a85978f78bc03f198273b01d7cda, raw metadata.json is c80806f155b246139b6cffba023d9c9b2359f42d38c1c7141db0af9cef8749c6, and the bundle sidecar SHA-256 is b56dbf03f08b76bf27f1060cc24928c8215fce9ffb12959febe4d2389987eeae. The normalized sequence metadata SHA-256 is f4952a6d3ecd31ab8b24b19e585b6a85b31d019b2cb4f622ba1a7e35a14be82b.

| Artifact | SHA-256 |
|---|---|
| [scripts/audit_native_av_times.py](../scripts/audit_native_av_times.py) | 75eb3d7a95011b31872d6ad367ed2fd6dbb64e5f7cb763ade6a628a76d54dcec |
| [docs/native_av_time_audit_20261014.json](native_av_time_audit_20261014.json) | 2e44d069a8ee3db871fb99ffc91f982abe9b80c37fdca4abfd2052c0436c1685 |

The five fixture tests passed. They cover offset alignment, a missing timestamp with no interpolation, monotonicity and zero-padding guards, partial support overlap, and the WavLM grid geometry.
