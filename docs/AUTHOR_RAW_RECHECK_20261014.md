# Author raw recheck — 2026-10-14

The author’s public CH-SIMS v2 Drive endpoint now serves two exact TRAIN MP4s
that returned quota HTML at preflight on 2026-10-12. This recheck acquired both
files in full, decoded them with CPU FFmpeg, and matched each file’s SHA-256 to
the recorded competition-source media hash for the same canonical ID.

The selection was checked against `data/simsv2_drive_listing.json` and
`data/author/simsv2/meta.csv`. The CSV structure was scanned to locate the two
TRAIN rows; only `mode`, `video_id`, and `clip_id` field values were used. The
reader did not inspect or numerically parse score fields, and no TEST row was
selected for acquisition. The prior failed probes are preserved in [the
2026-10-13 receipt](author_train_raw_acquisition_20261013.json).
The exact headers, timestamps, hashes, and output paths are in the
[2026-10-14 machine-readable receipt](author_train_raw_recheck_20261014.json).

| Canonical TRAIN ID | Author Drive file ID | Exact full response range | Last-Modified | Bytes | SHA-256 | CPU decode |
|---|---|---|---|---:|---|---|
| `video_0003/0002` | `1knNivNIvYHVVMp20ET7fEugrqCcjlSBC` | `bytes 0-1755106/1755107` | `Thu, 08 Sep 2022 09:13:08 GMT` | 1,755,107 | `40cddabf922ef30c67e2be2fb49fb5023a89c57b9c03674ba51fcca87c9f3e1e` | FFmpeg return code 0 |
| `video_0006/0006` | `1d4I2zWU4DnDZpf4I7Cxc64iNiSXAH1Hn` | `bytes 0-2119370/2119371` | `Thu, 08 Sep 2022 09:25:47 GMT` | 2,119,371 | `b37619116720abfc82ffae00355db07b0a38260123db9733a5ce8d8f6ab27933` | FFmpeg return code 0 |

Each one-byte preflight returned HTTP 206 with the exact `bytes 0-0/total`
range, `video/mp4`, and the same `Last-Modified` value later returned by its
single full finite-range request. Each transfer stayed below the 33,554,432
byte per-file limit. There were no retries, account or cookie use, alternate
endpoints, range chunking, GPU work, or feature extraction.

The downloaded files are in the ignored directory
`data/author/simsv2/raw_recheck_20261014/`. Their respective SHA-256 values
equal the competition-cache `metadata.decode.source_sha256` for these two IDs.
That establishes byte identity for these two comparisons only; it does not
establish identity for the remaining CH-SIMS v2 raw corpus or upgrade any
corpus-wide acquisition claim.
