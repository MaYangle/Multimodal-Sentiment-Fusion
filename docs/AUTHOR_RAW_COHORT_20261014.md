# Author raw TRAIN/VALID cohort acquisition — 2026-10-14

The acquisition queue targets all **3,369** official CH-SIMS v2 raw clips in
the author metadata: 2,722 TRAIN and 647 VALID. Each selected canonical ID is
bound to one exact `Raw\\<video_id>\\<clip_id>.mp4` path and Drive file ID in
the pinned public listing. TEST rows are not selected; the CSV reader checks
split markers and uses only `mode`, `video_id`, and `clip_id`. It does not
inspect or numerically parse score fields.

The exact source pins and per-ID status, byte counts, SHA-256 values, partial
state, and competition-cache SHA comparisons are kept in the ignored local
manifest at `data/author/simsv2/raw_cohort_v1/manifest.json` and mirrored in
[the dated cohort receipt](author_raw_cohort_20261014.json). Completed files
have an adjacent `.provenance.json` receipt. Preserved `.partial` files are
marked incomplete and are never reported as available training media.

Each new item uses one serial author `drive.usercontent.google.com` byte-zero
preflight. A complete payload is requested only after an exact HTTP 206 range,
total length, and Last-Modified version check; its finite whole-object range
must fit the 32 MiB per-file ceiling. Requests use a 15-second connect and
60-second read timeout, with at most one transport retry in a run. A quota
HTML response stops the entire queue immediately. Local partials are resumed
only after a fresh preflight confirms the same publisher version and the saved
file ID, byte count, prefix, and partial SHA.

The queue reuses the three previously verified author MP4s when their exact
file ID, official source URL, receipt, size, and SHA match: `video_0001/0003`,
`video_0003/0002`, and `video_0006/0006`. New downloads and verified copies are
stored under the ignored `data/author/simsv2/raw_cohort_v1/` tree. The process
is serial, CPU-only, and performs no feature extraction or training.

The plan check matched the pinned metadata and listing SHA-256 values and
confirmed all 3,369 exact TRAIN/VALID ID-to-Drive-path mappings. Seven offline
unit tests passed, including fake transport coverage for a complete file and
receipt, verified cache reuse, interruption and same-version suffix resume,
version-change preservation of a partial, and stopping the queue at quota HTML.

The live serial process was PID `33184`. It wrote stdout to
`work/author_raw_cohort_20261014.out.log` and stderr to
`work/author_raw_cohort_20261014.err.log`. It stopped at the first quota HTML
response, on `video_0001/0020`, after 15 new whole MP4s totaling 45,964,616
bytes. The failed payload response was HTTP 200 `text/html`, with 2,040
diagnostic bytes and a detected quota phrase; the valid preflight had reported
1,617,149 bytes and Last-Modified `Thu, 08 Sep 2022 09:09:42 GMT`. The request
added zero media bytes, no partial was left, and no later ID was attempted.

Two verified cached MP4s were copied locally into the cohort tree after the
quota stop: `video_0003/0002` and `video_0006/0006`, totaling 3,874,478 bytes.
The legacy `video_0001/0003` sample is outside the selected TRAIN/VALID plan
and was not copied. The stopped manifest now has 17 complete files and
49,839,094 available bytes out of 3,369 expected, with zero partial bytes.
The corrected offline `--check` verified all 17 local receipts and hashes.
No network request followed the quota response.

No corpus-wide byte-identity claim is made: each completed file is compared
independently to its competition-cache source SHA only when that exact ID has
a recorded source hash.
