# Cohort runner stop and code addendum — 2026-10-14

PID `33184` stopped on the first quota HTML response, for author TRAIN ID
`video_0001/0020` (Drive ID `1pAXGO-TbvXzwyvDPLN1Ja4WSvipqLHWO`). Its single
`bytes=0-0` preflight returned HTTP 206, `bytes 0-0/1617149`, and Last-Modified
`Thu, 08 Sep 2022 09:09:42 GMT`. The one full finite-range attempt returned
HTTP 200 `text/html`, 2,040 diagnostic bytes, and a quota phrase. It delivered
zero media bytes; the queue stopped before requesting another ID. The cohort
manifest records 15 new complete files and no partials at that stop.

After the process exited, inspection found two runner defects in paths the
live queue had not exercised: a missing local prefix variable in cross-run
partial validation and an unrelated-variable reference in complete-file
receipt validation. The 15 payloads had already been fully received; a later
offline check with the corrected validator confirmed every one against its
receipt size and SHA-256. The corrected runner now passes nine offline tests,
including same-version partial resume across calls, rejecting a corrupted
partial prefix, complete-file receipt validation, verified cache reuse, and
quota-stop behavior.

For code identity, the exact source file preserved at
`work/author_raw_cohort_20261014.started.py` has SHA-256
`2efb0a7f0008983064dd9dca8fe00f22a9859f517fbf5a04936d8804a6b2998e`. Its
LastWriteTimeUtc was `2026-10-14T00:17:00Z`; PID 33184 started at local
`2026-10-14 08:17:19` (`00:17:19Z`). This is a post-stop observed source
snapshot supported by file-mtime evidence. No runtime code hash was recorded
at process launch, so the receipt does not claim launch-time hash binding.
The corrected runner SHA-256 is
`a1bca34680ffb90001f271c9f058573abdc0cc8743ad7009e4ac2e6fa1238639`; the
manifest records the observed and corrected hashes as separate versions and
marks the initial runtime hash as unpinned.

The two-file receipt used by the CPU media demo remains unchanged at SHA-256
`b265ba4f1984b91916f01f3b441bd31e507f6402b4390f11ad63603e1607ccec`. The
only action after the quota stop was offline reuse of two verified cached
author MP4s; no request to the quota-limited file or subsequent queue item was
made.
