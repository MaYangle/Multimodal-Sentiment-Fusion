# Source publication — 2026-10-14

This research snapshot publishes the implemented model architecture, staged
training, native baselines, data preparation/audit scripts, experiment configs,
seven-arm three-seed results, negative studies, English/Chinese documentation,
contribution templates and a CPU contract workflow.

Verification before pushing:

- 341 contract tests pass in the independent publication checkout.
- The synthetic CPU architecture demo runs with 2,084,871 parameters and a
  finite feature-only forward pass; it uses random weights, not accuracy data.
- A wheel builds from the publication checkout without changing training dependencies.
- All Markdown local-file links under docs and the current front doors resolve.
- Ten frozen model/trainer/config/plan dependencies retain the exact SHA-256
  bytes bound by existing checkpoints, including in staged Git blobs.
- Final independent production audit: 27/27 cells; 1,733 checks; 1,729 pass.
  All four failures are scientific candidate gain gates, not integrity failures.
  All 21 fusion checkpoints replay 647 VALID rows on CPU with maximum
  saved-device difference below 4.93e-7; full feature byte hashes are checked.

The final auditor fixes a truth-map reset and ranks teacher controls using
native clipped CH MAE, with a production-path regression fixture. The fixed
full audit was rerun. Trained models, registered budgets and frozen configs
were not changed. Existing upstream/historical whitespace is retained.

Data, foundation weights, checkpoints, local runs/work/backups and per-sample
records are excluded; redacted JSON reports disclose original hashes and
removed paths. The repository has no published checkpoint download. This is
source publication, not confirmation of an original-method gain, final TEST
performance or fresh complete gate-matrix reproduction. The CPU GitHub workflow
is newly configured; its remote outcome is a separate verification.

Current result: registered candidate VALID MAE 0.378497 ± 0.007455, versus
matched distilled-static 0.378253 ± 0.008851. See [experiments](EXPERIMENTS.md)
and [reproduction limits](REPRODUCING.md). The prior old-repository deletion
was separately authorized and followed verified history/checkpoint backup.
