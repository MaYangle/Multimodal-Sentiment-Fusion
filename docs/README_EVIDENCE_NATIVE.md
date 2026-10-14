# README evidence and publication synchronization — 2026-10-14

Mode: restructure and publication sync for an experimental research repository.
English/Chinese front doors now describe the implemented architecture, completed
three-seed study, negative result, entrypoints and actual data limitations.
Historical READMEs preserve commands and attribution.

| Claim | Status | Evidence |
|---|---|---|
| Current GitHub identity | Verified | MaYangle/Multimodal-Sentiment-Fusion; migration receipt |
| Architecture implemented | Verified | sequence_gates.py, CPU synthetic demo and fixtures |
| Formal matrix complete | Verified | 27/27 cells, 21 fusion fits; sequence_gate_results_20261014.json |
| Candidate gain | Negative | .378497 versus matched distilled-static .378253; all-seed gate fails |
| Native metrics/splits | Verified with cohort discrepancies disclosed | Author CSV, native protocol, independent MOSEI audit |
| Full author CH raw input | Missing | 17 TRAIN/VALID videos; full payload quota/incompleteness remains |
| Current gate matrix fresh-environment repeat | Missing | Earlier baseline repeats exist; current matrix repeat pending |
| Independent final audit | Reported separately | NATIVE_SEQUENCE_GATE_AUDIT_20261014.md supersedes partial snapshots |
| Publication-copy tests | Verified | 341 tests passed in the final publication checkout, including the production audit regression fixture |
| Linux CPU CI | Verified | Commit cfefa2c; 341 tests, one CUDA-only resume test skipped; synthetic demo passed |
| Current document links | Verified | English/Chinese READMEs, architecture, experiments, reproduction, contributing, notices |
| Frozen study bytes | Verified | Ten identity-bound files unchanged; Git attributes preserve line endings |
| Data/weight redistribution | Excluded | Explicit source allowlist and redacted report copies with original SHA |
| Legacy deletion | Verified after backup | 33 commits, restore/fsck, acquired checkpoint SHA/CRC, authenticated HTTP 404 |
| Original algorithm, final TEST, public checkpoints | Missing | No success or superiority claim |

Stale live-session and pending-GitHub statements were replaced in current
navigation; historical records keep their time-specific context. Added CI is a
CPU contract workflow. The [verified Linux run](https://github.com/MaYangle/Multimodal-Sentiment-Fusion/actions/runs/38558604043)
passes with fixed measured numerical dependencies. The initial broad-version
run failed a historical shape fixture and is retained in Actions. A whitespace
check finds existing upstream/historical whitespace; frozen
training bytes and upstream code were preserved rather than reformatted.
Code licensing does not grant dataset/encoder-weight redistribution rights.
