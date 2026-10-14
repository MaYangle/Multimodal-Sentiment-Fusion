# GitHub home and legacy deletion

User instruction on2026-10-14 explicitly changed the sequence: create the new
GitHub home first, then delete the old repository now. A later instruction
requested a more specific name. These instructions supersede the previous
requirement to wait for completed research before repository creation/deletion.
They do not waive the research, native evaluation or reproduction requirements.

The public repository is now
[MaYangle/Multimodal-Sentiment-Fusion](https://github.com/MaYangle/Multimodal-Sentiment-Fusion),
GitHub repository ID1418766312. It was initially created as ModalAffect, then
renamed with the same ID. The task-specific name describes multimodal sentiment
analysis and fusion research. Initial commits83b78d0 andf5bd7d3 contain only
English/Chinese research-status README files and `.gitignore`. This is a
repository home, not a completed source, checkpoint or benchmark release.

Before deletion, the exact old repository identity and remote refs were checked
again. Main f6f9aaf6d80f596e3600fca0aa0d452cfcca0414 and Memor
22a53e35864b281eb65360bc86a27f7c5e0d4a80 still matched the durable backup. A fresh
restore from `backups/legacy_20261012/legacy-all.bundle` recovered both refs and
all33 reachable commits; `git fsck --full` passed. Bundle and main-source archive
SHA256 values matched the historical receipt. `best_models.zip` matched SHA256
4faf129c1cc15f80c2faa0a32647d2f8bea4f97242e038143ce86fa5f551a5c2; both en.pt and
zh.pt (72,998,565bytes each) passed ZIP CRC. No model weights were executed.

The old GitHub metadata, issue/PR list and release list were exported into
`backups/legacy_20261014_predelete`. Existing source/data/model links remain in
the full Git bundle and source archive. The additional external pretrained
archive remains incomplete; deletion of GitHub did not delete its Drive channel
or local partials. The backup claim covers Git history/source and the acquired
best checkpoints, not every externally hosted payload.

`gh repo delete MaYangle/Multimodal-Sentiment-Analysis --yes` succeeded. The
authenticated API subsequently returned404 for that exact repository. The new
repository remains public and its main branch is present. The user's explicit
request authorized the deletion; no research-performance claim was used to
justify it.

The workspace remains `D:/MYL/JevAffect`, preserving live experiment paths and
checkpoint identities. Its source has not yet been uploaded in this initial
GitHub scaffold. The complete project goal remains active and incomplete.
