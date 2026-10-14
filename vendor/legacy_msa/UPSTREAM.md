# Vendored source and attribution

This directory contains the model subset copied from the read-only snapshot
`D:\MYL\JevAffect-backup\legacy-source\MSAbypkl\src\models`:

- `fusion.py`
- `attention.py`
- `text.py`
- `audio.py`
- `visual.py`

The upstream snapshot identifies the project as JevAffect / Cross-Lingual
Multimodal Sentiment Analysis. Its README credits YanSong Hu, YangLe Ma,
ZhouYang Wang, and RiJie Hao. The original root MIT `LICENSE` is copied here
unchanged; retain it and this attribution when redistributing the vendored code.

Local edits are limited to package-relative imports so these files can be
loaded below `vendor.legacy_msa` without a global `config` or top-level `src`
module collision. Learnable modules, tensor operations, layer dimensions,
forward values, and state-dict parameter names remain as in the copied model.

The historical snapshot's README links checkpoints from a shared Google Drive
folder. No `.pt`, `.pth`, or dataset pickle was present in the audited snapshot;
this copy does not include or claim those weights, training results, or their
reproducibility.
