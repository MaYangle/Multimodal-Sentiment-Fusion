# Vendored source and attribution

`almt_layer.py` is copied from the ALMT author's repository:

- Repository: <https://github.com/Haoyu-ha/ALMT>
- Source path: `models/almt_layer.py`
- Source commit: `da9461071f21e75eb8bf2bedff14ac23f5954605`
- Commit metadata: Haoyu Zhang, 2025-04-09, “Update”
- License: MIT, copyright (c) 2024 Haoyu Zhang; the repository `LICENSE` is
  copied here unchanged.

The upstream README cites the ALMT paper by Zhang, Wang, Yin, Liu, Liu, and Yu
(EMNLP 2023). Keep this file and `LICENSE` when redistributing the vendored
layer. The source file also retains its comment that some code references the
Vision Transformer implementation at `gupta-abhay/pytorch-vit`; that reference
has not been independently license-audited in this task.

No upstream source files were modified. The adapter and classifier live in
`jevaffect/almt.py` outside this vendor directory.
