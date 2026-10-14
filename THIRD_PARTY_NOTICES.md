# Third-party attribution and redistribution boundary

JevAffect's original source is licensed under the root [MIT License](LICENSE). Bundled research-source ports retain their own notices:

| Directory | Upstream | License / original notice |
| --- | --- | --- |
| `vendor/almt/` | [ALMT](https://github.com/Haoyu-ha/ALMT) | MIT, copyright 2024 Haoyu Zhang; see `vendor/almt/LICENSE`. |
| `vendor/cmc/` | [CMC](https://github.com/gw-zhong/CMC) | MIT, copyright 2025 Guowei Zhong; see `vendor/cmc/LICENSE_CMC`. |
| `vendor/bcmlf/` | [BC-MLF](https://github.com/sunyulin0421/BC-MLF) | MIT, copyright 2026 Yulin Sun; see `vendor/bcmlf/LICENSE`. |
| `vendor/legacy_msa/` | MaYangle historical project (now deleted after verified local backup; see [migration record](docs/GITHUB_MIGRATION_20261014.md)) | MIT, copyright 2025 MaYangle; see `vendor/legacy_msa/LICENSE`. |

`jevaffect/source_robust.py` independently implements known exponentiated
group-risk weighting and gradient reversal. It does not bundle upstream source.
Credit [GroupDRO](https://arxiv.org/abs/1911.08731) and
[DANN](https://jmlr.org/papers/v17/15-239.html); adaptations and fixed settings
are documented in [the source baseline plan](docs/NATIVE_SOURCE_ROBUST_BASELINE_PLAN_V1.md).
They are known-method controls, not an original algorithm claim. No GEAR code
is copied; its paper/repository mismatch is documented separately.

`jevaffect/anchored_bernstein.py` and `shape_controls.py` independently
implement classical Beta/Bernstein mixture CDFs, censored logistic mixtures,
implicit median differentiation and a constrained monotone spline reference.
No upstream implementation is bundled for these modules. The
[primary-source prior-art review](docs/NATIVE_ANCHORED_BASIS_REVIEW_20261013.md)
credits density/quantile/calibration precedents; these numerical constructions
and the negative shape study are not an original-algorithm claim.

The repository does **not** distribute the competition media, labels, locally derived feature caches, downloaded foundation-model weights, or trained checkpoints. Users obtain those separately under their source terms. The frozen text/audio/vision pipelines currently reference revision-pinned [multilingual BERT](https://huggingface.co/google-bert/bert-base-multilingual-cased), [Chinese MacBERT](https://huggingface.co/hfl/chinese-macbert-base), [WavLM](https://huggingface.co/microsoft/wavlm-base-plus), [DINOv2](https://huggingface.co/facebook/dinov2-small), and optional [Whisper](https://huggingface.co/openai/whisper-base). Their weight redistribution permissions must be assessed separately before any model Release. In particular, the project's captured WavLM model-card metadata does not specify a license; this notice makes no redistribution claim for it.

Data provenance and permitted uses are separate from code licensing. See the competition's [rules](https://www.kaggle.com/competitions/msa-challenge-at-the-4th-pazhou-ai-competition/rules), the dataset manifests under `docs/`, and each source repository. TypeSafe Jev is not a dependency or bundled model; “Jev-inspired” refers only to independent structured probabilistic decisions, as explained in `docs/JEV_SCOPE.md`.

`jevaffect/sequence_gates.py` is an independent attention/adapter/policy
implementation, not distributed KuDA source or weights. Its score-error
teacher and knowledge-guided comparison are credited in the
[KuDA adaptation review](docs/NATIVE_KUDA_ADAPTATION_DESIGN_20261014.md), with
established normalization, routing and distillation precedents in the
[closest-work review](docs/NATIVE_SEQUENCE_GATE_PRIOR_REVIEW_20261014.md).
The failed candidate does not establish a new algorithmic contribution.
