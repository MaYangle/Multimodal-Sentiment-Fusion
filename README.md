# ModalAffect

[English](README.md) · [中文](README.zh-CN.md)

ModalAffect is a multimodal sentiment research project for speech, video and
text. It studies how modality evidence should influence a prediction when
signals disagree, while keeping each dataset's native task and evaluation rules.

**Status: research in progress.** This repository is the project's new GitHub
home. Source code, configurations and audited experiment records are being
prepared for publication. No verified original method, final-test result or
released checkpoint is claimed by this initial repository.

## Research scope

- CH-SIMS v2.0 first, followed by CMU-MOSEI, with separate native scores and splits.
- Competitive text, simple fusion and sequence baselines before new mechanisms.
- Controlled studies of target-free modality policies and normalization in fusion.
- Real training, three-seed comparisons, ablations and independent prediction replay.
- Clear records of failed hypotheses, missing author data and reproduction limits.

The project was developed locally under the working name **JevAffect**. Jev is
design inspiration; this project does not use Jev weights or claim to reproduce
its private training recipe. Routing, attention and distillation have established
prior art; implementation alone is not evidence of novelty or performance gain.

## Data and reproducibility

Author-source data acquisition and integrity checks remain in progress. Data,
third-party encoder weights and experimental checkpoints are not included in
this initial repository. Their provenance and usage conditions will be documented
with the runnable source release. Do not treat this scaffold as a complete
benchmark reproduction.

## Contributions

Planned contribution areas include encoder integration, native evaluation,
fusion algorithms, controlled experiment reproduction and error analysis.
Runnable instructions and contribution guidance will accompany the source release.

## Reference entry points

- [CH-SIMS v2 author repository](https://github.com/thuiar/ch-sims-v2)
- [CH-SIMS v2 data page](https://thuiar.github.io/sims.github.io/chsims)
- [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)
- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
