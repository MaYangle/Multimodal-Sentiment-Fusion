# Native CH-SIMS text adaptation baseline

Registered2026-10-13 before real training. Current frozen-text MAE0.4373 and
simple-fusion0.4221 do not establish a competitive fine-tuned language baseline.
Earlier top-four MacBERT training used historical five-class/source-holdout
targets and cannot substitute for native continuous-score experiments.

## Controlled comparison

Use the pinned MacBERT-base, author-provided transcripts, original TRAIN2722 /
VALID647 IDs and M scores on[-1,1]. Only these two split labels are parsed.
No test tokens, labels, features or score-based source selection. Same CLS,
dropout0.1 and scalar head for frozen versus top-four-layer adaptation.
Bottom eight encoder layers, embeddings and pooler remain frozen. The frozen
control retains the same encoder architecture and train-mode dropout; only
gradient eligibility changes. Report trainable versus total parameters.

Pretrained revisiona986e004d2a7f2a1c2f5a3edef4e20604a974ed1 and weight SHA
db0506d985574b80c33eec1cf13bd4c130585568753175871095ca13dbad9e23, plus all
tokenizer/config/ancillary hashes, must pass before local-only loading. Eager
attention, no AMP, max64 tokens, float32 output. Keep original pre-trained
files and old cached experiments unchanged.

Same seeds13/42/2026, raw L1, AdamW encoder1e-5/head2e-4, weight decay0.01,
norm1 clipping, batch32, max12 epochs, patience3, step-linear scheduler with
10% warmup and fixed12-epoch total step denominator. Best model minimizes
native VALID MAE, first tie; clipping is evaluation-only under author rules.
Frozen and adapted cases share head LR, scheduler and patience. This budget
differs from old frozen-feature MLPs and is reported; gains over those are not
attributed to fusion or an original mechanism.

## Verification and next decision

CPU fixtures cover unparsed test targets, original float labels, source/model
hash checks, gradient eligibility and exact epoch resume/RNG/identity guards.
Real32/32 forward/backward/checkpoint reload/resume precede serial six formal
runs. Save identities, configs, runtime, raw validation predictions, histories,
checkpoints, actual updates/time/GPU memory. No concurrent GPU jobs.

Evaluate native MAE/Corr/Acc2 and weak/strong/neutral error groups; keep prior
baselines visible. Fine-tuning is a known baseline, not a new contribution.
If it improves, freeze the selected adaptation and give multimodal controls
the same text features/backbone and encoder-training budget before studying
the next mechanism. If not, retain the negative result and avoid claiming the
current fusion failures were caused by a weak text backbone. Source-disjoint
robustness, cross-dataset method replication and final test remain separate
future gates; do not substitute this adaptation study for them.
