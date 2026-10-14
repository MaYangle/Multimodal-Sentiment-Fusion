# Native feature inference and local demonstration

The label-free CLI loads the project's trusted native text, early-fusion,
scalar ALMT, context-NIG or annotation-reference checkpoint and returns original
continuous scores. The rejected reference experiment can expose its trained
T/D components with decoder/supervision flags; they are not causal explanations
or calibrated confidence.
This is a feature-input demo. The isolated-environment feature route has passed
[preparation/test/smoke/inference verification](FRESH_NATIVE_REPRODUCTION_20261013.md);
full formal reproduction remains incomplete. A separate
[native CH-SIMS MP4 plus transcript path](NATIVE_MEDIA_INFERENCE.md) now runs
on two genuine TRAIN examples; full author-media acquisition and full-cohort
encoder reproduction remain incomplete. The historical
`predict_raw.py` still produces five-class pilot outputs.

Install `pip install -e ".[train]"` and obtain the exact trained cache and
checkpoint locally. Restricted features/checkpoints are not bundled. Create
two label-free examples from official VALID features (no test or label reads):

```powershell
python scripts/make_native_feature_demo.py --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --output-dir work/native_feature_demo
python scripts/predict_native_features.py --checkpoint runs/native_simsv2_v1/early_fusion/seed_13/best.pt --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --input-jsonl work/native_feature_demo/input.jsonl --output-jsonl work/native_feature_demo/predictions.jsonl
```

The same two examples have actually run through early fusion, scalar ALMT and
joint_dynamic checkpoints on CPU. The [comparison audit](native_feature_demo_audit.json)
compares outputs to saved GPU validation predictions. No labels or confidence
scores are supplied to the inference model; this rehearsal is not a new score
or a final-test result.

## Feature record contract

One JSONL record has `id`, `features_npz` and `metadata_sha256`. Paths can be
absolute or relative to the manifest directory. IDs must be unique; `target`
and `label` fields are rejected. `metadata_sha256` declares the exact trained
encoder/cache descriptor. Descriptor and architecture implementation hashes
are checked against checkpoint identity; caller-supplied new features must
come from that encoder/preprocessing contract. A matching declaration alone
does not prove a new recording was encoded correctly.

The NPZ stores each required `text`, `audio`, `vision` sequence as float
`[max_steps,feature_dim]` plus the corresponding integer scalar `text_length`,
`audio_length`, `vision_length`. The text-only model requires text only. Arrays
match trained shapes, valid regions must be finite, and lengths lie within the
sequence. CLS pools text; audio/vision use valid-prefix means. Sequence models
zero the invalid suffix. Unavailable modalities have length0, and an all-missing
record is rejected. Models trained on clean complete inputs have no verified
missing/noise guarantee just because the loader accepts masks.

The existing Chinese cache uses text64×768, audio400×768, vision55×384, but the
loader reads dimensions from the bound cache rather than fixing them globally.
The acquired MOSEI cache has aligned50 GloVe300/audio74/vision35 sequences;
its bound `text_pooling=valid_mean` averages text steps instead of selecting CLS.
Real MOSEI fusion/ALMT and Chinese reference-sum feature demos have also run.
The [six-path comparison](native_feature_demo_expansion_20261013.json) checks
every score against saved selected-model GPU outputs and checks additive
component identities; maximum difference is1.8e-7. Re-run after the six
prediction files are created:

```powershell
python scripts/audit_native_feature_demo.py --manifest configs/native_feature_demo_audit_v1.json --output docs/native_feature_demo_expansion_20261013.json
```
Native scales are kept separate: outputs include `score_raw` and
`score_range_clipped` (CH-SIMS[-1,1], MOSEI[-3,3]), feature/checkpoint SHA and
dataset identity. Use the dataset-specific protocol when evaluating: CH-SIMS
metrics clip scores; MOSEI raw MAE does not. No calibrated confidence or
classification probabilities are invented from a point-regression score.

Checkpoints use PyTorch's training artifact serialization and must be trusted
local outputs. Feature NPZ uses `allow_pickle=False`. Demonstration assets stay
outside Git and require the source data/model conditions. This command path
was verified locally and in an isolated fresh environment; it does not complete
the full formal/data/encoder reproduction requirement.
