# Contributing to Multimodal Sentiment Fusion

The Python package remains `jevaffect`. Start with the current
[architecture](docs/ARCHITECTURE.md), [three-seed experiments](docs/EXPERIMENTS.md)
and [reproduction paths](docs/REPRODUCING.md). The latest candidate completed all
27 formal cells and failed its matched-control gate. Preserve the negative
result and register a new hypothesis before tuning a replacement.

Concrete next contributions: add a target-free standalone loader for the
sequence-gate `best_state` checkpoint schema; audit more author raw-media bytes
without changing official IDs; repeat the complete gate matrix in a fresh
environment; or propose a utility-based teacher with equal pretraining,
auxiliary supervision and compute controls. MOSEI lacks true channel labels,
so its replication needs a separately registered M-only pretraining protocol.

JevAffect is an experimental research repository. Useful contributions include better verified dataset adapters, recent public baselines on a matched feature protocol, efficient raw-video inference, missing-modality handling, model calibration, and source-group generalization tests.

Concrete current tasks: verify author payload IDs/continuous scores against
published splits without silent joins; validate/re-extract complete author media
through the [native MP4 entry](docs/NATIVE_MEDIA_INFERENCE.md); reproduce a
competitive recent baseline with shared frozen encoders; or investigate source
generalization with the [capacity-matched diagnostic](docs/NATIVE_SOURCE_CAPACITY_CONTROL_PLAN_V1.md).
Correlation and disagreement probability gates failed their matched-control
tests. Preserve [negative evidence](docs/NATIVE_ANCHOR_CDF_FINDINGS_V1.md) and
register new comparisons before training or tuning. Use the
[stronger known standardized fusion reference](docs/NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md)
and retain raw inputs as controls. Source-risk methods did not pass their
[matched baseline gates](docs/NATIVE_SOURCE_ROBUST_FINDINGS_V1.md); proposed
mechanisms need stronger evidence than a gain over an ill-conditioned input.

The [nested shape study](docs/NATIVE_NESTED_SHAPE_FINDINGS_V1.md) also failed
promotion. Use its full point/calibration/outer source isolation when proposing
new regression-probability models. Native-score categorical distributions can
be legitimate regression forecasts when support, proper scores and native
error-event probabilities are explicit; five-class confidence alone does not
establish regression uncertainty. Compare with simple and rich matched
controls before claiming a new construction or calibrated confidence.

Start with the [README](README.md) and [native evaluation protocol](docs/NATIVE_EVALUATION_PROTOCOL.md). A code-only check needs no dataset download:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c configs/ci_cpu_constraints.txt -e ".[train,data,encoders,verify,uncertainty]"
python -m unittest discover -s tests
```

For a model change, describe the exact author cohort, original continuous label range, backbone versions, official source-video/speaker split policy, validation selection rule, compute budget, random seeds, and an ablation that isolates the proposed mechanism. Report native MAE, Corr and author-derived classification metrics from independently checked per-ID predictions. Keep the final test split frozen while choosing methods. Explicitly distinguish native author features, project-derived frozen encoders and historical competition classifiers. Claims of novelty require comparison with the closest public methods and a positive controlled result; extra unimodal supervision and teacher compute need matched controls.

For a data or inference change, preserve every canonical ID, a real mask for absent modalities, source/version SHA-256 records, and raw-media-to-native-score traceability. Regression uncertainty requires coverage/width or another error-matched evaluation; a classifier softmax is insufficient. Add a small regression test when it protects an important contract; fixtures are not research results.

Do not commit media, transcripts from restricted sources, credentials, API tokens, downloaded encoder weights, feature caches, or training checkpoints. Keep upstream notices and licenses with any adapted source. Open an issue before a change that would alter the evaluation protocol or add a dependency with unclear redistribution terms.

Pull requests should include the problem, resulting behavior, precise validation commands/results, and any remaining limitation. The root MIT license covers original code; third-party code and model/data rights are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
