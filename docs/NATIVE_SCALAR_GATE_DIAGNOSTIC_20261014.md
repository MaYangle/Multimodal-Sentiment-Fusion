# Scalar routing and normalization: mechanism diagnosis

The proposed pooled KuDA adaptation is on hold. Reading the pinned author
source revealed a mechanism that must be checked before another training study:
each branch applies LayerNorm to `(1 + R_m) * z_m`, where `R_m` is a positive
per-example scalar. At zero epsilon, LayerNorm exactly cancels positive scalar
rescaling. This is an established normalization property, not a new formula.
[Pinned source](https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/models/DyRoutFusion_CLS.py)
contains the branch operation in `DyRout_block.forward`.

For centered variance `s²`, the normalized part of a branch is
`c*(z-mean(z)) / sqrt(c²*s² + epsilon)`. With positive `c` and epsilon zero,
this is independent of `c`. Nonzero epsilon permits an effect, especially at
very low variance. The branch's fixed affine gamma/beta does not restore lost
scale information; preceding learned layers can change variance and later
layers can amplify small differences. Adaptive optimization also matters.
This does not establish sensitivity of a trained KuDA checkpoint or invalidate
its reported performance.

A second issue is the proposed one-token-per-modality adaptation: cross-attention
with one key has softmax weight exactly one, hence zero Q/K parameter gradients.
This concerns the singleton adaptation, not the author's multi-token architecture.
The next comparison must retain multiple valid keys and genuine attention rather
than count unused Q/K parameters as useful model capacity.

## Actual numerical evidence

The diagnostic verifies hashes of all 2,722 TRAIN standardized pooled vectors
and IDs, projects each modality to 128 dimensions with a fixed random seed, and
normalizes those projections. They proxy initial unit-scale branch states;
they are not learned KuDA states or a reproduction of its full forward. No
sentiment scores, VALID features, TEST data or pretrained weights are read.
No model is trained.

Changing the same gate coefficients from `1+[.1,.2,.7]` to `1+[.7,.2,.1]`:

| Placement | Epsilon | Max coordinate change | Median relative output L2 change |
|---|---:|---:|---:|
| Gate inside each branch LayerNorm | 0 | 1.33e-15 | 1.44e-16 |
| Gate inside each branch LayerNorm | 1e-5 | 9.49e-6 | 1.95e-6 |
| Gate after branch LayerNorm, before mixing | 1e-5 | 1.7455 | .35769 |

Both placements use the same coefficients and outer normalization. Moving the
gate preserves relative branch weighting in this proxy. It does not prove lower
sentiment error, calibration, robustness or novelty of the placement.

The autograd sanity check gives Q/K gradient norms exactly zero with one key,
and nonzero norms 2.18/3.25 with two keys; V remains active in both cases. Three
fixtures check scale invariance, the finite-epsilon formula and low-variance
exception, and singleton versus multi-key gradients.
[Machine-readable evidence](native_scalar_gate_normalization_20261014.json).

```powershell
python scripts/diagnose_scalar_gate_normalization.py --cache-dir data/feature_cache/simsv2_native_pooled_standardized_v1 --output docs/native_scalar_gate_normalization_20261014.json
python -m unittest tests.test_scalar_gate_normalization -v
```

## Next research decision

The next core hypothesis is whether preserving a scalar modality signal through
normalization makes its intended training effect identifiable and useful for
native sentiment regression. Build a competitive multi-key sequence adaptation
with matched static, published placement and normalize-then-gate arms. Keep
inputs, true TRAIN channel supervision, pretraining checkpoints, parameters and
updates identical; protect VALID/TEST boundaries. Check gate Jacobians before
fitting and measure actual branch variance and gate effects during training.
Source/code inference differences must be controlled separately.

The [source-verified blueprint](NATIVE_KUDA_ADAPTATION_DESIGN_20261014.md) describes
paper/code differences; its singleton proposal is not approved for formal fits.
A separate precise plan is required before a new model fit. Any original-method
claim still requires prior-art review, real three-seed gains, controlled ablation,
source-held support and MOSEI native replication. This diagnostic alone does not
complete that requirement. The failed covariance and CDF studies remain stopped.
