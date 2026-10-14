# Source-risk baselines: no verified improvement over strong references

2026-10-13. The [registered plan](NATIVE_SOURCE_ROBUST_BASELINE_PLAN_V1.md)
defines independently implemented known GroupDRO and DANN regression controls.
Eight new operator/resume/isolation tests and six real source-group GPU
smoke/resume paths passed before54 full models. Only official TRAIN is used.

| Source-OOF condition | MAE, seeds13/42/2026 |
|---|---|
| Clip ERM | .419411 / .414890 / .405405 |
| Source-balanced ERM | .417103 / .423665 / .415289 |
| Fixed uniform group weights | .418871 / .422369 / .413777 |
| GroupDRO | .417691 / .424213 / .413591 |
| Independent domain probe | .417103 / .423665 / .415289 |
| DANN | .415217 / .419125 / .413943 |

[Independent54-model audit](native_source_robust_audit_20261013.json) verifies
source folds/fit maps, code/features/checkpoints/history/optimizer state,
per-epoch sampling and q state, all18 merged OOF tables, and CPU-reloaded
selection/held-source predictions. Every one of the2722 TRAIN clips has one
held-source prediction percase/seed. The nine clip-ERM paths exactly reproduce
the original fusion point tensors, optimizer state, selected epoch, history
and OOF forecasts. Domain-probe point paths exactly reproduce balanced ERM;
extra source-head updates and clipping do not secretly change point training.

All point models have492,033 parameters. Every case stores the same additional
linear source classifier (its size depends on the fold's optimization sources).
Only Probe/DANN train that head; active gradient counts are reported. Point and
source-head gradients are clipped separately. Dynamic group weights use the
original exponentiated-loss operator; the fixed-weight and balanced-sampling
controls isolate different changes. Sampling exposure and actual epochs/updates
are saved, not asserted equal after early stopping. Loops total166.20s,
excluding loading/hash/startup/I/O. The [plan](NATIVE_SOURCE_ROBUST_BASELINE_PLAN_V1.md)
links the primary GroupDRO/DANN sources; this is not a GEAR reproduction.

DANN-minus-Probe mean MAE is −.002590, with5000 paired whole-source draws
giving95% interval[-.005456,.000132]. Its three mean directions improve,
but the interval includes zero. GroupDRO-minus-fixed-q is+.000160 with interval
[-.001900,.002264], and one seed worsens. Neither passes the declared known
baseline gain gate. Neither method consistently beats original clip ERM or
either text width. This fixed recipe does not disprove all tuned variants.

## Input conditioning changes the research decision

[TRAIN feature geometry](native_source_feature_geometry_20261013.json) finds
coordinate-STD medians .290(text), .0236(audio), 1.804(vision) and variance
traces66.73/1.34/1247.56. Source variance fractions include real content and
sentiment; they do not identify nuisance. These unequal numeric scales require
a preprocessing check before attributing raw-fusion failure to source bias.

The [registered fit-source normalization control](NATIVE_SOURCE_STANDARDIZATION_PLAN_V1.md)
then trains nine more fusion teachers on exactly the same raw bytes/labels/
folds/backbone/budget. Only optimization-source mean/STD is fitted. Source-inner
selection and OOF rows never enter those statistics. Wrapper statistics/fit IDs/
source SHA are part of each checkpoint configuration; the original point
trainer stays frozen. Real smoke/resume and two leakage/identity tests pass.

[Normalization audit](native_source_standardized_audit_20261013.json) rebuilds
all statistics and reloads every checkpoint. OOF MAEs .392652/.398457/.389615
are lower than raw fusion in all seeds; paired mean delta−.019661 with95%
interval[-.027669,-.011351]. The result is approximately tied with narrow text
on average, with mixed seed directions. This closes much of the observed
fusion gap using **known standardization**, not an original source-invariance
mechanism. Normalized teacher loops total17.51s; no official VALID/TEST used.

Next freeze the better-conditioned native baseline, retain raw controls, and
choose a narrower mechanism from its errors. Do not expand raw-source DANN
or intercept-removal claims into an original result. GEAR's linked repository
does not expose its paper GMAE/IPW recipe in the audited commit; the
[review](GEAR_NATIVE_ADAPTATION_REVIEW_20261013.md) records the exact boundary.
The [conditional acquisition review](NATIVE_CONDITIONAL_ACQUISITION_REVIEW_20261013.md)
also rejects a generic regret gate as a new routing principle. Closest methods,
source nesting and actual encoder cost need stronger evidence before promotion.

## Commands

Requires the verified Chinese encoder cache and `train,uncertainty` extras.
Only one GPU training suite runs at a time. Existing runs require `--resume`.

```powershell
python scripts/train_source_robust.py --config configs/native_source_robust_v1.json --output docs/native_source_robust_formal_20261013.json
python scripts/audit_source_robust.py --suite docs/native_source_robust_formal_20261013.json --config configs/native_source_robust_v1.json --output docs/native_source_robust_audit_20261013.json
python scripts/diagnose_native_feature_geometry.py --config configs/native_source_robust_v1.json --output docs/native_source_feature_geometry_20261013.json
python scripts/train_standardized_source.py --config configs/native_source_standardized_v1.json --output docs/native_source_standardized_formal_20261013.json
python scripts/audit_standardized_source.py --config configs/native_source_standardized_v1.json --suite docs/native_source_standardized_formal_20261013.json --baseline-suite docs/native_anchor_crossfit_formal_20261013.json --capacity-audit docs/native_source_text_capacity_audit_20261013.json --output docs/native_source_standardized_audit_20261013.json
```

The project original-method, cross-dataset method replication, final-test,
author Chinese acquisition, reproduction/release and legacy-cleanup gates
remain open. No paid compute or test scoring occurred.
