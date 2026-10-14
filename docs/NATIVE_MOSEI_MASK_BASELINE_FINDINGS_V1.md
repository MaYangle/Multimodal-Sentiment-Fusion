# ALMT valid-step correction: native MOSEI findings

Three formal seeds and15 frozen VALID sequence evaluations are complete and
audited. No TEST evaluation or original algorithm contribution is claimed.
The [registered plan](NATIVE_MOSEI_MASK_BASELINE_PLAN_V1.md) treats masks and
valid-rank positions as common baseline corrections.

| Input/model | VALID MAE ± sample SD ↓ | Corr ↑ | Non0 Acc2 ↑ |
|---|---:|---:|---:|
| Frozen upstream structural adaptation |0.611524 ±0.005063|0.604384|76.787%|
| Valid-step/rank correction |0.612737 ±0.004018|0.605625|77.298%|

Both use the exact raw aligned50 author-linked TRAIN/VALID cohort and
3,047,937 parameters, identical initialization recipe/L1/optimizer/max budget
and minimum VALID MAE selection. This is still a GloVe structural adaptation,
not the author's fine-tuned BERT recipe. Masked epochs21/22/31 and updates
10689/11198/15779 compare with old25/32/36; identical stopping rules yield
different executed budgets, which are reported rather than hidden. Masked
training loops totaled1192.95 seconds on the shared desktop GPU, excluding
loading/hashing/checkpoints; timings are not isolated speed benchmarks.

| Seed | Original MAE | Masked MAE | Masked minus original |
|---|---:|---:|---:|
|13|0.605682|0.608253|+0.002570|
|42|0.614638|0.613947|−0.000691|
|2026|0.614252|0.616011|+0.001759|

There is no stable clean-input MAE gain. Small average Corr/Non0 Acc differences
do not establish universal dominance or a new mechanism.

## Padding and sequence diagnostics

CPU evaluation of the three frozen masked models uses the same per-ID
seed105 transforms and exact VALID targets as the earlier raw diagnostic.
Explicit masks retain published interior zero vectors. All15 prediction files
passed independent hash/ID/target/native-metric recomputation.

| Condition | Mean masked MAE |
|---|---:|
|Identity|0.612737|
|Equivalent front padding|0.612737|
|Joint valid-step permutation|0.622384|
|Independent A/V permutation|0.612966|
|Joint reverse|0.619850|

Identity versus front-padding predictions differ by at most4.8e-7 across the
three full VALID cohorts, compared with the old adapter's MAE shift from
0.6115 to0.6699. CPU versus saved GPU identity differences also stay within
the registered1e-5 tolerance. The position/mask correction fixes dependence
on storage layout while preserving measurable order sensitivity. Independent
A/V permutation still barely changes mean MAE; this does not support a claim
that precise AV alignment explains the clean regression gain over pooling.
These are frozen-feature tests, not causal or raw-media robustness evidence.

All-valid outputs match the upstream adapter exactly with identical weights.
Unit tests verify invalid values/layouts, valid gradients and exact CPU resume;
a real32/32 GPU smoke preceded training. Padding-equivalent model views are
already invariant, so adding an identically zero consistency target would
not provide a useful new objective.

## Evidence and next decision

[Formal suite](native_mosei_masked_formal_v1.json),
[run audit](native_mosei_masked_audit_v1.json),
[sequence report](native_mosei_masked_sequence_v1.json), and
[paired/independent comparison](native_mosei_masked_comparison_v1.json).
Keep both adaptations as transparent references. Choose the next research
hypothesis from their actual remaining errors and closest prior work; do not
promote this correction as the required original algorithm. Full isolated
environment reproduction is currently running separately.

```powershell
python scripts/train_masked_author_baseline.py --config configs/native_mosei_masked_almt_v1.json --output docs/native_mosei_masked_formal_v1.json
python scripts/audit_author_baselines.py --suite docs/native_mosei_masked_formal_v1.json --models almt_masked --output docs/native_mosei_masked_audit_v1.json
python scripts/diagnose_author_sequences.py --suite docs/native_mosei_masked_formal_v1.json --models almt_masked --device cpu --output docs/native_mosei_masked_sequence_v1.json
```

Existing training requires`--resume`. The actual15 diagnostic evaluations run
on CPU to avoid overlapping the separate GPU reproduction. Official VALID is
development/selection data; final test, new-method replication and source-
disjoint method assessment remain incomplete.
