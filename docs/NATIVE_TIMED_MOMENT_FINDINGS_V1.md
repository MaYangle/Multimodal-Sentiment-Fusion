# Timed-moment controls: no promotion

2026-10-14. Twelve real CUDA fits completed after the
[registered plan](NATIVE_TIMED_MOMENT_PLAN_V1.md), CPU fixtures and a real32-row
checkpoint/resume smoke. Complete source data is the frozen author-ID-matched
TRAIN2722/VALID647 cache; TEST is not exported or scored. The paired feature
uses four interior interpolated decoder-time query positions, not verified
physical synchronization. It is a known projected second-order operator.

## Native official VALID development results

All moment cases have494101 parameters, the same projections/head/optimizer/
seed/budget and the same frozen clip-pooled TRAIN coordinate statistics.
Mean ± sample SD over13/42/2026:

| Model | Native MAE ↓ | Corr ↑ | Acc2 ↑ |
|---|---:|---:|---:|
| Strong conditioned pooled fusion (492033 params) | .411524 ± .003855 | .635776 | .76816 |
| Strong conditioned sequence ALMT (3288577 params) | .408680 ± .006690 | .615730 | .73622 |
| Static projected mean product | .412228 ± .005954 | .631926 ± .008261 | .759402 ± .013326 |
| Marginal grid variance | .411311 ± .008400 | .635452 ± .012522 | .770737 ± .006246 |
| Paired grid covariance | .409240 ± .002539 | .639228 ± .005819 | .762494 ± .013853 |
| One-row shifted grid covariance | .410610 ± .001977 | .634646 ± .004994 | .768676 ± .010056 |

Paired seed MAEs are .411170/.410186/.406364. Against pooled fusion, seed13
worsens; against ALMT, seed42 worsens and the three-seed mean is higher. Atseed42,
static product.407525 and shifted covariance.409941 both beat paired.410186.
Variance atseed2026 also beats paired (.402849 versus.406364). Therefore the
all-seed gate fails independently of any uncertainty interval. This is a negative
result for this four-query diagonal representation, not a disproof of every
temporal architecture. Do not retune querycount/rank/shift on these outcomes.

The12 loops total48.560098 seconds excluding initialization/hash/preparation/I/O;
peak allocated training GPU memory79,437,824bytes per fit. Selected/executed
epochs for13/42/2026 are product6/18,3/15,4/16; variance11/23,2/14,5/17;
paired11/23,3/15,5/17; shifted5/17,2/14,4/16. All stop atpatience12. These
wall-time comparisons are descriptive and depend on the shared desktop load.
The known pooled baseline is smaller/faster, ALMT larger/slower; equal budget
means matched optimizer policy, not artificially equal achieved walltime.

CH author MAE/Corr clip prediction and truth to[-1,1]. Binary positive iff>0,
with neutral negative; original five/three-bin thresholds remain unchanged.
Training uses unbounded L1. Higher correlation here does not replace the
registered MAE criterion or establish a classification gain.

## Input and numerical checks

[Timestamp audit](NATIVE_AV_TIME_AUDIT_20261014.md) verifies hash-bound per-stream
grids, masks/padding and provenance. Corpus physical synchrony remains UNVERIFIED.
The20 payload artifacts plusmetadata contain original global vectors and four
linear-interpolation queries per eligible record. Valid numeric pairs2715/2722
TRAIN and646/647 VALID; other records retain global fallback withzero moments.
Median retained overlap2.488/2.738seconds.69 audio andonevision prefix truncations
remain; no dropped tails are recovered. WavLM state timestamps anchor clip-wide
transformer features, not isolated local receptive fields.

Before fitting,16 scoped tests pass: moments/gradients/494101 matching, constant
offset interaction invariance, common paired permutations, independently rolled
pairings, masks/padding, original-score/split/hash guards and exactfixture resume.
Real32-row CUDA smoke resumed fromepoch1 to2; no formal completion claim. After
allfits the then-current304-test integrated suite passed.

The [independent CPU/NumPy audit](native_timed_moment_audit_20261014.json) is
complete:326 checks, zero integrity/replay findings, all12fits plus6reference
runs. It validates175 cached file hashes, recomputes the saved normalizer from
TRAIN vectors and checks source/code/config/CSV/checkpoint identity, parameters,
selection/earlystop/scheduler/optimizer/RNG and native metrics. NumPyfloat64 and
functional TorchCPU outputs match all647 saved predictions perfit within4.63e-7
and4.77e-7 respectively. Six independentaudit fixtures pass. Common paired
permutations preserve outputs; the one-modality shift does not reliably remove
any gain.

Paired covariance minusreference clipped absolute error, averaged acrossseeds;
5000 shared whole-source draws over120 VALID sourcevideos, seed20261014:

| Reference | Mean MAE delta (negative favors paired) | Conditional95% source interval |
|---|---:|---:|
| Static product | -.002988 | [-.008548,.002413] |
| Marginal variance | -.002072 | [-.009695,.005277] |
| Unpaired covariance | -.001370 | [-.009271,.006328] |
| Conditioned pooled fusion | -.002285 | [-.011529,.006355] |
| Conditioned sequence ALMT | +.000559 | [-.017470,.018112] |

Allfive intervals includezero. The647 unique examples are repeated acrossthree
seed predictions, not1941 independent clips. Conditional intervals do not
correct repeated development/selection onVALID. Fixed V-roll inference changes
MAE by+.002794/-.000439/+.001941; seed42 improves ratherthan worsens andeachseed
interval includeszero. This also fails the pairing-specificity gate. Stop this
representation; neither source-held promotion nor MOSEI candidate transfer is
justified by these results.

## Reproduction entry points

Requires the pinned cached backbones/features, originalauthorCSV and saved TRAIN
pooled scaler. No new encoding is performed:

```powershell
python scripts/prepare_native_timed_grid.py --cache-dir data/feature_cache/simsv2_full_macbert_raw_av --author-csv data/author/simsv2/meta.csv --output-dir data/feature_cache/simsv2_native_timed_grid_v1
python scripts/train_timed_moments.py --config configs/native_timed_moments_v1.json --case paired_cov --seed 13 --smoke32 --stop-after 1
python scripts/train_timed_moments.py --config configs/native_timed_moments_v1.json --case paired_cov --seed 13 --smoke32 --stop-after 1 --resume
python scripts/train_timed_moments.py --config configs/native_timed_moments_v1.json
python scripts/audit_timed_moments.py
```

Use`--resume` for an interrupted formal matrix; source/config/cache/scaler/plan
identity must match. Checkpoints, logs and647 selected VALID predictions perfit
are under`runs/native_timed_moments_v1/formal`. The complete12-run receipt is
`runs/native_timed_moments_v1/matrix.json`, process16929 exited0. No original-method
promotion, MOSEI transfer of this rejected mechanism, finalTEST, finalfresh full
workflow, publication or legacy deletion is claimed. Full projectgoal remains.
