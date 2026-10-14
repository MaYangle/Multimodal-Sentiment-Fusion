# Joint-context public-method adaptation — registered before training

2026-10-12. Previous goal turn made progress through twelve real reliability
runs and132 stress evaluations; none passed the competitive/calibration gates.
This study strengthens the public baseline before a new mechanism is claimed.

Author source pinned at `2be09edc0893e97e80eb6a6db18c2f70977b1734`:
[CombModel](https://github.com/MaHuanAAA/MoNIG/blob/2be09edc0893e97e80eb6a6db18c2f70977b1734/src/models.py),
[training](https://github.com/MaHuanAAA/MoNIG/blob/2be09edc0893e97e80eb6a6db18c2f70977b1734/src/train.py),
[active utility fusion](https://github.com/MaHuanAAA/MoNIG/blob/2be09edc0893e97e80eb6a6db18c2f70977b1734/src/utils.py).
The training function imports utility `moe_nig`, whose means use gamma weights;
class-local gamma+2alpha helpers are not called by that path. The joint Comb
branch is trained against the common target and included in fusion. Authors
sum individual and fused NIG losses. V1 used a mean branch loss instead and
omitted this branch; its results remain correctly labeled simplifications.

## Scope and cases

Use exactly the same frozen MacBERT CLS/WavLM/DINOv2 pooled bytes, 2,722 train
and647 validation samples, native M score, minimum valid MAE selection, and
seeds13/42/2026. No real T/A/V auxiliary targets, no teacher or additional data.
Transforms/sequence backbones are replaced with common pooled MLP experts;
this remains a source-attributed adaptation, not the author's full Transformer
reproduction. No novelty follows from the joint/pseudo-modality expert.

Six cases: three experts(width256) × uniform/dynamic; joint expert(width64)
plus three experts(width192) × uniform/dynamic; same joint architecture with
its joint contribution disabled × uniform/dynamic. The wider three-expert
reference has about0.2% more parameters than the joint model, isolating effects
from a larger overall parameter budget. The joint-off case preserves weights
and initialization while eliminating this component; effective active capacity
falls and must not be hidden. Report total/requires-grad parameters and actual
nonzero-gradient parameters. Width reallocation is an explicit comparison
choice, not evidence of a pure isolated architectural causal effect by itself.

Joint features concatenate zero-filled unavailable modalities. The joint
expert is eligible whenever at least one actual modality remains. Training
has clean inputs; missing/noise stress is diagnostic extrapolation. It is a
derived expert, never a fourth acquired modality. All-missing is rejected.

Loss follows the authors' **sum of available branch mean NIG losses plus
fused NIG loss**, regularizer .01, common M target. No auxiliary loss. AdamW
lr1e-4/weight decay1e-4, batch32, max80 epochs, patience12, gradient norm1,
validation-MAE plateau scheduler. Tuning budget: one fixed config per case.
Executed epochs/updates/time are recorded under the shared selection policy.

## Gates

Real prefix32/32 smoke and exact-resume tests precede formal training. Evaluate
all six cases × three seeds and retain failures. Compare MAE, Corr, native
Acc2, raw Student-t density NLL, coverage/width and error/variance ranking.
Only validation is opened; final test remains independent. Stress uses the
same eleven conditions as V1 (seven subsets/four shared pooled-feature noises).
Baseline selection cannot ignore the simple fusion0.4221 MAE reference.

If the joint expert recovers competitive point prediction, next inspect
whether correlated evidence summation drives undercoverage; no correction is
yet designed or claimed effective. If it also loses to simple fusion, improve
the baseline/representation before celebrating a small gate effect. Source
and cohort access, three-seed final method validation, fresh environment,
publication and conditional legacy cleanup remain required and unfinished.
