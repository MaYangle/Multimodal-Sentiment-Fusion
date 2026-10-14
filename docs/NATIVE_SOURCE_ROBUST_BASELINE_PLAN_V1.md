# Native source-video robustness controls, registered before fitting

2026-10-13. These are independently implemented **known baselines**, not a
JevAffect original contribution or a reproduction of GEAR. The source capacity
study finds a fusion disadvantage on source-OOF rows even with matched text
capacity; source offsets explain only26.6% of extra MSE. Neither teacher spread
nor fixed source-intercept removal is a justified reliability mechanism.

Test whether established source-balanced risk, GroupDRO or domain-adversarial
training changes this result before designing another mechanism. All models
predict the original CH-SIMS M[-1,1] label using the same frozen pooled
MacBERT/WavLM/DINOv2 bytes, 256-unit GELU/dropout.1 fusion MLP (492,033 point
parameters). A linear source-video classifier is constructed after the point
model inside a saved RNG fork so it cannot change point initialization/dropout
RNG. Store it in every case and report its parameter/gradient cost; source
IDs are not claimed verified speaker IDs. Inference needs no source ID.

| Case | Batch sampler | Regression objective / source branch |
|---|---|---|
| clip_erm | Uniform clip permutation | Original unclipped L1; original teacher parity reference |
| balanced_erm | Replacement, probability proportional to inverse source size | Batch mean L1; known balanced-source sampling reference |
| dro_uniform | Same balanced sampler | Fixed uniform q dot source-batch mean L1, absent-source mean0 |
| group_dro | Same balanced sampler | q proportional to q exp(.01 source-batch mean L1), then q dot group means |
| domain_probe | Same balanced sampler | Batch L1 plus source CE on detached context; probe/control |
| dann | Same balanced sampler | Batch L1 plus source CE, gradient reversal into point context |

GroupDRO follows the exponentiated group-weight operator in the
[original code](https://github.com/kohpangwei/group_DRO/blob/master/loss.py)
and [ICLR2020 paper](https://arxiv.org/abs/1911.08731). Log-space storage is
algebraically equivalent. Group count is only optimization sources; absent
batch groups get zero risk and retain their relative pre-normalization weight.
The fixed-q control isolates dynamic weights, and balanced_erm isolates batch
group averaging/scaling. Source-balanced sampling uses exactly the same number
of rows/optimizer steps per epoch as clip_erm, with replacement; it does not
see more unique data. Log sampled source counts to disclose exposure.

DANN uses the known gradient reversal principle from the
[JMLR paper](https://jmlr.org/papers/v17/15-239.html), adapted to a regression
head and source-video groups rather than source/target domain adaptation.
The linear source classifier trains with coefficient1; reversed feature
gradient is scaled by `.1 * (2/(1+exp(-10p))-1)` with p the completed step
fraction of the fixed max80-epoch budget. No target/VALID/OOF features train
the domain classifier. Domain_probe detaches context, allowing the same
source-head capacity/compute without adversarial feature training. Clip point
and source-head gradients separately at1, so domain_probe cannot alter point
updates merely through shared gradient clipping. Its point weights/predictions
must reproduce balanced_erm exactly under identical initialization/batches.

Use the prior three source folds (seed20261012), source-inner .1 split and
seeds13/42/2026. Only TRAIN labels/features are loaded; VALID/TEST are unused.
Optimization, inner selection and OOF source sets are disjoint for each model.
Same AdamW1e-4/WD1e-4, batch32, max80/patience12, plateau patience3/factor.5.
Select the first minimum inner native MAE in all cases. This is a comparable
early-stop policy, not an equal-epoch assertion or a hyperparameter optimum.
No search on OOF scores. Report actual epochs, updates and classifier compute.

Before formal54 models, verify source sampling/formulas/gradient directions,
Train-only and source-split contracts, exact epoch resume, and real smoke on
eight full source groups. Source/domain state must restore with optimizer/RNG.
Do not modify the already frozen AnchorCDF or native baseline trainers.

On canonical2722 OOF rows percase/seed, report native MAE/Corr/author
classifications, source-equal mean MAE, worst source MAE among sources with
at least5 clips, and clipped MSE source/within decomposition. Use5000 paired
whole-source draws with seed20261013; intervals are conditional development
evidence. A useful baseline improvement needs lower MAE in all three seeds
and a pooled source interval wholly below0 against its direct control.
GroupDRO compares to dro_uniform and balanced_erm; DANN compares to
domain_probe/balanced_erm. Also compare against both text widths and original
fusion. Such a gain is a known-method result, not novelty. Failure of this
fixed recipe does not disprove all tuned GroupDRO/DANN variants.

If none beats strong text, do not claim a competitive source-invariant fusion
mechanism. Inspect source/prediction/representation failures, primary GEAR
mechanism and temporal AV information before registering one narrow new
experiment. Cross-dataset original-method replication, final test, full fresh
method reproduction, data acquisition and release/legacy cleanup remain open.
