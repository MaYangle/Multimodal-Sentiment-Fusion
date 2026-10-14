# Registered train-only dependence study — 2026-10-12

Registered before any OOF training or correction fitting. Frozen joint-dynamic
context NIG validation MAE is 0.434643, worse than early fusion 0.422070.
This is an uncertainty investigation, not an accuracy improvement. Duplicating
experts preserves means but reduces NIG variance; the proper Student-t linear
pool already improves coverage with wider intervals. These diagnostics motivate
a falsifiable comparison and do not establish that dependence correction wins.

## Prior work and hypothesis

[Residual-ICM](https://arxiv.org/html/2609.30085v1) estimates standardized OOF
residual correlations and forms D R D for different regression targets.
We test the same known covariance construction across predictors of ONE scalar
sentiment target, with gamma-weighted fixed means. Var(w'e)=w'Sigma w and global
scale calibration are established techniques; no new covariance formula is
claimed. [EviDep](https://arxiv.org/html/2604.16579v3) already uses shared/private
NIG experts. Joint experts and evidential fusion are not original contributions.
The possible application hypothesis is that training-source-disjoint residual
dependence improves uncertainty beyond equally fitted scalar controls.

## Cohort and training

Only the 2,722 official CH-SIMS v2.0 TRAIN rows enter cross-fitting. No official
validation/test target is parsed by the OOF trainer. Cache hashes and original
continuous M scores are checked. Source video is ID prefix before `/`.
Three outer folds use seed 20261012, label-free shuffled group assignment to
the least-populated fold. Every training sample is predicted exactly once by a
model whose optimization AND selection exclude its entire source video.
Within each outer training cohort, a label-free 10% of source groups form
an inner selection cohort (seed 20261012 + fold + 1). Fit and select groups
are disjoint from the OOF groups. This changes no official benchmark split.

Use the registered joint_dynamic model and source-summed NIG objective, same
80-epoch/patience12, batch32, AdamW and scheduler settings as the context study.
Seeds 13/42/2026 each produce three fold models (nine total); fold RNG seed is
seed + 10000*fold. Every uncertainty comparator shares the exact OOF models,
predictions and final official-train checkpoint for that seed. Extra OOF cost
is reported separately from final-model cost. Epoch resume saves random states,
optimizer, scheduler, manifest, code and cache hashes. Real prefix32 smoke and
group/leakage/resume checks precede formal training.

## Fixed comparators

For expert i, z_i=(y-mu_i)/sqrt(v_i), where v_i is its Student-t predictive
variance. Fit Pearson R on all TRAIN OOF predictions, shrink R by fixed
lambda=0.1: R_s=.9R+.1I. No lambda tuning. For gamma weights w and D=diag(sqrt(v)):

- Original summed NIG distribution; no fit.
- Original NIG variance times one fitted scalar q (global calibration).
- Independent-expert variance w'D^2w times its own fitted scalar q.
- Correlated-expert variance w'D R_s D w times its own fitted scalar q.
- Proper gamma-weighted linear Student-t pool; no fit.
- Proper pool with one fitted common component-variance scalar q; means fixed.

Each fitted q minimizes OOF raw continuous Student-t density NLL in predeclared
log(q) bounds [-6,6]. For covariance comparators, preserve original summed-NIG
mean and degrees of freedom, replace scale to match desired variance. This is
a moment-based Student-t readout, not the exact density of a correlated sum.
For the pool, multiply each component beta by q; keep means/gamma/alpha.
Do not use official validation residuals to fit q or R. OOF evaluation itself
is training-fit diagnostics, not held-out evidence for the correction.

## Evaluation and rejection rules

Evaluate all six distributions on the frozen 647 validation IDs for each seed.
Report native MAE/Corr/Acc2 (point means unchanged within dtype tolerance),
raw density NLL, 50/80/90/95 coverage and width, variance/error Spearman and
selective MAE where supported. No final test. NLL is a continuous density for
quantized bounded labels, not discrete likelihood. Store fitted matrices,
scalars, hashes, individual predictions, exact manifests, counts and time.

The correlation hypothesis fails if gains are inconsistent across seeds, are
matched by scalar-only or proper-pool controls, or require much wider intervals
without better density. Examine R stability across seeds and within OOF folds.
Even a successful result requires competitive point backbone, MOSEI replication,
independent final test and reproduction before a verified contribution claim.
