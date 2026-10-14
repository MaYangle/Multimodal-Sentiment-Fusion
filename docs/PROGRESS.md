# Latest update — 2026-10-14 source publication

Current repository: MaYangle/Multimodal-Sentiment-Fusion. All 27 gate-study
cells are complete; the candidate fails its three-seed matched-control gain
requirement. [Current architecture](ARCHITECTURE.md), [complete results](EXPERIMENTS.md),
and [reproduction paths](REPRODUCING.md) supersede older live-status notes below.
The old repository was deleted after the verified history/checkpoint backup,
under the user's later explicit authorization. Research success, MOSEI method
replication, final TEST and fresh repetition of the latest matrix remain open.

---

# JevAffect current progress

Updated 2026-10-14 (Asia/Shanghai). Goal active; not complete.

Latest user steering: new public home
[MaYangle/Multimodal-Sentiment-Fusion](https://github.com/MaYangle/Multimodal-Sentiment-Fusion)
created with a research-status scaffold. Old GitHub repository deleted after
actual backup recovery and weight checks, as explicitly requested now rather
than after research completion. This changes repository timing/name only;
original-method validation and complete publication remain pending.
[Migration receipt](GITHUB_MIGRATION_20261014.md).

## Scope
The 2026-10-13 replacement task is authoritative. Preserve native continuous
scores and author splits, establish true baselines, choose one evidence-based
hypothesis, and verify it before fresh-environment reproduction and release.
Earlier common five-class requirements are superseded, not completed native
research. [ACTIVE_SCOPE.md](ACTIVE_SCOPE.md) records the current full scope.

## Current round: implemented deployment policy; formal matrix still running

The previous goal turn made verified progress (12 fits, independent checks,
author media). This turn implements a multi-key sequence model and adds a
target-free deployed policy learned from TRAIN score-error preferences. It
addresses both normalization suppression and the teacher-only inference gap.
Known routing/distillation components and close primary sources are explicit;
this is a research candidate, not a verified innovative release.

[Frozen plan](NATIVE_SEQUENCE_GATE_PLAN_V1.md): source-held TRAIN T/A/V head
selection, shared full-TRAIN refit, seven matched fusion arms and seeds13/42/2026.
There are21 fusion fits plus6 selection/refit cells. All fusion fits have50
epochs/4300updates, total2084871 params; target-free policy/placement/distillation
controls share the same pretraining. Real32-row CUDA flow/resume succeeds.
The integrated suite passes340 tests (24.329s); no new TEST or paid resource.

Formal session8199 is confirmed live and must be re-polled before resuming.
Seed13 select/refit and six fusion arms have completed. Its static/pre-LN
controls both give native VALID MAE.379826; post-LN teacher.379814.
Distilled static/pre-LN give.379619/.379619; the proposed distilled-post-LN
network gives.380256, worse than its same-supervision controls. This seed
already fails the registered candidate gate; finish the fixed three-seed
matrix and retain the outcome without retuning or claiming original gain.
[Architecture/status](NATIVE_SEQUENCE_GATE_STATUS_20261014.md).

Partial independent audit (five complete cells at its snapshot) passes295
checks and replays three647-row forecasts within3.73e-7; full array byte hashes
and remaining cells are pending. Its first replay used a documented bundled
Python fallback; original isolated Python3.12.10/Torch2.11.0/NumPy2.3.5 was
subsequently independently confirmed working. Training CPs omit version fields;
[supplemental runtime observation](native_sequence_gate_environment_observation_20261014.json)
is not launch-time version binding or final fresh reproduction. Do not change
the frozen live implementation to fix that metadata omission.

The user's status follow-up is answered directly: an independently implemented
candidate now exists, but stable innovation/performance has not been achieved.
Publication and backup-gated legacy deletion remain unperformed. Original
author raw acquisition remains17/3369 with an explicit quota stop. A privileged
VALID-label teacher diagnostic is being prepared strictly outside deployment/
selection, to check whether the teacher's preferences actually help value paths.
Goal remains active; no smaller completion criterion is substituted.

## Earlier round: controlled moments rejected, author media acquisition advances

The previous turn obeyed the user's explicit pause. This resumed turn verifies
per-stream time/length/hash coverage and registers one point-fusion study before
fitting. Full-corpus physical synchrony remains UNVERIFIED. Four decoder-time
queries produce 20 payload artifacts, with 2715/646 numeric paired TRAIN/VALID
rows and zero-moment fallback for the other records.

Twelve real matched 494101-parameter CUDA fits complete 48.560 seconds of loops
after real 32-row smoke/resume. Paired covariance MAE .409240 ± .002539 fails
the seed gates: sequence ALMT's mean .408680 is better, and seed 42 static and
unpaired controls win. Independent CPU/NumPy audit completes 326 checks across
12 fits and 6 references, zero issues, prediction differences 4.63e-7/4.77e-7.
All five reference source intervals cross zero; the inference V-roll improves
seed 42 and fails the pairing specificity criterion. Stop this covariance study.
[Full result](NATIVE_TIMED_MOMENT_FINDINGS_V1.md).

The final integrated suite passes 325 tests (15.791 seconds), including the
new transport recovery and scalar-gate fixtures. TRAIN has 2722 complete M/T/A/V annotations: 838
modality sign conflicts (30.79%). Existing source-held fusion OOF MAE .432424
(conflict) versus .376294 (agreement) is descriptive, not causal utility.
[Diagnostic](NATIVE_ANNOTATION_DIAGNOSTIC_20261014.md) and
[primary-source review](NATIVE_DISAGREEMENT_REVIEW_20261014.md) identify KuDA/
AV-MC/CMC-GT prior art. Root's source inspection identifies a scalar-gate
suppression mechanism inside LayerNorm. The established zero-epsilon scale
invariance also yields finite-epsilon relative sensitivity near 1.95e-6 in a
normalized random-projection proxy over all 2722 TRAIN vectors. Moving the same
gate after normalization preserves relative weighting, but proves no prediction
gain. Singleton-key Q/K gradients are zero; that pooled blueprint is on hold.
[Diagnosis and limits](NATIVE_SCALAR_GATE_DIAGNOSTIC_20261014.md). Next: a
competitive multi-key sequence control and a precise gate-placement study.
No new gate model has been fitted.

After more than 27 hours, two author TRAIN MP4 requests complete 1,755,107 and
2,119,371 bytes, decode and match their historical source SHAs. Their full CPU
encoder/scaler/fusion rehearsal matches cached predictions within 1.669e-6.
[Author-media demo](NATIVE_AUTHOR_MEDIA_DEMO_20261014.md) establishes two examples,
not full-corpus equivalence. The full 3369 TRAIN/VALID acquisition queue then
downloads 15 files and stops on quota at video_0001/0020; offline reuse adds
the two recheck files. All 17 available files (49,839,094 bytes) pass receipt/SHA
checks, with no partials. PID 33184 is terminal; no network retry follows quota.
Cross-run partial restoration was fixed and meaningful transport fixtures added.
The pre-fix source snapshot is observed after termination, not hash-bound at launch.
[Cohort receipt](author_raw_cohort_20261014.json). No paid compute. Original
contribution, complete author acquisition, MOSEI method repeat, frozen TEST,
full final fresh workflow, GitHub release and legacy deletion remain open.
Goal active; not complete or blocked.

## Earlier round: CRPS rejection, exact repeat and stronger sequence baseline

Previous goal turn made real progress:9points/90shape fits/independent audits,
267 tests and scalar-score plan. This round completes36 native-CRPS fits and
987 independent checks/32664rows/12merges. Degree6 fails its registered
comparison with Mixture and native-hit confidence gates;90% interval score
also deteriorates. [Full negative result](NATIVE_CDF_CRPS_FINDINGS_V1.md).
Stop the CDF-shape line; no new family/degree is promoted.

633-file independent source copy and existing isolated dependencies reproduce
all36 heads exactly, including tensors/optimizer/32664records/metrics/epochs/
history/budget; root and source copy each pass288 tests. This is head-only
full retraining with reused audited point tables, not full raw/encoder/point
or final-method reproduction. [Verified scope](NATIVE_CDF_CRPS_REPRODUCTION_20261014.md).

Sequence preparation applies the SAME saved TRAIN pooled-coordinate statistics
to valid tokens, preserving T64/A400/V55, IDs/labels/lengths and4timestamps
byte-for-byte. 21artifacts+metadata; original PADnonzero count0 inallmodalities,
no additional cleanup intervention; pooled equivalence max1.216e-5.
Three normalized-sequence ALMT seeds complete362.454s loops and independently
recompute all647 predictions onCPU within9.239e-7. MAE .408680±.006690 vsraw
.427086±.006547; allseed means improve, paired source delta−.018406,
95% interval[-.029832,−.007161]. Vsstrong pooled fusion, mean delta−.002844
but interval[-.021147,.015315] includeszero andseed13 worsens. Acc2 islower.
[Baseline result/commands](NATIVE_SEQUENCE_CONDITIONING_FINDINGS_V1.md).

Next: verify A/V time origins and test one controlled point-fusion hypothesis
against BOTHconditioned baselines, with static-product, marginal-variance and
broken-pairing controls. [Review](NATIVE_TIMED_COVARIATION_REVIEW_20261013.md)
calls covariance priorart; it establishes no originality or physical sync.
Author acquisition, verified original contribution, MOSEI method repeat,
frozen TEST, final full reproduction/release and legacy deletion remain open.
No paid compute. Direct .kaggle-companion/project.json check reportsabsent;
the replaced AGENTS instructions do not activate competition lifecycle here.

## Earlier round: honest nested probability-shape test

New goal work inspected current files and confirmed no live prior training.
Registered [one bounded shape study](NATIVE_NESTED_SHAPE_PLAN_V1.md) before
fitting. Nine source-nested standardized fusion points complete109.246s of
loops; calibration sources never enter point fitting/selection/scalers.
27 exports pass646 independent CPU checks. Ten shapes xthree seeds xthree
outer folds complete90 fits (49.822s loops), with real-source smoke/resume
first. Integrated267 tests pass after adding the probability-audit checks
and four native-support prototype tests. No official VALID/TEST is used.

Degree6 binNLL1.307789/1.307807/1.330469 versusMixture
1.311071/1.311670/1.318897 fails the all-seed gate; raw-point HitBrier also
worsens in13/2026. [Negative study](NATIVE_NESTED_SHAPE_FINDINGS_V1.md)
retains allcases, same-median MDN, degree/uniform/tied ablations and raw-point
versus author-clipped CH metric distinctions. Independent probability replay
passes1711 checks/90models/81660rows/30merges. Paired source binNLL gain
againstMixture−.001476,95% interval[-.014306,.011128]; event gates alsofail.
No formula novelty claim.

TRAIN-only diagnostic finds11 actualnative scores merged intofive author
bins and a large fit-to-held gap for flexible shapes. Native-score probability
supervision [review](NATIVE_SCORE_RESOLUTION_REVIEW_20261013.md) identifies
ordinal/native-support scoring as a known baseline, not a new mechanism.
[One native-support plan](NATIVE_SUPPORT_PROBABILITY_PLAN_V1.md) was registered
before fitting, then deferred when the review rechecked an existing off-grid
author-metadata annotation. Its four numerical prototype tests pass, including
independent KL optimum/gradient and off-grid CRPS without rounding. No native
PMF fits were run. It is a TRAIN-only diagnostic, not a full protocol model.
The [next complete-protocol plan](NATIVE_CDF_CRPS_PLAN_V1.md) trains the same
bounded CDF families on original scalar-score CRPS, with primary confidence
matching author-clipped CH predictions and raw-point events auxiliary. Its
subsequent36 fits also failed and the line is stopped; the earlier raw-event
gate remains unchanged. Author
media, original contribution, MOSEI method replication, final TEST/release
and legacy deletion remain incomplete. Goal active; no paid compute.

## Earlier round: stronger native baseline and source controls

### Follow-up: source-risk controls, numeric conditioning and full repeat

Previous goal work made real progress: candidate rejection,48 checkpoint
reloads, nine capacity controls and exact21-run calibration reproduction.
This round completes54 known source-risk/control models and18 merged OOFs.
Independent CPU/code/data/checkpoint/optimizer/history/sampling audits pass.
Clip-ERM exactly reproduces original fusion; detached domain probe exactly
reproduces balanced ERM point updates. DANN improves balanced mean MAE in
allseeds, but paired95% source interval[-.005456,.000132] includeszero.
GroupDRO is inconsistent; neither beats strong text/original fusion.
[Negative controls](NATIVE_SOURCE_ROBUST_FINDINGS_V1.md).

STD medians .290/.0236/1.804 expose unequal T/A/V numeric scales. Nine
fit-source standardized fusion teachers improve raw fusion OOF MAE in every
seed: .392652/.398457/.389615, pairedmean delta−.019661,95% interval
[-.027669,-.011351]. Optimization-only statistics and all checkpoints pass
independent audit. Known preprocessing closes much of the raw fusion gap,
approximately tying narrow text on average; no original mechanism claim.

Six official2722/647 standardized text/fusion models also complete/audit.
Fusion MAE .411524±.003855 vsraw .422070 improves allseed means, but official
VALID source interval[-.023699,.001989] includeszero. Text worsens (.442491).
[Stronger known baseline/results/commands](NATIVE_CH_STANDARDIZATION_FINDINGS_V1.md).
Two real TRAIN MP4+text CPU forecasts match normalized cached vectors within
3.815e-6. Descriptor-only deployment loads without training arrays; statistics
and model redistribution still require final review.

Current and569-file source-copy235 tests pass. Independent re-preparation
reproduces all18 files exactly; six full source-copy retrainings match every
tensor/prediction/metric/epoch/history/update. [Scope](NATIVE_STANDARDIZED_REPRODUCTION_20261013.md)
excludes full raw re-extraction/original-method/final-test proof.

Two previously unattempted author TRAIN Drive targets returned quota HTML,
stopped at preflight with0media bytes each. [Receipts](author_train_raw_acquisition_20261013.json).
GEAR's linked code differs from its paper GMAE/IPW chain, and generic regret
acquisition gates have close defer/VOI precedents; independent reviews preserve
these boundaries. Verified original mechanism, method replication, final TEST,
release and legacy deletion remain open. No paid resource used.

### Follow-up: AnchorCDF rejection and source capacity control

The preceding execution turn made real progress; the separate instruction to
wait for the new goal was honored. New goal work resumed by inspecting current
files/results and testing the contracts, rather than relying on old status.
48 formal teachers are complete, with source-disjoint fit/select/OOF and exact
once-only2722 TRAIN coverage perview/seed. Full teachers never parse VALID
labels. Table export checks48 models/source/code/features/IDs/history/checkpoint
hashes and excludes in-sample teacher forecasts. Separate CPU reloads of all48
recover GPU forecasts within5.663e-7. Teacher loops total112.69s/73872 updates.

Seven real-sample calibration smoke cases and epoch resume passed before all21
formal CPU fits. Independent reloading/recomputed probabilities, intervals,
native scores, fit-only statistics and selected objectives pass the
[V2 audit](native_anchor_calibration_audit_v2_20261013.json). Disagreement bin
log score1.326491 vsMixture1.326072 is worse in allthree seeds; paired5000-draw
source interval[-.0008396,.0017330] does not support benefit. Tiny Hit-Brier
gain also has a zero-crossing interval. MDN noninferiority is unestablished
without a registered margin. The gate is rejected; no TEST evaluation, nested
source-transfer claim or MOSEI gate expansion. Preserve the
[complete negative study](NATIVE_ANCHOR_CDF_FINDINGS_V1.md).

Source-OOF textbeatsfusion in allseeds; nine new640-unit text teachers match
fusion parameters within.16%, after real source-group smoke/resume. Allnine
passed checkpoint/prediction/source/hash/selected-epoch/budget audits. Wide
text MAE .396386/.396074/.397116 vsfusion .419411/.414890/.405405; pairedmean
delta−.016710,95% interval[-.025688,-.007441]. Wider textis worse than original
256-unit text, so extra capacity is not beneficial here. This supports further
source diagnostics, not causal AVnuisance or originalmethod. Primary-source
[mechanism review](NATIVE_SOURCE_GENERALIZATION_REVIEW_20261013.md) shows
source pairs/debiasing already have close precedents. Sourceerror decomposition
must precede any newmechanism registration. A510-file independent source copy
passes214 tests and reproduces all21 calibration fits exactly (model tensors,
all predictions/probabilities/intervals/metrics/selected epochs/updates). It
reuses audited OOF tables and the previously installed isolated environment;
[reproduction scope](ANCHOR_CALIBRATION_REPRODUCTION_20261013.md). Earlier
nine-model MOSEI exactfresh reproduction stays separately verified.

The source error decomposition now shows source-mean offsets explain only
26.6% of fusion's extra MSE relative to capacity-matched text (34.0% versus
narrow text); within-source variation explains most. This weakens the proposed
fixed-source-offset remedy. Two mathematical tests pass; current root now has
216 tests including these checks. [Source diagnostic findings](NATIVE_SOURCE_CAPACITY_FINDINGS_V1.md)
record limits and next mature debiasing control. All author-data/original-method/
final-test/release/legacy-deletion gates remain open; no paid resource used.

### Follow-up: native media entry and text adaptation control

Previous goal turn made real progress: no-system-package/source reproduction,
six exact full pooled retrainings and completed three-seed mask baseline/15
diagnostics. The remaining ALMT reproduction session68439 has now ended
successfully. All nine models reproduce every prediction, metric, selected
epoch and update count exactly in the isolated/source-only environment.
[Full comparison](fresh_author_baseline_comparison_20261013.json).

Native CH-SIMS MP4+provided-transcript inference is implemented and run on two
genuine TRAIN clips in isolated dependencies/CPU. Their media SHA matches the
training cache records; all kept lengths match, prediction differences are
≤1.7e-6, while feature bytes differ slightly across CPU/prior GPU. This is an
encoder/software flow check, not held-out accuracy or full original-author
media equivalence. [Entry/evidence](NATIVE_MEDIA_INFERENCE.md).

Two exact author-original VALID clips were attempted, but finite payload ranges
returned quota HTML; neither completed. No test media was predicted. The
[native text tuning control plan](NATIVE_TEXT_TUNING_BASELINE_PLAN_V1.md)
registers frozen/top4 MacBERT scalar baselines before real training. Parent
review repaired fixture pin, completed-resume and float32 metric-boundary
issues before real32/32 GPU smoke/resume and all six formal runs. The actual
187-test integrated suite passed. Fine-tuning MAE0.427238±0.002180 improves
the matched frozen scalar control0.457426±0.006726 but still trails simple
fusion0.422070. Six runs/source/point/epoch/checkpoint audits completed;
fine-tuning is a known baseline, not the original contribution. Two label-free
text CPU predictions match saved GPU outputs within1.1e-6. Current integrated
suite(in isolated dependencies/current source) now passes191 tests, including
four new CDF numerical checks. A new452-file independent source snapshot also
passed all191 tests in the isolated environment and all source/import checks;
[environment evidence](source_native_cdf_environment_20261013.json). The older
snapshot's176-test/nine-full-run reproduction remains separately preserved.

The next [AnchorCDF plan](NATIVE_ANCHOR_CDF_PLAN_V1.md) freezes the strongest
point predictor and tests probability shape/gate against global, conditional
scale, feature-only mixture and direct error-band calibration. Common-M view
disagreement avoids inventing MOSEI T/A/V annotations. The median-preserving
CDF prototype passes numerical/probability/gradient checks, not calibration or
originality gates. New source-group OOF point/view predictions, real calibration
training, source-transfer checks and cross-dataset verification remain required.

### Follow-up: isolated reproduction and padding baseline correction

2026-10-13: six-path native inference audit passed(max difference1.8e-7).
Created a genuine no-system-packages venv and411-file source-only snapshot.
Actual install/import/hash audit, all176 tests, complete author MOSEI
re-preparation(19 artifacts and metadata exactly match), three real32/32 CPU
train/resume exercises and two selected-model feature inference paths passed.
[Fresh evidence and commands](FRESH_NATIVE_REPRODUCTION_20261013.md) separate
these gates from full formal/original-method/raw-media reproduction.

The measured +0.0584 MAE padding shift motivated a registered common ALMT
mask/valid-rank correction. All-valid parity is exact, changed padding/invalid
values preserve outputs and valid gradients, exact resume and real GPU smoke
passed. Three-seed formal training and15 CPU sequence diagnostics are now
complete/audited. Masked MAE0.612737±0.004018 versus old0.611524±0.005063;
two seeds worsen, one improves. Front-padding prediction change is at most
4.8e-7, while permutation/reversal sensitivity remains. It is a baseline
correctness control, not a new-method gain. [Findings](NATIVE_MOSEI_MASK_BASELINE_FINDINGS_V1.md).
A nine-run full fresh-environment suite waited on the observed training process
and is now training serially; it will audit and compare with the old numerical suite.
The six text/fusion full runs first reproduced all1,869 predictions, metrics,
selected epochs and executed budgets exactly. ALMT then completed and the
[full nine-run comparison](fresh_author_baseline_comparison_20261013.json)
also passed exactly. Original-method/full-encoder gates remain open.
Revalidate tool handles before interpreting these snapshots.

New CH missing-offset probes were exact64KiB206. The subsequent one-worker
8MiB resume ended with200 HTML containing a quota phrase, added0 bytes;
1,604,845,568 partial/cached bytes remain. This particular response is now
classified, earlier unsampled200 failures are not. No download job remains live.
Original contribution, complete Chinese author inputs, final test, raw native
inference and release/deletion gates remain unmet. Goal active.

Previous goal turn made progress through author package acquisition, nine
cross-fit models, feature demo and verified old Git/en/zh backups. This round
implemented an independent author-feature adapter and completed nine raw
MOSEI runs, six known normalization controls,45 frozen sequence diagnostics
and twelve new CH-SIMS structure/annotation-supervision runs. All actual
formal runs and per-ID/hash/epoch audits are complete; no test scoring.

| MOSEI author-linked aligned50 input/model | VALID MAE ± SD | Corr | Non0 Acc2 |
|---|---:|---:|---:|
| Raw text valid-mean | 0.6697 ± 0.0001 | 0.5325 | 78.64% |
| Raw early fusion | 0.6709 ± 0.0013 | 0.5209 | 77.16% |
| Raw ALMT scalar adaptation | 0.6115 ± 0.0051 | 0.6044 | 76.79% |
| TRAIN-standardized text | 0.6516 ± 0.0005 | 0.5646 | 77.18% |
| TRAIN-standardized fusion | 0.6450 ± 0.0017 | 0.5735 | 77.74% |

Source version16265/1869/4643, native[-3,3], temporal IDs, all four current-SDK
membership exceptions retained. Full independent data audit matches every
TRAIN/VALID ID/target/suffix float32 feature and19 artifact hashes. Root tests
cover API/masks/resume; outside-repository preparation reuse is also verified.
Mean/std are fit on TRAIN clip means only. These are common-input public
adaptations, not paper reproductions or native raw-video results.

ALMT's regression gains hold in three source-cluster comparisons but binary
accuracy is worse. It helps neutral/strong sentiment and hurts weak nonzero
sentiment relative to text. Joint-step shuffle changes its MAE to0.6223;
independent AV shuffle does not hurt (0.6108), while front padding raises it
to0.6699. Pooled references are invariant within1e-5. Evidence does not justify
claiming all benefit comes from AV temporal alignment. See
[complete MOSEI findings](NATIVE_MOSEI_BASELINE_FINDINGS_V1.md).

The CH annotation-reference study uses the same492,691-parameter architecture
for direct M versus T+D, with/without real T and M-T auxiliary targets. Four
real smoke cases and exact-resume/target isolation checks preceded12 runs:
flat0 MAE0.4237, flat1 0.4205, sum0 0.4286, sum1 0.4218. Additive vs equally
supervised direct differences vary across seeds and additive mean is worse.
Auxiliary supervision helps both; the structure-specific promotion gate fails.
It is a negative result, not an original contribution. See
[reference findings](NATIVE_REFERENCE_RESIDUAL_FINDINGS_V1.md).

Before the padding follow-up, the integrated suite passed172 tests. Native label-free feature
inference now supports GloVe valid-mean and reference components; real MOSEI
fusion/ALMT and CH additive demo runs passed. English/Chinese documentation
is synchronized to actual outcomes. Generic acquisition and AV residual
claims have close primary-source precedents, preserved in research notes.

Acquisition: a new CH first-missing64KiB probe returned valid206/version, but
the bounded resume added only16,777,216 bytes before another wrong200 response.
Received prefixes/blocks/partials total1,604,845,568, still incomplete. Legacy
extra pretrained4.5GB archive attempt also ended incomplete; verified Git and
en/zh backups remain intact. No unchanged retry or GPU training is implied
by stale state files; inspect current tool/process handles before starting jobs.

Next: choose a specific mechanism based on measured error/sensitivity and
closest methods; keep the competitive native baselines and negative controls.
Author original-media/Chinese feature coverage, verified original gain,
source-disjoint method validation, cross-dataset method replication, independent
final test, full native raw inference, fresh environment, GitHub publication
and conditional old-repository deletion remain incomplete. Goal stays active.

## Previous round: joint experts and train-only dependence (2026-10-12)

Eighteen joint-context runs and198 saved stress evaluations are complete and
independently audited. Joint dynamic MAE0.434643±0.003459 remains worse than
simple fusion0.422070; nominal90% coverage is61.67%. Duplicate expert evidence
preserves means but reduces variance15.5%. The known proper Student-t pool
widens intervals and improves coverage, which motivated a train-only controlled
uncertainty investigation, not a novelty claim.

Nine new source-disjoint OOF models (three folds × seeds13/42/2026) completed.
Only2,722 official TRAIN labels are parsed;129 source videos have disjoint
optimization, inner selection and OOF cohorts. Every training ID is predicted
once per seed. Six frozen-mean controls share these predictions/final models.
R and scalar calibration use OOF only. Official valid is evaluation/development
data on an already validation-selected backbone, not independent final test.

| Readout | Valid density NLL | Coverage90 | Width90 |
|---|---:|---:|---:|
| Original NIG | 1.2600 | 61.67% | 0.9633 |
| Global variance scaling | 0.8006 | 85.27% | 1.5378 |
| Independent covariance + scaling | 0.7892 | 86.76% | 1.6159 |
| Residual-correlated covariance + scaling | 0.8002 | 85.78% | 1.5600 |
| Proper Student-t pool | 0.8260 | 85.11% | 1.5820 |
| Scaled proper pool | 0.8094 | 88.97% | 1.7038 |

Correlation-minus-global NLL is+0.002375/+0.002900/-0.006338 across seeds.
Source-cluster bootstrap over120 validation videos yields conditional95%
interval[-0.010632,0.009622] for the mean-over-seeds difference. The correlation
candidate fails the registered consistency gate; no accuracy/originality gain
is claimed. Stronger point representation takes priority over further tweaking
this rejected correction. Nine OOF loops totaled122.3 seconds, excluding
startup/hashing/checkpoints; all point MAEs stay0.434643. Current integrated
suite actually passed163 tests, including native metrics, sequence masks,
cross-fit label isolation, exact parameter/loss resume and covariance controls.

Evidence: [context findings](NATIVE_CONTEXT_NIG_FINDINGS_V1.md),
[dependence plan](NATIVE_DEPENDENCE_PLAN_V1.md),
[dependence findings](NATIVE_DEPENDENCE_FINDINGS_V1.md), formal suites,
prediction/hash audits, fitted matrices/scalars and bootstrap artifacts. A
Luna Max independent read-only review confirmed hashes, IDs, means/SDs and the
rejection decision. It correctly limits validation independence and resume
claims; no byte-identical whole-checkpoint claim is made.

Author acquisition advanced: one exact original MP4 fully downloaded,
SHA-verified and decoded for blinded media integrity. Both author large
feature channels returned fresh exact206/Last-Modified. CH-SIMS added about
1.21GB before a wrong200 range response stopped its new handle; cached bytes
remain. MOSEI's bounded resume completed3,730,509,179 bytes with full SHA
31007381c01dd346945ed8a64c7321239de5aaf1ef05bf112e952848a823186d;
its actual16265/1869/4643 temporal-ID cohort is audited, with disjoint source
splits but four current-SDK membership errors, leading text padding and audio
-inf. A separate source-bound adapter is required before training;
[cohort details](MOSEI_MULTIBENCH_COHORT.md). CH-SIMS author features remain incomplete; see
[current acquisition status](AUTHOR_ACQUISITION_STATUS.md). Earlier failed
raw attempts and transfers remain historical, not current availability claims.

The native feature-only demo now runs three model families on two label-free
VALID inputs; CPU vs saved GPU differences are at most1.8e-7. Raw native
MP4-to-score and fresh environment are still pending. Old backup directory
was absent; a fresh33-commit/all-ref mirror and bundle passed actual restore
and fsck, and en.pt/zh.pt archive passed whole SHA and ZIP CRC. Durable copies
are under ignored backups; extra pretrained archive is still incomplete.
[Backup verification](LEGACY_BACKUP_STATUS.md) keeps deletion gates explicit.

Next: build the actual MOSEI author-feature adapter and register native baselines;
finish the other author transfers and retain source/version checks;
train source-attributed native author-feature baselines and select a more
competitive representation before another original hypothesis. Original
contribution, MOSEI multimodal replication, final test, native raw-media inference,
fresh environment, publication and conditional legacy cleanup remain open.
The earlier research turn made progress; this turn changes code, authoritative
experiment results and data acquisition state. Goal remains active.

## Previous round: native reliability factor experiment (2026-10-12)

Twelve formal GPU runs (four conditions × three seeds), independently audited
against original labels and exact validation IDs, are complete. Each condition
uses identical feature bytes, 496,143 parameters and the same registered
optimizer/selection policy. Four real prefix32/32 smoke runs and mathematical
NIG/Student-t tests preceded formal training; exact CPU epoch resume, auxiliary
gradient isolation and validation auxiliary-label isolation are verified.
The integrated suite actually passed **149 tests**.

| Auxiliary T/A/V supervision | Fusion | Native valid MAE ± SD | Nominal90% interval actual coverage |
|---|---|---:|---:|
| off | equal means | 0.4397 ± 0.0021 | 56.67% |
| off | dynamic NIG | 0.4363 ± 0.0036 | 80.32% |
| on | equal means | 0.4393 ± 0.0026 | 56.21% |
| on | dynamic NIG | 0.4360 ± 0.0035 | 82.33% |

All remain worse than the simple-fusion 0.4221 MAE baseline. Auxiliary × fusion
interaction changes sign across seeds; dynamic variance/error ranking worsens
with auxiliary supervision. The completed 132 stress evaluations find some
visual-noise resilience but worse audio+vision-only performance; visual noise
increases visual mean weight in two seeds. No calibrated-reliability or
original-contribution claim. Full findings and next baseline rationale:
[NATIVE_RELIABILITY_FINDINGS_V1.md](NATIVE_RELIABILITY_FINDINGS_V1.md).

Author-download progress: public usercontent finite ranges and Last-Modified
were independently measured. The original MOSEI2MiB and Chinese371,720,192-byte
prefixes are immutable and SHA-bound; new blocks have exact Content-Range,
version and checksums. Unbounded direct GET still failed. A first MOSEI finite
range transfer ended with a ConnectionError after completed-byte progress
228,589,568; a second lower-concurrency transfer also stopped after progress
362,807,296 bytes. Cached blocks/partials are retained. A bounded resume
supervisor subsequently stopped on a200 response lacking Content-Range;
completed-byte progress reached396,361,728 and actual block/partial bytes
413,138,944. No whole payload is available. The stopped handles are recorded
in work/native_active_sessions.json; no live download is implied. CH-SIMS
gained a verified8MiB block, then returned quota HTML
even on a64KiB Range probe; unchanged quota retries are deferred. Neither
complete author payload nor its schema is claimed. See
[AUTHOR_DOWNLOAD_DIAGNOSIS.md](AUTHOR_DOWNLOAD_DIAGNOSIS.md) and
simsv2_range_response_diagnosis.json.

The author's individual Raw directory has6,245 MP4 entries and exact matches
for every2,722/647/1,034 supervised CSV ID (author_simsv2_raw_id_coverage.json).
Two blinded test-media retrieval attempts also failed direct prefix preflight;
no new video or test score is claimed. This is an alternate official channel
ready for later availability checks, not proof that the media is downloaded.

Next: finish/revalidate the MOSEI source transfer; inspect actual IDs, raw
continuous labels and SDK folds before author-feature training. Add the
original MoNIG joint/pseudo-modality predictor as a matched public baseline,
then use that evidence to test a focused uncertainty mechanism. Final test,
competitive original method, MOSEI replication, native inference/demo, fresh
environment, publication and conditional legacy cleanup remain incomplete.
Previous goal turn was progress; this round also completed real experiments
and changed the next research decision. Goal remains active.

## Previous round: native CH-SIMS v2.0 baseline (2026-10-12)

- RTX 5070 Ti, 16 GB; D disk about 1.05 TiB free. GPU is also used by desktop
  applications. No user applications were stopped, no paid/cloud work started.
- Author Drive metadata independently downloaded with source receipt; exact
  SHA matches local competition metadata. Actual 4,403 rows, 2,722/647/1,034;
  author homepage says 4,402. Native -0.1 remains a valid score.
- Only train/valid cache features were reused, with file SHA verification and
  exact author-ID/continuous-label checks. No class indices enter regression.
  Native manifests freeze test IDs without exporting test scores.
- Text, masked pooled fusion and scalar-head ALMT passed real 32/32 smoke
  training and checkpoint reload; text resumed from its completed epoch.
- Four native metric tests and three native data/resume tests passed. An
  independent CPU fixture verifies resumed and uninterrupted model parameters
  and losses exactly; identity changes are rejected. Author evaluator
  conformance passed 200 samples × 2 dtypes × 2 datasets, including boundaries.
  The current integrated suite actually ran and passed 138 tests (historical
  test counts below are earlier records, not the current observed count).
- Nine full runs completed serially on local GPU (seeds 13/42/2026). Per-ID
  prediction/target/hash/best-epoch audit passed for all. Native validation
  selection results, mean ± sample SD:

| Model | MAE lower | Corr higher | Acc2 higher |
|---|---:|---:|---:|
| Text | 0.4373 ± 0.0024 | 0.5820 ± 0.0003 | 75.06% ± 0.78 pp |
| Simple fusion | 0.4221 ± 0.0007 | 0.6234 ± 0.0060 | 77.13% ± 0.00 pp |
| ALMT adaptation | 0.4271 ± 0.0065 | 0.5994 ± 0.0091 | 74.19% ± 1.17 pp |

Evidence: native_simsv2_baselines_v1.json, native_simsv2_baseline_audit.json,
native_metric_conformance.json, native_simsv2_error_analysis.json. Per-example
predictions and best/last checkpoints are in ignored runs/native_simsv2_v1.
Feature input forms and truncation/hardware caveats remain explicit. These
are validation selection scores, not test results or author-feature scores.

**Conclusion:** simple fusion is the strongest first-round MAE comparator.
For 117 opposite-sign unimodal-annotation validation samples, its MAE is
0.4718 versus ALMT 0.5651 (no opposite signs: 0.4111 vs 0.3966). Annotation
disagreement is an analysis signal, not evidence of unreliable modalities.
MoNIG/AV-MC/SELF-MM comparison suggests a controlled supervision × reliability
fusion experiment; no hypothesis is yet supported by candidate results.

**Actual gaps:** complete author CH-SIMS media/native features are missing.
CMU endpoints yielded HTTP 500/HTTPS SSL errors. New MOSEI CSV retrieval timed
out with a partial file preserved; existing MMSA CSV is hash-verified. Four
MMSA train rows are not members of current SDK folds and require investigation;
competition media/IDs will not be silently substituted. No Python process was
live at the end of the completed-run process inventory; old download pointers
are not proof of live work. The channel probe now parses the SDK's pinned
recipe instead of guessing URLs; the corrected WordVectors endpoint also
returned HTTP 500/HTTPS SSL errors. The incorrect older Glove paths are
explicitly recorded as rejected probes. MOSEI author metadata was then reused
from exact verified bytes; both native manifests and label-free test-ID hashes
are saved. The SDK-linked MultiBench inventory succeeded: four processed
payloads are listed in data/author/mosei/multibench_listing.json. Its
mosei_senti_data.pkl reports 3,730,509,179 remote bytes; transfer terminated
after read timeouts, preserving a 2,097,152-byte resumable part and a failure
receipt. No complete MultiBench package, schema or final checksum is claimed.
The independent fold audit identifies one MMSA-only train video and three
SDK-only videos; see MOSEI_FOLD_DISCREPANCY.md. Do not delete/reassign rows.

**Next experiment:** resolve author data alternatives and MOSEI cohort evidence;
pre-register one reliability/uncertainty hypothesis after the error and closest-
work comparison, then implement matched controls and real smoke/full runs.
Final test, selected-method three seeds/ablations, MOSEI replication, full native
inference/demo, fresh-environment reproduction and release remain unfinished.
Legacy repository deletion remains conditional on successful release and
verified backups. Goal remains active.

## Historical five-class work (superseded task, retained evidence)

## Real results, independently recomputed
Frozen revision-pinned mBERT CLS + 256-wide MLP, provided transcripts; seeds 13/42/2026. Selection uses official validation accuracy with early stopping. These are selection-set results, not test scores or strong multimodal reproduction results.

| Dataset/cohort | Train / validation | Accuracy mean +/- sample SD | Macro-F1 mean +/- sample SD |
|---|---:|---:|---:|
| CH-SIMS v2.0 | 2722 / 647 | 39.16% +/- 0.39 pp | 33.82% +/- 2.09 pp |
| CMU-MOSEI MMSA author | 16326 / 1871 | 53.47% +/- 0.43 pp | 38.70% +/- 1.17 pp |

Evidence: pretrained_text_validation_results.json and mosei_text_validation_results.json. Each validation ID/label/probability was checked against its own frozen cohort; metrics recomputed from predictions. Chinese runs completed on private Kaggle T4; English runs completed locally on RTX 5070 Ti. All best checkpoints and predictions are present locally; the Chinese seed-42 last-resume checkpoint still needs retrieval. No test inference/scoring.

English feature encoding took 15.87s; classifier train+validation loop totals were 26.00s / 27.15s / 22.50s. These exclude downloads, process/model startup and artifact serialization. Encoder truncation at 64 tokens affected 653 train and 74 validation rows. Full end-to-end inference timing remains pending.

## Verified foundations
- System Python 3.12.10 + Torch 2.11.0+cu128 supports this RTX 5070 Ti; project work/train-env inherits it with pinned encoder-library overrides. This is not yet a clean-environment reproduction.
- 136 integrated tests passed, including real CPU/CUDA training/resume machinery, cross-token/missing-modality controls, and strict BC-MLF initialization/loss checks. Synthetic fixtures are not research results.
- mBERT and WavLM pretrained weights are complete and SHA256 verified. WavLM encoded audio from two actual training MP4s into 63 and 106 temporal vectors, 768 dimensions. A small real-data ALMT pipeline pilot is now trained; see below.
- Author MOSEI CSV cohort is separately frozen: 16326/1871/4659. Literal filenames cannot be joined to the competition's 16327/1871/4662 cohort: 13258 common IDs have different raw scores. The provisional within-video ordering crosswalk is not used for training joins; media identity still needs checking.
- SIMSv2 full CSV and split manifests verified. One test raw label -0.1 remains unresolved; 129 source groups overlap official splits and speakers are unknown.
- 496 complete media members from the partial ZIP independently pass inventory size, CRC and SHA checks. Of these, 339 train/43 valid were joined to the official CSV for the pilot; 91 test clips and 23 unknown IDs were excluded from model work. The ZIP remains incomplete and this prefix is not representative.
- Historical Git mirror/bundle independently restored; best_models.zip and external MSA.zip verified. Old GitHub repository remains untouched.

## Recent multimodal baseline
BC-MLF (Sep 2026 preprint, MIT) is a recent candidate, not an asserted independently verified SOTA. Complete GPT/AV architecture, BI_ENC raw-score initialization, five-class auxiliary losses and selective freezing are ported. Native Stage 1 AV length 39 and Stage 2 length 55 are preserved. Drivers use train/valid only, checkpoint identities and exact epoch resume. Source accumulation/tail corrections are documented. No real BC-MLF training result yet.

Private cloud pilot v1 stopped before training because the runner's Drive ID contained an erroneous extra character. The log was diagnosed and the source ID corrected against the independent local downloader. See bcmlf_cloud_pilot_v1_diagnosis.json. Do not interpret this as a publisher quota failure. Corrected version 2 terminated ERROR before training: the valid official Drive ID hit a too-many-users download quota, independently confirmed locally. Do not repeat unchanged Drive attempts; native-feature reproduction remains pending. No competition submission or paid compute.


## Raw-video milestone (real pilot, not a formal comparison)
The mBERT/WavLM/DINOv2/Whisper weights are verified and the local raw pipeline runs. All 382 selected train/valid videos were encoded with supplied transcripts, retaining sequence features, masks and audio/video timestamps. The padded trainer cache has text64/audio400/vision55 steps; one audio sequence loses85 tail frames, explicitly recorded. ALMT still follows its source unmasked padding behavior.

ALMT seed13 trained on339 and validated on43 prefix-selected examples, stopping after16epochs (best epoch8). The43-sample selection score is53.49% accuracy/44.97% Macro-F1; **this biased tiny pilot must not be compared with full-cohort scores**. Training/validation loops took16.02s and peak allocated GPU memory557,619,200bytes, excluding encoding and I/O.

Two actual validation videos now run from MP4 through supplied-transcript or Whisper-ASR features into the trained three-modality classifier: four legal probability/label records, with all four low-confidence records retained in the main output and separately logged. Supplied-transcript predictions match the saved validation probabilities within8e-8. ASR-path warm per-video times were0.39s and0.21s; initialization was6.28s. This is a two-video functional check, not a latency benchmark or robustness result. See raw_video_pipeline_pilot_results.json. Missing-modality support, reliable interpretation, final model and competition packaging remain unfinished.

The first private raw-feature job ended at **3,226/3,369** records when its fixed extraction budget expired; 50 verified archives contain 3,200 records. The input manifest interleaves splits: 2,594 train and 632 valid were encoded before the stop. A first resume job failed on an input-mount path; the second returned ERROR without establishing a complete artifact. Independently, all 50 archive hashes were verified locally, the remaining 169 train/valid MP4s were fetched through the authenticated competition API and hashed, then re-encoded with identical pinned mBERT/WavLM/DINOv2 weights. The merged source has exactly 2,722 train and 647 valid IDs; the verified padded cache loaded in the CPU trainer. The final 169 feature rows were produced on RTX 5070 Ti while the first 3,200 came from Kaggle T4. See the raw cache provenance and explicit hardware caveat. No test encoding or scoring; this common-backbone result is separate from native MMSA-feature reproduction.

CMC pseudo architecture/loss/transfer helpers are also ported and pass actual CPU tensor/cache fixtures, without ground-truth unimodal labels. A complete five-class mBERT common-backbone seed 13 later reached 40.80% accuracy / 37.13% Macro-F1; this is not its published native-feature result. Its frozen-contextual-text route is explicitly distinct from its source fine-tuned BERT route.

## Preliminary public-method and candidate pilots
On the same biased 339/43 raw-feature subset, a short two-stage CMC pseudo-label adaptation reached 41.86% accuracy / 29.05% Macro-F1 (seed 13, best fusion epoch 12). The independent mask-aware Jev-inspired temporal candidate, **without OOF utility supervision**, reached 44.19% / 36.79% (seed 13, best epoch 10). Across three pilot seeds, it averaged 44.19% +/- 2.33 pp accuracy and 35.19% +/- 2.22 pp Macro-F1. Seed 13's last training accuracy was 100%, selected validation NLL was 2.76, and overfitting is evident. Seed 42's interrupted run resumed from its epoch checkpoint; seed 2026's completed checkpoint was verified as an identity-checked no-op on resume. ALMT on this subset was 53.49% / 44.97%. Schedules differ and validation has only 43 rows; none of these ranks methods on the full cohort. See cmc_raw_pilot_results.json, jev_temporal_three_seed_pilot_results.json, and raw_video_pipeline_pilot_results.json.

A strict merger and full-cohort cache contract have been prepared for all 3,369 independently encoded records. ALMT was trained locally on the complete raw-derived train/valid cache in all three fixed seeds. Independent per-example audit: mean 44.87% +/- 0.50 pp five-class accuracy and 41.21% +/- 0.82 pp Macro-F1 on the validation selection set. Total train+validation loops were 101.0s, 86.1s and 104.8s (excluding preprocessing, I/O and model startup); peak CUDA allocation 994 MB. See almt_full_validation_results.json. This is a common-backbone ALMT adaptation, not the paper's native-feature result. Later full candidate/ablation/calibration results are recorded below; native BC-MLF and frozen tests remain.

A three-fold source-group OOF teacher pass on the pilot training split produced 4,670 finite modality-window utility targets without validation/test labels. A matched seed-13 utility/no-utility comparison gave the same 44.19% validation accuracy; NLL fell from 2.76 to 1.75, Macro-F1 fell from 36.79% to 26.86%, and teacher compute was extra. This does not establish an accuracy contribution or novelty. See jev_utility_pilot_ablation.json.

A full 2,722/647 mBERT all-layer text fine-tuning trial (seed13) selected 38.49% accuracy / 27.66% Macro-F1, below the frozen mBERT text control. It was not promoted; see mbert_finetune_trial.json. The root English README and synchronized Chinese guide now document verified commands, measured scores, attribution and open release gates. Original code now uses MIT with separate upstream notices; distributable trained weights and clean-environment reproduction remain outstanding.

## Stronger-backbone and source-disjoint model audit (2026-10-12)
Pinned Apache-2.0 MacBERT-base provided-transcript features replaced only the text channel. WavLM audio and DINOv2 visual arrays were byte-for-byte copied; exact 2,722/647 ID and label order and every file hash passed the strict cache loader. MacBERT CLS text alone reached 45.29% on full official validation in one seed. On the exact same MacBERT/WavLM/DINOv2 cohort, three-seed ALMT was 47.04% +/- 0.64 pp accuracy and 44.69% +/- 1.19 pp Macro-F1; the independent temporal model was 49.72% +/- 0.24 pp and 47.13% +/- 1.00 pp. Calibration cross-fitted by source group improved NLL for all three temporal seeds but ECE15 worsened slightly for seed 2026. See macbert_full_model_paired_audit.json and jev_macbert_calibration_seed*.json. These are selection-set scores, and published native MMSA five-class figures are higher under a different feature protocol.

Critical split audit: 646/647 official CH-SIMS validation clips share a source video with official TRAIN. A fixed StratifiedGroupKFold diagnostic split derived solely from official TRAIN gives 2,182 train / 540 holdout with zero shared source-video groups. Three-seed results on that diagnostic selection set: ALMT 48.95% +/- 0.47 pp accuracy / 43.48% +/- 3.18 pp Macro-F1; previous temporal model 49.75% +/- 0.47 pp / 44.24% +/- 2.53 pp; projected 128-D token cross-attention 49.94% +/- 0.47 pp / 43.04% +/- 3.15 pp. Cross-attention's seed-13 no-attention control scored 49.81% versus 49.44% with attention, so direct token attention is not a verified contribution. See group_holdout_model_paired_audit.json. This group holdout has been used for model choice and is not a pristine final test.

The earlier temporal gate's score-to-true-window-masking-loss Pearson correlation was only 0.031 (9,103 observed windows). A three-fold source-group OOF teacher on the 2,182 diagnostic training clips produced 30,377 finite signed intervention targets without any diagnostic validation or official validation/test labels. JevAffect V2 with bounded utility-biased cross-token correction and utility supervision reached 49.26% accuracy in seed13; the same architecture without utility supervision reached 50.37%. Temporal-base gate/masking-effect correlation was 0.032 with supervision and 0.021 without it. The proposed utility mechanism failed both accuracy and faithfulness tests; no originality claim. See architecture_reassessment.md, jev_affect_v2_utility_gate_audit.json and jev_affect_v2_control_gate_audit.json.

English competition MOSEI has a frozen label-free 16,327/1,871 train/valid extraction manifest; 249 of 18,198 clips lack video while all have WAV. A real 54-train/22-valid pilot fetched 146 media files from the authenticated competition inventory, independently SHA-checked every file and resulting NPZ, and encoded all 76 clips locally with separate audio and correct missing-video masks for six of them. It is only a flow check, not a model result; see mosei_pilot_feature_flow.json. The 26.49 GB full competition ZIP is being resumed locally with official ETag/object-size checks after a stalled first attempt; only train/valid MOSEI media will be opened for training. No English full multimodal score yet. A prepared private Kaggle feature-extraction upload was rejected by automatic permission review because it included transcript/code payload and account-ownership/authorization evidence was insufficient; no such English kernel was uploaded. All subsequent training and feature extraction will stay local, per the user's instruction.

The 54/22 English pilot additionally completed real masked cross-token GPU training with exact epoch resume. Its selected 9/22 accuracy and 100% last-epoch training accuracy show the tiny sample is unsuitable for method ranking. All 22 prediction IDs, source labels, class order and probability sums passed independent checks; see mosei_pilot_training_flow.json. A reusable MOSEI competition-ID dataset reader keeps absent video at zero valid frames and pools long audio/video across the entire physical timeline. The full English cohort has not yet been trained. The official 26.49 GB ZIP server supports exact HTTP 206 byte ranges; a local four-worker, 128-MiB-chunk transfer is under trial with ETag, per-chunk SHA, final whole-ZIP CRC and train/valid-only extraction checks. The slower serial client and per-file pilot download remain recoverable fallbacks.

## Text representation controls
Three-seed masked-mean pooling changes full validation accuracy from39.16% to39.31% onSIMSv2 and53.47% to53.91% onMOSEI; Macro-F1 falls from33.82% to33.59% and38.70% to36.43%. No robust gain or originality claim is supported. Retain both configurations in tuning history. The local Chinese CLS repeats recover the same accuracy for each cloud seed. Class errors remain large (initial CLS NEUT recall10.9% inChinese, SPOS recall3.6% inEnglish); text-only errors do not establish audiovisual conflict. See TEXT_BASELINE_ERROR_ANALYSIS.md and text_pooling_control_results.json.

## Historical next experiments (superseded by native scope above)
1. Finish ETag-verified local competition ZIP download, selectively extract MOSEI train/valid WAV/MP4, build a missing-modality-aware full English cache and train current public/new methods locally on the 16,327/1,871 competition cohort. Keep author and competition ID namespaces separate.
2. Reassess model design using the negative cross-attention and OOF gate results, reproduce the closest recent public method under a clearly matched feature protocol, and test a new hypothesis only with matched controls. Do not present unfaithful gates or a small selection-set gain as originality.
3. Complete candidate three seeds where justified, noise/missingness robustness, physical-window interventions, calibration, frozen-test reporting, bilingual raw-video inference and clean-environment reproduction.
4. Publish after reproducibility and rights checks. Delete the old repository only after the new release and final backup audit succeed.

Tool handles are recorded in work/active_sessions.json. A saved record is not proof of a live process; revalidate before resuming or restarting.
