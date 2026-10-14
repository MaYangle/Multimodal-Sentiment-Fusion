# Saved text-baseline validation error analysis

These are official validation-set selection diagnostics for the frozen mBERT CLS + MLP text baselines. Each run selected checkpoints using validation accuracy/early stopping. The source metadata CSVs contain all official splits; the analyzer maps train/valid labels only, and it does not open test prediction files or compute test metrics.

The SIMS v2 cohort uses the official train/valid rows in `meta.csv`; the MOSEI cohort uses MMSA author's `mmsa_label.csv` train/valid rows. They are analyzed independently. The MOSEI score bins use the project's adopted application of the competition's stated possible mapping.

## CH-SIMS v2.0

Validation rows: 647; three seeds. Mean accuracy 0.392 (seed SD 0.004); macro-F1 0.338 (seed SD 0.021).

Class-level recall and the most common errors pooled over the three runs:

| True class | Valid rows/seed | Recall (seed mean ± SD) | Most frequent destination(s) |
|---|---:|---:|---|
| SNEG | 182 | 58.8% ± 4.4% | WNEG (20%), WPOS (11%) |
| WNEG | 143 | 33.8% ± 11.1% | SNEG (33%), WPOS (20%) |
| NEUT | 92 | 10.9% ± 6.6% | SNEG (34%), WPOS (24%) |
| WPOS | 131 | 49.6% ± 1.5% | SNEG (23%), WNEG (12%) |
| SPOS | 99 | 23.2% ± 13.2% | WPOS (45%), SNEG (20%) |

Recall ± SD is descriptive across these three seeds, not a confidence interval or significance test.

| Confidence/consistency diagnostic | Result |
|---|---:|
| Mean max class probability | 0.457 |
| Mean entropy | 1.298 nats (80.7% of ln 5) |
| ECE15 (raw model probabilities) | 0.072 (seed SD 0.030) |
| Mean max probability on correct / wrong predictions | 0.485 / 0.439 |
| Mean entropy on correct / wrong predictions | 1.253 / 1.327 nats |
| Wrong predictions with max probability >= .80 / >= .90 | 9 / 2 across all three runs |
| IDs wrong in >=2 seeds / all 3 seeds | 393 / 337 of 647 |
| Same predicted class across all 3 seeds | 71.6% of rows |
| Source groups with >=5 valid rows; median group accuracy [p25,p75] | 52 groups; 40.0% [29.1%, 49.6%] |

Neutral and nearby source-score diagnostics (accuracy averaged over rows and seeds):

- `neutral_and_adjacent_source_scores_-0.2_0_0.2`: 174 validation rows per seed; accuracy 20.9%.
- `adjacent_WNEG`: 48 validation rows per seed; accuracy 38.9%.
- `adjacent_WPOS`: 34 validation rows per seed; accuracy 22.5%.

Examples wrong in at least two seeds (IDs only): `video_xjzw/00054` (WNEG), `video_xjzw/00045` (WNEG), `video_xjzw/00038` (SNEG), `video_0060/0039` (WPOS), `video_0060/0038` (WPOS).

## CMU-MOSEI (MMSA author cohort)

Validation rows: 1,871; three seeds. Mean accuracy 0.535 (seed SD 0.004); macro-F1 0.387 (seed SD 0.012).

Class-level recall and the most common errors pooled over the three runs:

| True class | Valid rows/seed | Recall (seed mean ± SD) | Most frequent destination(s) |
|---|---:|---:|---|
| SNEG | 77 | 26.4% ± 9.4% | WNEG (48%), NEUT (17%) |
| WNEG | 301 | 40.9% ± 8.3% | NEUT (43%), WPOS (9%) |
| NEUT | 832 | 68.2% ± 4.7% | WPOS (23%), WNEG (8%) |
| WPOS | 550 | 51.9% ± 4.0% | NEUT (41%), WNEG (5%) |
| SPOS | 111 | 3.6% ± 0.9% | WPOS (62%), NEUT (27%) |

Recall ± SD is descriptive across these three seeds, not a confidence interval or significance test.

| Confidence/consistency diagnostic | Result |
|---|---:|
| Mean max class probability | 0.549 |
| Mean entropy | 1.030 nats (64.0% of ln 5) |
| ECE15 (raw model probabilities) | 0.043 (seed SD 0.012) |
| Mean max probability on correct / wrong predictions | 0.574 / 0.519 |
| Mean entropy on correct / wrong predictions | 0.985 / 1.081 nats |
| Wrong predictions with max probability >= .80 / >= .90 | 84 / 19 across all three runs |
| IDs wrong in >=2 seeds / all 3 seeds | 857 / 641 of 1,871 |
| Same predicted class across all 3 seeds | 69.3% of rows |
| Source groups with >=5 valid rows; median group accuracy [p25,p75] | 150 groups; 53.3% [42.7%, 63.8%] |

Neutral and nearby source-score diagnostics (accuracy averaged over rows and seeds):

- `NEUT_raw_zero`: 433 validation rows per seed; accuracy 71.1%.
- `NEUT_raw_±1/3_near_mapping_boundary`: 397 validation rows per seed; accuracy 65.3%.
- `NEUT_other_nonzero_interior_scores`: 2 validation rows per seed; accuracy 33.3%.

Examples wrong in at least two seeds (IDs only): `zjYEBwXGD8I_4` (NEUT), `zjYEBwXGD8I_13` (NEUT), `zjYEBwXGD8I_10` (NEUT), `zhE_68GMJIk_8` (NEUT), `zhE_68GMJIk_5` (WPOS).

## MOSEI author-cohort train/valid transcript overlap

Using NFKC + casefold + whitespace collapse, 18 normalized transcript groups appear in both train and valid (75 rows total). Of these, 7 groups have different five-class labels and 13 have different raw scores. Within-split duplicate groups: train 79 groups (219 rows), valid 2 groups (5 rows). Blank train/valid transcript rows excluded from duplicate grouping: 0.

These counts flag repeated language strings; they do not establish that samples share a speaker or media segment. No records were dropped. Source-group accuracy is descriptive and is not a significance test.

## Baseline change before a multimodal novelty claim

Keep validation-accuracy checkpoint selection as the primary protocol. Before claiming a multimodal gain, run a three-seed text-only class-weighted cross-entropy control with weights derived from train labels only, then report accuracy, macro-F1, and all five recalls; the weighted loss may trade accuracy for minority-class recall. If macro-F1 checkpoint selection is also explored, label it as a separate protocol diagnostic, not a replacement for the primary run. The existing accuracy-selected control has severe class blind spots: MOSEI SPOS recall is 3.6%, and SIMS NEUT recall is 10.9%. Temperature scaling can assess probability calibration but leaves argmax labels unchanged, so it cannot fix these confusion patterns.

Max-probability and entropy figures describe raw softmax outputs; this analysis does not claim the baseline probabilities are calibrated. The text-only errors do not establish an audio/video conflict. Validation calibration metrics are selection-set diagnostics too.

## Reproduction and integrity checks

The companion `analysis.json` records SHA-256 hashes for both metadata CSVs, both source result JSONs, and all six validation prediction files. It verifies the prediction hashes against the result JSONs, exact ID coverage of each official validation split, and each target against the project's label mapping.

Run from this directory with `python analyze.py --project-root D:\MYL\JevAffect --output-dir outputs`.
