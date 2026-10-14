# CMU-MOSEI author cohort manifest

This directory keeps the author's `mmsa_label.csv` cohort independent from the competition CSV cohort. The script imports the shared `jevaffect.protocol.map_mosei_sentiment` API and the existing MOSEI filename parser; it does not change either shared implementation. All manifests, frozen ID files, transcript CSV, and the provisional crosswalk are under `generated/`.

Author `ID` is built as the exact source `video_id`, underscore, and source `clip_id` text. The clip suffix remains a string, including leading zeros. The competition `FileName` is retained as a separate `competition_id` only inside the provisional crosswalk audit; no author row is rewritten or merged by literal ID.

The author source CSV SHA-256 is `52d885bd1e8b29a4c06a69e2b1dd67dd02f7bcbdfa38ad918144fd5673a83521`. Its official `mode` counts are train 16,326, valid 1,871, test 4,659 (22,856 total). Competition CSV counts are train 16,327, valid 1,871, test 4,662. They are separate cohorts with separately named manifests and frozen ID files.

The frozen author test-ID file is `generated/mosei_author_test_ids.txt`, containing 4,659 sorted author IDs with SHA-256 `ca44c54b295e689bfb143bf2ac25e4d9c8cf5d636833150232a678563e29c341`. The encoder transcript file, `generated/mosei_author_train_valid_encoder.csv`, contains 18,197 train/valid rows and no label or annotation columns. The generated summary omits test label distribution and records that no model scoring was performed.

Literal-ID overlap is only an audit statistic: 17,460 author/competition IDs overlap, but 13,258 have unequal raw scores and 9,294 map to different project classes. The literal-ID audit is not used to create author records or the crosswalk.

The separate `generated/mosei_provisional_order_crosswalk.jsonl` recomputes the stated within-video ordering hypothesis: preserve original author CSV row order and pair only equal-count source-video groups to competition rows ordered by numeric final segment suffix. It contains 22,846 provisional pairs across 3,221 equal-count groups; raw label, mapped class, and split mismatches are all zero. It leaves 10 author and 14 competition rows unpaired. The 0 mismatch result is evidence for this ordering hypothesis only; it does not establish media identity and is not applied to author IDs, splits, or manifests.

Validation: 5 unit tests pass on the existing project interpreter, including preservation of leading zeros, separation of author test IDs, train/valid-only encoder transcripts, provisional order pairing, unequal-group exclusion, duplicate/invalid-input rejection, and reporting literal-ID label conflicts. The real CSV audit was run with CPU-only metadata processing. No model scoring, GPU execution, media inspection, data download, or modification outside `work/mosei_author_impl` was performed.

Reproduction command used:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
D:\MYL\JevAffect\work\train-env\Scripts\python.exe `
  work\mosei_author_impl\scripts\prepare_mosei_author_manifest.py `
  --author-csv D:\MYL\JevAffect\data\mosei\mmsa_label.csv `
  --competition-csv-dir D:\MYL\JevAffect\data\kaggle `
  --order-hypothesis-json D:\MYL\JevAffect\data\mosei\order_hypothesis_comparison.json `
  --output-dir work\mosei_author_impl\generated
```
