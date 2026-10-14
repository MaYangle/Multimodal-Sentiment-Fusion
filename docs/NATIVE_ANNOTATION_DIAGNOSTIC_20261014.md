# TRAIN unimodal annotation diagnostic

2026-10-14. The pinned authorCSV contains allfour native scores M/T/A/V for every
one of2722 TRAIN records; no missing channel scores. The script skips VALID and
TEST rows before parsing scorevalues. No training or new selectedmodel evaluation
was performed. Numerical source:
[native_annotation_disagreement_20261014.json](native_annotation_disagreement_20261014.json).

There are838/2722 (30.79%) clips where T/A/V binary signs disagree and446/2722
(16.39%) where text and multimodal M signs disagree. Zero is on the negative
side, matching CH authorbinary scoring. Text differs from M infive authorbins
on1334 records. Meanabsolute channel-to-M difference is T.291918/A.210287/V.244673;
sign agreement is83.61%/88.17%/86.08%. These are annotation relationships, not
claims that audio is inherently more reliable or that these labels measure
physical noise, annotator variance or causal modality utility.

The already-audited standardized fusion source-held TRAIN OOF predictions cover
eachrecord exactlyonce perseed13/42/2026. Average nativeMAE is.393575 overall,
.376294 on1884 sign-agreement records and.432424 on838 sign-conflict records.
Text-versus-M signconflicts haveMAE.500613 andfive-bin conflicts.440935.
These are descriptive outcome-conditioned slices with differing samplemixes,
not controlled effects, independenttest significance or improvement evidence.

This supports checking a competitive known unimodal-supervised baseline before
designing another reliability mechanism. Inference cannot access authorchannel
labels; a router must infer any channeldisagreement from features using TRAIN
supervision. Any claimed new mechanism must be compared against models using
exactly the same added annotations and comparable parameter/optimization budget.
The [independent literature review](NATIVE_DISAGREEMENT_REVIEW_20261014.md) will
determine whether a specific defensible gap exists; no originality claim here.

```powershell
python scripts/diagnose_native_annotations.py --author-csv data/author/simsv2/meta.csv --oof-suite docs/native_source_standardized_formal_20261013.json --output docs/native_annotation_disagreement_20261014.json
```

Three focused fixtures pass, including unparsable VALID/TEST scores, missing
channel masks and duplicate/out-of-native-range TRAIN score rejection.
