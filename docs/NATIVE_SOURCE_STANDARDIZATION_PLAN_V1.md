# Fit-source feature standardization control, registered before fitting

2026-10-13. A known preprocessing control, not a novel fusion algorithm.
All54 source GroupDRO/DANN runs completed; none has a lower mean OOF MAE than
the original clip-ERM fusion or either text width in the preliminary table.
Independent audit is still required. DANN improves its balanced-sampling
reference in each seed, but this alone does not close the strong-text gap.

The [TRAIN-only geometry](native_source_feature_geometry_20261013.json)
measures coordinate-STD medians .29023(text), .02357(audio), 1.80432(vision);
total variance traces66.73/1.34/1247.56. Different encoders and temporal mean
pooling produce very different input scales. This suggests checking numerical
conditioning before treating raw fusion failures as identified source nuisance.
Source variance fractions are descriptive and include real content/sentiment.

Fit coordinate-wise mean/population STD only on each source fold's optimization
IDs. STD below1e-6 becomes1; no clipping, label normalization or fit on selection/
OOF rows. Transform text CLS, masked-mean audio and masked-mean vision using
those frozen statistics. Preserve all raw feature bytes, IDs, original M and
source folds. Train the identical492,033-parameter pooled fusion MLP using the
already frozen source teacher trainer and the exact old seed/initialization,
AdamW/scheduler/80-epoch/patience12/MAE-selection policy. Only input scaling
changes. The per-fold preprocessing statistics, fit IDs, raw feature hashes and
preprocessor source SHA must enter each checkpoint's configuration identity.

Run source-group real smoke and restore before nine formal models. Do not load
official VALID or TEST. Independent checkpoint forward/statistics/once-only ID
audit is required. Compare allthree seed native OOF MAEs against original raw
fusion and both text widths with5000 whole-source paired resamples. Treat lower
MAE in every seed plus pooled upper95% delta below0 as useful preprocessing
evidence. Gains are attributed to known normalization, not a new mechanism or
raw encoder improvement. Keep actual updates, epochs and timing.

The wrapper uses the frozen AnchorCDF point trainer without editing it.
Its raw-input SHA fields describe source artifacts; its additional config
preprocessing fields describe transformations. It is not a claim that the
normalized model saw unmodified numeric feature vectors. Existing raw models,
formal source robustness data and calibration tables remain unchanged.

Only after this control establishes a well-conditioned competitive backbone
can the next narrow hypothesis be selected. Official native-task evaluation,
cross-dataset original-method verification and release remain incomplete.
