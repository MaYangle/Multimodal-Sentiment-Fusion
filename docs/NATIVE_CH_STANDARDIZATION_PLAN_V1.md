# Official CH-SIMS pooled standardization control, registered before fitting

2026-10-13. Source-fold normalization improves fusion OOF MAE in each seed
(.39265/.39846/.38961 versus .41941/.41489/.40540). It is approximately tied
with narrow text on average and does not beat text in every seed. This is
known preprocessing evidence; it cannot be promoted as a novel algorithm.

Extend the same conditioning check to the original author TRAIN/VALID protocol.
Fit pooled-coordinate mean/population STD on all2722 TRAIN rows only; transform
TRAIN/VALID and preserve original M scores and all647 VALID IDs. Export compact
pooled vectors as one-step sequences in a separate cache, with explicit source
hashes/encoder attribution and input-form description. Text remains CLS, A/V
remain length-masked means; no temporal representation or label scale change.
Missing modalities retain length0/zero vector. No TEST features/labels export.

Train text and simple fusion with the unchanged native point trainer and the
same original configurations except cache/output paths: hidden256/dropout.1,
AdamW1e-4/WD1e-4/batch32/max80/patience12, original validation MAE selection,
seeds13/42/2026. Six formal runs follow real32/32 smoke and epoch restore.
Compare raw/standardized text and fusion, report every native metric and
actual budget, independently recompute all647 forecasts from checkpoints.
Only then decide whether to freeze a better-conditioned research backbone.

Normalization gains are known baseline improvements. Official VALID is used
for development/selection and is not independent final-test evidence. Its
scores cannot be compared directly with different source-OOF cohorts to claim
causal source effects. Original raw models and caches stay unchanged.
