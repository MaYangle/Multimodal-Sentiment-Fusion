# Historical report evidence

Source: old README's shared Drive folder, `Experimental report.pdf`, SHA256 828d65edb062c817ddb44ab8b19eacd165dd54fde2596021b11e5a3717c0f98f. The five-page document was read in full with text extraction; the performance page was rendered and visually checked.

The report describes Whisper plus language-specific BERT, clip-level acoustic/visual descriptors, and transformer/cross-attention fusion. It reports internal accuracy above 50% without a per-split table, exact split hashes, seeds, checkpoint hashes, or reproducible logs. It reports public Kaggle score 0.4350 and says enhanced models were not trained to convergence, with many experiments on CPU. These are historical author statements, not results independently reproduced in this project.

The PDF describes Macro-F1 and attention explanations, but the audited repository actually computes weighted F1 in its main metric branch and reduces modalities to single tokens. No numeric ablation table or heatmap is present in the five pages. Do not turn descriptions of planned/claimed ablations into measured evidence. Public Kaggle score cannot be compared directly to author-split five-class accuracy or the unavailable private final-set score.

Use this report to preserve provenance and choose faithful retraining settings; use new saved per-sample predictions and source/split hashes for all current comparisons.
