# Jev inspiration and scope

Updated native-task boundary, 2026-10-12: CH-SIMS v2 and MOSEI predict original
continuous scores separately. Structured regression outputs may contain a
score and predictive interval/scale with its calibration protocol. No such
uncertainty mechanism has yet been validated here. The five-class Choice
example below describes historical experiments only; it is not a confidence
definition for regression. See NATIVE_EVALUATION_PROTOCOL.md.

JevAffect is an independent discriminative multimodal research project inspired by typed probability outputs and empirically calibrated decisions. It does not use TypeSafe Jev weights, does not reproduce RLCD, and is not affiliated with TypeSafe.

Official sources inspected on 2026-10-11:
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://docs.typesafe.ai/introduction
- https://docs.typesafe.ai/confidence
- https://github.com/typesafe-ai

The inspected introduction and public repository listing provide API/SDK, workflow and adapter material, not an accessible full Jev model/training recipe. This is evidence about the inspected sources, not proof that no other release exists.

Jev's typed output guarantees are about schema conformance, not factual or sentiment prediction correctness. Our five-class softmax is similarly constrained but can be wrong. Calibration must be measured on held-out data; a normalized softmax alone is not evidence of calibration.

For five-class Choice outputs the documented TypeSafe confidence statistic is (max_probability - 0.2) / 0.8. We may expose this separately as choice_confidence for conceptual compatibility. It is not the probability of a correct prediction. Report max_probability for top-label calibration metrics, and keep the complete probability vector. An ordinal dispersion statistic could be reported separately, but must not be confused with Choice confidence.

Neither a softmax classifier nor post-hoc temperature scaling is claimed as a new contribution. Novelty, if supported, must come from a specific mechanism and controlled comparisons against closest multimodal methods. Counterfactual or occlusion sensitivity is inspectable evidence of model behavior, not proof of human-interpretable causal reasoning.
