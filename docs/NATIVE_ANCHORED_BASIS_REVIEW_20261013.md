# Native anchored two-sided Bernstein CDF: prior-art and test review

2026-10-13. This review covers the corrected candidate: two side-specific degree-6 Bernstein/Beta mixtures, each assigned probability 0.5 around a frozen point, with latent supports extended beyond the native score limits and then censored to those limits. The earlier version whose supports stopped at the score limits created no endpoint atoms. A central interval with coverage below 100% would then exclude exact boundary outcomes; endpoint-bin probabilities would come only from integrating a continuous tail, with no explicit endpoint mass.

## Decision

**Approve one limited, source-honest probability-shape test; do not call the formula an original algorithm or promote it on this prior-art review.** This is a concrete small CDF family that hard-preserves the strong point forecast’s median, allows distinct lower and upper shapes, and gives endpoint labels nonzero probability through censoring. Those properties make it worth one controlled fit against strong same-median references. Its ingredients are established: Beta mixtures are the classical density form of Bernstein polynomials; neural conditional distributions with Bernstein bases already exist; censoring a latent forecast to create endpoint mass is established; and regression calibration already operates on frozen predictive models. The narrow task-specific combination may be useful in MSA, but a performance win would support an empirical adaptation unless a broader search and a clearer method distinction establish more.

The choice of degree 6, positive simplex weights, the fixed half-mass median constraint, affine side coordinates, and endpoint censoring are not by themselves originality claims. No human-rater latent state or annotation-conflict interpretation follows from these coefficients.

## Candidate and exact constraints

For each side let

\[
 h_{\omega}(t)=\sum_{k=0}^{6}\omega_k\,\mathrm{Beta}(t;k+1,7-k),\qquad
 \omega_k>0,\quad \sum_{k=0}^{6}\omega_k=1,\quad 0\le t\le1.
\]

The feature-conditioned heads produce separate left and right simplex weights and two positive latent margins. For native bounds \([-B,B]\), anchor \(m\in[-B,B]\), and raw margin outputs \(a_L,a_R\), use

\[
 w_L=(m+B)+\mathrm{softplus}(a_L)+10^{-4},\qquad
 w_R=(B-m)+\mathrm{softplus}(a_R)+10^{-4}.
\]

Draw a side with probability 0.5. On the left draw \(T_L\sim h_{\omega_L}\) and set \(Z=m-w_LT_L\); on the right draw \(T_R\sim h_{\omega_R}\) and set \(Z=m+w_RT_R\). The predictive outcome is the pushforward \(Y=\mathrm{clip}(Z,[-B,B])\). Both side weight heads and both margin heads should read the same frozen fusion representation and \(m\); no modality teacher or annotation label is needed.

For each component, the CDF is an integer-parameter beta tail, so no general numerical Beta implementation is needed:

\[
 I_t(k+1,7-k)=\sum_{j=k+1}^{7}\binom{7}{j}t^j(1-t)^{7-j}.
\]

The mixture CDF \(H_\omega(t)\) is the corresponding weighted sum. The censoring-induced endpoint masses have simple forms:

\[
 P(Y=-B)=\tfrac12\left[1-H_{\omega_L}\left(\frac{m+B}{w_L}\right)\right],\qquad
 P(Y=B)=\tfrac12\left[1-H_{\omega_R}\left(\frac{B-m}{w_R}\right)\right].
\]

The native CDF follows by evaluating the left or right affine coordinate and accounting for censoring. Since each latent left draw is at or below \(m\), each right draw is at or above \(m\), and censoring is monotone, an interior \(m\) has \(F_Y(m)=0.5\). At \(m=-B\), the censored left half gives \(F_Y(-B)=0.5\); at \(m=B\), \(F_Y(B^-)=0.5\) and \(F_Y(B)=1\). In all cases \(F_Y(m^-)\le0.5\le F_Y(m)\), so \(m\) is an exact generalized median. The two margins cross the native bounds, so clipping creates endpoint probability mass; that mass is a pushforward of the predictive law, not a latent human annotation atom. If exact continuous-target likelihood is also reported, it must use the mixed measure (endpoint atom probabilities plus interior density), not evaluate only a continuous density at \(\pm B\).

The Beta basis also has a useful sanity check: setting all seven weights to \(1/7\) produces a uniform density on the unit interval. This is a natural fixed-weight ablation. The head emits 7+7 mixture logits and two margin values (16 outputs), close to the existing 15-weight-plus-scale MDN head. That is a useful capacity match, though the shared trunk, regularization and training budget still need to match.

## Prior-art boundary

The proposed density basis is directly within established Bernstein density estimation. A Bernstein density on a bounded interval is a finite Beta mixture with data-dependent weights; conditional Bernstein density estimation has also been published. The particular component shapes \(\mathrm{Beta}(k+1,7-k)\) are standard, and choosing a finite degree is an ordinary model-capacity choice. Therefore neither the Beta basis, softmax weights, nor a closed-form polynomial CDF should be presented as the method contribution.

Bernstein quantile networks (BQN) are a close distribution-shape precedent, though not the same parameterization. BQN predicts a conditional quantile function as a Bernstein polynomial whose ordered coefficients come from a neural network. Bayesian spatial quantile regression likewise uses Bernstein bases to model monotone conditional quantile curves. These works do not establish the exact two-sided fixed-median density construction here, but they do establish that conditional Bernstein distribution shape, quantile constraints, and neural feature-conditioned coefficients are old ideas. Fixing one quantile is a linear/shape constraint in a quantile representation; splitting probability mass at 0.5 is another direct way to satisfy the median constraint. The median algebra alone is not novel.

The revised margins and shared clipping control address a real modeling requirement, but censoring itself is established. Censored BQN has been proposed for precipitation with a point mass at zero; other censored quantile-regression models cover mixed discrete-continuous outcomes. Thus the novelty boundary must not rely on endpoint atoms either. The plausible narrow difference is the *joint task design*: start from an already strong, frozen native MSA point predictor; allocate half the probability to each side of that exact point; learn separate, low-order Beta-mixture shapes and latent margins from common fusion features; then censor to the bounded sentiment scale and train/evaluate on the project’s shared author bins and regression-error events.

Post-hoc regression calibration is also established. Kuleshov, Fenner and Ermon recalibrate the CDF output of a previously trained regression model using an isotonic map; Song et al. propose distribution calibration for regression using a post-hoc Beta-link model. These are frozen-base calibration precedents, even though neither implies this specific neural side-mixture. A plain isotonic CDF map \(G\circ F\) need not preserve the anchor median. A fair same-median version should constrain the calibration map at \(G(0.5)=0.5\), with \(G(0)=0,G(1)=1\); for censored outcomes, fit it using randomized PIT values or an explicitly discrete/bin-aware calibration procedure. Fit the map only from source-held calibration predictions. This baseline tests whether a generic recalibration layer explains the gain.

The existing MSA history already sets the right boundary: previous AnchorCDF notes reject the disagreement gate and say not to claim the median-preserving mixture algebra as new. This candidate removes the failed teacher-disagreement hypothesis and changes the distribution family. It does not erase the previous overfit signal or make the previous VALID comparison an honest source-transfer result. See [AnchorCDF findings](NATIVE_ANCHOR_CDF_FINDINGS_V1.md), [AnchorCDF pre-fit plan](NATIVE_ANCHOR_CDF_PLAN_V1.md), and [the earlier prior-work review](NATIVE_ANNOTATION_OBSERVATION_RESEARCH_20261013.md).

## Strong fair comparison

All CDF models must receive the same frozen point outputs, fusion features, source folds, normalization data, seeds, selection rule, objective, optimization budget, censoring bounds and endpoint conventions. The simplex weights are feature-conditioned using the common fusion feature plus \(m\), without extra T/A/V teachers. Use the common author-bin negative log score as the primary fitted and selected objective; it is a proper score for the declared coarsened outcomes, not evidence that the continuous density is correctly specified.

| Reference | Why it is needed | Fairness condition |
|---|---|---|
| Existing censored logistic scale and feature-only central-kernel mixture | Shows whether the Bernstein family improves on the project’s existing point-anchored shapes | Same anchor, bin objective, censoring rule and inputs; report the prior disagreement model only as a rejected historical result |
| Median-recentered MDN | Strongest rich same-median shape control; distinguishes the Bernstein basis from general mixture capacity | Recenter the latent MDN by its predicted median, then apply the same censoring bounds; train/select on the final recentered-and-censored bin probabilities; use the 16-output-matched trunk and budget |
| Constrained Kuleshov/isotonic CDF recalibration | Tests whether a generic frozen-model calibration layer explains any gain | Fit on separate source-held predictions; require the map to pass through (0,0), (0.5,0.5), (1,1); handle censoring with randomized PIT or a declared bin-aware method |
| Existing unconstrained MDN (and optionally BQN) | A point-unconstrained probability reference shows the price, if any, of exact median preservation | Keep the original frozen point for native MAE and error-band centers, even if the distribution median differs; apply the same bounds/censoring and score |
| Uniform side weights and shared-side simplex weights | Tests whether learned coefficients and independent asymmetry add value | Uniform weights fix each standardized side shape; shared-side weights use one simplex for both sides while retaining separate margins |
| Degree sensitivity, e.g. degree 3 / 6 / 12 | Checks whether the declared degree 6 is arbitrary or whether gains simply follow added flexibility | Degree 6 is the sole primary candidate; treat other degrees as predeclared ablations selected only inside TRAIN source folds |

The median-recentered MDN is particularly important. Its latent forecast can be shifted by \(m-Q_{0.5}^{MDN}(x)\) so that the shifted distribution has median \(m\); applying the common monotone clip afterwards preserves that median for in-range \(m\). Its full distribution can then provide a rich same-median control without the degree-6 basis. Use a differentiable or otherwise training-consistent median calculation, and score the exact final distribution rather than an unshifted surrogate.

## Test and promotion gate

The first evidence should be on CH-SIMS with the verified native point predictor and labels, TRAIN only, and all three registered seeds. Build an honest outer source-held pipeline: each outer held source must be excluded from point-model fitting/selection, feature normalization, shape fitting, calibration fitting and early stopping. The earlier 21-fit AnchorCDF head split was source-disjoint at the head stage but not nested through the teacher models, so it is development evidence and cannot serve as the new source-transfer result. No TEST labels are needed or allowed for this decision.

Preserve the original frozen point predictions bit-for-bit and report their native MAE/Corr unchanged. The primary candidate must beat both the existing feature-only central-mixture reference and the median-recentered MDN on common-bin log score in the same direction for all three seeds, with a paired source/video-cluster interval for the gain excluding zero. It must also improve the predeclared \(P(|Y-m|\le0.1)\) Brier/log score and retain useful error-band probability calibration, while showing credible 50/80/90/95% interval coverage and widths. Add a CDF-wide proper score such as CRPS or integrated threshold Brier if available; coverage or ECE by itself is not sufficient. Keep the probability event centered at the frozen point and report its calibration separately from bin reliability.

The gain should survive the tied-simplex and degree controls and should not be explainable by extra parameters, optimization, or a looser margin range. A benefit only over the lower-capacity logistic model establishes a shape-family benefit at most; a gain over a same-capacity, same-median MDN or a useful complexity/robustness advantage is needed to support a method-specific claim. If seed directions disagree, the paired interval includes zero, event/coverage calibration worsens, the rich same-median control matches it, or it wins only after selecting a degree on VALID, reject promotion and record the result as another negative shape study.

On MOSEI, the native target bounds are \([-3,3]\), while an unbounded anchor can fall outside them. Define one common probabilistic anchor control \(m_c=\mathrm{clip}(m_{native},[-3,3])\) for every bounded CDF reference, and report the frequency and source pattern of clipping. Keep native MAE on the untouched \(m_{native}\); do not silently replace it with MAE of \(m_c\). For absolute-error events, keep the event centered at \(m_{native}\) and disclose when \(m_c\ne m_{native}\). In those cases the CDF median is the clipped control, not the original out-of-range point, so do not claim exact native-point median preservation for those rows. A separate unclipped native MAE reference and shared clipping policy are required before any cross-dataset comparison.

## Primary references

- Vitale (1975), [A Bernstein polynomial approach to density function estimation](https://doi.org/10.1016/B978-0-12-568002-8.50011-2): classical density construction as a finite Beta mixture.
- Belalia, Bouezmarni and Leblanc (2019), [Bernstein conditional density estimation with application to conditional distribution and regression functions](https://doi.org/10.1016/j.jkss.2019.05.005): conditional Bernstein density with data-dependent Beta-mixture weights.
- Bremnes (2020), [Ensemble postprocessing using quantile function regression based on neural networks and Bernstein polynomials](https://doi.org/10.1175/MWR-D-19-0227.1): neural feature-conditioned Bernstein quantile distributions.
- Reich, Fuentes and Dunson (2011), [Bayesian spatial quantile regression](https://doi.org/10.1198/jasa.2010.ap09237): Bernstein basis for conditional quantile curves with monotonicity constraints.
- Bremnes (2023), [Censored Bernstein quantile networks for probabilistic precipitation forecasting](https://doi.org/10.5194/egusphere-egu23-13849): Bernstein quantiles plus censoring for a mixed distribution with a boundary point mass.
- Kuleshov, Fenner and Ermon (2018), [Accurate uncertainties for deep learning using calibrated regression](https://proceedings.mlr.press/v80/kuleshov18a.html): post-hoc isotonic CDF recalibration for regression forecasts.
- Song et al. (2019), [Distribution calibration for regression](https://proceedings.mlr.press/v97/song19a.html): post-hoc calibration of frozen regression distributions using a Beta link.
- Wu and Yao (2016), [Mixtures of quantile regressions](https://doi.org/10.1016/j.csda.2014.04.014): mixture regression models that directly include conditional median/quantile modeling.
