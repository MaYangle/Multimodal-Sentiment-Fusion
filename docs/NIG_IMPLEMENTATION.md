# Native regression NIG module

This implementation provides three independent modality experts for the shared `label_M` regression target. Each branch consumes one pooled `[B, D]` text, audio, or vision feature tensor and predicts `(mu, gamma, alpha, beta)` plus a scalar auxiliary prediction. Branch MLPs use a common configurable hidden width, but have independent parameters. They contain `Linear → GELU → Dropout` encoders and separate output heads; there is no LayerNorm or cross-modal feature-fusion branch. The auxiliary heads always exist. The training script decides whether to include their supervised loss.

The model call is `ReliabilityRegressor(dims, hidden=128, dropout=0.1, fusion="dynamic")(features)`. `features` contains `text`, `audio`, `vision`, and optionally boolean `availability` shaped `[B, 3]` in that order. Omitting the availability tensor treats all three inputs as available. The model never infers absence from all-zero or padded feature values. An explicitly unavailable branch still produces finite branch parameters and an auxiliary prediction, but receives zero fusion weight. Every sample must have at least one available modality.

The output is a dictionary with `prediction [B]`, `fused [B,4]`, `branches [B,3,4]`, `unimodal_prediction [B,3]`, `weights [B,3]`, and `availability [B,3]`. The NIG parameter order is `(mu, gamma, alpha, beta)`: `mu` is unconstrained, `gamma` and `beta` use `softplus(raw)+1e-4`, and `alpha` uses `softplus(raw)+1`, so the Student-t predictive variance exists. The fusion helpers accept any positive branch values and apply small numeric floors at their boundary.

## Dynamic MoNIG fusion

For a scalar target, NIG means `y | theta, sigma² ~ Normal(theta, sigma²)`, `theta | sigma² ~ Normal(mu, sigma²/gamma)`, and `sigma² ~ InverseGamma(alpha, beta)`. The first parameter `mu` is the prior location of the predictive mean. Ma et al. define the binary NIG summation in Eq. 8–10 of the paper. For two components it is

```text
gamma' = gamma1 + gamma2
mu'    = (gamma1*mu1 + gamma2*mu2) / gamma'
alpha' = alpha1 + alpha2 + 1/2
beta'  = beta1 + beta2
         + 1/2 * [gamma1*(mu1-mu')² + gamma2*(mu2-mu')²]
```

Their operator is associative, so for the set `A` of available modalities this implementation evaluates the same result in vector form:

```text
gamma_fused = sum_A gamma_m
weights_m   = gamma_m / gamma_fused       (m in A; otherwise 0)
mu_fused    = sum_A weights_m * mu_m
alpha_fused = sum_A alpha_m + (|A|-1)/2
beta_fused  = sum_A beta_m
              + 1/2 * sum_A gamma_m*(mu_m-mu_fused)²
```

Thus the returned dynamic `weights` are each branch's normalized `gamma` confidence, matching the authors' operator. Missing branches contribute to none of these terms. If only one branch is available, fusion returns that branch's NIG unchanged.

## Uniform control G0

The fixed-weight control sets `weights_m=1/|A|` for available branches and computes `mu_fused` as the equal-weight mean. It keeps the same total evidence and alpha as dynamic fusion: `gamma_fused=sum_A gamma_m` and `alpha_fused=sum_A alpha_m+(|A|-1)/2`. For the between-prediction dispersion term only, it uses the equalized per-branch confidence `gamma_eff=gamma_fused/|A|`:

```text
beta_fused = sum_A beta_m
             + 1/2 * sum_A gamma_eff*(mu_m-mu_fused)²
```

This defines a useful fixed-mean control distribution while holding total `gamma` and the alpha-combination rule constant. It is **not** the authors' full MoNIG rule: it disables confidence-weighted means and equalizes the dispersion contribution. It is also not the literal mixture distribution in the paper's Eq. 7, which the paper notes does not in general reduce to a closed-form NIG. The experiment should call it the uniform/equal-weight control, not a complete reproduction of MoNIG or an independently established method.

## Predictive distribution and loss

Marginalizing a NIG gives a Student-t prediction with

```text
df = 2*alpha
loc = mu
scale² = beta*(1+gamma)/(alpha*gamma)
Var[y] = beta*(1+gamma)/(gamma*(alpha-1))
```

`predictive_distribution` returns this `torch.distributions.StudentT`; `predictive_variance` returns its variance. `nig_loss(parameters, target, regularization=0.01, reduction="mean"|"none")` computes its exact negative log likelihood plus the evidential penalty used in the authors' implementation:

```text
omega = 2*beta*(1+gamma)
NLL = 0.5*log(pi/gamma) - alpha*log(omega)
      + (alpha+0.5)*log(gamma*(y-mu)² + omega)
      + lgamma(alpha) - lgamma(alpha+0.5)
L = NLL + regularization*|y-mu|*(2*gamma+alpha)
```

The NLL's sum inside the logarithm is evaluated with `logaddexp`, and the evidence regularizer is the absolute residual times evidence strength. `reduction="none"` preserves all leading dimensions, so `[B,3,4]` branch parameters yield `[B,3]` losses for availability-based filtering by the trainer. Positive floors guard logarithms and denominators against floating-point underflow.

## Attribution and implementation boundary

The parent agent subsequently ran the math/missingness tests in the actual
project PyTorch environment, including extreme evidence underflow. Model
outputs use a1e-4 floor for gamma/beta and alpha>1, preserving finite variance.
Three training-contract tests verify auxiliary-gradient isolation, validation
annotation isolation and exact resume. The integrated suite passed149 tests;
the subagent's initial inability to import torch was an environment access
limitation and does not describe the verified training runtime.

The dynamic operator and loss are mathematical formulas from Huan Ma et al., “Trustworthy Multimodal Regression with Mixture of Normal-inverse Gamma Distributions,” NeurIPS 2021, especially Eqs. 3–12. The authors' repository independently confirms Eq. 9 in `src/utils.py::moe_nig` and its NLL/evidence regularizer in `src/utils.py::criterion_nig`. References: [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2021/file/371bce7dc83817b7893bcdeed13799b5-Paper.pdf), [arXiv version](https://arxiv.org/abs/2111.08456), [authors' repository](https://github.com/MaHuanAAA/MoNIG), [authors' utility source](https://github.com/MaHuanAAA/MoNIG/blob/main/src/utils.py).

The Python implementation in `jevaffect/nig.py` and `jevaffect/reliability.py` is written independently and does not copy source from the authors' repository. The cited equations and prior method are attributed to their authors. The model omits MoNIG's pseudo-modality branch and is a targeted three-branch regression adaptation; no novelty or performance claim follows from implementing these equations.
