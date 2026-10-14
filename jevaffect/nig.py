"""Numerical helpers for scalar Normal--Inverse-Gamma regression.

Parameters use the order ``(mu, gamma, alpha, beta)``.  The equations are
implemented independently from their mathematical descriptions; see
``docs/NIG_IMPLEMENTATION.md`` for attribution and exact conventions.
"""
from __future__ import annotations

import math
from typing import Literal

import torch


def _check_parameters(parameters: torch.Tensor) -> None:
    if not isinstance(parameters, torch.Tensor):
        raise TypeError("parameters must be a torch.Tensor")
    if parameters.ndim < 1 or parameters.shape[-1] != 4:
        raise ValueError("parameters must have shape [..., 4] in (mu, gamma, alpha, beta) order")
    if not parameters.is_floating_point():
        raise TypeError("parameters must have a floating point dtype")


def _safe_parameters(parameters: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return positive distribution parameters with small numerical floors."""
    mu, gamma, alpha, beta = parameters.unbind(dim=-1)
    eps = torch.finfo(parameters.dtype).eps
    gamma = gamma.clamp_min(eps)
    alpha = alpha.clamp_min(1.0 + eps)
    beta = beta.clamp_min(eps)
    return mu, gamma, alpha, beta


def nig_loss(
    parameters: torch.Tensor,
    target: torch.Tensor,
    regularization: float = 0.01,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Student-t NIG negative log likelihood plus evidence regularization.

    ``parameters`` may be ``[B, 4]`` or have additional leading dimensions,
    such as ``[B, M, 4]`` for branchwise losses.  Targets follow ordinary
    broadcasting; for ``[B, M, 4]``, a ``[B]`` target is expanded over M.
    ``regularization`` is the nonnegative evidence-penalty coefficient.
    """
    _check_parameters(parameters)
    if reduction not in ("mean", "none"):
        raise ValueError("reduction must be 'mean' or 'none'")
    if not math.isfinite(float(regularization)) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")

    mu, gamma, alpha, beta = _safe_parameters(parameters)
    target = torch.as_tensor(target, device=parameters.device, dtype=parameters.dtype)
    # A batch target [B] should align with the batch axis of [B, M, 4], not
    # with the final modality axis. Repeatedly append singleton axes as needed.
    while target.ndim < mu.ndim:
        target = target.unsqueeze(-1)
    try:
        error = mu - target
        error = torch.broadcast_to(error, mu.shape)
    except RuntimeError as exc:
        raise ValueError("target is not broadcastable to the NIG parameter shape") from exc

    # Compute log(gamma * error^2 + omega) with logaddexp.  Clamping only the
    # log argument at |error|=0 avoids log(0), while preserving the exact sum.
    log_gamma = torch.log(gamma)
    log_beta = torch.log(beta)
    log_omega = math.log(2.0) + log_beta + torch.log1p(gamma)
    tiny = torch.finfo(parameters.dtype).tiny
    log_abs_error = torch.log(error.abs().clamp_min(tiny))
    log_evidence_term = torch.logaddexp(log_gamma + 2.0 * log_abs_error, log_omega)

    nll = (
        0.5 * (math.log(math.pi) - log_gamma)
        - alpha * log_omega
        + (alpha + 0.5) * log_evidence_term
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    evidence_penalty = float(regularization) * error.abs() * (2.0 * gamma + alpha)
    loss = nll + evidence_penalty
    return loss.mean() if reduction == "mean" else loss


def fuse_nig(
    parameters: torch.Tensor,
    availability: torch.Tensor | None = None,
    mode: Literal["dynamic", "uniform"] = "dynamic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``[..., M, 4]`` NIG predictions and return ``(fused, weights)``.

    ``dynamic`` applies the authors' pairwise NIG summation, vectorized over M.
    ``uniform`` is the fixed-weight control documented in
    ``docs/NIG_IMPLEMENTATION.md``. Missing branches have zero weight and make
    no contribution. Every leading sample must have at least one available
    modality.
    """
    _check_parameters(parameters)
    if parameters.ndim < 2:
        raise ValueError("parameters must have shape [..., M, 4] with M >= 1")
    if mode not in ("dynamic", "uniform"):
        raise ValueError("mode must be 'dynamic' or 'uniform'")
    modality_count = parameters.shape[-2]
    if modality_count < 1:
        raise ValueError("at least one modality dimension is required")

    leading_shape = parameters.shape[:-1]
    if availability is None:
        mask = torch.ones(leading_shape, dtype=torch.bool, device=parameters.device)
    else:
        mask = torch.as_tensor(availability, device=parameters.device)
        if mask.shape != leading_shape:
            raise ValueError(f"availability must have shape {tuple(leading_shape)}")
        if mask.dtype != torch.bool:
            raise TypeError("availability must be boolean")

    count = mask.sum(dim=-1)
    if torch.any(count == 0):
        raise ValueError("every sample must have at least one available modality")

    mu, gamma, alpha, beta = _safe_parameters(parameters)
    gamma_available = torch.where(mask, gamma, torch.zeros_like(gamma))
    gamma_total = gamma_available.sum(dim=-1)
    denominator = gamma_total.clamp_min(torch.finfo(parameters.dtype).eps)

    mu_available = torch.where(mask, mu, torch.zeros_like(mu))
    if mode == "dynamic":
        weights = gamma_available / denominator.unsqueeze(-1)
        fused_mu = (weights * mu_available).sum(dim=-1)
        # This is the vector form of repeating the author's binary ⊕ operator:
        # the between-mean term is weighted by each branch's own confidence.
        between_confidence = gamma_available
    else:
        weights = mask.to(parameters.dtype) / count.to(parameters.dtype).unsqueeze(-1)
        fused_mu = (weights * mu_available).sum(dim=-1)
        # Keep total evidence identical to dynamic fusion, while assigning the
        # same effective confidence to each available branch in the dispersion term.
        effective_gamma = gamma_total / count.to(parameters.dtype)
        between_confidence = torch.where(mask, effective_gamma.unsqueeze(-1), torch.zeros_like(gamma))

    alpha_available = torch.where(mask, alpha, torch.zeros_like(alpha))
    beta_available = torch.where(mask, beta, torch.zeros_like(beta))
    fused_alpha = alpha_available.sum(dim=-1) + 0.5 * (count.to(parameters.dtype) - 1.0)
    centered = torch.where(mask, mu - fused_mu.unsqueeze(-1), torch.zeros_like(mu))
    between = 0.5 * (between_confidence * centered.square()).sum(dim=-1)
    fused_beta = beta_available.sum(dim=-1) + between

    fused = torch.stack((fused_mu, gamma_total, fused_alpha, fused_beta), dim=-1)
    return fused, weights


def predictive_variance(parameters: torch.Tensor) -> torch.Tensor:
    """Return the marginal Student-t predictive variance for NIG parameters."""
    _check_parameters(parameters)
    _, gamma, alpha, beta = _safe_parameters(parameters)
    return beta * (1.0 + gamma) / (gamma * (alpha - 1.0))


def predictive_distribution(parameters: torch.Tensor) -> torch.distributions.StudentT:
    """Return the marginal Student-t predictive distribution for NIG inputs."""
    _check_parameters(parameters)
    mu, gamma, alpha, beta = _safe_parameters(parameters)
    scale = torch.sqrt(beta * (1.0 + gamma) / (alpha * gamma))
    return torch.distributions.StudentT(df=2.0 * alpha, loc=mu, scale=scale)


__all__ = ["fuse_nig", "nig_loss", "predictive_distribution", "predictive_variance"]
