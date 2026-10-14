"""Five-class CE adapter for the vendored BC-MLF model.

The model's native scalar regression path remains available with task_out=1.
For the JevAffect protocol port, set task_out=5 and pass integer class targets
from the project's authoritative mapping. Keep raw regression targets
separately for BC-MLF's label-distance contrastive objective.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


NUM_CLASSES = 5
CLM_IGNORE_INDEX = -1


def _require_logits(name: str, value: torch.Tensor, expected_ndim: int) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != expected_ndim or value.shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"{name} must have {expected_ndim} dimensions and last size "
            f"{NUM_CLASSES}; got {tuple(value.shape)}"
        )


def soft_label_contrastive_loss(
    features: torch.Tensor,
    raw_y: torch.Tensor,
    *,
    temperature: float = 0.2,
    label_sigma: float = 0.8,
) -> torch.Tensor:
    """BC-MLF's soft supervised contrastive loss using raw numeric labels."""
    if features.ndim != 2:
        raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")
    batch_size = features.shape[0]
    if raw_y.ndim == 0:
        raise ValueError("raw_y must have a batch dimension")
    if raw_y.shape[0] != batch_size:
        raise ValueError("raw_y and features must have the same batch size")
    if batch_size < 2:
        return features.new_zeros(())

    temperature = max(float(temperature), 1e-6)
    label_sigma = max(float(label_sigma), 1e-6)
    z = F.normalize(features.float(), p=2, dim=1)
    y = raw_y.to(device=features.device).reshape(batch_size, -1).float()
    label_distance = torch.cdist(y, y, p=1)
    positive_weights = torch.exp(
        -(label_distance ** 2) / (2.0 * label_sigma ** 2)
    )
    non_self = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_weights = positive_weights * non_self.float()
    row_sum = positive_weights.sum(dim=1, keepdim=True)
    valid_rows = row_sum.squeeze(1) > 1e-8
    if not torch.any(valid_rows):
        return features.new_zeros(())

    positive_weights = positive_weights / row_sum.clamp_min(1e-8)
    logits = torch.matmul(z, z.transpose(0, 1)) / temperature
    logits = logits.masked_fill(~non_self, -1e9)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_row = -(positive_weights * log_probability).sum(dim=1)
    return per_row[valid_rows].mean()


def build_causal_lm_targets(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Match BC-MLF's trainer next-token shift and padding mask exactly.

    The source rolls token IDs left by one position, sets the final target to
    token ID 0, then masks positions where the input attention mask is false
    with ignore index -1.
    """
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must both have shape [B,L]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have identical shapes")
    targets = torch.roll(input_ids, shifts=-1, dims=1).clone()
    targets[:, -1] = 0
    targets.masked_fill_(~attention_mask.bool(), CLM_IGNORE_INDEX)
    return targets.long()


def bcmlf_five_class_loss(
    outputs: Mapping[str, Any],
    class_targets: torch.Tensor,
    raw_y: torch.Tensor,
    *,
    l_bn: float = 1.0,
    l_av: float = 1.0,
    l_text: float = 1.0,
    contrast_weight: float = 0.005,
    contrast_temperature: float = 0.2,
    contrast_label_sigma: float = 0.8,
    use_clm: bool = True,
    lm_targets: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute BC-MLF main/branch/CLM losses plus raw-label contrast.

    Expected forward keys are the native MSALM dictionary keys. `bn_logits`
    is `[B,N,5]`; text, AV, and task logits are `[B,5]`. The fusion branch
    keeps the source loss semantics by supervising each bottleneck token with
    the sample class. With `use_clm=True` (the supplied config default), callers
    must supply either pre-shifted `lm_targets` or both `input_ids` and
    `attention_mask`; missing LM targets raise instead of silently dropping
    the source's causal language-model objective. `lm_logits` must be returned
    by MSALM forward. `gamma` is the source CLM loss weight.
    """
    if "task_logits" not in outputs:
        raise KeyError("forward outputs must contain 'task_logits'")
    task_logits = outputs["task_logits"]
    _require_logits("task_logits", task_logits, expected_ndim=2)
    if class_targets.ndim != 1 or class_targets.shape[0] != task_logits.shape[0]:
        raise ValueError("class_targets must have shape [B]")
    if class_targets.dtype != torch.long:
        raise TypeError("class_targets must use torch.long class indices")
    if torch.any(class_targets < 0) or torch.any(class_targets >= NUM_CLASSES):
        raise ValueError("class_targets must be in [0, 4]")

    batch_size = task_logits.shape[0]
    if "bn_logits" not in outputs:
        raise KeyError("forward outputs must contain 'bn_logits'")
    bn_logits = outputs["bn_logits"]
    _require_logits("bn_logits", bn_logits, expected_ndim=3)
    if bn_logits.shape[0] != batch_size:
        raise ValueError("bn_logits batch size does not match task_logits")
    if bn_logits.shape[1] < 1:
        raise ValueError("bn_logits must contain at least one fusion token")
    bn_targets = class_targets[:, None].expand(-1, bn_logits.shape[1]).reshape(-1)
    bn_loss = F.cross_entropy(
        bn_logits.reshape(-1, NUM_CLASSES), bn_targets
    )

    branch_losses: dict[str, torch.Tensor] = {"bn_loss": bn_loss}
    for key, loss_name in (("av_logits", "av_loss"), ("text_logits", "text_loss")):
        if key not in outputs:
            raise KeyError(f"forward outputs must contain {key!r}")
        logits = outputs[key]
        _require_logits(key, logits, expected_ndim=2)
        if logits.shape[0] != batch_size:
            raise ValueError(f"{key} batch size does not match task_logits")
        branch_losses[loss_name] = F.cross_entropy(logits, class_targets)

    task_loss = F.cross_entropy(task_logits, class_targets)
    lm_loss = task_logits.new_zeros(())
    if use_clm:
        if attention_mask is None:
            raise ValueError(
                "use_clm=True requires attention_mask and either lm_targets "
                "or input_ids"
            )
        if lm_targets is None:
            if input_ids is None:
                raise ValueError(
                    "use_clm=True requires lm_targets or both input_ids and "
                    "attention_mask"
                )
            lm_targets = build_causal_lm_targets(input_ids, attention_mask)
        elif lm_targets.ndim != 2:
            raise ValueError("lm_targets must have shape [B,L]")

        if "lm_logits" not in outputs or outputs["lm_logits"] is None:
            raise KeyError("use_clm=True requires forward output 'lm_logits'")
        lm_logits = outputs["lm_logits"]
        if lm_logits.ndim != 3 or lm_logits.shape[0] != batch_size:
            raise ValueError("lm_logits must have shape [B,L,V]")
        if lm_targets.shape != attention_mask.shape:
            raise ValueError("lm_targets and attention_mask must have identical shapes")
        if lm_logits.shape[1] < lm_targets.shape[1]:
            raise ValueError(
                "lm_logits sequence is shorter than LM targets: "
                f"{lm_logits.shape[1]} < {lm_targets.shape[1]}"
            )
        lm_targets = lm_targets.to(device=lm_logits.device, dtype=torch.long).clone()
        valid_positions = attention_mask.to(device=lm_logits.device).bool()
        lm_targets.masked_fill_(~valid_positions, CLM_IGNORE_INDEX)
        if not torch.any(lm_targets != CLM_IGNORE_INDEX):
            raise ValueError("all LM target positions are masked")

        # MSALM appends bottleneck tokens to the decoder sequence.  The source
        # trainer computes CLM loss only on the leading text-token positions.
        text_length = lm_targets.shape[1]
        text_lm_logits = lm_logits[:, :text_length, :].contiguous()
        lm_loss = F.cross_entropy(
            text_lm_logits.reshape(-1, text_lm_logits.shape[-1]),
            lm_targets.reshape(-1),
            ignore_index=CLM_IGNORE_INDEX,
        )

    contrast_loss = task_logits.new_zeros(())
    if contrast_weight > 0:
        if "Feature_bn" not in outputs or outputs["Feature_bn"] is None:
            raise KeyError("Feature_bn is required when contrast_weight is positive")
        contrast_loss = soft_label_contrastive_loss(
            outputs["Feature_bn"],
            raw_y,
            temperature=contrast_temperature,
            label_sigma=contrast_label_sigma,
        )

    total = (
        task_loss
        + float(l_bn) * branch_losses["bn_loss"]
        + float(l_av) * branch_losses["av_loss"]
        + float(l_text) * branch_losses["text_loss"]
        + float(gamma) * lm_loss
        + float(contrast_weight) * contrast_loss
    )
    return {
        "loss": total,
        "task_loss": task_loss,
        **branch_losses,
        "lm_loss": lm_loss,
        "contrast_loss": contrast_loss,
    }


def five_class_forward_view(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Expose class probabilities and BC branch diagnostics separately.

    The softmax values are uncalibrated class probabilities. BC contribution
    weights describe branch mixing, not confidence in the selected class.
    """
    task_logits = outputs["task_logits"]
    _require_logits("task_logits", task_logits, expected_ndim=2)
    bn_logits = outputs["bn_logits"]
    _require_logits("bn_logits", bn_logits, expected_ndim=3)
    batch_size = task_logits.shape[0]
    for key in ("text_logits", "av_logits"):
        _require_logits(key, outputs[key], expected_ndim=2)
        if outputs[key].shape[0] != batch_size:
            raise ValueError(f"{key} batch size does not match task_logits")
    if bn_logits.shape[0] != batch_size:
        raise ValueError("bn_logits batch size does not match task_logits")
    weights = outputs.get("Contribution_weights")
    if weights is not None and tuple(weights.shape) != (batch_size, 3):
        raise ValueError(
            "Contribution_weights must have shape [B,3], "
            f"got {tuple(weights.shape)}"
        )
    return {
        "class_logits": task_logits,
        "class_probabilities_uncalibrated": torch.softmax(task_logits.float(), dim=-1),
        "branch_logits": {
            "fusion": bn_logits.mean(dim=1),
            "text": outputs["text_logits"],
            "av": outputs["av_logits"],
        },
        "branch_mixing_weights": weights,
    }
