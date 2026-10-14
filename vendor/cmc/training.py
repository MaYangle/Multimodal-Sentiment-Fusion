"""CMC's two-stage pseudo-label training logic adapted to five classes.

The functions preserve the source's unsupervised-for-modalities path: each
modality begins with the *multimodal* label, generates a modality-specific
counterfactual unified pseudo label, and updates its per-ID distribution by
EMA. No true T/A/V labels are accepted by this API.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .models import CMCClassifier, CMCUnimodalModel, MODALITIES, NUM_CLASSES, SupConLoss


class CMCTrainingError(RuntimeError):
    """Raised on invalid pseudo labels, batch contracts, or training state."""


def get_source_momentum(initial: float, epoch: int, gamma: float) -> float:
    """CMC source schedule: `1 - (1-initial)/epoch**gamma`."""
    if not math.isfinite(initial) or not 0.0 <= initial < 1.0:
        raise CMCTrainingError("initial momentum must be finite in [0,1)")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise CMCTrainingError("epoch must be a positive integer")
    if not math.isfinite(gamma) or gamma < 0:
        raise CMCTrainingError("momentum gamma must be finite and non-negative")
    return 1.0 - (1.0 - float(initial)) / (float(epoch) ** float(gamma))


def polarity_group(class_indices: Tensor) -> Tensor:
    """CMC SIMS contrastive grouping generalized to five classes.

    In source SIMS, class index 2 is positive while 0/1 are non-positive. With
    JevAffect order SNEG,WNEG,NEUT,WPOS,SPOS, positive groups are WPOS/SPOS
    (indices greater than 2). This keeps the source's binary-polarity SupCon
    auxiliary rather than replacing it with five-class contrastive learning.
    """
    return class_indices.reshape(-1).gt(2).to(dtype=torch.float32)


def generate_unified_pseudo_labels(logits: Tensor, base_one_hot: Tensor) -> Tensor:
    """Five-class generalization of CMC's reverse-label/unified-label rule.

    The source chooses the one-hot label nearest to the current softmax vector;
    mathematically that is the argmax class. Its alpha interpolates between the
    current pseudo one-hot and this nearest prediction according to relative
    Euclidean distances. The small denominator clamp only resolves the source
    zero-over-zero case when both labels are identical.
    """
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError("pseudo logits must be [B,5]")
    if base_one_hot.shape != logits.shape:
        raise ValueError("base pseudo one-hot labels must match logits [B,5]")
    if not torch.isfinite(logits).all() or not torch.isfinite(base_one_hot).all():
        raise ValueError("pseudo logits/targets must be finite")
    base = base_one_hot.to(device=logits.device, dtype=torch.float32)
    sums = base.sum(dim=-1)
    if torch.any(base < 0) or not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
        raise ValueError("base pseudo labels must be probability vectors summing to one")
    probs = torch.softmax(logits.detach().float(), dim=-1)
    reverse = F.one_hot(probs.argmax(dim=-1), num_classes=NUM_CLASSES).to(probs.dtype)
    distance_base = torch.linalg.vector_norm(probs - base, ord=2, dim=-1)
    distance_reverse = torch.linalg.vector_norm(probs - reverse, ord=2, dim=-1)
    denominator = (distance_base + distance_reverse).clamp_min(1e-12)
    alpha = ((distance_base - distance_reverse) / denominator).clamp(0.0, 1.0)
    pseudo = (1.0 - alpha[:, None]) * base + alpha[:, None] * reverse
    return pseudo / pseudo.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class PseudoLabelBank:
    """Per-canonical-ID, modality-specific soft pseudo labels."""

    values: dict[str, dict[str, Tensor]]
    num_classes: int = NUM_CLASSES

    def __post_init__(self) -> None:
        if self.num_classes != NUM_CLASSES:
            raise CMCTrainingError("this protocol adaptation uses exactly five classes")
        for sample_id, modal_map in self.values.items():
            if not isinstance(sample_id, str) or not sample_id:
                raise CMCTrainingError("pseudo-bank IDs must be non-empty canonical strings")
            if set(modal_map) != set(MODALITIES):
                raise CMCTrainingError("pseudo bank must contain T/A/V for every ID")
            for modality, vector in modal_map.items():
                if not torch.is_tensor(vector) or vector.shape != (NUM_CLASSES,):
                    raise CMCTrainingError("{} pseudo label for {} must be Tensor[5]".format(modality, sample_id))
                if not torch.isfinite(vector).all() or torch.any(vector < 0):
                    raise CMCTrainingError("pseudo label distributions must be finite/non-negative")
                if not torch.allclose(vector.float().sum(), vector.new_tensor(1.0).float(), atol=1e-5):
                    raise CMCTrainingError("pseudo label distributions must sum to one")

    @classmethod
    def from_multimodal_labels(
        cls, ids: Sequence[str], labels: Sequence[int] | Tensor
    ) -> "PseudoLabelBank":
        if isinstance(labels, np.ndarray):
            labels = np.array(labels, copy=True)
        label_tensor = torch.as_tensor(labels, dtype=torch.long).reshape(-1).cpu()
        if len(ids) != label_tensor.numel() or not ids:
            raise CMCTrainingError("IDs and multimodal labels must have the same non-zero length")
        if len(set(ids)) != len(ids):
            raise CMCTrainingError("cannot initialize pseudo bank with duplicate IDs")
        if torch.any(label_tensor < 0) or torch.any(label_tensor >= NUM_CLASSES):
            raise CMCTrainingError("multimodal labels must be class indices 0..4")
        values: dict[str, dict[str, Tensor]] = {}
        for sample_id, label in zip(ids, label_tensor.tolist()):
            one_hot = F.one_hot(torch.tensor(int(label)), num_classes=NUM_CLASSES).float()
            values[str(sample_id)] = {modality: one_hot.clone() for modality in MODALITIES}
        return cls(values)

    def batch(self, ids: Sequence[str], *, device: torch.device) -> dict[str, Tensor]:
        missing = [sample_id for sample_id in ids if sample_id not in self.values]
        if missing:
            raise CMCTrainingError("pseudo bank is missing canonical ID {!r}".format(missing[0]))
        return {
            modality: torch.stack([self.values[sample_id][modality] for sample_id in ids]).to(device)
            for modality in MODALITIES
        }

    @torch.no_grad()
    def update(
        self,
        ids: Sequence[str],
        pseudo_values: Mapping[str, Tensor],
        momenta: Mapping[str, float],
    ) -> None:
        if set(pseudo_values) != set(MODALITIES) or set(momenta) != set(MODALITIES):
            raise CMCTrainingError("pseudo updates require matching T/A/V values and momentum")
        for modality in MODALITIES:
            current = pseudo_values[modality].detach().float().cpu()
            momentum = float(momenta[modality])
            if current.shape != (len(ids), NUM_CLASSES):
                raise CMCTrainingError("{} pseudo update must be [B,5]".format(modality))
            if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
                raise CMCTrainingError("pseudo momentum must be in [0,1)")
            if not torch.isfinite(current).all() or torch.any(current < 0):
                raise CMCTrainingError("updated pseudo labels must be finite/non-negative")
            current = current / current.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            for row, sample_id in enumerate(ids):
                if sample_id not in self.values:
                    raise CMCTrainingError("pseudo bank is missing canonical ID {!r}".format(sample_id))
                old = self.values[sample_id][modality]
                updated = momentum * old + (1.0 - momentum) * current[row]
                updated = updated / updated.sum().clamp_min(1e-12)
                self.values[sample_id][modality] = updated.detach().clone()

    def snapshot(self) -> dict[str, dict[str, Tensor]]:
        return {
            sample_id: {modality: value.clone() for modality, value in modal_map.items()}
            for sample_id, modal_map in self.values.items()
        }


def _batch_modal_pseudo(
    bank: PseudoLabelBank, batch: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    ids = list(batch["id"])
    distributions = bank.batch(ids, device=device)
    hard = {modality: distribution.argmax(dim=-1) for modality, distribution in distributions.items()}
    return distributions, hard


@torch.no_grad()
def update_ema_parameters(
    source_models: Mapping[str, nn.Module],
    ema_models: Mapping[str, nn.Module],
    momenta: Mapping[str, float],
) -> None:
    """Match the source's EMA update over parameters (buffers are not averaged)."""
    for modality in MODALITIES:
        source = source_models[modality]
        ema = ema_models[modality]
        momentum = float(momenta[modality])
        source_params = dict(source.named_parameters())
        ema_params = dict(ema.named_parameters())
        if source_params.keys() != ema_params.keys():
            raise CMCTrainingError("EMA/source {} model parameter keys differ".format(modality))
        for name, parameter in source_params.items():
            ema_params[name].mul_(momentum).add_(parameter.detach(), alpha=1.0 - momentum)


def _forward_unimodal(
    models: Mapping[str, CMCUnimodalModel], batch: Mapping[str, Any]
) -> dict[str, dict[str, Tensor]]:
    return {
        "text": models["text"](batch["text"]),
        "audio": models["audio"](batch["audio"]),
        "vision": models["vision"](batch["vision"]),
    }


def _move_modalities(batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    values = {}
    for modality in MODALITIES:
        tensor = batch.get(modality)
        if not torch.is_tensor(tensor):
            raise CMCTrainingError("batch[{}] must be a Tensor".format(modality))
        values[modality] = tensor.to(device, non_blocking=device.type == "cuda")
    return values


def _contrastive_loss(
    outputs: Mapping[str, Mapping[str, Tensor]],
    hard_pseudo: Mapping[str, Tensor],
    criterion: SupConLoss,
) -> Tensor:
    features = torch.cat(
        (outputs["text"]["h_l"], outputs["audio"]["h_a"], outputs["vision"]["h_v"]),
        dim=0,
    )
    classes = torch.cat(
        (hard_pseudo["text"], hard_pseudo["audio"], hard_pseudo["vision"]), dim=0
    )
    # Source SIMS polarity grouping generalized from class>1 (3 classes) to
    # class>2 (five-class SNEG/WNEG/NEUT/WPOS/SPOS order).
    polarity = polarity_group(classes)
    return criterion(features, polarity)


def train_pseudo_pretrain_epoch(
    models: Mapping[str, CMCUnimodalModel],
    ema_models: Mapping[str, CMCUnimodalModel],
    optimizers: Mapping[str, torch.optim.Optimizer],
    loader,
    pseudo_bank: PseudoLabelBank,
    *,
    epoch: int,
    device: torch.device,
    contrastive_criterion: SupConLoss,
    initial_pseudo_momenta: Mapping[str, float],
    pseudo_momentum_gamma: Mapping[str, float],
    initial_ema_momenta: Mapping[str, float],
    ema_momentum_gamma: Mapping[str, float],
) -> dict[str, Any]:
    """One CMC pseudo pretraining epoch; only multimodal labels seed the bank."""
    if set(models) != set(MODALITIES) or set(ema_models) != set(MODALITIES) or set(optimizers) != set(MODALITIES):
        raise CMCTrainingError("pseudo pretraining requires T/A/V model, EMA, and optimizer maps")
    pseudo_momenta = {
        modality: get_source_momentum(initial_pseudo_momenta[modality], epoch, pseudo_momentum_gamma[modality])
        for modality in MODALITIES
    }
    ema_momenta = {
        modality: get_source_momentum(initial_ema_momenta[modality], epoch, ema_momentum_gamma[modality])
        for modality in MODALITIES
    }
    for model in models.values():
        model.train()
    for model in ema_models.values():
        model.eval()
    criterion = nn.CrossEntropyLoss()
    total = {"text_ce": 0.0, "audio_ce": 0.0, "vision_ce": 0.0, "supcon": 0.0, "total": 0.0}
    correct = 0
    samples = 0
    for batch in loader:
        ids = list(batch["id"])
        labels = batch["label"].to(device).long().reshape(-1)
        if labels.shape[0] != len(ids) or torch.any(labels < 0) or torch.any(labels >= NUM_CLASSES):
            raise CMCTrainingError("batch label indices must align with IDs and be in 0..4")
        _distributions, hard_pseudo = _batch_modal_pseudo(pseudo_bank, batch, device)
        model_batch = _move_modalities(batch, device)
        current_outputs = _forward_unimodal(models, model_batch)
        generated = {
            "text": generate_unified_pseudo_labels(
                current_outputs["text"]["pred"],
                F.one_hot(hard_pseudo["text"], NUM_CLASSES).float(),
            ),
            "audio": generate_unified_pseudo_labels(
                current_outputs["audio"]["pred"],
                F.one_hot(hard_pseudo["audio"], NUM_CLASSES).float(),
            ),
            "vision": generate_unified_pseudo_labels(
                current_outputs["vision"]["pred"],
                F.one_hot(hard_pseudo["vision"], NUM_CLASSES).float(),
            ),
        }
        losses = {
            modality: criterion(current_outputs[modality]["pred"], hard_pseudo[modality])
            for modality in MODALITIES
        }
        contrastive = _contrastive_loss(current_outputs, hard_pseudo, contrastive_criterion)
        loss = losses["text"] + losses["audio"] + losses["vision"] + contrastive
        if not torch.isfinite(loss):
            raise CMCTrainingError("pseudo pretraining loss is non-finite")
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()

        pseudo_bank.update(ids, generated, pseudo_momenta)
        update_ema_parameters(models, ema_models, ema_momenta)
        batch_size = int(labels.numel())
        total["text_ce"] += float(losses["text"].detach().item()) * batch_size
        total["audio_ce"] += float(losses["audio"].detach().item()) * batch_size
        total["vision_ce"] += float(losses["vision"].detach().item()) * batch_size
        total["supcon"] += float(contrastive.detach().item()) * batch_size
        total["total"] += float(loss.detach().item()) * batch_size
        correct += int((current_outputs["text"]["pred"].detach().argmax(-1) == hard_pseudo["text"]).sum().item())
        samples += batch_size
    if samples == 0:
        raise CMCTrainingError("empty CMC pseudo pretraining loader")
    return {
        **{key: value / samples for key, value in total.items()},
        "pseudo_text_accuracy": correct / samples,
        "samples": samples,
        "pseudo_momenta": pseudo_momenta,
        "ema_momenta": ema_momenta,
    }


@torch.no_grad()
def evaluate_pseudo_pretrain_epoch(
    models: Mapping[str, CMCUnimodalModel],
    ema_models: Mapping[str, CMCUnimodalModel],
    loader,
    pseudo_bank: PseudoLabelBank,
    *,
    epoch: int,
    device: torch.device,
    contrastive_criterion: SupConLoss,
    initial_pseudo_momenta: Mapping[str, float],
    pseudo_momentum_gamma: Mapping[str, float],
) -> dict[str, Any]:
    """Source-like EMA validation with train/valid pseudo banks only.

    The source trained against pseudo-label argmaxes and evaluated an EMA
    network against its moving pseudo bank. This intentionally reports that
    auxiliary validation objective; it does not substitute real unimodal
    labels or request the test loader.
    """
    for model in models.values():
        model.eval()
    for model in ema_models.values():
        model.eval()
    momenta = {
        modality: get_source_momentum(initial_pseudo_momenta[modality], epoch, pseudo_momentum_gamma[modality])
        for modality in MODALITIES
    }
    criterion = nn.CrossEntropyLoss()
    sums = {"text_ce": 0.0, "audio_ce": 0.0, "vision_ce": 0.0, "supcon": 0.0, "total": 0.0}
    samples = 0
    for batch in loader:
        ids = list(batch["id"])
        _, hard_pseudo = _batch_modal_pseudo(pseudo_bank, batch, device)
        model_batch = _move_modalities(batch, device)
        current_outputs = _forward_unimodal(models, model_batch)
        ema_outputs = _forward_unimodal(ema_models, model_batch)
        generated = {
            modality: generate_unified_pseudo_labels(
                current_outputs[modality]["pred"],
                F.one_hot(hard_pseudo[modality], NUM_CLASSES).float(),
            )
            for modality in MODALITIES
        }
        losses = {
            modality: criterion(ema_outputs[modality]["pred"], hard_pseudo[modality])
            for modality in MODALITIES
        }
        contrastive = _contrastive_loss(ema_outputs, hard_pseudo, contrastive_criterion)
        total_loss = losses["text"] + losses["audio"] + losses["vision"] + contrastive
        if not torch.isfinite(total_loss):
            raise CMCTrainingError("pseudo validation loss is non-finite")
        pseudo_bank.update(ids, generated, momenta)
        batch_size = len(ids)
        sums["text_ce"] += float(losses["text"].item()) * batch_size
        sums["audio_ce"] += float(losses["audio"].item()) * batch_size
        sums["vision_ce"] += float(losses["vision"].item()) * batch_size
        sums["supcon"] += float(contrastive.item()) * batch_size
        sums["total"] += float(total_loss.item()) * batch_size
        samples += batch_size
    if samples == 0:
        raise CMCTrainingError("empty CMC pseudo validation loader")
    return {**{key: value / samples for key, value in sums.items()}, "samples": samples, "pseudo_momenta": momenta}


def train_fusion_epoch(
    model: CMCClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    pseudo_bank: PseudoLabelBank,
    *,
    device: torch.device,
    contrastive_criterion: SupConLoss,
) -> dict[str, float | int]:
    """Train source CMC PFM/MCR with its main plus three pseudo-branch CE losses."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    sums = {"main_ce": 0.0, "text_ce": 0.0, "audio_ce": 0.0, "vision_ce": 0.0, "supcon": 0.0, "total": 0.0}
    correct = 0
    samples = 0
    for batch in loader:
        ids = list(batch["id"])
        labels = batch["label"].to(device).long().reshape(-1)
        _, hard_pseudo = _batch_modal_pseudo(pseudo_bank, batch, device)
        features = _move_modalities(batch, device)
        outputs = model(features)
        losses = {
            "main": criterion(outputs["pred"], labels),
            "text": criterion(outputs["pred_t"], hard_pseudo["text"]),
            "audio": criterion(outputs["pred_a"], hard_pseudo["audio"]),
            "vision": criterion(outputs["pred_v"], hard_pseudo["vision"]),
        }
        contrastive = _contrastive_loss(
            {"text": {"h_l": outputs["h_l"]}, "audio": {"h_a": outputs["h_a"]}, "vision": {"h_v": outputs["h_v"]}},
            hard_pseudo,
            contrastive_criterion,
        )
        total_loss = sum(losses.values()) + contrastive
        if not torch.isfinite(total_loss):
            raise CMCTrainingError("CMC fusion loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        batch_size = int(labels.numel())
        for key, loss_value in losses.items():
            sums[key + "_ce"] += float(loss_value.detach().item()) * batch_size
        sums["supcon"] += float(contrastive.detach().item()) * batch_size
        sums["total"] += float(total_loss.detach().item()) * batch_size
        correct += int((outputs["pred"].detach().argmax(dim=-1) == labels).sum().item())
        samples += batch_size
    if samples == 0:
        raise CMCTrainingError("empty CMC fusion training loader")
    return {**{key: value / samples for key, value in sums.items()}, "accuracy": correct / samples, "samples": samples}


@torch.no_grad()
def evaluate_fusion(
    model: CMCClassifier,
    loader,
    pseudo_bank: PseudoLabelBank,
    *,
    device: torch.device,
    contrastive_criterion: SupConLoss,
) -> dict[str, Any]:
    """Valid-only CMC metrics, including main five-class accuracy/NLL."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    samples = 0
    correct = 0
    nll = 0.0
    ids_all: list[str] = []
    labels_all: list[int] = []
    probs_all: list[list[float]] = []
    for batch in loader:
        ids = list(batch["id"])
        labels = batch["label"].to(device).long().reshape(-1)
        _, hard_pseudo = _batch_modal_pseudo(pseudo_bank, batch, device)
        outputs = model(_move_modalities(batch, device))
        logits = outputs["pred"].float()
        if logits.shape != (len(ids), NUM_CLASSES) or not torch.isfinite(logits).all():
            raise CMCTrainingError("CMC output must be finite [B,5] logits")
        nll += float(criterion(logits, labels).item())
        correct += int((logits.argmax(-1) == labels).sum().item())
        probabilities = torch.softmax(logits, dim=-1)
        ids_all.extend(ids)
        labels_all.extend(int(value) for value in labels.cpu().tolist())
        probs_all.extend(probabilities.cpu().tolist())
        samples += len(ids)
    if samples == 0:
        raise CMCTrainingError("empty CMC fusion validation loader")
    return {
        "accuracy": correct / samples,
        "nll": nll / samples,
        "samples": samples,
        "ids": ids_all,
        "labels": labels_all,
        "probabilities_uncalibrated": probs_all,
    }
