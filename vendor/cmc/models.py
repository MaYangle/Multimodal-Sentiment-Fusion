"""CMC architecture port with the author's PFM/MCR computation preserved.

Structure is adapted from `gw-zhong/CMC` commit
`c695f719f9cd0136e52dc2f7a9e75a40aabd37ad`, `src/models.py`. The source's
hard-coded BERT directory is replaced by an explicit, local-only/injected text
encoder so this module can consume JevAffect's canonical `text_bert.npy`
`[N,3,T]` token triplets without downloading weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


NUM_CLASSES = 5
MODALITIES = ("text", "audio", "vision")
_HEAD_NAMES = ("proj1", "proj2", "out_layer")


class CMCModelError(ValueError):
    """Raised when an input or CMC adaptation config violates the port contract."""


class LocalBertTokenEncoder(nn.Module):
    """The source CMC BERT interface backed by an explicitly local HF model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.input_mode = "bert_token_triplets"
        self.model = model
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise CMCModelError("local BERT model must expose config.hidden_size")
        self.output_dim = hidden_size

    @classmethod
    def from_local(cls, model_dir: str) -> "LocalBertTokenEncoder":
        """Load only a pre-staged local BERT directory; never fetch remote weights."""
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        return cls(model)

    def forward(self, token_triplets: Tensor, *, use_finetune: bool) -> Tensor:
        if token_triplets.ndim != 3 or token_triplets.shape[1] != 3:
            raise ValueError("CMC text tokens must be [B,3,T] (IDs, attention mask, token types)")
        input_ids = token_triplets[:, 0, :].long()
        attention_mask = token_triplets[:, 1, :].long()
        token_type_ids = token_triplets[:, 2, :].long()
        if use_finetune:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError("local BERT must return last_hidden_state [B,T,D]")
        return hidden


class ContextualTextEncoder(nn.Module):
    """Adapter for a cache that already stores contextual `[B,T,D]` text."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_mode = "cached_contextual_text"
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise CMCModelError("input_dim must be a positive integer")
        self.output_dim = input_dim

    def forward(self, contextual_text: Tensor, *, use_finetune: bool) -> Tensor:
        del use_finetune
        if contextual_text.ndim != 3 or contextual_text.shape[-1] != self.output_dim:
            raise ValueError("contextual text must be [B,T,configured_hidden_dim]")
        return contextual_text.float()


class TopkRouter(nn.Module):
    """CMC's sparse top-k gating: softmax only across selected experts."""

    def __init__(self, embed_dim: int, num_experts: int = 3, top_k: int = 3) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise CMCModelError("top_k must be in [1, num_experts]")
        self.top_k = int(top_k)
        self.linear = nn.Linear(embed_dim, num_experts)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.linear(value)
        top_logits, indices = logits.topk(self.top_k, dim=-1)
        sparse = torch.full_like(logits, float("-inf")).scatter(-1, indices, top_logits)
        return F.softmax(sparse, dim=-1), indices


class SupConLoss(nn.Module):
    """CMC supervised contrastive objective over the source polarity groups.

    The source normalizes each modality representation before calling this
    module. The implementation uses log-sum-exp for numerical stability while
    retaining the source ratio objective: negative log of summed positive
    mass over all non-self mass, averaged over rows with a positive.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise CMCModelError("contrastive temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("features must be [N,D]")
        labels = labels.reshape(-1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("contrastive labels must align with features")
        count = features.shape[0]
        if count < 2:
            return features.new_zeros(())
        logits = features.float() @ features.float().T / self.temperature
        non_self = ~torch.eye(count, dtype=torch.bool, device=features.device)
        positives = labels[:, None].eq(labels[None, :]) & non_self
        valid = positives.any(dim=1)
        if not bool(valid.any()):
            return features.new_zeros(())
        log_denominator = torch.logsumexp(logits.masked_fill(~non_self, float("-inf")), dim=1)
        positive_logits = logits.masked_fill(~positives, float("-inf"))
        positive_logits = torch.where(valid[:, None], positive_logits, torch.zeros_like(positive_logits))
        log_positive_mass = torch.logsumexp(positive_logits, dim=1)
        per_row = -(log_positive_mass - log_denominator)
        return per_row[valid].mean().to(dtype=features.dtype)


def _validate_hyperparameters(
    *, input_dim: int, embed_dim: int, output_dim: int, nhead: int, transformer_layers: int
) -> None:
    for name, value in (
        ("input_dim", input_dim), ("embed_dim", embed_dim),
        ("output_dim", output_dim), ("nhead", nhead),
        ("transformer_layers", transformer_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CMCModelError("{} must be a positive integer".format(name))
    if embed_dim % nhead:
        raise CMCModelError("embed_dim must be divisible by nhead")


def _transformer(embed_dim: int, nhead: int, layers: int) -> nn.TransformerEncoder:
    # Matches source nn.TransformerEncoderLayer defaults except for the
    # explicit norm_first=True in CMC's models.py.
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            norm_first=True,
        ),
        num_layers=layers,
    )


class CMCUnimodalModel(nn.Module):
    """Source CMC TextModel/AudioModel/VisionModel with five-logit heads."""

    def __init__(
        self,
        modality: Literal["text", "audio", "vision"],
        input_dim: int,
        *,
        embed_dim: int = 32,
        output_dim: int = NUM_CLASSES,
        nhead: int = 2,
        transformer_layers: int = 4,
        out_dropout: float = 0.3,
        text_encoder: nn.Module | None = None,
        finetune_text: bool = False,
    ) -> None:
        super().__init__()
        if modality not in MODALITIES:
            raise CMCModelError("modality must be text, audio, or vision")
        _validate_hyperparameters(
            input_dim=input_dim,
            embed_dim=embed_dim,
            output_dim=output_dim,
            nhead=nhead,
            transformer_layers=transformer_layers,
        )
        if output_dim != NUM_CLASSES:
            raise CMCModelError("CMC protocol port requires five-logit unimodal heads")
        if not 0.0 <= float(out_dropout) < 1.0:
            raise CMCModelError("out_dropout must be in [0,1)")
        self.modality = modality
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.output_dim = int(output_dim)
        self.out_dropout = float(out_dropout)
        self.finetune_text = bool(finetune_text)

        if modality == "text":
            if text_encoder is None:
                raise CMCModelError("text modality requires a BERT/token encoder or contextual adapter")
            encoder_width = getattr(text_encoder, "output_dim", None)
            if encoder_width is not None and int(encoder_width) != self.input_dim:
                raise CMCModelError("text encoder output width differs from input_dim")
            self.text_model = text_encoder
            self.proj_l = nn.Conv1d(self.input_dim, embed_dim, kernel_size=1, bias=False)
            self.encoder_l = _transformer(embed_dim, nhead, transformer_layers)
            self._projector_name = "l"
        elif modality == "audio":
            self.norm_a = nn.BatchNorm1d(self.input_dim)
            self.proj_a = nn.Conv1d(self.input_dim, embed_dim, kernel_size=1, bias=False)
            self.encoder_a = _transformer(embed_dim, nhead, transformer_layers)
            self._projector_name = "a"
        else:
            self.norm_v = nn.BatchNorm1d(self.input_dim)
            self.proj_v = nn.Conv1d(self.input_dim, embed_dim, kernel_size=1, bias=False)
            self.encoder_v = _transformer(embed_dim, nhead, transformer_layers)
            self._projector_name = "v"

        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.out_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        if inputs.ndim != 3:
            raise ValueError("CMC unimodal features must be rank 3")
        modality = self.modality
        if modality == "text":
            sequence = self.text_model(inputs, use_finetune=self.finetune_text)
            if sequence.ndim != 3 or sequence.shape[0] != inputs.shape[0]:
                raise ValueError("text encoder must return [B,T,D]")
            if sequence.shape[-1] != self.input_dim:
                raise ValueError("text encoder hidden width differs from CMC projection")
            projected = self.proj_l(sequence.transpose(1, 2)).permute(2, 0, 1)
            hidden_sequence = self.encoder_l(projected)
            representation_key = "h_l"
        elif modality == "audio":
            projected = self.proj_a(self.norm_a(inputs.transpose(1, 2))).permute(2, 0, 1)
            hidden_sequence = self.encoder_a(projected)
            representation_key = "h_a"
        else:
            projected = self.proj_v(self.norm_v(inputs.transpose(1, 2))).permute(2, 0, 1)
            hidden_sequence = self.encoder_v(projected)
            representation_key = "h_v"

        representation = F.normalize(hidden_sequence.mean(dim=0), p=2, dim=-1)
        residual = self.proj2(
            F.dropout(F.relu(self.proj1(representation)), p=self.out_dropout, training=self.training)
        )
        residual = residual + representation
        return {
            "pred": self.out_layer(residual),
            "last_hs_proj": residual,
            representation_key: representation,
        }


class CMCClassifier(nn.Module):
    """Source CMC multimodal consensus model with a protocol-bound five-class head."""

    def __init__(
        self,
        *,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        text_encoder: nn.Module,
        embed_dim: int = 32,
        num_classes: int = NUM_CLASSES,
        nhead: int = 2,
        transformer_layers: int = 4,
        top_k: int = 3,
        out_dropout: float = 0.3,
        temperature: float = 0.07,
        finetune_text: bool = False,
        text_input_mode: Literal["bert_token_triplets", "cached_contextual_text"] = "bert_token_triplets",
    ) -> None:
        super().__init__()
        _validate_hyperparameters(
            input_dim=text_dim,
            embed_dim=embed_dim,
            output_dim=num_classes,
            nhead=nhead,
            transformer_layers=transformer_layers,
        )
        if num_classes != NUM_CLASSES:
            raise CMCModelError("CMC protocol port requires exactly five output classes")
        if audio_dim <= 0 or vision_dim <= 0:
            raise CMCModelError("audio_dim and vision_dim must be positive")
        if not 0.0 <= float(out_dropout) < 1.0:
            raise CMCModelError("out_dropout must be in [0,1)")
        if temperature <= 0:
            raise CMCModelError("PFM temperature must be positive")
        self.text_dim = int(text_dim)
        self.audio_dim = int(audio_dim)
        self.vision_dim = int(vision_dim)
        self.embed_dim = int(embed_dim)
        self.output_dim = int(num_classes)
        self.top_k = int(top_k)
        self.out_dropout = float(out_dropout)
        self.temperature = float(temperature)
        self.finetune_text = bool(finetune_text)
        if text_input_mode not in ("bert_token_triplets", "cached_contextual_text"):
            raise CMCModelError("unsupported text_input_mode {!r}".format(text_input_mode))
        self.text_input_mode = text_input_mode

        if int(getattr(text_encoder, "output_dim", text_dim)) != text_dim:
            raise CMCModelError("text encoder output width differs from text_dim")
        self.text_model = text_encoder
        self.norm_a = nn.BatchNorm1d(self.audio_dim)
        self.norm_v = nn.BatchNorm1d(self.vision_dim)
        self.proj_l = nn.Conv1d(self.text_dim, embed_dim, kernel_size=1, bias=False)
        self.proj_a = nn.Conv1d(self.audio_dim, embed_dim, kernel_size=1, bias=False)
        self.proj_v = nn.Conv1d(self.vision_dim, embed_dim, kernel_size=1, bias=False)
        self.encoder_l = _transformer(embed_dim, nhead, transformer_layers)
        self.encoder_a = _transformer(embed_dim, nhead, transformer_layers)
        self.encoder_v = _transformer(embed_dim, nhead, transformer_layers)
        self.router = TopkRouter(embed_dim=embed_dim, num_experts=3, top_k=top_k)
        self.proj1s = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in range(3))
        self.proj2s = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in range(3))
        self.out_layers = nn.ModuleList(nn.Linear(embed_dim, num_classes) for _ in range(3))

    def forward(
        self,
        features: Mapping[str, Tensor] | Tensor,
        audio: Tensor | None = None,
        vision: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if isinstance(features, Mapping):
            text = features.get("text")
            audio = features.get("audio")
            vision = features.get("vision")
        else:
            text = features
        if not all(torch.is_tensor(value) for value in (text, audio, vision)):
            raise TypeError("CMCClassifier expects text, audio, and vision tensors")
        if text.ndim != 3 or audio.ndim != 3 or vision.ndim != 3:
            raise ValueError("CMC input features must be [B,T,D]")
        text_shape_matches = (
            text.shape[1] == 3
            if self.text_input_mode == "bert_token_triplets"
            else text.shape[-1] == self.text_dim
        )
        if not text_shape_matches or audio.shape[-1] != self.audio_dim or vision.shape[-1] != self.vision_dim:
            raise ValueError("CMC feature widths differ from the configured cache schema")

        text_hidden = self.text_model(text, use_finetune=self.finetune_text)
        if text_hidden.ndim != 3 or text_hidden.shape[0] != text.shape[0] or text_hidden.shape[-1] != self.text_dim:
            raise ValueError("CMC text encoder must return [B,T,text_dim]")
        projected_text = self.proj_l(text_hidden.transpose(1, 2)).permute(2, 0, 1)
        projected_audio = self.proj_a(self.norm_a(audio.transpose(1, 2))).permute(2, 0, 1)
        projected_vision = self.proj_v(self.norm_v(vision.transpose(1, 2))).permute(2, 0, 1)
        h_l_seq = self.encoder_l(projected_text)
        h_a_seq = self.encoder_a(projected_audio)
        h_v_seq = self.encoder_v(projected_vision)
        h_l = F.normalize(h_l_seq.mean(dim=0), p=2, dim=-1)
        h_a = F.normalize(h_a_seq.mean(dim=0), p=2, dim=-1)
        h_v = F.normalize(h_v_seq.mean(dim=0), p=2, dim=-1)

        # PFM: source modality-consensus attention over the three modality tokens.
        modality_tokens = torch.stack((h_l, h_a, h_v), dim=1)
        cosine_similarity = torch.bmm(modality_tokens, modality_tokens.transpose(1, 2))
        consensus = torch.bmm(
            F.softmax(cosine_similarity / self.temperature, dim=-1), modality_tokens
        )

        # MCR: sparse top-k expert selection and per-sample weighted consensus.
        fused = consensus.sum(dim=1)
        routing_weights, expert_indices = self.router(fused)
        batch_size = consensus.shape[0]
        final_logits = consensus.new_zeros((batch_size, self.output_dim))
        branch_logits = [consensus.new_zeros((batch_size, self.output_dim)) for _ in range(3)]
        for expert in range(3):
            selected = (expert_indices == expert).any(dim=-1)
            if not bool(selected.any()):
                continue
            expert_features = consensus[selected, expert]
            projected = self.proj2s[expert](
                F.dropout(
                    F.relu(self.proj1s[expert](expert_features)),
                    p=self.out_dropout,
                    training=self.training,
                )
            ) + expert_features
            logits = self.out_layers[expert](projected)
            branch_logits[expert][selected] = logits
            weights = routing_weights[selected, expert].unsqueeze(1)
            final_logits[selected] += weights * logits

        return {
            "pred": final_logits,
            "pred_t": branch_logits[0],
            "pred_a": branch_logits[1],
            "pred_v": branch_logits[2],
            "h_l": h_l,
            "h_a": h_a,
            "h_v": h_v,
            "routing_weights": routing_weights,
            "expert_indices": expert_indices,
        }

    def forward_modalities(self, text: Tensor, audio: Tensor, vision: Tensor) -> dict[str, Tensor]:
        """Source-shaped call signature retained for direct adapter use."""
        return self.forward(text, audio, vision)


def transfer_unimodal_to_cmc(
    model: CMCClassifier,
    unimodal_models: Mapping[str, CMCUnimodalModel],
) -> dict[str, Any]:
    """Port source `transfer_models`: copy encoders and map 3 heads by modality.

    The returned CMC state keeps the five-class classifier widths. Source
    stage-one models are expected to have the same five-class `out_layer`.
    """
    if set(unimodal_models) != set(MODALITIES):
        raise CMCModelError("transfer requires text, audio, and vision pseudo-pretrained models")
    state = model.state_dict()
    copied: list[str] = []
    for modality_index, modality in enumerate(MODALITIES):
        source = unimodal_models[modality]
        expected_modality = modality
        if not isinstance(source, CMCUnimodalModel) or source.modality != expected_modality:
            raise CMCModelError("{} pseudo model has the wrong source modality".format(modality))
        for name, tensor in source.state_dict().items():
            mapped = name
            for source_head, target_head in (
                ("proj1", "proj1s"),
                ("proj2", "proj2s"),
                ("out_layer", "out_layers"),
            ):
                prefix = source_head + "."
                if name.startswith(prefix):
                    mapped = "{}.{}.{}".format(target_head, modality_index, name[len(prefix):])
                    break
            if mapped not in state:
                raise CMCModelError("source {} tensor {} has no CMC target".format(modality, name))
            if tuple(state[mapped].shape) != tuple(tensor.shape):
                raise CMCModelError(
                    "source {} tensor {} shape {} differs from target {} shape {}".format(
                        modality, name, tuple(tensor.shape), mapped, tuple(state[mapped].shape)
                    )
                )
            state[mapped] = tensor.detach().clone()
            copied.append(mapped)
    model.load_state_dict(state, strict=True)
    return {"copied_tensor_count": len(copied), "copied_state_keys": copied}
