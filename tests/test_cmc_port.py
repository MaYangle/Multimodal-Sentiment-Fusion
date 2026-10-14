from __future__ import annotations

import hashlib
import json
import pickle
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jevaffect.cache import convert_source_pickle, load_cached_split
from vendor.cmc.cache_adapter import load_cmc_train_valid
from vendor.cmc.builders import build_fusion_model, build_fusion_optimizer, build_pseudo_pretraining_bundle
from vendor.cmc.models import CMCClassifier, SupConLoss, transfer_unimodal_to_cmc
from vendor.cmc.training import (
    PseudoLabelBank,
    evaluate_fusion,
    evaluate_pseudo_pretrain_epoch,
    generate_unified_pseudo_labels,
    train_fusion_epoch,
    train_pseudo_pretrain_epoch,
)


CLASS_NAMES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")
RAW_SCORES = (-1.0, -0.6, 0.0, 0.2, 1.0)


class TinyTokenEncoder(nn.Module):
    """CPU fixture encoder at CMC's text-token boundary, not a fusion stand-in."""

    def __init__(self, output_dim=8):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(32, output_dim)

    def forward(self, token_triplets, *, use_finetune):
        if use_finetune:
            hidden = self.embedding(token_triplets[:, 0].long())
        else:
            with torch.no_grad():
                hidden = self.embedding(token_triplets[:, 0].long())
        return hidden * token_triplets[:, 1].to(hidden.dtype).unsqueeze(-1)


def _normalized_hash(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_fixture_cache(root: Path) -> Path:
    source = {}
    manifest = []
    for split, count in (("train", 5), ("valid", 5), ("test", 1)):
        labels = np.arange(count, dtype=np.int64) % 5
        scores = np.asarray([RAW_SCORES[index] for index in labels], dtype=np.float32)
        ids = ["v_{}_{:02d}$_$c{:02d}".format(split, index, index) for index in range(count)]
        texts = ["fixture {} class {}".format(split, int(label)) for label in labels]
        token_bundle = np.zeros((count, 3, 6), dtype=np.float32)
        token_bundle[:, 0, :] = np.tile(np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32), (count, 1))
        token_bundle[:, 1, :] = 1.0
        token_bundle[:, 2, :] = 0.0
        centers = labels.astype(np.float32)[:, None, None]
        audio = np.broadcast_to(centers, (count, 4, 2)).copy()
        vision = np.broadcast_to((centers * 0.5), (count, 4, 3)).copy()
        split_record = {
            "id": ids,
            "text_bert": token_bundle,
            "raw_text": texts,
            "audio": audio,
            "vision": vision,
            "regression_labels": scores,
            "audio_lengths": np.full(count, 4, dtype=np.int16),
            "vision_lengths": np.full(count, 4, dtype=np.int16),
        }
        if split == "test":
            split_record["regression_labels"] = np.full(count, np.nan, dtype=np.float32)
        source[split] = split_record
        for index, (sample_id, score, text) in enumerate(zip(ids, scores, texts)):
            mapped = CLASS_NAMES[int(labels[index])] if split != "test" else None
            status = "mapped" if mapped is not None else "unresolved"
            manifest.append({
                "ID": sample_id.replace("$_$", "/"),
                "mode": split,
                "raw_label": float(score) if split != "test" else None,
                "Label": mapped,
                "label_status": status,
                "transcript_sha256": _normalized_hash(text),
            })
    source_path = root / "fixture.pkl"
    with source_path.open("wb") as handle:
        pickle.dump(source, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cache_path = root / "cache"
    convert_source_pickle(
        source_path,
        cache_path,
        expected_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        trust_source_pickle=True,
        manifest_path=manifest_path,
        source_id_format="video_dollar_clip",
        manifest_id_format="video_clip_path",
        source_reference="CMC CPU fixture only",
    )
    return cache_path


class CMCPortTests(unittest.TestCase):
    def test_five_class_unified_pseudo_distribution(self):
        logits = torch.tensor([[0.0, 0.1, 0.2, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
        base = torch.nn.functional.one_hot(torch.tensor([2, 0]), 5).float()
        pseudo = generate_unified_pseudo_labels(logits, base)
        self.assertEqual(tuple(pseudo.shape), (2, 5))
        self.assertTrue(torch.isfinite(pseudo).all())
        self.assertTrue(torch.allclose(pseudo.sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.all(pseudo >= 0))

    def test_real_cache_adapter_and_two_stage_cpu_forward_backward(self):
        torch.manual_seed(123)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache = _build_fixture_cache(Path(directory.name))
        calls = []

        def reader(cache_dir, split, **kwargs):
            calls.append(split)
            return load_cached_split(cache_dir, split, **kwargs)

        train, valid, identity = load_cmc_train_valid(cache, data_loader=reader)
        self.addCleanup(train.close)
        self.addCleanup(valid.close)
        self.assertEqual(calls, ["train", "valid"])
        self.assertEqual(train.text_mode, "bert_token_triplets")
        self.assertEqual(train.text.shape[1:], (3, 6))
        self.assertEqual(train[0]["label"].dtype, torch.long)
        self.assertIn("raw_label", train[0])
        self.assertEqual(identity["source_sha256"], train.meta["source_sha256"])
        self.assertEqual(set(train[0]), {"text", "audio", "vision", "label", "raw_label", "id"})

        model_cfg = {
            "embed_dim": 8,
            "nhead": 2,
            "transformer_layers": 1,
            "out_dropout": 0.0,
            "top_k": 3,
            "temperature": 0.07,
        }
        pseudo_cfg = {
            "lr": 1e-3,
            "lr_bert": 5e-5,
            "weight_decay_bert": 1e-3,
        }
        encoder_factory = lambda _split: TinyTokenEncoder(output_dim=8)
        bundle = build_pseudo_pretraining_bundle(
            train,
            model_config=model_cfg,
            optimizer_config=pseudo_cfg,
            bert_model_dir=None,
            finetune_text=False,
            device="cpu",
            text_encoder_factory=encoder_factory,
        )
        train_bank = PseudoLabelBank.from_multimodal_labels(train.ids, train.labels)
        valid_bank = PseudoLabelBank.from_multimodal_labels(valid.ids, valid.labels)
        loader_train = DataLoader(train, batch_size=5, shuffle=False)
        loader_valid = DataLoader(valid, batch_size=5, shuffle=False)
        initial = {"text": 0.8, "audio": 0.99, "vision": 0.99}
        gamma = {"text": 0.5, "audio": 0.5, "vision": 0.5}
        ema_initial = {"text": 0.8, "audio": 0.9, "vision": 0.6}
        ema_gamma = {"text": 2.5, "audio": 5.0, "vision": 2.0}
        pretrain = train_pseudo_pretrain_epoch(
            bundle["models"], bundle["ema_models"], bundle["optimizers"],
            loader_train, train_bank,
            epoch=1, device=torch.device("cpu"), contrastive_criterion=SupConLoss(0.07),
            initial_pseudo_momenta=initial, pseudo_momentum_gamma=gamma,
            initial_ema_momenta=ema_initial, ema_momentum_gamma=ema_gamma,
        )
        self.assertEqual(pretrain["samples"], 5)
        self.assertTrue(np.isfinite(pretrain["total"]))
        self.assertEqual(set(train_bank.values[train.ids[0]]), {"text", "audio", "vision"})
        self.assertAlmostEqual(float(train_bank.values[train.ids[0]]["text"].sum()), 1.0, places=5)

        valid_result = evaluate_pseudo_pretrain_epoch(
            bundle["models"], bundle["ema_models"], loader_valid, valid_bank,
            epoch=1, device=torch.device("cpu"), contrastive_criterion=SupConLoss(0.07),
            initial_pseudo_momenta=initial, pseudo_momentum_gamma=gamma,
        )
        self.assertEqual(valid_result["samples"], 5)
        self.assertTrue(np.isfinite(valid_result["total"]))

        fusion, transfer = build_fusion_model(
            train,
            bundle["ema_models"],
            model_config=model_cfg,
            bert_model_dir=None,
            finetune_text=True,
            device="cpu",
            text_encoder_factory=encoder_factory,
        )
        self.assertGreater(transfer["copied_tensor_count"], 0)
        optimizer = build_fusion_optimizer(fusion, lr=1e-3, lr_bert=5e-5, weight_decay_bert=1e-3)
        batch = next(iter(loader_train))
        output = fusion({key: batch[key] for key in ("text", "audio", "vision")})
        self.assertEqual(tuple(output["pred"].shape), (5, 5))
        self.assertEqual(tuple(output["pred_t"].shape), (5, 5))
        self.assertEqual(tuple(output["routing_weights"].shape), (5, 3))
        self.assertTrue(torch.allclose(output["routing_weights"].sum(-1), torch.ones(5), atol=1e-6))
        fusion_result = train_fusion_epoch(
            fusion, loader_train, optimizer, train_bank,
            device=torch.device("cpu"), contrastive_criterion=SupConLoss(0.07),
        )
        self.assertEqual(fusion_result["samples"], 5)
        self.assertTrue(np.isfinite(fusion_result["total"]))
        selected = evaluate_fusion(
            fusion, loader_valid, valid_bank,
            device=torch.device("cpu"), contrastive_criterion=SupConLoss(0.07),
        )
        self.assertEqual(selected["samples"], 5)
        self.assertEqual(len(selected["probabilities_uncalibrated"]), 5)

    def test_transfer_rejects_incompatible_head_shapes(self):
        encoder_factory = lambda _split: TinyTokenEncoder(output_dim=8)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache = _build_fixture_cache(Path(directory.name))
        train, valid, _identity = load_cmc_train_valid(cache)
        self.addCleanup(train.close)
        self.addCleanup(valid.close)
        model_cfg = {"embed_dim": 8, "nhead": 2, "transformer_layers": 1, "top_k": 3}
        bundle = build_pseudo_pretraining_bundle(
            train, model_config=model_cfg, optimizer_config={}, bert_model_dir=None,
            text_encoder_factory=encoder_factory,
        )
        fusion = CMCClassifier(
            text_dim=8, audio_dim=2, vision_dim=3,
            text_encoder=TinyTokenEncoder(output_dim=8), num_classes=5,
            embed_dim=8, nhead=2, transformer_layers=1, top_k=3,
        )
        bundle["ema_models"]["text"].out_layer = nn.Linear(8, 3)
        with self.assertRaisesRegex(ValueError, "shape"):
            transfer_unimodal_to_cmc(fusion, bundle["ema_models"])


if __name__ == "__main__":
    unittest.main()
