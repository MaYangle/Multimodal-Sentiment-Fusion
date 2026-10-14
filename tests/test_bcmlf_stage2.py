from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from jevaffect.bcmlf_stage2 import (
    BCMLFStage2Error,
    _FiveClassSplit,
    _load_model_config,
    _AttrDict,
    _configure_source_trainability,
    _optimizer_groups,
    _validate_stage1_run,
    _attrify,
    normalize_stage2_config,
    train_stage2_seed,
)


REPO = Path(__file__).resolve().parents[1]
MSALM_CONFIG = REPO / "vendor" / "bcmlf" / "config" / "bcmlf_simsv2_5class_v1.json"


class _Encoding(dict):
    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class _ToyTokenizer:
    def __call__(self, texts, *, return_tensors, padding, truncation):
        self.assertions = (return_tensors, padding, truncation)
        length = 39
        ids = torch.zeros((len(texts), length), dtype=torch.long)
        mask = torch.zeros((len(texts), length), dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = [1 + (ord(char) % 7) for char in text[: length - 1]]
            if not tokens:
                tokens = [1]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, : len(tokens)] = 1
        return _Encoding(input_ids=ids, attention_mask=mask)


class _ToyCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(3, 3, bias=False)
        self.c_fc = nn.Linear(3, 3)
        self.ln_1 = nn.LayerNorm(3)

    def forward(self, value):
        return self.ln_1(value + self.attn(value) + self.c_fc(value))


class _ToyDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ca_layer = _ToyCA()
        self.base = nn.Linear(3, 3)


class _ToyWTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 3))
        self.bn_embedding = nn.Linear(3, 3)

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight)


class _ToyLanguageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_ToyDecoderBlock()])
        self.transformer.wte = _ToyWTE()


class _ToyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lang_encoder = _ToyLanguageEncoder()
        self.av_encoder = nn.Linear(2, 3)
        self.W_task = nn.Linear(3, 5)
        self.W_bn = nn.Linear(3, 5)
        self.W_text = nn.Linear(3, 5)
        self.W_av = nn.Linear(3, 5)
        self.LN = nn.LayerNorm(3)
        self.global_branch_logits = nn.Parameter(torch.zeros(3))
        self.dropout = nn.Dropout(0.2)
        self.tokenizer = _ToyTokenizer()


class _ToyMSALM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Model = _ToyInner()

    def forward(self, input_ids, audio, vision, *, attention_mask):
        inner = self.Model
        av_input = torch.cat((audio.mean(dim=(1, 2)).unsqueeze(1), vision.mean(dim=(1, 2)).unsqueeze(1)), dim=1)
        av = inner.av_encoder(av_input)
        text = inner.lang_encoder.transformer.wte(input_ids).mean(dim=1)
        hidden = inner.lang_encoder.transformer.h[0].ca_layer(av + text)
        hidden = inner.LN(hidden)
        if self.training:
            hidden = inner.dropout(hidden)
        bn_logits = inner.W_bn(hidden).unsqueeze(1).expand(-1, 2, -1)
        text_logits = inner.W_text(text)
        av_logits = inner.W_av(av)
        task_logits = inner.W_task(hidden)
        lm_logits = F.linear(
            hidden[:, None, :].expand(-1, input_ids.shape[1], -1),
            inner.lang_encoder.transformer.wte.weight,
        )
        weights = torch.softmax(inner.global_branch_logits, dim=0).expand(len(input_ids), -1)
        return {
            "task_logits": task_logits,
            "bn_logits": bn_logits,
            "text_logits": text_logits,
            "av_logits": av_logits,
            "lm_logits": lm_logits,
            "Feature_bn": hidden,
            "Contribution_weights": weights,
        }


def _toy_factory(args):
    return _ToyMSALM(args)


def _fixture_splits():
    rng = np.random.default_rng(20261012)

    def make_split(name, count):
        labels = np.arange(count, dtype=np.int64) % 5
        # Make class-dependent A/V means while retaining a small feature
        # variation, enough to exercise the complete train/valid path.
        centers = labels.astype(np.float32) - 2.0
        audio = np.broadcast_to(centers[:, None, None], (count, 400, 33)).copy()
        vision = np.broadcast_to(centers[:, None, None] * 0.5, (count, 55, 709)).copy()
        audio += rng.normal(0.0, 0.02, size=audio.shape).astype(np.float32)
        vision += rng.normal(0.0, 0.02, size=vision.shape).astype(np.float32)
        return {
            "split": name,
            "ids": ["{}-{:03d}".format(name, i) for i in range(count)],
            "raw_text": ["sample class {} row {}".format(labels[i], i) for i in range(count)],
            "labels": labels,
            "label_valid": np.ones(count, dtype=np.bool_),
            "raw_labels": centers.astype(np.float32),
            "audio": audio,
            "vision": vision,
            "meta": {
                "source_sha256": "a" * 64,
                "text_features_ready": False,
                "feature_dims": {"text": None, "audio": 33, "vision": 709},
                "sequence_lengths": {"text": None, "audio": 400, "vision": 55},
            },
        }

    return {"train": make_split("train", 5), "valid": make_split("valid", 5)}


class BCMLFStage2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.splits = _fixture_splits()
        self.stage1 = self.root / "stage1" / "seed-{seed}"
        for seed in (13, 42, 2026):
            run = self.root / "stage1" / "seed-{}".format(seed)
            run.mkdir(parents=True, exist_ok=True)
            identity = ("b" if seed == 13 else "c" if seed == 42 else "d") * 64
            (run / "run_manifest.json").write_text(
                json.dumps({
                    "stage": "BI_ENC_raw_label_L1_pretrain",
                    "cache_source_sha256": "a" * 64,
                    "seed": seed,
                    "identity_sha256": identity,
                }),
                encoding="utf-8",
            )
            torch.save({"Model.fixture": torch.ones(1)}, run / "bienc_best_encoder.pth")
            torch.save({
                "identity_sha256": identity,
                "seed": seed,
                "completed": True,
                "epoch_completed": 4,
                "next_epoch": 4,
                "best_epoch": 3,
                "best_mae": 0.25,
                "rng_state": {"numpy": np.random.get_state()},
            }, run / "last.pt")
        self.config = normalize_stage2_config(
            {
                "repo_root": str(REPO),
                "cache_dir": str(self.root / "cache"),
                "run_root": str(self.root / "runs"),
                "msalm_config": str(MSALM_CONFIG),
                "stage1_run_template": str(self.stage1),
                "seeds": [13, 42, 2026],
                "training": {
                    "max_epochs": 4,
                    "batch_size": 2,
                    "update_epochs": 2,
                    "early_stop": 10,
                    "warmup_epochs": 1,
                    "grad_clip_value": 1.0,
                    "num_workers": 0,
                    "device": "cpu",
                },
            }
        )

    def tearDown(self):
        self.temp.cleanup()

    def _loader(self, calls):
        def load(cache_dir, split, **kwargs):
            calls.append((split, kwargs))
            return self.splits[split]
        return load

    def test_native_av_length_is_independent_of_text_token_length(self):
        args, _ = _load_model_config(
            MSALM_CONFIG, REPO, _FiveClassSplit(self.splits["train"], "train"),
            self.root / "stage1" / "seed-13" / "bienc_best_encoder.pth",
            self.config["training"], torch.device("cpu"),
        )
        self.assertEqual(args.max_token_len, 39)
        self.assertEqual(args.av_enc.maxlen, 55)

    def test_freeze_policy_and_native_four_optimizer_groups(self):
        args = _attrify({
            "mmgpt": {"type": "gpt2", "use_lora": False},
            "use_ulgm": False,
            "use_lnorm": True,
            "learning_rate_mmgpt": 1e-4,
            "learning_rate_av": 2e-4,
            "weight_decay_mmgpt": 0.1,
            "weight_decay_av": 0.2,
            "beta_1": 0.9,
            "beta_2": 0.95,
        })
        model = _ToyMSALM(args)
        _configure_source_trainability(model, args)
        inner = model.Model
        self.assertTrue(inner.lang_encoder.transformer.h[0].ca_layer.attn.weight.requires_grad)
        self.assertTrue(inner.lang_encoder.transformer.h[0].ca_layer.c_fc.weight.requires_grad)
        self.assertFalse(inner.lang_encoder.transformer.h[0].base.weight.requires_grad)
        self.assertFalse(inner.lang_encoder.transformer.wte.weight.requires_grad)
        self.assertTrue(inner.lang_encoder.transformer.wte.bn_embedding.weight.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in inner.av_encoder.parameters()))
        self.assertTrue(inner.W_task.weight.requires_grad)
        self.assertTrue(inner.LN.weight.requires_grad)
        self.assertTrue(inner.global_branch_logits.requires_grad)

        groups, detail = _optimizer_groups(model, args)
        self.assertEqual(len(groups), 4)
        self.assertEqual([row["name"] for row in detail["groups"]], [
            "non_av_decay", "non_av_no_decay", "av_decay", "av_no_decay"
        ])
        self.assertEqual(groups[0]["lr"], 1e-4)
        self.assertEqual(groups[2]["lr"], 2e-4)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[3]["weight_decay"], 0.0)
        self.assertEqual(sum(row["parameters"] for row in detail["groups"]), sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ))

    def test_bad_stage1_transfer_fails_closed(self):
        run = self.root / "invalid-stage1"
        run.mkdir()
        (run / "run_manifest.json").write_text(
            json.dumps({
                "stage": "BI_ENC_raw_label_L1_pretrain",
                "cache_source_sha256": "a" * 64,
            }), encoding="utf-8"
        )
        (run / "run_manifest.json").write_text(
            json.dumps({
                "stage": "BI_ENC_raw_label_L1_pretrain",
                "cache_source_sha256": "a" * 64,
                "seed": 13,
                "identity_sha256": "e" * 64,
            }), encoding="utf-8"
        )
        torch.save({
            "identity_sha256": "e" * 64,
            "seed": 13,
            "completed": True,
            "epoch_completed": 1,
            "next_epoch": 1,
            "best_epoch": 1,
            "best_mae": 0.5,
            "rng_state": {"numpy": np.random.get_state()},
        }, run / "last.pt")
        torch.save({"av_encoder.weight": torch.ones(1)}, run / "bienc_best_encoder.pth")
        with self.assertRaisesRegex(BCMLFStage2Error, "flat Model"):
            _validate_stage1_run(run, "a" * 64, 13)

    def test_stage1_must_be_completed_and_seed_matched(self):
        run = self.root / "stage1" / "seed-13"
        with self.assertRaisesRegex(BCMLFStage2Error, "seed"):
            _validate_stage1_run(run, "a" * 64, 42)

        checkpoint_path = run / "last.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["completed"] = False
        torch.save(checkpoint, checkpoint_path)
        with self.assertRaisesRegex(BCMLFStage2Error, "not marked completed"):
            _validate_stage1_run(run, "a" * 64, 13)

    def test_stage1_checkpoint_identity_must_match_manifest(self):
        run = self.root / "stage1" / "seed-13"
        checkpoint_path = run / "last.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["identity_sha256"] = "f" * 64
        torch.save(checkpoint, checkpoint_path)
        with self.assertRaisesRegex(BCMLFStage2Error, "identity"):
            _validate_stage1_run(run, "a" * 64, 13)

    def test_only_train_valid_and_exact_epoch_resume(self):
        calls_a = []
        uninterrupted = self.root / "runs" / "whole"
        result_whole = train_stage2_seed(
            self.config, 13, uninterrupted, data_loader=self._loader(calls_a), model_factory=_toy_factory
        )
        calls_b = []
        resumed = self.root / "runs" / "split"
        first = train_stage2_seed(
            self.config, 13, resumed, max_epochs_this_call=2,
            data_loader=self._loader(calls_b), model_factory=_toy_factory,
        )
        self.assertEqual(first["last_epoch"], 2)
        result_resumed = train_stage2_seed(
            self.config, 13, resumed, resume=True, max_epochs_this_call=2,
            data_loader=self._loader(calls_b), model_factory=_toy_factory,
        )
        self.assertEqual(result_whole["best_valid_accuracy"], result_resumed["best_valid_accuracy"])
        self.assertEqual(
            (uninterrupted / "metrics.jsonl").read_text(encoding="utf-8"),
            (resumed / "metrics.jsonl").read_text(encoding="utf-8"),
        )
        whole_last = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=False)
        resumed_last = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
        for name, whole_value in whole_last["model_state"].items():
            self.assertTrue(torch.equal(whole_value, resumed_last["model_state"][name]), name)
        metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["optimizer_updates"] == 2 for row in metrics))
        self.assertTrue(all(row["valid_accuracy"] >= 0.0 and row["valid_accuracy"] <= 1.0 for row in metrics))
        predictions = [json.loads(line) for line in (resumed / "validation_predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(predictions), len(self.splits["valid"]["ids"]))
        self.assertTrue(all(len(row["probabilities_uncalibrated"]) == 5 for row in predictions))
        self.assertEqual({split for split, _ in calls_a + calls_b}, {"train", "valid"})
        self.assertTrue(all(kwargs["require_text_features"] is False for _, kwargs in calls_a + calls_b))
        self.assertTrue(all(kwargs["require_raw_text"] is True for _, kwargs in calls_a + calls_b))

    def test_config_rejects_nonzero_workers(self):
        raw = dict(self.config)
        raw["training"] = dict(raw["training"])
        raw["training"]["num_workers"] = 1
        with self.assertRaisesRegex(BCMLFStage2Error, "num_workers must be 0"):
            normalize_stage2_config(raw)


if __name__ == "__main__":
    unittest.main()
