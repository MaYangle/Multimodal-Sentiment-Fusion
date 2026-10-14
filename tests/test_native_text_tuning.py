import csv
import hashlib
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import jevaffect.native_text_tuning as tuning


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class ToyLayer(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class ToyLayerStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([ToyLayer() for _ in range(12)])


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8)
        self.embeddings = nn.Embedding(64, 8)
        self.encoder = ToyLayerStack()
        self.pooler = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        hidden = self.dropout(hidden)
        return types.SimpleNamespace(last_hidden_state=hidden)


class FakeTokenizer:
    seen_texts = []

    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        type(self).seen_texts.extend(texts)
        self_args = (padding, truncation, return_tensors)
        if self_args != ("max_length", True, "pt"):
            raise AssertionError("Unexpected tokenizer settings")
        ids, masks = [], []
        for text in texts:
            values = [ord(char) % 63 + 1 for char in text[:max_length]]
            mask = [1] * len(values)
            values += [0] * (max_length - len(values))
            mask += [0] * (max_length - len(mask))
            ids.append(values)
            masks.append(mask)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


def fake_model_factory(model_dir, mode):
    return tuning.MacBERTScalarRegressor(ToyEncoder(), mode=mode, dropout=0.1)


def fake_tokenizer_factory(model_dir):
    return FakeTokenizer()


class NativeTextTuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "macbert"
        self.model_dir.mkdir()
        self.csv_path = self.root / "author.csv"
        self._write_fixture_csv()
        self._write_manifest()
        self.pins = patch.multiple(
            tuning,
            CSV_SHA256=file_sha(self.csv_path),
            MODEL_SHA256=file_sha(self.model_dir / "pytorch_model.bin"),
            EXPECTED_ROWS={"train": 4, "valid": 2, "test": 1},
            PINNED_MANIFEST_FILES=self.manifest_files,
        )
        self.pins.start()
        FakeTokenizer.seen_texts = []

    def tearDown(self):
        self.pins.stop()
        self.temp.cleanup()

    def _write_fixture_csv(self):
        rows = [
            ("a", "1", "train", "-0.8", "train alpha"),
            ("a", "2", "train", "-0.2", "train beta"),
            ("b", "1", "train", "0.3", "train gamma"),
            ("b", "2", "train", "0.9", "train delta"),
            ("c", "1", "valid", "-0.4", "valid alpha"),
            ("c", "2", "valid", "0.5", "valid beta"),
            ("heldout", "1", "test", "DO NOT PARSE", "DO NOT TOKENIZE"),
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video_id", "clip_id", "mode", "label", "text"])
            writer.writerows(rows)

    def _write_manifest(self):
        files = {}
        for name, contents in {
            "added_tokens.json": b"{}",
            "config.json": b"{}",
            "pytorch_model.bin": b"fixture-weights-only",
            "README.md": b"fixture manifest entry",
            "special_tokens_map.json": b"{}",
            "tokenizer_config.json": b"{}",
            "vocab.txt": b"[PAD]\n[UNK]\n",
        }.items():
            path = self.model_dir / name
            path.write_bytes(contents)
            files[name] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
        manifest = {
            "model": tuning.MODEL_ID,
            "revision": tuning.MODEL_REVISION,
            "files": files,
        }
        self.manifest_files = files
        (self.model_dir / "jevaffect_download_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")

    def config(self, run_root):
        return {
            "dataset": "simsv2",
            "author_csv": str(self.csv_path),
            "pretrained_dir": str(self.model_dir),
            "run_root": str(run_root),
            "model": "native_macbert_text",
            "modes": ["top4", "frozen"],
            "seeds": [13, 42, 2026],
            "device": "cpu",
            "batch_size": 32,
            "max_tokens": 64,
            "max_epochs": 12,
            "patience": 3,
            "encoder_lr": 1e-5,
            "head_lr": 2e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "dropout": 0.1,
            "warmup_fraction": 0.1,
            "torch_num_threads": 1,
            "model_id": tuning.MODEL_ID,
            "model_revision": tuning.MODEL_REVISION,
            "model_sha256": tuning.MODEL_SHA256,
            "author_csv_sha256": tuning.CSV_SHA256,
            "expected_rows": dict(tuning.EXPECTED_ROWS),
        }

    def test_parser_does_not_read_test_label_or_tokenize_test_text(self):
        records, cohort = tuning.parse_author_csv(self.csv_path)
        self.assertEqual({key: len(value) for key, value in records.items()},
                         {"train": 4, "valid": 2})
        self.assertEqual(cohort["counts"], {"train": 4, "valid": 2, "test": 1})
        self.assertNotIn("DO NOT PARSE", str(records))
        self.assertNotIn("DO NOT TOKENIZE", str(records))
        verified = tuning.verify_pinned_pretrained(self.model_dir)
        self.assertEqual(len(verified["files"]), 7)

    def test_all_manifest_files_are_verified(self):
        (self.model_dir / "README.md").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "README.md"):
            tuning.verify_pinned_pretrained(self.model_dir)

    def test_duplicate_id_guard_covers_test_rows(self):
        rows = self.csv_path.read_text(encoding="utf-8").replace(
            "heldout,1,test,DO NOT PARSE,DO NOT TOKENIZE",
            "a,1,test,DO NOT PARSE,DO NOT TOKENIZE",
        )
        self.csv_path.write_text(rows, encoding="utf-8")
        with patch.object(tuning, "CSV_SHA256", file_sha(self.csv_path)):
            with self.assertRaisesRegex(ValueError, "Duplicate author sample ID"):
                tuning.parse_author_csv(self.csv_path)

    def test_top4_and_frozen_trainable_counts_and_dropout_mode(self):
        top4 = tuning.MacBERTScalarRegressor(ToyEncoder(), "top4")
        frozen = tuning.MacBERTScalarRegressor(ToyEncoder(), "frozen")
        self.assertEqual(sum(p.numel() for p in top4.encoder.parameters() if p.requires_grad),
                         sum(p.numel() for layer in top4.encoder.encoder.layer[-4:]
                             for p in layer.parameters()))
        self.assertEqual(sum(p.numel() for p in frozen.encoder.parameters() if p.requires_grad), 0)
        self.assertFalse(top4.encoder.embeddings.weight.requires_grad)
        self.assertFalse(top4.encoder.pooler.weight.requires_grad)
        self.assertFalse(any(p.requires_grad for layer in top4.encoder.encoder.layer[:8]
                             for p in layer.parameters()))
        top4.train()
        self.assertTrue(top4.encoder.training)
        self.assertTrue(top4.encoder.dropout.training)

    def test_float32_native_threshold_matches_native_training(self):
        prediction = np.nextafter(np.float32(0.1), np.float32(np.inf))
        target = np.float32(0.1)

        class FixedPrediction(nn.Module):
            def forward(self, input_ids, attention_mask):
                return torch.tensor([prediction], dtype=torch.float32)

        batch = {
            "input_ids": torch.ones((1, 64), dtype=torch.long),
            "attention_mask": torch.ones((1, 64), dtype=torch.long),
            "target": torch.tensor([target], dtype=torch.float32),
            "id": ["valid/boundary"],
        }
        metrics, predictions = tuning._evaluate(FixedPrediction(), [batch], torch.device("cpu"))
        expected = tuning.native_metrics(
            np.asarray([prediction], dtype=np.float32),
            np.asarray([target], dtype=np.float32), "simsv2")
        self.assertEqual(metrics, expected)
        self.assertEqual(metrics["Mult_acc_3"], 0.0)
        self.assertEqual(metrics["Mult_acc_5"], 0.0)
        self.assertEqual(predictions[0]["id"], "valid/boundary")

    def test_exact_epoch_resume_matches_uninterrupted_training(self):
        whole_config = self.config(self.root / "whole")
        whole = tuning.train(whole_config, 13, mode="top4", model_factory=fake_model_factory,
                             tokenizer_factory=fake_tokenizer_factory)

        resumed_config = self.config(self.root / "resumed")
        first = tuning.train(resumed_config, 13, mode="top4", stop_after=1,
                             model_factory=fake_model_factory,
                             tokenizer_factory=fake_tokenizer_factory)
        self.assertEqual(first["last_epoch"], 1)
        self.assertFalse(first["complete"])
        with self.assertRaisesRegex(ValueError, "resume identity mismatch"):
            tuning.train(dict(resumed_config, torch_num_threads=2), 13, mode="top4",
                         resume=True, model_factory=fake_model_factory,
                         tokenizer_factory=fake_tokenizer_factory)
        resumed = tuning.train(resumed_config, 13, mode="top4", resume=True,
                               model_factory=fake_model_factory,
                               tokenizer_factory=fake_tokenizer_factory)
        self.assertEqual(whole["best_epoch"], resumed["best_epoch"])
        self.assertEqual(whole["valid_metrics"], resumed["valid_metrics"])
        self.assertEqual(whole["complete"], resumed["complete"])
        self.assertTrue(whole["suite_complete"])
        self.assertEqual(whole["optimizer_updates"], whole["last_epoch"])
        self.assertGreater(whole["total_parameters"], whole["trainable_parameter_count"])
        self.assertGreaterEqual(whole["train_valid_seconds"], 0.0)
        self.assertIsNone(whole["peak_cuda_allocated_bytes"])
        self.assertFalse(first["suite_complete"])
        self.assertNotIn("DO NOT TOKENIZE", FakeTokenizer.seen_texts)
        saved_predictions = json.loads(
            (self.root / "resumed/top4/seed_13/valid_predictions.json").read_text(encoding="utf-8"))
        self.assertEqual({row["id"] for row in saved_predictions}, {"c/1", "c/2"})
        self.assertTrue(all(set(row) == {"id", "target", "prediction"} for row in saved_predictions))

        whole_state = torch.load(self.root / "whole/top4/seed_13/last.pt",
                                 map_location="cpu", weights_only=False)
        resumed_state = torch.load(self.root / "resumed/top4/seed_13/last.pt",
                                   map_location="cpu", weights_only=False)
        for name, value in whole_state["model"].items():
            self.assertTrue(torch.equal(value, resumed_state["model"][name]), name)
        self.assertEqual(whole_state["scheduler"], resumed_state["scheduler"])
        self.assertEqual(len(whole_state["history"]), len(resumed_state["history"]))
        for left, right in zip(whole_state["history"], resumed_state["history"]):
            self.assertEqual(left["train_L1"], right["train_L1"])
            self.assertEqual(left["valid"], right["valid"])

    def test_completed_early_stop_resume_does_not_advance_epoch_or_model(self):
        config = self.config(self.root / "completed")
        first = tuning.train(config, 13, mode="frozen", stop_after=1,
                             model_factory=fake_model_factory,
                             tokenizer_factory=fake_tokenizer_factory)
        self.assertFalse(first["complete"])
        last_path = self.root / "completed/frozen/seed_13/last.pt"
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        # Mark the epoch checkpoint as an early-stopped run, as if patience was
        # exhausted immediately after this saved epoch.
        state["stale"] = 3
        state["completed"] = True
        torch.save(state, last_path)
        before = torch.load(last_path, map_location="cpu", weights_only=False)

        resumed = tuning.train(config, 13, mode="frozen", resume=True,
                               model_factory=fake_model_factory,
                               tokenizer_factory=fake_tokenizer_factory)
        after = torch.load(last_path, map_location="cpu", weights_only=False)
        self.assertTrue(resumed["complete"])
        self.assertEqual(after["epoch"], before["epoch"])
        self.assertEqual(after["optimizer_updates"], before["optimizer_updates"])
        for name, value in before["model"].items():
            self.assertTrue(torch.equal(value, after["model"][name]), name)

    def test_pinned_loader_requests_local_revision_and_eager_attention(self):
        calls = {}

        class StubEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    hidden_size=768, hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                )
                self.embeddings = nn.Embedding(2, 1)
                self.encoder = types.SimpleNamespace(layer=nn.ModuleList([nn.Linear(1, 1) for _ in range(12)]))
                self.pooler = nn.Linear(1, 1)

            def forward(self, input_ids, attention_mask):
                raise NotImplementedError

        class AutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls["model"] = (args, kwargs)
                return StubEncoder()

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls["tokenizer"] = (args, kwargs)
                return FakeTokenizer()

        fake_transformers = types.SimpleNamespace(AutoModel=AutoModel, AutoTokenizer=AutoTokenizer)
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            model = tuning._load_pinned_model(self.model_dir, "top4")
            tuning._load_pinned_tokenizer(self.model_dir)
        self.assertEqual(model.mode, "top4")
        self.assertEqual(calls["model"][1]["revision"], tuning.MODEL_REVISION)
        self.assertTrue(calls["model"][1]["local_files_only"])
        self.assertEqual(calls["model"][1]["attn_implementation"], "eager")
        self.assertEqual(calls["tokenizer"][1]["revision"], tuning.MODEL_REVISION)
        self.assertTrue(calls["tokenizer"][1]["local_files_only"])


if __name__ == "__main__":
    unittest.main()
