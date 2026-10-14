"""CPU fixtures for strict BI_ENC transfer and deterministic epoch resume."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from jevaffect.bcmlf_training import (  # noqa: E402
    _RegressionSplit,
    _load_bienc_args,
    train_bienc_stage,
)
from jevaffect.training import RunIdentityError  # noqa: E402
from vendor.bcmlf.models.singleTask.MMGPT import AV_Enc  # noqa: E402


def _split(name: str, count: int, offset: int) -> dict:
    rng = np.random.default_rng(900 + offset)
    raw_y = np.linspace(-0.8, 0.8, count, dtype=np.float32)
    return {
        "split": name,
        "ids": ["{}/{}".format(name, offset + index) for index in range(count)],
        "labels": (np.arange(count, dtype=np.int64) + offset) % 5,
        "label_valid": np.ones(count, dtype=np.bool_),
        "raw_labels": raw_y,
        "audio": rng.normal(size=(count, 4, 3)).astype(np.float32),
        "vision": rng.normal(size=(count, 2, 2)).astype(np.float32),
        "meta": {
            "source_sha256": "b" * 64,
            "feature_dims": {"text": 4, "audio": 3, "vision": 2},
            "sequence_lengths": {"text": 3, "audio": 4, "vision": 2},
        },
    }


def _bienc_model_config(path: Path, repo_root: Path) -> None:
    config = {
        "datasetCommonParams": {
            "dataset_root_dir": str(repo_root),
            "sims": {
                "unaligned": {
                    "featurePath": "unused-for-cached-features.pkl",
                    "seq_lens": [3, 4, 2],
                    "feature_dims": [4, 3, 2],
                    "num_classes": 5,
                    "language": "cn",
                    "KeyEval": "Loss",
                }
            },
        },
        "bienc": {
            "commonParams": {
                "need_data_aligned": False,
                "need_model_aligned": False,
                "early_stop": 3,
                "use_bert": False,
                "use_bert_finetune": False,
                "attn_mask": True,
                "update_epochs": 2,
            },
            "datasetParams": {
                "sims": {
                    "hfPath": "unused-for-BI-ENC",
                    "av_enc": {
                        "d_enc": 4,
                        "n_embd": 4,
                        "n_head": 2,
                        "nlevels": 1,
                        "d_enc_out": 4,
                        "maxlen": 3,
                        "p_mask": 0.0,
                        "enc_attn_dropout": 0.0,
                        "enc_res_dropout": 0.0,
                        "enc_dropout": 0.0,
                        "use_softperm": False,
                        "p_perm": 0.0,
                        "tf_fusion": False,
                        "use_ln": False,
                        "use_bn": False,
                        "use_m3": False,
                    },
                    "learning_rate": 0.001,
                    "grad_clip": 1.0,
                    "patience": 2,
                    "weight_decay": 0.001,
                }
            },
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def _config(temp_root: Path, model_config_path: Path) -> dict:
    return {
        "repo_root": str(REPO_ROOT),
        "cache_dir": str(temp_root / "synthetic-cache"),
        "run_root": str(temp_root / "runs"),
        "bienc_config": str(model_config_path),
        "seeds": [13, 42, 2026],
        "training": {
            "max_epochs": 2,
            "batch_size": 2,
            "accumulation_steps": 2,
            "early_stop": 4,
            "scheduler_patience": 2,
            "learning_rate": 0.001,
            "grad_clip_value": 1.0,
            "weight_decay": 0.01,
            "device": "cpu",
        },
    }


class BiencStage1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache_dir = self.root / "synthetic-cache"
        self.cache_dir.mkdir()
        self.model_config_path = self.root / "bienc-tiny.json"
        _bienc_model_config(self.model_config_path, REPO_ROOT)
        self.config = _config(self.root, self.model_config_path)
        self.data = {"train": _split("train", 5, 0), "valid": _split("valid", 4, 20)}
        self.loader_calls = []

    def tearDown(self):
        self.temp.cleanup()

    def _load_fixture(self, _cache_dir, split, **kwargs):
        self.loader_calls.append((split, kwargs))
        return self.data[split]

    def test_cpu_pretrain_resumes_exactly_and_transfers_strictly_to_av_enc(self):
        partial_dir = self.root / "partial"
        full_dir = self.root / "full"
        with patch(
            "jevaffect.bcmlf_training.load_cached_split",
            side_effect=self._load_fixture,
        ):
            first = train_bienc_stage(
                self.config, 13, partial_dir, max_epochs_this_call=1
            )
            self.assertFalse(first["completed"])
            last = torch.load(partial_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(last["epoch_completed"], 1)
            self.assertIn("optimizer_state", last)
            self.assertIn("scheduler_state", last)
            self.assertIn("rng_state", last)

            resumed = train_bienc_stage(self.config, 13, partial_dir, resume=True)
            uninterrupted = train_bienc_stage(self.config, 13, full_dir)

        self.assertTrue(resumed["completed"])
        self.assertTrue(uninterrupted["completed"])
        self.assertEqual(resumed["best_epoch"], uninterrupted["best_epoch"])
        self.assertAlmostEqual(resumed["best_valid_mae"], uninterrupted["best_valid_mae"])
        partial_last = torch.load(partial_dir / "last.pt", map_location="cpu", weights_only=False)
        full_last = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
        for name, tensor in full_last["model_state"].items():
            self.assertTrue(torch.equal(partial_last["model_state"][name], tensor), name)

        best_state = torch.load(
            partial_dir / "bienc_best_encoder.pth", map_location="cpu", weights_only=True
        )
        self.assertTrue(best_state)
        self.assertTrue(all(key.startswith("Model.") for key in best_state))
        args = _load_bienc_args(
            self.model_config_path,
            REPO_ROOT,
            _RegressionSplit(self.data["train"], "train"),
            torch.device("cpu"),
            self.config["training"],
        )
        transferred = AV_Enc.from_pretrained(
            str(partial_dir / "bienc_best_encoder.pth"), args
        )
        transferred_state = transferred.state_dict()
        for key, tensor in transferred_state.items():
            self.assertTrue(torch.equal(best_state["Model." + key], tensor), key)

        self.assertEqual(
            [split for split, _ in self.loader_calls],
            ["train", "valid", "train", "valid", "train", "valid"],
        )
        self.assertTrue(
            all(not kwargs.get("require_raw_text", False) for _, kwargs in self.loader_calls)
        )
        history = [
            json.loads(line)
            for line in (partial_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([row["epoch"] for row in history], [1, 2])
        self.assertEqual(history[0]["optimizer_updates"], 2)  # tail group was stepped
        valid_rows = [
            json.loads(line)
            for line in (partial_dir / "validation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(valid_rows), 4)
        self.assertTrue(all(row["split"] == "valid" for row in valid_rows))

    def test_changed_config_cannot_resume_or_mutate_last_checkpoint(self):
        run_dir = self.root / "mismatch"
        with patch(
            "jevaffect.bcmlf_training.load_cached_split",
            side_effect=self._load_fixture,
        ):
            train_bienc_stage(self.config, 13, run_dir, max_epochs_this_call=1)
            before = (run_dir / "last.pt").read_bytes()
            changed = json.loads(json.dumps(self.config))
            changed["training"]["learning_rate"] = 0.0005
            with self.assertRaises(RunIdentityError):
                train_bienc_stage(changed, 13, run_dir, resume=True)
            self.assertEqual((run_dir / "last.pt").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
