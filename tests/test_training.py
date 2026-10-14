"""CPU smoke tests for cache handling, exact epoch-boundary resume, and run safety.

These tests require the project's declared optional torch dependency. They are
provided for the main agent to run in its prepared environment; this work turn
does not install torch or execute training.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevaffect.training import (  # noqa: E402
    RunIdentityError,
    TrainingError,
    train_run,
)


def _write_split(
    cache_dir: Path,
    split: str,
    count: int,
    offset: int,
    *,
    text_only: bool = False,
    relative_path: str | None = None,
) -> None:
    directory = cache_dir / (relative_path or split)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(100 + offset)
    np.save(directory / "text.npy", rng.normal(size=(count, 3, 4)).astype(np.float32))
    if not text_only:
        np.save(directory / "audio.npy", rng.normal(size=(count, 2, 3)).astype(np.float32))
        np.save(directory / "vision.npy", rng.normal(size=(count, 2, 2)).astype(np.float32))
    np.save(directory / "labels.npy", (np.arange(count) + offset).astype(np.int64) % 5)
    np.save(directory / "label_valid.npy", np.ones(count, dtype=np.bool_))
    modalities = (("text", 3),) if text_only else (("text", 3), ("audio", 2), ("vision", 2))
    for modality, steps in modalities:
        np.save(directory / (modality + "_lengths.npy"), np.full(count, steps, dtype=np.int64))
    (directory / "ids.json").write_text(
        json.dumps(["{}_{}.mp4".format(split, offset + index) for index in range(count)]),
        encoding="utf-8",
    )


def _make_cache(
    cache_dir: Path,
    *,
    text_features_ready: bool = True,
    text_only: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_dims = {"text": 4} if text_only else {"text": 4, "audio": 3, "vision": 2}
    sequence_lengths = {"text": 3} if text_only else {"text": 3, "audio": 2, "vision": 2}
    split_paths = (
        {"train": "splits/train", "valid": "splits/valid", "test": "splits/test"}
        if text_only
        else {}
    )
    metadata = {
        "feature_dims": feature_dims,
        "sequence_lengths": sequence_lengths,
        "source_sha256": "a" * 64,
        "text_features_ready": text_features_ready,
        "split_paths": split_paths,
    }
    (cache_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    _write_split(
        cache_dir, "train", 12, 0, text_only=text_only,
        relative_path=split_paths.get("train"),
    )
    _write_split(
        cache_dir, "valid", 8, 30, text_only=text_only,
        relative_path=split_paths.get("valid"),
    )
    # This split is deliberately present but incomplete. The trainer must not
    # inspect or load it during ordinary fitting and validation.
    test_dir = cache_dir / split_paths.get("test", "test")
    test_dir.mkdir(parents=True)
    (test_dir / "sentinel.txt").write_text("must remain untouched", encoding="utf-8")


def _config(cache_dir: Path, run_root: Path) -> dict:
    return {
        "dataset": "synthetic",
        "language": "en",
        "cache_dir": str(cache_dir),
        "run_root": str(run_root),
        "seeds": [13, 42, 2026],
        "model": {
            "name": "text_mlp",
            "params": {"pooling": "masked_mean", "hidden_dim": 8, "dropout": 0.2},
        },
        "training": {
            "max_epochs": 3,
            "batch_size": 4,
            "patience": 5,
            "min_delta": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 1.0,
            "scheduler_patience": 1,
            "scheduler_factor": 0.5,
            "num_workers": 0,
            "device": "cpu",
            "amp": False,
            "amp_dtype": "float16",
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache_dir = self.root / "cache"
        _make_cache(self.cache_dir)
        self.config = _config(self.cache_dir, self.root / "runs")

    def tearDown(self):
        self.temp.cleanup()

    def test_epoch_boundary_resume_matches_uninterrupted_run(self):
        partial_dir = self.root / "partial"
        full_dir = self.root / "full"

        partial = train_run(
            self.config,
            13,
            partial_dir,
            max_epochs_this_call=1,
        )
        self.assertFalse(partial["completed"])
        partial_checkpoint = torch.load(partial_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(partial_checkpoint["next_epoch"], 1)
        self.assertIn("optimizer_state", partial_checkpoint)
        self.assertIn("scheduler_state", partial_checkpoint)
        self.assertIn("scaler_state", partial_checkpoint)
        self.assertIn("rng_state", partial_checkpoint)

        resumed = train_run(self.config, 13, partial_dir, resume=True)
        uninterrupted = train_run(self.config, 13, full_dir)
        self.assertTrue(resumed["completed"])
        self.assertTrue(uninterrupted["completed"])
        self.assertEqual(resumed["best_epoch"], uninterrupted["best_epoch"])
        self.assertAlmostEqual(resumed["best_accuracy"], uninterrupted["best_accuracy"])

        resumed_best = torch.load(partial_dir / "best.pt", map_location="cpu", weights_only=False)
        uninterrupted_best = torch.load(full_dir / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(resumed_best["epoch"], uninterrupted_best["epoch"])
        for name, tensor in uninterrupted_best["model_state"].items():
            self.assertTrue(torch.equal(resumed_best["model_state"][name], tensor), name)
        self.assertEqual(
            (partial_dir / "metrics.jsonl").read_text(encoding="utf-8"),
            (full_dir / "metrics.jsonl").read_text(encoding="utf-8"),
        )
        resources = [
            json.loads(line)
            for line in (partial_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([row["epoch"] for row in resources], [1, 2, 3])
        self.assertEqual(resources[-1]["train_samples_processed"], 12)
        self.assertEqual(resources[-1]["valid_samples_processed"], 8)
        self.assertIn("train_seconds", resources[-1])
        if self.config["training"]["device"] == "cuda":
            self.assertGreater(resources[-1]["peak_cuda_allocated_bytes"], 0)
        else:
            self.assertIsNone(resources[-1]["peak_cuda_allocated_bytes"])
        metric_rows = [
            json.loads(line)
            for line in (partial_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertNotIn("train_seconds", metric_rows[-1])

        model_summary = json.loads(
            (partial_dir / "model_summary.json").read_text(encoding="utf-8")
        )
        self.assertGreater(model_summary["total_parameters"], 0)
        self.assertEqual(
            model_summary["total_parameters"], model_summary["trainable_parameters"]
        )

        prediction_lines = (partial_dir / "validation_predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        records = [json.loads(line) for line in prediction_lines]
        self.assertEqual(len(records), 8)
        self.assertTrue(all(record["split"] == "valid" for record in records))
        self.assertTrue((self.cache_dir / "test" / "sentinel.txt").is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU resume check")
    def test_cuda_epoch_boundary_resume_matches_uninterrupted_run(self):
        self.config["training"]["device"] = "cuda"
        self.test_epoch_boundary_resume_matches_uninterrupted_run()

    def test_resume_with_changed_config_fails_without_mutating_run(self):
        run_dir = self.root / "mismatch"
        train_run(self.config, 13, run_dir, max_epochs_this_call=1)
        before = _sha256(run_dir / "last.pt")
        changed = json.loads(json.dumps(self.config))
        changed["training"]["learning_rate"] = 0.0005
        with self.assertRaises(RunIdentityError):
            train_run(changed, 13, run_dir, resume=True)
        self.assertEqual(_sha256(run_dir / "last.pt"), before)

    def test_nonempty_new_run_is_never_overwritten(self):
        run_dir = self.root / "occupied"
        run_dir.mkdir()
        sentinel = run_dir / "important.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaises(RunIdentityError):
            train_run(self.config, 13, run_dir)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_cache_without_contextual_text_features_is_refused(self):
        bad_cache = self.root / "token_ids_cache"
        _make_cache(bad_cache, text_features_ready=False)
        config = _config(bad_cache, self.root / "unused_runs")
        run_dir = self.root / "not_started"
        with self.assertRaises(TrainingError):
            train_run(config, 13, run_dir)
        self.assertFalse(run_dir.exists())

    def test_text_only_cache_needs_no_audio_vision_metadata_or_arrays(self):
        text_cache = self.root / "text_only_cache"
        _make_cache(text_cache, text_only=True)
        config = _config(text_cache, self.root / "text_only_runs")
        run_dir = self.root / "text_only_run"

        result = train_run(config, 13, run_dir, max_epochs_this_call=1)
        self.assertEqual(result["last_epoch"], 1)
        self.assertFalse((text_cache / "splits" / "train" / "audio.npy").exists())
        self.assertFalse((text_cache / "splits" / "valid" / "vision.npy").exists())
        model_summary = json.loads(
            (run_dir / "model_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(model_summary["feature_dims"], {"text": 4})
        self.assertEqual(model_summary["sequence_lengths"], {"text": 3})

    def test_split_paths_must_resolve_inside_the_cache(self):
        text_cache = self.root / "escaping_cache"
        _make_cache(text_cache, text_only=True)
        metadata_path = text_cache / "meta.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["split_paths"]["train"] = "../../outside"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        config = _config(text_cache, self.root / "escaping_runs")
        run_dir = self.root / "escape_run"
        with self.assertRaises(TrainingError):
            train_run(config, 13, run_dir)
        self.assertFalse(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
