import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from jevaffect.native_training import save_checkpoint
from jevaffect.sequence_gate_training import (
    CASES,
    MODALITIES,
    _load_checkpoint,
    fit_fusion_case,
    read_author_annotations,
    refit_heads,
    select_heads,
    source_holdout,
)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _make_split(cache, split, ids, labels, channels):
    directory = Path(cache) / split
    directory.mkdir(parents=True)
    artifacts = {}
    _json(directory / "ids.json", ids)
    _json(directory / "raw_labels.json", labels)
    artifacts[f"{split}/ids.json"] = _sha(directory / "ids.json")
    artifacts[f"{split}/raw_labels.json"] = _sha(directory / "raw_labels.json")
    for mi, modality in enumerate(MODALITIES):
        values = np.zeros((len(ids), 4, 2), dtype=np.float32)
        lengths = np.full(len(ids), 3, dtype=np.int64)
        for row in range(len(ids)):
            values[row, :3, :] = np.asarray([
                [0.1 * (row + 1) + mi * .03, 0.04 * (row + 1) + mi * .02],
                [0.08 * (row + 1) + mi * .05, 0.02 * (row + 1) + mi * .01],
                [0.03 * (row + 1) + mi * .01, 0.09 * (row + 1) + mi * .03],
            ], dtype=np.float32)
        np.save(directory / f"{modality}.npy", values)
        np.save(directory / f"{modality}_lengths.npy", lengths)
        artifacts[f"{split}/{modality}.npy"] = _sha(directory / f"{modality}.npy")
        artifacts[f"{split}/{modality}_lengths.npy"] = _sha(directory / f"{modality}_lengths.npy")
    return artifacts


def _fixture(root):
    root = Path(root)
    author_path = root / "author.csv"
    train_ids = [f"video{i}/clip{i}" for i in range(8)]
    valid_ids = [f"valid{i}/clip{i}" for i in range(3)]
    train_m = [-.78, -.55, -.32, -.08, .13, .37, .62, .86]
    valid_m = [-.61, .12, .73]
    with author_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"])
        writer.writeheader()
        for sid, m in zip(train_ids, train_m):
            video, clip = sid.split("/")
            writer.writerow({"video_id": video, "clip_id": clip, "text": "x", "label": m,
                             "label_T": m * .8, "label_A": m * .7, "label_V": m * .6,
                             "annotation": "", "mode": "train"})
        for sid, m in zip(valid_ids, valid_m):
            video, clip = sid.split("/")
            # T/A/V are deliberately unparsable on VALID.
            writer.writerow({"video_id": video, "clip_id": clip, "text": "x", "label": m,
                             "label_T": "do not parse", "label_A": "do not parse", "label_V": "do not parse",
                             "annotation": "", "mode": "valid"})
        writer.writerow({"video_id": "test-video", "clip_id": "test-clip", "text": "x", "label": "do not parse",
                         "label_T": "do not parse", "label_A": "do not parse", "label_V": "do not parse",
                         "annotation": "", "mode": "test"})
    author_sha = _sha(author_path)
    raw_cache = root / "raw"
    std_cache = root / "standardized"
    pooled_cache = root / "pooled"
    for cache in (raw_cache, std_cache, pooled_cache):
        cache.mkdir()
    raw_artifacts = _make_split(raw_cache, "train", train_ids, train_m, None)
    _json(raw_cache / "metadata.json", {"dataset": "simsv2", "test_exported": False, "artifact_sha256": raw_artifacts})
    std_artifacts = {}
    std_artifacts.update(_make_split(std_cache, "train", train_ids, train_m, None))
    std_artifacts.update(_make_split(std_cache, "valid", valid_ids, valid_m, None))
    _json(std_cache / "metadata.json", {"dataset": "simsv2", "test_exported": False, "author_csv_sha256": author_sha,
                                         "artifact_sha256": std_artifacts})
    _json(pooled_cache / "normalization.json", {"fixture": True})
    _json(pooled_cache / "metadata.json", {"dataset": "simsv2", "test_exported": False,
                                            "author_csv_sha256": author_sha})
    config = {
        "schema": "fixture",
        "dataset": "simsv2",
        "author_csv": str(author_path),
        "author_csv_sha256": author_sha,
        "raw_cache": str(raw_cache),
        "standardized_cache": str(std_cache),
        "pooled_cache": str(pooled_cache),
        "run_root": str(root / "runs"),
        "device": "cpu",
        "seeds": [13],
        "cases": list(CASES),
        "source_holdout": {"seed": 20261014, "fraction": .1},
        "model": {"dim": 8, "queries": 2, "depth": 1, "heads": 2, "ff_dim": 16, "dropout": .1},
        "expected_parameters": {},
        "stage1": {"epochs": 2, "batch_size": 4, "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1.0},
        "refit": {"batch_size": 4, "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1.0},
        "fusion": {"epochs": 2, "batch_size": 3, "learning_rate": .001, "weight_decay": .0001,
                   "gradient_clip": 1.0, "contrastive_weight": .01, "distillation_weight": .1},
    }
    return config, author_path


def _assert_state_equal(testcase, left, right):
    testcase.assertEqual(set(left), set(right))
    for name in left:
        testcase.assertTrue(torch.equal(left[name], right[name]), name)


def _force_two_epoch_refit_counts(root):
    path = Path(root) / "select" / "seed_13" / "checkpoint.pt"
    checkpoint = _load_checkpoint(path)
    checkpoint["selected_epochs"] = {m: 2 for m in MODALITIES}
    save_checkpoint(path, checkpoint)


class SequenceGateTrainingTests(unittest.TestCase):
    def test_source_holdout_uses_sorted_train_sources_and_ceil_ten_percent(self):
        ids = [f"video{i:03d}/clip{j}" for i in range(129) for j in range(2)]
        first = source_holdout(ids)
        second = source_holdout(ids)
        self.assertEqual(first, second)
        self.assertEqual(len(first["all_sources"]), 129)
        self.assertEqual(len(first["held_sources"]), 13)
        self.assertEqual(len(first["opt_sources"]), 116)
        self.assertFalse(set(first["opt_sources"]) & set(first["held_sources"]))
        self.assertEqual(len(first["opt_ids"]) + len(first["held_ids"]), len(ids))

        smoke_ids = [f"source{i}/clip{j}" for i, count in enumerate((20, 12)) for j in range(count)]
        smoke = source_holdout(smoke_ids)
        self.assertEqual(len(smoke["all_sources"]), 2)
        self.assertEqual(smoke["held_count"], 1)
        self.assertEqual(len(smoke["held_sources"]), 1)

    def test_author_reader_never_parses_forbidden_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            config, author_path = _fixture(temp)
            train_only = read_author_annotations(author_path, channels=True, allow_fixture=True)
            self.assertIn("channels", train_only["records"]["video0/clip0"])
            self.assertNotIn("score", train_only["records"]["valid0/clip0"])
            self.assertNotIn("channels", train_only["records"]["valid0/clip0"])
            self.assertNotIn("score", train_only["records"]["test-video/test-clip"])
            self.assertNotIn("channels", train_only["records"]["test-video/test-clip"])

            native_only = read_author_annotations(author_path, channels=False, allow_fixture=True)
            self.assertIn("score", native_only["records"]["valid0/clip0"])
            self.assertNotIn("channels", native_only["records"]["video0/clip0"])
            self.assertNotIn("score", native_only["records"]["test-video/test-clip"])

    def test_selection_refit_and_fusion_resume_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = _fixture(root)
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"

            select_heads(config, 13, output=uninterrupted, allow_fixture=True)
            select_heads(config, 13, output=resumed, stop_after=1, allow_fixture=True)
            select_heads(config, 13, output=resumed, stop_after=1, resume=True, allow_fixture=True)
            selected_a = _load_checkpoint(uninterrupted / "select" / "seed_13" / "checkpoint.pt")
            selected_b = _load_checkpoint(resumed / "select" / "seed_13" / "checkpoint.pt")
            self.assertEqual(selected_a["selected_epochs"], selected_b["selected_epochs"])
            _assert_state_equal(self, selected_a["last_state"], selected_b["last_state"])
            self.assertTrue(torch.equal(selected_a["rng"]["validation_loader_generator"], selected_b["rng"]["validation_loader_generator"]))
            for modality in MODALITIES:
                self.assertTrue(torch.equal(selected_a["best_branches"][modality]["state"][f"branches.{modality}.projector.0.weight"],
                                            selected_b["best_branches"][modality]["state"][f"branches.{modality}.projector.0.weight"]))

            # Give the fixture a fixed two-epoch schedule so the refit resume
            # path always crosses a checkpoint boundary, independent of its
            # tiny held-source score ordering.
            _force_two_epoch_refit_counts(uninterrupted)
            _force_two_epoch_refit_counts(resumed)
            refit_heads(config, 13, output=uninterrupted, allow_fixture=True)
            refit_heads(config, 13, output=resumed, stop_after=1, allow_fixture=True)
            refit_heads(config, 13, output=resumed, stop_after=1, resume=True, allow_fixture=True)
            refit_a = _load_checkpoint(uninterrupted / "refit" / "seed_13" / "checkpoint.pt")
            refit_b = _load_checkpoint(resumed / "refit" / "seed_13" / "checkpoint.pt")
            _assert_state_equal(self, refit_a["last_state"], refit_b["last_state"])
            self.assertIsNone(refit_a["rng"]["validation_loader_generator"])

            fit_fusion_case(config, 13, "distilled_pre_norm", output=uninterrupted, allow_fixture=True)
            fit_fusion_case(config, 13, "distilled_pre_norm", output=resumed, stop_after=1, allow_fixture=True)
            fit_fusion_case(config, 13, "distilled_pre_norm", output=resumed, stop_after=1, resume=True, allow_fixture=True)
            fusion_a = _load_checkpoint(uninterrupted / "fusion" / "seed_13" / "distilled_pre_norm" / "checkpoint.pt")
            fusion_b = _load_checkpoint(resumed / "fusion" / "seed_13" / "distilled_pre_norm" / "checkpoint.pt")
            self.assertEqual(fusion_a["best_epoch"], fusion_b["best_epoch"])
            _assert_state_equal(self, fusion_a["last_state"], fusion_b["last_state"])
            _assert_state_equal(self, fusion_a["best_state"], fusion_b["best_state"])
            self.assertEqual(fusion_a["best_valid_predictions"], fusion_b["best_valid_predictions"])
            self.assertTrue(torch.equal(fusion_a["rng"]["validation_loader_generator"], fusion_b["rng"]["validation_loader_generator"]))
            distilled_result = json.loads((uninterrupted / "fusion" / "seed_13" / "distilled_pre_norm" / "result.json").read_text())
            self.assertIn("source_none", distilled_result["gate_intervention_stats"])
            self.assertEqual(distilled_result["policy_head"]["status"], "trained_by_distillation_and_used_for_label_free_routing")
            fit_fusion_case(config, 13, "static", output=uninterrupted, allow_fixture=True)
            static = _load_checkpoint(uninterrupted / "fusion" / "seed_13" / "static" / "checkpoint.pt")
            self.assertEqual(fusion_a["initial_fusion_state_sha256"], static["initial_fusion_state_sha256"])
            static_result = json.loads((uninterrupted / "fusion" / "seed_13" / "static" / "result.json").read_text())
            self.assertEqual(static_result["policy_head"]["status"], "present_but_unused")
            self.assertTrue(static_result["unused_policy_unchanged"])
            self.assertTrue(all("distillation" in row for row in fusion_a["history"]))
            self.assertTrue(all(row["distillation"] == 0.0 for row in static["history"]))
            self.assertTrue(fusion_a["policy_head"]["used_for_routing"])

            fit_fusion_case(config, 13, "learned_post_norm", output=uninterrupted, allow_fixture=True)
            learned = _load_checkpoint(uninterrupted / "fusion" / "seed_13" / "learned_post_norm" / "checkpoint.pt")
            learned_result = json.loads((uninterrupted / "fusion" / "seed_13" / "learned_post_norm" / "result.json").read_text())
            self.assertEqual(learned_result["policy_head"]["status"],
                             "trained_by_prediction_loss_and_used_for_label_free_routing_without_distillation")
            self.assertIsNone(learned_result["unused_policy_unchanged"])
            self.assertFalse(learned_result["policy_state_unchanged"])
            self.assertTrue(all(row["distillation"] == 0.0 for row in learned["history"]))
            self.assertEqual(fusion_a["initial_fusion_state_sha256"], learned["initial_fusion_state_sha256"])
            self.assertEqual(fusion_a["parameter_count"], learned["parameter_count"])

            expected = {
                "pre_norm": ("present_but_unused", False),
                "post_norm": ("present_but_unused", False),
                "distilled_static": ("trained_by_distillation_but_not_used_for_routing", True),
                "distilled_post_norm": ("trained_by_distillation_and_used_for_label_free_routing", True),
            }
            for case, (status, uses_kl) in expected.items():
                fit_fusion_case(config, 13, case, output=uninterrupted, allow_fixture=True)
                case_checkpoint = _load_checkpoint(uninterrupted / "fusion" / "seed_13" / case / "checkpoint.pt")
                case_result = json.loads((uninterrupted / "fusion" / "seed_13" / case / "result.json").read_text())
                self.assertEqual(case_result["policy_head"]["status"], status)
                self.assertEqual(case_result["policy_head"]["used_for_routing"], case in ("distilled_post_norm",))
                self.assertEqual(case_checkpoint["parameter_count"], fusion_a["parameter_count"])
                self.assertEqual(case_checkpoint["initial_fusion_state_sha256"], fusion_a["initial_fusion_state_sha256"])
                if not uses_kl:
                    self.assertTrue(all(row["distillation"] == 0.0 for row in case_checkpoint["history"]))


if __name__ == "__main__":
    unittest.main()
