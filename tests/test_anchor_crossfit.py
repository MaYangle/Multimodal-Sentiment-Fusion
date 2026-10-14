import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from jevaffect.anchor_crossfit import (
    FeatureOnlyPooledSplit,
    PooledView,
    VIEWS,
    collect_oof,
    full_selection_manifest,
    prepare,
    run_suite,
    train_teacher,
)
from jevaffect.native_training import digest
from jevaffect.source_crossfit import group_of, training_rows_only


class AnchorCrossfitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / "author.csv"
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.sources = [f"src{i:02d}" for i in range(12)]
        self.rows = []
        self.ids = {"train": [], "valid": []}
        for source_index, source in enumerate(self.sources):
            for clip_index in range(2):
                self.rows.append((source, f"tr{clip_index}", "train", (-.8 + .1 * source_index + .2 * clip_index)))
            self.rows.append((source, "va0", "valid", "DO_NOT_PARSE_VALID_LABEL"))
            self.rows.append((source, "te0", "test", "DO_NOT_PARSE_TEST_LABEL"))
        with self.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("video_id", "clip_id", "mode", "label"))
            writer.writeheader()
            for source, clip, split, label in self.rows:
                writer.writerow({"video_id": source, "clip_id": clip, "mode": split, "label": label})
                if split in self.ids:
                    self.ids[split].append(f"{source}/{clip}")
        hashes = {}
        rng = np.random.default_rng(321)
        dimensions = {"text": 4, "audio": 5, "vision": 3}
        for split, ids in self.ids.items():
            directory = self.cache / split
            directory.mkdir()
            (directory / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
            for modality, dim in dimensions.items():
                x = rng.normal(size=(len(ids), 3, dim)).astype(np.float32)
                lengths = np.array([1 + (i % 3) for i in range(len(ids))], np.int64)
                np.save(directory / f"{modality}.npy", x)
                np.save(directory / f"{modality}_lengths.npy", lengths)
                for name in (f"{modality}.npy", f"{modality}_lengths.npy"):
                    hashes[f"{split}/{name}"] = digest(directory / name)
        (self.cache / "meta.json").write_text(json.dumps({"artifact_sha256": hashes}), encoding="utf-8")
        self.config = {
            "dataset": "simsv2",
            "author_csv": str(self.csv),
            "cache_dir": str(self.cache),
            "run_root": str(self.root / "runs"),
            "seeds": [13, 42, 2026],
            "device": "cpu",
            "torch_threads": 1,
            "fold_count": 3,
            "fold_seed": 20261012,
            "inner_fraction": .1,
            "full_selection_seed": 20261013,
            "full_selection_fraction": .1,
            "expected_valid_count": 12,
            "predict_valid": True,
            "hidden": 8,
            "dropout": .25,
            "batch_size": 4,
            "learning_rate": .0001,
            "weight_decay": .0001,
            "max_epochs": 3,
            "patience": 12,
            "gradient_norm": 1.0,
            "scheduler_patience": 3,
            "scheduler_factor": .5,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_train_only_parser_and_feature_only_valid_do_not_read_labels(self):
        rows = training_rows_only(self.csv)
        self.assertEqual(set(rows), set(self.ids["train"]))
        self.assertEqual(len(rows), 24)
        data = prepare(self.config, allow_fixture=True)
        self.assertEqual(data["ids"], [
            f"{source}/{clip}" for source, clip, split, _ in self.rows if split == "train"
        ])
        self.assertEqual(len(data["train"]), 24)
        self.assertEqual(len(data["valid_features"]), 12)
        self.assertIsInstance(data["valid_features"], FeatureOnlyPooledSplit)
        self.assertFalse(data["valid_features"].__dict__.get("targets"))
        self.assertFalse(data["valid_features"].__dict__.get("labels"))
        self.assertTrue(data["fixture_smoke"])
        result = train_teacher(self.config, data, view="text", seed=13, fold="full",
                               output_root=self.root / "full")
        self.assertFalse(result["official_validation_labels_parsed"])
        self.assertEqual(result["train_prediction_count"], 24)
        self.assertTrue(Path(result["full_train_predictions_file"]).exists())
        self.assertTrue(result["valid_predictions_file"].endswith("full_valid_predictions.csv"))
        valid_csv = Path(result["root"]) / "full_valid_predictions.csv"
        with valid_csv.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["id", "prediction"])
            records = list(reader)
        self.assertEqual(len(records), 12)

    def test_three_fold_oof_merge_has_each_train_id_once(self):
        data = prepare(self.config, allow_fixture=True)
        output_root = self.root / "oof_runs"
        for fold in range(3):
            result = train_teacher(self.config, data, view="vision", seed=2026, fold=fold,
                                   output_root=output_root)
            self.assertEqual(result["oof_count"], len(data["fold_manifest"]["folds"][fold]["oof_ids"]))
            self.assertIn("predictions_file", result)
        merged = collect_oof(output_root, "smoke", "vision", 2026, data["ids"])
        self.assertTrue(merged["coverage_exactly_once"])
        self.assertEqual(merged["count"], len(data["ids"]))
        with Path(merged["path"]).open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual([row["id"] for row in predictions], data["ids"])

    def test_partial_suite_report_never_claims_full_completion(self):
        output = self.root / "selected.json"
        report = run_suite(self.config, output=output, seed=13, view="text", fold=0,
                           stop_after=1, allow_fixture=True)
        self.assertFalse(report["suite_complete"])
        self.assertFalse(report["complete"])
        self.assertTrue(report["smoke_only"])
        manifest = json.loads(Path(str(output) + ".manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["has_full"])
        self.assertEqual(manifest["selected_folds"], [0])
        self.assertEqual(manifest["ids"], [
            f"{source}/{clip}" for source, clip, split, _ in self.rows if split == "train"
        ])
        self.assertIn("nested outer source evaluation", manifest["source_transfer_limit"])

    def test_source_cohorts_are_disjoint_and_every_train_id_is_held_once(self):
        data = prepare(self.config, allow_fixture=True)
        validate_ids = []
        for fold in data["fold_manifest"]["folds"]:
            source_sets = [
                {group_of(sid) for sid in fold[role + "_ids"]}
                for role in ("fit", "select", "oof")
            ]
            self.assertFalse(source_sets[0] & source_sets[1])
            self.assertFalse(source_sets[0] & source_sets[2])
            self.assertFalse(source_sets[1] & source_sets[2])
            validate_ids.extend(fold["oof_ids"])
        self.assertEqual(len(validate_ids), len(set(validate_ids)))
        self.assertEqual(set(validate_ids), set(data["ids"]))
        full = full_selection_manifest(data["ids"])
        self.assertFalse(set(full["fit_sources"]) & set(full["select_sources"]))

    def test_views_filter_pooled_inputs_and_keep_the_same_m_target(self):
        data = prepare(self.config, allow_fixture=True)
        expected = {sid: score for sid, score in training_rows_only(self.csv).items()}
        for view, modalities in VIEWS.items():
            dataset = PooledView(data["train"], modalities)
            features, target, sid = dataset[0]
            self.assertEqual(tuple(features), modalities)
            self.assertAlmostEqual(float(target), expected[sid]["score"], places=6)
            self.assertEqual({name: tensor.shape[-1] for name, tensor in features.items()},
                             {name: data["train"].features[name].shape[-1] for name in modalities})

    def test_smoke_selection_uses_complete_source_groups(self):
        data = prepare(self.config, allow_fixture=True, smoke_groups=8)
        self.assertEqual(len({group_of(sid) for sid in data["ids"]}), 8)
        self.assertEqual(len(data["ids"]), 16)
        for source in {group_of(sid) for sid in data["ids"]}:
            self.assertEqual(sum(group_of(sid) == source for sid in data["ids"]), 2)

    def test_exact_dropout_epoch_resume(self):
        data = prepare(self.config, allow_fixture=True)
        whole_root = self.root / "whole"
        resumed_root = self.root / "resumed"
        train_teacher(self.config, data, view="audio", seed=42, fold=1, output_root=whole_root)
        partial = train_teacher(self.config, data, view="audio", seed=42, fold=1,
                                stop_after=1, output_root=resumed_root)
        self.assertFalse(partial["complete"])
        train_teacher(self.config, data, view="audio", seed=42, fold=1,
                      resume=True, output_root=resumed_root)
        whole = torch.load(whole_root / "smoke/audio/seed_42/fold_1/last.pt",
                           map_location="cpu", weights_only=False)
        resumed = torch.load(resumed_root / "smoke/audio/seed_42/fold_1/last.pt",
                             map_location="cpu", weights_only=False)
        self.assertEqual(len(whole["history"]), len(resumed["history"]))
        for left, right in zip(whole["history"], resumed["history"]):
            for key in ("epoch", "train_L1", "inner_select", "optimizer_updates", "learning_rate"):
                self.assertEqual(left[key], right[key], (left, right, key))
        for key, value in whole["model"].items():
            self.assertTrue(torch.equal(value, resumed["model"][key]), key)
        self.assertEqual(
            (whole_root / "smoke/audio/seed_42/fold_1/oof_predictions.csv").read_bytes(),
            (resumed_root / "smoke/audio/seed_42/fold_1/oof_predictions.csv").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
