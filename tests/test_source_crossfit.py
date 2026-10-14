import csv
import json
import unittest
from pathlib import Path
import numpy as np
import torch
import test_native_training as fixtures
from jevaffect.source_crossfit import make_folds, validate_folds, training_rows_only
from jevaffect.crossfit_training import prepare, train_fold
from jevaffect.native_training import digest


class SourceCrossfitTests(unittest.TestCase):
    def test_entire_sources_excluded_from_fit_and_selection(self):
        ids = [f"video_{g}/clip_{c}" for g in range(11) for c in range(g + 1)]
        a = make_folds(ids); self.assertEqual(a, make_folds(ids))
        validate_folds(ids, a["folds"])
        corrupt = json.loads(json.dumps(a))
        corrupt["folds"][0]["select_ids"].append(corrupt["folds"][0]["oof_ids"][0])
        with self.assertRaises(ValueError):
            validate_folds(ids, corrupt["folds"])

    def test_official_heldout_labels_never_parsed(self):
        fixture = fixtures.NativeTrainingTests(); fixture.setUp()
        try:
            with fixture.csv.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
            for r in rows:
                if r["mode"] != "train":
                    r["label"] = "not-a-number-and-must-not-be-read"
            with fixture.csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            self.assertEqual(len(training_rows_only(fixture.csv)), 2)
        finally:
            fixture.tearDown()

    def test_exact_resume_of_real_fold_and_identity_rejection(self):
        fixture = fixtures.NativeTrainingTests(); fixture.setUp()
        try:
            with fixture.csv.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f); fields = reader.fieldnames; original = list(reader)
            templates = [r for r in original if r["mode"] == "train"]
            rows = []
            for g in range(6):
                for c, template in enumerate(templates):
                    rows.append(dict(template, video_id=f"source_{g}", clip_id=str(c)))
            with fixture.csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            ids = [r["video_id"] + "/" + r["clip_id"] for r in rows]
            (fixture.cache / "train/ids.json").write_text(json.dumps(ids))
            meta = json.loads((fixture.cache / "meta.json").read_text())
            for m in ("text", "audio", "vision"):
                path = fixture.cache / "train" / (m + ".npy")
                x = np.load(path); np.save(path, np.tile(x, (6, 1, 1)))
                lp = path.with_name(m + "_lengths.npy"); np.save(lp, np.tile([1, 2], 6).astype(np.int64))
                for file in (path, lp):
                    meta["artifact_sha256"]["train/" + file.name] = digest(file)
            (fixture.cache / "meta.json").write_text(json.dumps(meta))
            config = {"dataset": "simsv2", "author_csv": str(fixture.csv), "cache_dir": str(fixture.cache),
                      "run_root": str(fixture.root / "whole"), "device": "cpu", "three_hidden": 8,
                      "joint_branch_hidden": 8, "joint_hidden": 4, "dropout": .1, "batch_size": 2,
                      "learning_rate": .0001, "weight_decay": .0001, "regularization": .01,
                      "max_epochs": 3, "patience": 12, "fold_count": 3, "fold_seed": 20261012, "inner_fraction": .1}
            dataset, manifest = prepare(config)
            train_fold(config, dataset, manifest, 13, 0)
            resumed = dict(config, run_root=str(fixture.root / "resumed"))
            train_fold(resumed, dataset, manifest, 13, 0, stop_after=1)
            train_fold(resumed, dataset, manifest, 13, 0, resume=True)
            states = [torch.load(Path(c["run_root"]) / "formal/seed_13/fold_0/last.pt", weights_only=False) for c in (config, resumed)]
            for key in states[0]["model"]:
                self.assertTrue(torch.equal(states[0]["model"][key], states[1]["model"][key]), key)
            for a, b in zip(states[0]["history"], states[1]["history"]):
                self.assertEqual(a["train_loss"], b["train_loss"]); self.assertEqual(a["inner_select"], b["inner_select"])
            with self.assertRaises(ValueError):
                train_fold(dict(resumed, learning_rate=.001), dataset, manifest, 13, 0, resume=True)
        finally:
            if "dataset" in locals():
                for values in dataset.lengths.values():
                    values._mmap.close()
            fixture.tearDown()
