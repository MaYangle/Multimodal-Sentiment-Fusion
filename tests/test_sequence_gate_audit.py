import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_sequence_gates import (
    first_minimum_epoch,
    independent_native_metrics,
    manual_teacher_kl,
    paired_source_bootstrap,
    source_holdout,
    _read_annotations,
)
from scripts import audit_sequence_gates as audit_module


class SequenceGateAuditTests(unittest.TestCase):
  def test_source_holdout_keeps_every_clip_from_a_source_together(self):
    ids = [f"v{source}/c{clip}" for source in range(20) for clip in range(source % 3 + 1)]
    split = source_holdout(ids, seed=20261014, fraction=0.1)
    self.assertEqual(split["held_count"], 2)
    self.assertEqual(len(split["held_sources"]), 2)
    self.assertTrue(set(split["opt_sources"]).isdisjoint(split["held_sources"]))
    held_prefixes = {sid.split("/", 1)[0] for sid in split["held_ids"]}
    opt_prefixes = {sid.split("/", 1)[0] for sid in split["opt_ids"]}
    self.assertEqual(held_prefixes, set(split["held_sources"]))
    self.assertEqual(opt_prefixes, set(split["opt_sources"]))


  def test_first_minimum_preserves_first_epoch_on_ties(self):
    rows = [{"m": 0.3}, {"m": 0.2}, {"m": 0.2}, {"m": float("nan")}]
    self.assertEqual(first_minimum_epoch(rows, lambda row: row["m"]), 2)
    self.assertIsNone(first_minimum_epoch([], lambda row: row["m"]))


  def test_native_metrics_use_float32_author_bins_and_report_full_schema(self):
    metrics = independent_native_metrics(
        np.asarray([-0.10000000149, 0.0, 0.10000000149, 0.8], dtype=np.float32),
        np.asarray([-0.1, 0.0, 0.1, 0.8], dtype=np.float32),
    )
    self.assertEqual(metrics["count"], 4)
    self.assertLess(metrics["MAE"], 1e-7)
    self.assertEqual(metrics["Mult_acc_2"], 1.0)
    self.assertEqual(metrics["Mult_acc_3"], 1.0)
    self.assertEqual(metrics["Mult_acc_5"], 1.0)
    self.assertEqual(metrics["weak_count"], 3)
    self.assertEqual(metrics["Mult_acc_2_weak"], 1.0)
    self.assertEqual(set(metrics), {"count", "MAE", "Corr", "Mult_acc_2", "F1_score", "weak_count",
                                    "Mult_acc_2_weak", "Mult_acc_3", "Mult_acc_5", "R_squre"})


  def test_manual_teacher_kl_zeros_absent_modalities_and_all_absent_rows(self):
    scores = np.asarray([[0.4, 0.1, -0.2], [0.2, 0.0, 0.0]], dtype=np.float32)
    target = np.asarray([0.1, -0.4], dtype=np.float32)
    present = np.asarray([[True, False, True], [False, False, False]])
    policy = np.asarray([[0.25, 0.0, 0.75], [0.0, 0.0, 0.0]], dtype=np.float32)
    teacher, kl = manual_teacher_kl(scores, target, present, policy)
    self.assertTrue(np.allclose(teacher.sum(1), [1.0, 0.0]))
    self.assertEqual(teacher[0, 1], 0.0)
    self.assertTrue(np.all(teacher[1] == 0.0))
    self.assertGreaterEqual(kl, 0.0)
    self.assertEqual(manual_teacher_kl(scores[1:], target[1:], present[1:], policy[1:])[1], 0.0)


  def test_source_bootstrap_is_paired_and_deterministic(self):
    truth = {"v1/a": 0.2, "v1/b": -0.1, "v2/a": 0.8, "v3/a": -0.6}
    candidate = {sid: value + 0.01 for sid, value in truth.items()}
    reference = {sid: value + 0.1 for sid, value in truth.items()}
    first = paired_source_bootstrap(candidate, reference, truth, draws=200, seed=7)
    second = paired_source_bootstrap(candidate, reference, truth, draws=200, seed=7)
    self.assertEqual(first, second)
    self.assertEqual(first["source_count"], 3)
    self.assertEqual(first["clip_count"], 4)
    self.assertLess(first["ci95_percentile"][1], 0)


  def test_annotation_reader_never_parses_valid_channels_or_test_scores(self):
    fields = ["video_id", "clip_id", "mode", "label", "label_T", "label_A", "label_V"]
    rows = [
        {"video_id": "v1", "clip_id": "c1", "mode": "train", "label": "0.2", "label_T": "0.1", "label_A": "-0.2", "label_V": "0.3"},
        {"video_id": "v2", "clip_id": "c1", "mode": "valid", "label": "-0.1", "label_T": "bad", "label_A": "bad", "label_V": "bad"},
        {"video_id": "v3", "clip_id": "c1", "mode": "test", "label": "bad", "label_T": "bad", "label_A": "bad", "label_V": "bad"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "author.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        train_only, _ = _read_annotations(path, channels=True, valid_m=False)
        self.assertTrue(np.allclose(train_only["v1/c1"]["TAV"], [0.1, -0.2, 0.3]))
        self.assertNotIn("M", train_only["v2/c1"])
        self.assertNotIn("TAV", train_only["v2/c1"])
        self.assertNotIn("M", train_only["v3/c1"])
        full_train_valid, _ = _read_annotations(path, channels=False, valid_m=True)
        self.assertAlmostEqual(full_train_valid["v2/c1"]["M"], -0.1)
        self.assertNotIn("M", full_train_valid["v3/c1"])
        self.assertNotIn("TAV", full_train_valid["v1/c1"])


  def test_teacher_arm_selection_uses_clipped_native_mae_not_raw_mae(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        author = root / "author.csv"
        fields = ["video_id", "clip_id", "mode", "label", "label_T", "label_A", "label_V"]
        valid_rows = []
        valid_ids = []
        targets = {}
        for i in range(647):
            sid = f"v{i // 100:02d}/c{i:04d}"
            target = np.float32(1.0)
            video, clip = sid.split("/")
            valid_ids.append(sid)
            targets[sid] = float(target)
            valid_rows.append({"video_id": video, "clip_id": clip, "mode": "valid", "label": str(float(target)),
                               "label_T": "ignored", "label_A": "ignored", "label_V": "ignored"})
        test_row = {"video_id": "test_video", "clip_id": "0001", "mode": "test", "label": "not-a-score",
                    "label_T": "ignored", "label_A": "ignored", "label_V": "ignored"}
        with author.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(valid_rows + [test_row])

        root_run = root / "runs/gates"
        references = ["static", "pre_norm", "post_norm", "distilled_static", "distilled_pre_norm", "distilled_post_norm", "learned_post_norm"]
        for case in references:
            if case == "distilled_post_norm":
                prediction_score = 1.0
            elif case == "pre_norm":
                prediction_score = 2.0
            elif case == "post_norm":
                prediction_score = 0.6
            else:
                prediction_score = 0.8
            prediction = [{"id": sid, "prediction": prediction_score} for sid in valid_ids]
            path = root_run / "fusion/seed_13" / case / "valid_predictions.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prediction), encoding="utf-8")
        for model in ("early_fusion", "almt"):
            prediction = [{"id": sid, "target": targets[sid], "prediction": 0.8} for sid in valid_ids]
            path = root / f"runs/native_simsv2_{'standardized_v1' if model == 'early_fusion' else 'sequence_standardized_v1'}/{model}/seed_13/valid_predictions.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prediction), encoding="utf-8")

        target_vector = np.asarray([targets[sid] for sid in valid_ids], np.float32)
        raw_pre = float(np.mean(np.abs(np.full(647, 2.0, np.float32) - target_vector)))
        raw_post = float(np.mean(np.abs(np.full(647, 0.6, np.float32) - target_vector)))
        native_pre = independent_native_metrics(np.full(647, 2.0, np.float32), target_vector)["MAE"]
        native_post = independent_native_metrics(np.full(647, 0.6, np.float32), target_vector)["MAE"]
        self.assertLess(raw_post, raw_pre, "raw MAE ranks post_norm first in this fixture")
        self.assertLess(native_pre, native_post, "native clipped MAE ranks pre_norm first")

        checks = []
        def record_check(name, passed, detail=None):
            checks.append({"name": name, "passed": bool(passed), "detail": detail})

        config = {"author_csv": "author.csv", "run_root": "runs/gates"}
        with patch.object(audit_module, "SEEDS", (13,)), patch.object(audit_module, "PROMOTION_BOOTSTRAP_DRAWS", 20):
            outcome = audit_module._audit_comparisons(root, config, record_check)
        self.assertEqual(outcome["status"], "complete")
        self.assertEqual(set(outcome["references"]), {
            "distilled_static", "distilled_pre_norm", "learned_post_norm",
            "strongest_teacher_pre_post", "conditioned_early_fusion", "conditioned_ALMT",
        })
        self.assertEqual(outcome["strongest_teacher_choice_by_seed"][13], "pre_norm")
        self.assertFalse(outcome["all_primary_gates_pass"])
        failures = [item["name"] for item in checks if not item["passed"]]
        self.assertEqual(failures, ["candidate_gate:strongest_teacher_pre_post"])
        self.assertTrue(all(item["passed"] for item in checks if item["name"] != "candidate_gate:strongest_teacher_pre_post"))
        self.assertEqual(outcome["references"]["distilled_static"]["bootstrap"]["draws"], 20)


if __name__ == "__main__":
    unittest.main()
