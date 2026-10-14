import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tests import test_anchor_crossfit as anchor_fixtures
from jevaffect import nested_shape_points
from jevaffect.native_training import digest
from jevaffect.source_crossfit import group_of


class NestedShapePointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = anchor_fixtures.AnchorCrossfitTests(
            methodName="test_source_cohorts_are_disjoint_and_every_train_id_is_held_once"
        )
        self.fixture.setUp()
        self.root = self.fixture.root
        self.config = copy.deepcopy(self.fixture.config)
        self.config["predict_valid"] = False
        self.config["fold_count"] = 3
        self.config["fold_seed"] = 20261012
        self.config["seeds"] = [13, 42, 2026]
        self.config["nested_pipeline"] = {
            "name": nested_shape_points.PIPELINE_NAME,
            "implementation_sha256": digest(Path(nested_shape_points.__file__)),
            "role_seed": 20261013,
            "calibration_source_fraction": .15,
            "point_selection_fraction": .1,
            "shape_selection_fraction": .3,
            "point_holdout_semantics": "shape_fit+shape_select+outer; not global once-only OOF",
            "scaler_fit_role": "point_fit",
            "context_definition": "selected standardized fusion Linear(1920,256) then GELU and eval-mode Dropout",
            "anchor_definition": "clip(selected raw native point output,-1,1)",
            "target_definition": "original native CH-SIMS M score, float32",
            "export_tables": ["shape_fit", "shape_select", "outer"],
            "outer_role": "fixed-stage final readout only; excluded from point selection and normalization",
            "official_VALID_used": False,
            "official_TEST_used": False,
        }
        self.config["run_root"] = str(self.root / "run")
        self.data = nested_shape_points.prepare_nested(self.config, allow_fixture=True)

    def tearDown(self):
        self.fixture.tearDown()

    def test_nested_roles_are_source_disjoint_and_outer_is_once_only(self):
        self.assertTrue(nested_shape_points.validate_nested_manifest(
            self.data["ids"], self.data["nested_manifest"]
        ))
        outer_ids = []
        for fold in self.data["nested_manifest"]["folds"]:
            role_sources = {
                role: {group_of(sid) for sid in fold[role + "_ids"]}
                for role in nested_shape_points.ROLE_NAMES
            }
            for i, left in enumerate(nested_shape_points.ROLE_NAMES):
                for right in nested_shape_points.ROLE_NAMES[i + 1:]:
                    self.assertFalse(role_sources[left] & role_sources[right])
            calibration = role_sources["shape_fit"] | role_sources["shape_select"]
            point = role_sources["point_fit"] | role_sources["point_select"]
            self.assertFalse(calibration & point)
            self.assertEqual(set(fold["point_scaler_fit_ids"]), set(fold["point_fit_ids"]))
            self.assertEqual(set(fold["point_oof_ids"]),
                             set(fold["shape_fit_ids"] + fold["shape_select_ids"] + fold["outer_ids"]))
            self.assertIn("not global once-only OOF", fold["point_holdout_semantics"])
            outer_ids.extend(fold["outer_ids"])
        self.assertEqual(len(outer_ids), len(set(outer_ids)))
        self.assertEqual(set(outer_ids), set(self.data["ids"]))

    def test_calibration_sources_are_excluded_from_point_fit_and_scaler(self):
        fold = 0
        nested = self.data["nested_manifest"]["folds"][fold]
        actual_config, transformed = nested_shape_points.standardized_source.transform_fold(
            self.config, self.data, fold
        )
        fit_ids = set(actual_config["preprocessing"]["fit_ids"])
        self.assertEqual(fit_ids, set(nested["point_fit_ids"]))
        self.assertFalse(fit_ids & set(nested["shape_fit_ids"] + nested["shape_select_ids"] + nested["outer_ids"]))
        for modality, stats in actual_config["preprocessing"]["statistics"].items():
            fit_rows = np.asarray([self.data["ids"].index(sid) for sid in nested["point_fit_ids"]])
            values = transformed["train"].features[modality][fit_rows]
            np.testing.assert_allclose(values.mean(0), 0, atol=1e-6)
            scale = np.asarray(stats["scale"])
            active = scale > 1e-6
            np.testing.assert_allclose(values.std(0)[active], 1, atol=1e-5)
        held_ids = set(nested["point_select_ids"] + nested["shape_fit_ids"] + nested["shape_select_ids"] + nested["outer_ids"])
        original = {name: array.copy() for name, array in self.data["train"].features.items()}
        for modality in original:
            held_rows = [self.data["ids"].index(sid) for sid in held_ids]
            self.data["train"].features[modality][held_rows] += 77
        changed_config, _ = nested_shape_points.standardized_source.transform_fold(
            self.config, self.data, fold
        )
        self.assertEqual(actual_config["preprocessing"]["statistics"],
                         changed_config["preprocessing"]["statistics"])
        for modality in original:
            self.data["train"].features[modality][...] = original[modality]

    def test_outside_point_roles_labels_do_not_change_point_model_or_scaler(self):
        fold = 1
        nested = self.data["nested_manifest"]["folds"][fold]
        changed = dict(self.data)
        targets = self.data["train"].targets.copy()
        index = {sid: i for i, sid in enumerate(self.data["train"].ids)}
        held = nested["shape_fit_ids"] + nested["shape_select_ids"] + nested["outer_ids"]
        targets[[index[sid] for sid in held]] *= -0.4
        permutation = np.arange(len(self.data["train"].ids))[::-1]
        changed["train"] = SimpleNamespace(
            ids=[self.data["train"].ids[i] for i in permutation],
            targets=targets[permutation],
            features={name: values[permutation].copy()
                      for name, values in self.data["train"].features.items()},
            hashes=self.data["train"].hashes,
        )
        changed["pooled_feature_sha256"] = nested_shape_points.feature_content_sha256(changed["train"])
        first = nested_shape_points.train_point_fold(
            self.config, self.data, seed=13, fold=fold,
            output_root=self.root / "labels_a" / "point_models",
        )
        second = nested_shape_points.train_point_fold(
            self.config, changed, seed=13, fold=fold,
            output_root=self.root / "labels_b" / "point_models",
        )
        a = torch.load(Path(first["point_model_root"]) / "last.pt", map_location="cpu", weights_only=False)
        b = torch.load(Path(second["point_model_root"]) / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(len(a["history"]), len(b["history"]))
        for left, right in zip(a["history"], b["history"]):
            for key in ("epoch", "train_L1", "inner_select", "optimizer_updates", "learning_rate"):
                self.assertEqual(left[key], right[key], (left, right, key))
        for name in a["model"]:
            self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)
        scaler_a = json.loads(Path(first["point_scaler_path"]).read_text(encoding="utf-8"))
        scaler_b = json.loads(Path(second["point_scaler_path"]).read_text(encoding="utf-8"))
        self.assertEqual(scaler_a["fit_ids"], scaler_b["fit_ids"])
        for modality in scaler_a["statistics"]:
            for key in ("mean", "scale"):
                self.assertEqual(scaler_a["statistics"][modality][key],
                                 scaler_b["statistics"][modality][key])
        self.assertEqual(set(first["tables"]), {"shape_fit", "shape_select", "outer"})
        outer = np.load(first["tables"]["outer"]["path"])
        self.assertEqual(outer["context"].shape[1], self.config["hidden"])
        self.assertEqual(outer["context"].dtype, np.float32)
        self.assertTrue(np.all(outer["anchor"] >= -1) and np.all(outer["anchor"] <= 1))
        reordered_outer = np.load(second["tables"]["outer"]["path"])
        changed_index = {sid: i for i, sid in enumerate(changed["train"].ids)}
        expected_targets = np.asarray([changed["train"].targets[changed_index[sid]]
                                       for sid in reordered_outer["ids"]], np.float32)
        np.testing.assert_array_equal(reordered_outer["target"], expected_targets)

    def test_exact_resume_matches_uninterrupted_point_checkpoint(self):
        full = nested_shape_points.train_point_fold(
            self.config, self.data, seed=42, fold=2,
            output_root=self.root / "whole" / "point_models",
        )
        partial = nested_shape_points.train_point_fold(
            self.config, self.data, seed=42, fold=2,
            output_root=self.root / "resumed" / "point_models", stop_after=1,
        )
        self.assertFalse(partial["complete"])
        resumed = nested_shape_points.train_point_fold(
            self.config, self.data, seed=42, fold=2,
            output_root=self.root / "resumed" / "point_models", resume=True,
        )
        left = torch.load(Path(full["point_model_root"]) / "last.pt", map_location="cpu", weights_only=False)
        right = torch.load(Path(resumed["point_model_root"]) / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(len(left["history"]), len(right["history"]))
        for a, b in zip(left["history"], right["history"]):
            for key in ("epoch", "train_L1", "inner_select", "optimizer_updates", "learning_rate"):
                self.assertEqual(a[key], b[key], (a, b, key))
        for name in left["model"]:
            self.assertTrue(torch.equal(left["model"][name], right["model"][name]), name)
        for role in ("shape_fit", "shape_select", "outer"):
            with np.load(full["tables"][role]["path"]) as expected, np.load(resumed["tables"][role]["path"]) as actual:
                for key in ("context", "anchor", "native_point", "target", "ids"):
                    np.testing.assert_array_equal(expected[key], actual[key])

    def test_code_feature_and_role_guards_reject_changed_inputs(self):
        wrong_code = copy.deepcopy(self.config)
        wrong_code["nested_pipeline"]["implementation_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "implementation SHA"):
            nested_shape_points.validate_registered_config(wrong_code, allow_fixture=True)
        original = self.data["train"].features["text"][0, 0]
        self.data["train"].features["text"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "feature content changed"):
            nested_shape_points.train_point_fold(
                self.config, self.data, seed=13, fold=0, output_root=self.root / "guard"
            )
        self.data["train"].features["text"][0, 0] = original
        broken = copy.deepcopy(self.data["nested_manifest"])
        broken["folds"][0]["point_fit_ids"].append(broken["folds"][0]["shape_fit_ids"][0])
        with self.assertRaises(ValueError):
            nested_shape_points.validate_nested_manifest(self.data["ids"], broken)

    def test_smoke_subset_keeps_full_cache_but_only_registered_source_roles(self):
        data = nested_shape_points.prepare_nested(self.config, allow_fixture=True, smoke_groups=8)
        self.assertLessEqual(len(data["ids"]), len(data["train"].ids))
        nested_shape_points._verify_data_guard(data)
        self.assertEqual(len({group_of(sid) for sid in data["ids"]}), 8)
        result = nested_shape_points.train_point_fold(
            self.config, data, seed=13, fold=0, output_root=self.root/"subset"/"point_models"
        )
        self.assertTrue(result["complete"])
        for role, table in result["tables"].items():
            actual = json.loads(Path(table["ids_path"]).read_text())
            self.assertEqual(actual, data["nested_manifest"]["folds"][0][role+"_ids"])


if __name__ == "__main__":
    unittest.main()
