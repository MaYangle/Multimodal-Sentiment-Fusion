import copy
import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from tests import test_nested_shape_points as point_fixtures
from jevaffect import native_crps_training as crps_training, shape_controls
from jevaffect.native_training import digest
from jevaffect.nested_shape_points import run_nested_suite
from jevaffect.nested_shape_training import load_stage


def quiet_call(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class NativeCRPSTrainingTests(unittest.TestCase):
    def setUp(self):
        self.points = point_fixtures.NestedShapePointTests()
        self.points.setUp()
        self.root = self.points.root
        self.suite = self.root / "point_suite.json"
        quiet_call(run_nested_suite, self.points.config, output=self.suite, allow_fixture=True)
        self.tables, self.binding = load_stage(self.suite, 13, 0, allow_smoke=True)
        self.config = {
            "dataset": "simsv2",
            "point_suite": "runs/native_nested_shape_points_v1/suite.json",
            "run_root": str(self.root / "crps"),
            "device": "cpu",
            "torch_threads": 1,
            "hidden": 8,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 4,
            "max_epochs": 3,
            "patience": 12,
            "gradient_norm": 1.0,
            "quadrature_nodes": 48,
            "readout_nodes": [48, 96, 192],
            "quadrature_node_tolerance": 1e-4,
            "bin_negative_roundoff_tolerance": 1e-12,
            "bin_sum_roundoff_tolerance": 1e-10,
            "cdf_range_roundoff_tolerance": 1e-12,
            "cases": list(crps_training.CASES),
            "seeds": list(crps_training.SEEDS),
            "primary_case": "bern_d6",
            "plan": "docs/NATIVE_CDF_CRPS_PLAN_V1.md",
            "selection": "first minimum shape_select CRPS_48",
            "official_VALID_used": False,
            "official_TEST_used": False,
        }

    def tearDown(self):
        self.points.tearDown()

    def test_all_registered_heads_train_and_keep_common_encoder_initialization(self):
        rows = []
        for case in crps_training.CASES:
            result = quiet_call(crps_training.train, self.config, self.tables, self.binding,
                                case=case, seed=13, fold=0)
            self.assertTrue(result["complete"])
            self.assertEqual(result["epochs"], 3)
            self.assertGreater(result["updates"], 0)
            self.assertEqual(result["implementation_sha256"]["cdf_crps.py"], digest(Path("jevaffect/cdf_crps.py")))
            self.assertEqual(result["implementation_sha256"]["native_crps_training.py"], digest(Path("jevaffect/native_crps_training.py")))
            records = json.loads((Path(result["root"]) / "outer_predictions.json").read_text())
            self.assertEqual([row["id"] for row in records], self.tables["outer"]["ids"])
            self.assertTrue(all(np.isfinite(row["CRPS_48"]) for row in records))
            self.assertTrue(all(np.isfinite(row["CRPS_96"]) for row in records))
            self.assertTrue(all(np.isfinite(row["CRPS_192"]) for row in records))
            self.assertTrue(all("native_hit_probability" in row and "raw_hit_probability" in row for row in records))
            self.assertIn("native_author_point", result["outer"])
            self.assertIn("raw_unclipped_MAE", result["outer"])
            self.assertIn("native_hit_Brier", result["outer"])
            self.assertIn("raw_hit_Brier", result["outer"])
            self.assertTrue(result["quadrature_gate_pass"])
            self.assertTrue(all(np.isfinite(row["CRPS_trapezoid_1001"]) for row in records))
            resumed = quiet_call(crps_training.train, self.config, self.tables, self.binding,
                                 case=case, seed=13, fold=0, resume=True)
            self.assertEqual(result["predictions_sha256"], resumed["predictions_sha256"])
            rows.append(result)
        self.assertEqual(len({row["initial_encoder_sha256"] for row in rows}), 1)

    def test_exact_resume_outer_label_isolation_and_off_grid_readout(self):
        case = "median_mdn"
        whole = quiet_call(crps_training.train, self.config, self.tables, self.binding,
                           case=case, seed=42, fold=1)
        partial_config = {**self.config, "run_root": str(self.root / "partial")}
        partial = quiet_call(crps_training.train, partial_config, self.tables, self.binding,
                             case=case, seed=42, fold=1, stop_after=1)
        self.assertFalse(partial["complete"])
        self.assertFalse((Path(partial["root"]) / "outer_predictions.json").exists())
        resumed = quiet_call(crps_training.train, partial_config, self.tables, self.binding,
                             case=case, seed=42, fold=1, resume=True)

        changed = copy.deepcopy(self.tables)
        changed["outer"]["target"][:] = np.float32(-0.1)
        changed_config = {**self.config, "run_root": str(self.root / "changed_outer")}
        outside = quiet_call(crps_training.train, changed_config, changed, self.binding,
                             case=case, seed=42, fold=1)
        states = [torch.load(Path(row["root"]) / "last.pt", map_location="cpu", weights_only=False)
                  for row in (whole, resumed, outside)]
        for state in states[1:]:
            for name, value in states[0]["model"].items():
                self.assertTrue(torch.equal(value, state["model"][name]), name)
            for left, right in zip(states[0]["history"], state["history"]):
                self.assertEqual({k: v for k, v in left.items() if k != "seconds"},
                                 {k: v for k, v in right.items() if k != "seconds"})
        self.assertEqual(whole["best_epoch"], outside["best_epoch"])
        identity_whole = json.loads((Path(whole["root"]) / "identity.json").read_text())
        identity_outside = json.loads((Path(outside["root"]) / "identity.json").read_text())
        self.assertEqual(identity_whole["normalization"], identity_outside["normalization"])
        records = json.loads((Path(outside["root"]) / "outer_predictions.json").read_text())
        self.assertTrue(all(row["target"] == float(np.float32(-0.1)) for row in records))
        self.assertTrue(all(np.isfinite(row["CRPS_48"]) for row in records))

    def test_extreme_mdn_edge_bins_and_boundary_label_roundoff(self):
        anchor = torch.tensor([-0.1, -0.7, 0.7], dtype=torch.float32)
        raw_scale = torch.full((3,), -100.0, dtype=torch.float32)
        logits = torch.full((3, 15), -1000.0, dtype=torch.float32)
        logits[:, 0] = 1000.0
        output = {
            "kind": "mdn",
            "parameters": shape_controls.mdn_parameters(anchor, raw_scale, logits, recenter=True),
        }
        raw_probabilities = crps_training._common_bin_probabilities(output).detach().cpu().numpy()
        repaired, audit = crps_training._repair_bin_probabilities(raw_probabilities)
        self.assertTrue(np.isfinite(repaired).all())
        np.testing.assert_allclose(repaired.sum(1), 1.0, atol=1e-15, rtol=0)
        self.assertLessEqual(audit["maximum_absolute_correction"], 1e-10)

        target = torch.tensor([-0.1, 0.1, 0.7], dtype=torch.float32)
        for nodes in (48, 96, 192):
            score = crps_training.cdf_crps.per_sample(output, target, nodes=nodes)
            self.assertEqual(score.dtype, torch.float64)
            self.assertTrue(torch.isfinite(score).all())

        noisy_cdf = torch.tensor([[0.0, .2, .4, .6, 1 + 5e-13, 1.0]], dtype=torch.float64)
        _, correction = crps_training._bins_from_cdf_values(noisy_cdf)
        self.assertAlmostEqual(correction, 5e-13, delta=1e-15)
        invalid_cdf = noisy_cdf.clone()
        invalid_cdf[0, 4] = 1.1
        with self.assertRaisesRegex(ValueError, "CDF values exceed"):
            crps_training._bins_from_cdf_values(invalid_cdf)

        simulated_roundoff = raw_probabilities.copy()
        simulated_roundoff[0, 0] -= 5e-13
        simulated_roundoff[0, 1] += 5e-13
        clipped, clip_audit = crps_training._repair_bin_probabilities(simulated_roundoff)
        self.assertGreaterEqual(clipped.min(), 0.0)
        self.assertAlmostEqual(clip_audit["minimum_before_repair"], -5e-13, delta=1e-15)
        self.assertLessEqual(clip_audit["maximum_absolute_correction"], 1e-10)
        too_negative = simulated_roundoff.copy()
        too_negative[0, 0] = -2e-12
        with self.assertRaisesRegex(ValueError, "roundoff repair bound"):
            crps_training._repair_bin_probabilities(too_negative)

    def test_smoke_guard_and_point_table_hash_validation(self):
        smoke_config = {**self.config, 'point_suite':str(self.suite)}
        crps_training._validate_training_contract(smoke_config, True)
        with self.assertRaisesRegex(ValueError, 'point_suite'):
            crps_training._validate_training_contract(smoke_config, False)
        with self.assertRaises(ValueError):
            load_stage(self.suite, 13, 0)
        suite = json.loads(self.suite.read_text())
        table_path = Path(suite["results"][0]["tables"]["shape_fit"]["path"])
        original = table_path.read_bytes()
        try:
            table_path.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "table changed"):
                load_stage(self.suite, 13, 0, allow_smoke=True)
        finally:
            table_path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
