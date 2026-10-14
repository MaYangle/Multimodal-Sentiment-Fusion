import unittest

import numpy as np
import torch

from jevaffect.source_crossfit import make_folds
from scripts.audit_source_robust import (
    compare_point_tensors,
    _compare_optimizer_point_state,
    compare_prediction_rows,
    merge_oof_records,
    paired_source_bootstrap,
    source_diagnostics,
    validate_epoch_accounting,
    replay_scheduler,
)


class SourceRobustAuditTests(unittest.TestCase):
    def test_scheduler_thresholded_best_differs_from_selected_minimum(self):
        config={"learning_rate":.0001,"weight_decay":.0001,"scheduler_patience":3,"scheduler_factor":.5}
        history=[{"inner_select":{"MAE":value},"learning_rate":.0001} for value in (.5,.49998)]
        state=replay_scheduler(history,config)
        self.assertEqual(state["best"],.5)
        self.assertNotEqual(state["best"],min(r["inner_select"]["MAE"] for r in history))
        history[-1]["learning_rate"]*=.5
        with self.assertRaises(ValueError):replay_scheduler(history,config)

    def setUp(self):
        self.ids = [f"video{i:02d}/{clip:02d}" for i in range(15) for clip in range(2)]
        self.manifest = make_folds(self.ids, fold_count=3, seed=20261012, inner_fraction=.1)
        self.targets = {sid: float(np.float32((index % 7 - 3) / 5)) for index, sid in enumerate(self.ids)}
        self.fold_records = {
            fold["fold"]: [
                {"id": sid, "target": self.targets[sid], "prediction": float(np.float32(index / 20))}
                for index, sid in enumerate(fold["oof_ids"])
            ]
            for fold in self.manifest["folds"]
        }

    def test_source_oof_merge_checks_exact_once_and_float32_targets(self):
        records = merge_oof_records(self.fold_records, self.manifest["folds"], self.ids, self.targets)
        self.assertEqual([row["id"] for row in records], self.ids)
        self.assertEqual(len(records), len(self.ids))

        duplicate = {fold: list(rows) for fold, rows in self.fold_records.items()}
        duplicate[0] = duplicate[0] + [dict(duplicate[0][0])]
        with self.assertRaises(ValueError):
            merge_oof_records(duplicate, self.manifest["folds"], self.ids, self.targets)

        missing = {fold: list(rows) for fold, rows in self.fold_records.items()}
        missing[1].pop()
        with self.assertRaises(ValueError):
            merge_oof_records(missing, self.manifest["folds"], self.ids, self.targets)

        wrong_target = {fold: [dict(row) for row in rows] for fold, rows in self.fold_records.items()}
        wrong_target[2][0]["target"] = float(np.float32(wrong_target[2][0]["target"] + .2))
        with self.assertRaises(ValueError):
            merge_oof_records(wrong_target, self.manifest["folds"], self.ids, self.targets)

    def test_epoch_budget_sampling_q_and_dann_schedule(self):
        q = [1 / 4] * 4
        fit_counts = [5, 7, 3, 6]
        rows = [{
            "epoch": epoch,
            "optimizer_updates": 2,
            "sampled_source_counts": fit_counts,
            "source_probability": q,
            "domain_CE": .2,
            "reversal_scale_end": .1 * (2 / (1 + np.exp(-10 * (((epoch - 1) * 2 + 1) / (80 * 2 - 1)))) - 1),
            "train_clip_L1": .4,
            "risk_objective": .4,
            "learning_rate": .0001,
            "seconds": .1,
        } for epoch in (1, 2)]
        self.assertEqual(validate_epoch_accounting(rows, fit_count=21, batch_size=16, case="dann",
                                                  source_counts=fit_counts, max_epochs=80, patience=12), 2)

        bad_counts = [{**rows[0], "sampled_source_counts": [5, 7, 3, 5]}, rows[1]]
        with self.assertRaises(ValueError):
            validate_epoch_accounting(bad_counts, 21, 16, "dann", fit_counts, 80, 12)

        bad_fixed_q = [{**rows[0], "source_probability": [.1, .2, .3, .4]}, rows[1]]
        with self.assertRaises(ValueError):
            validate_epoch_accounting(bad_fixed_q, 21, 16, "balanced_erm", fit_counts, 80, 12)

    def test_clipped_source_within_mse_decomposition_and_source_equal_mae(self):
        records = []
        for source, errors in (("a", [.2, .2, .2, .2, .2]),
                               ("b", [-.1, .1, 0., .1, -.1]),
                               ("c", [.4, .4])):
            for index, error in enumerate(errors):
                target = 0.0
                records.append({"id": f"{source}/{index}", "target": target,
                                "prediction": error})
        diagnostic = source_diagnostics(records)
        self.assertEqual(diagnostic["source_count"], 3)
        self.assertAlmostEqual(diagnostic["source_equal_mean_MAE"], (0.2 + .08 + .4) / 3)
        self.assertEqual(diagnostic["worst_eligible_source"], "a")
        self.assertEqual(diagnostic["worst_source_clip_count"], 5)
        self.assertAlmostEqual(diagnostic["clipped_error_total_MSE"],
                               diagnostic["clipped_error_source_between_MSE"]
                               + diagnostic["clipped_error_within_source_MSE"], places=12)

    def test_paired_bootstrap_is_fixed_and_shared_across_seeds(self):
        ids = [f"src{i}/{clip}" for i in range(5) for clip in range(2)]
        deltas = {
            13: np.array([-.2, -.1, .1, .2, -.1, -.2, .2, .1, -.3, .1]),
            42: np.array([-.1, -.1, .2, .1, -.2, -.1, .1, .2, -.1, .2]),
            2026: np.array([-.3, -.2, .1, .1, -.1, -.1, .2, .2, -.2, .1]),
        }
        one = paired_source_bootstrap(deltas, ids, replicates=2000, random_seed=20261013)
        two = paired_source_bootstrap(deltas, ids, replicates=2000, random_seed=20261013)
        self.assertEqual(one, two)
        self.assertEqual(one["source_count"], 5)
        self.assertTrue(one["three_seed_same_source_average"]["shared_source_draws_across_seeds"])
        self.assertEqual(one["three_seed_same_source_average"]["replicates"], 2000)
        self.assertTrue(one["three_seed_same_source_average"]["ci95"][0]
                        <= one["three_seed_same_source_average"]["estimate"]
                        <= one["three_seed_same_source_average"]["ci95"][1])
        with self.assertRaises(ValueError):
            paired_source_bootstrap(deltas, ids, replicates=1999)

    def test_source_point_state_prefix_and_prediction_parity_helpers(self):
        reference = {"head.0.weight": torch.tensor([[1., 2.]]),
                     "head.0.bias": torch.tensor([.1])}
        wrapped = {f"regressor.{key}": value.clone() for key, value in reference.items()}
        wrapped["domain_classifier.weight"] = torch.ones(2, 2)
        state = compare_point_tensors(wrapped, reference)
        self.assertTrue(state["exact"])
        second_wrapped = {key: value.clone() for key, value in wrapped.items()}
        second_wrapped["domain_classifier.weight"] *= 2
        self.assertTrue(compare_point_tensors(wrapped,second_wrapped)["exact"])
        wrapped["regressor.head.0.bias"][0] += .01
        with self.assertRaises(ValueError):
            compare_point_tensors(wrapped, reference)

        rows = [{"id": "a/1", "target": .2, "prediction": -.4}]
        self.assertTrue(compare_prediction_rows(rows, [dict(rows[0])], exact=True)["exact"])
        with self.assertRaises(ValueError):
            compare_prediction_rows(rows, [{"id": "a/1", "target": .2, "prediction": -.3}], exact=True)

        source_last = {"optimizer": {"state": {0: {"step": torch.tensor(2.),
                                                   "exp_avg": torch.tensor([.1, .2])}}}}
        reference_last = {"optimizer": {"state": {0: {"step": torch.tensor(2.),
                                                      "exp_avg": torch.tensor([.1, .2])}}}}
        _compare_optimizer_point_state(source_last, reference_last, 1)
        reference_last["optimizer"]["state"][0]["exp_avg"][0] += .01
        with self.assertRaises(ValueError):
            _compare_optimizer_point_state(source_last, reference_last, 1)


if __name__ == "__main__":
    unittest.main()
