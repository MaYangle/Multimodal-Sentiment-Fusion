import unittest

import numpy as np

from scripts.audit_anchor_calibration import (
    _compare_payload,
    _verify_probabilities_and_point,
    expected_head_split,
    first_minimum_epoch,
    paired_source_cluster_bootstrap,
    summarize_registered_gates,
    validate_head_split,
)


class AnchorCalibrationAuditTests(unittest.TestCase):
    def test_overlapping_interval_does_not_establish_noninferiority(self):
        summary = {"by_seed": {str(s): {"estimate": -.01, "ci95": [-.1, .1]} for s in (13,42,2026)},
                   "three_seed_same_source_average": {"estimate": -.01, "ci95": [-.1,.1]}}
        contrasts = {c: {m: summary for m in ("bin_log_score","hit_Brier","hit_log_score")}
                     for c in ("mixture","views","scale","mdn","direct_hit")}
        gates = summarize_registered_gates({"comparisons": contrasts}, True)
        self.assertIsNone(gates["gates"]["mdn_competitiveness"]["pass"])
        self.assertTrue(gates["gates"]["mdn_competitiveness"]["no_clear_disadvantage_detected"])
        self.assertFalse(gates["gates"]["direct_hit_competitiveness"]["pass"])
        self.assertFalse(gates["all_score_gates_pass"])

    def test_registered_source_selection_is_fixed_and_disjoint(self):
        ids = [f"source{i:03d}/{clip:02d}" for i in range(129) for clip in range(2)]
        split = expected_head_split(ids)
        self.assertEqual(split, expected_head_split(ids))
        self.assertEqual(split["seed"], 20261013)
        self.assertEqual(split["select_source_count"], 13)
        fit, select = validate_head_split(ids, split)
        self.assertEqual(fit, split["fit_indices"])
        self.assertEqual(select, split["select_indices"])
        self.assertFalse(set(split["fit_sources"]) & set(split["select_sources"]))

        damaged = dict(split)
        damaged["select_sources"] = damaged["fit_sources"][:1]
        with self.assertRaises(ValueError):
            validate_head_split(ids, damaged)

    def test_selection_rule_keeps_the_first_exact_tie(self):
        history = [
            {"epoch": 1, "selection_objective": .3},
            {"epoch": 2, "selection_objective": .2},
            {"epoch": 3, "selection_objective": .2},
        ]
        self.assertEqual(first_minimum_epoch(history), (2, .2))
        with self.assertRaises(ValueError):
            first_minimum_epoch([{"epoch": 1, "selection_objective": float("nan")}])

    def test_prediction_comparison_checks_every_nested_value(self):
        _compare_payload(
            {"bins": [.1, .2], "interval": {"0.5": [-.1, .1]}},
            {"bins": [.1 + 1e-9, .2], "interval": {"0.5": [-.1, .1]}},
            tolerance=1e-7,
        )
        with self.assertRaises(ValueError):
            _compare_payload({"p_hit": .8}, {"p_hit": .7}, tolerance=1e-7)

    def test_source_cluster_bootstrap_is_paired_fixed_and_reproducible(self):
        ids = [f"source{i}/{clip}" for i in range(4) for clip in range(2)]
        ids_by_seed = {seed: list(ids) for seed in (13, 42, 2026)}
        differences = {
            13: np.array([-.2, -.1, .1, .2, -.1, -.2, .2, .1]),
            42: np.array([-.1, -.1, .2, .1, -.2, -.1, .1, .2]),
            2026: np.array([-.3, -.2, .1, .1, -.1, -.1, .2, .2]),
        }
        first = paired_source_cluster_bootstrap(differences, ids_by_seed, reps=2000, random_seed=17)
        second = paired_source_cluster_bootstrap(differences, ids_by_seed, reps=2000, random_seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["three_seed_same_source_average"]["resamples"], 2000)
        self.assertTrue(first["three_seed_same_source_average"]["shared_source_resamples_across_seeds"])
        self.assertEqual(first["three_seed_same_source_average"]["resampled_sources"], 4)
        self.assertTrue(first["three_seed_same_source_average"]["ci95"][0]
                        <= first["three_seed_same_source_average"]["estimate"]
                        <= first["three_seed_same_source_average"]["ci95"][1])
        with self.assertRaises(ValueError):
            paired_source_cluster_bootstrap(differences, ids_by_seed, reps=1999)

    def test_mdn_median_may_differ_while_original_point_stays_fixed(self):
        anchor = np.array([-.8, -.2, .4, .9], np.float32)
        mdn_records = [
            {"prediction": float(m), "median": float(m + .03), "p_hit": .5,
             "bin_probabilities": [.1, .2, .3, .2, .2], "scale": .25,
             "intervals": {level: [-.1, .1] for level in ("0.5", "0.8", "0.9", "0.95")}}
            for m in anchor
        ]
        result = _verify_probabilities_and_point("mdn", mdn_records, anchor)
        self.assertTrue(result["mdn_median_may_differ"])
        self.assertLessEqual(result["max_point_delta"], 1e-6)
        bad_median_case = [dict(row) for row in mdn_records]
        with self.assertRaises(ValueError):
            _verify_probabilities_and_point("mixture", bad_median_case, anchor)


if __name__ == "__main__":
    unittest.main()
