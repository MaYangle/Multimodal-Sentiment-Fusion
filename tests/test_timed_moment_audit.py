import unittest

import numpy as np
import torch

from scripts.audit_timed_moments import (
    CASES,
    EXPECTED_DIMS,
    Recorder,
    _source_draws,
    _validate_stats_and_normalize,
    native_metrics,
    numpy_forward,
    paired_source_delta,
    torch_cpu_forward,
)


class TimedMomentAuditTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(712)
        self.features = {
            "text": rng.normal(size=(3, 5)).astype(np.float32),
            "audio_mean": rng.normal(size=(3, 7)).astype(np.float32),
            "vision_mean": rng.normal(size=(3, 3)).astype(np.float32),
            "audio_grid": rng.normal(size=(3, 4, 7)).astype(np.float32),
            "vision_grid": rng.normal(size=(3, 4, 3)).astype(np.float32),
            "pair_valid": np.asarray([True, True, False]),
        }
        self.weights = {
            "audio_projection.weight": rng.normal(size=(2, 7)).astype(np.float32),
            "vision_projection.weight": rng.normal(size=(2, 3)).astype(np.float32),
            "head.0.weight": rng.normal(size=(6, 17)).astype(np.float32),
            "head.0.bias": rng.normal(size=(6,)).astype(np.float32),
            "head.3.weight": rng.normal(size=(1, 6)).astype(np.float32),
            "head.3.bias": rng.normal(size=(1,)).astype(np.float32),
        }

    def test_numpy_and_torch_cpu_replay_all_moment_cases(self):
        torch_weights = {key: torch.from_numpy(value.copy()) for key, value in self.weights.items()}
        for case in CASES:
            with self.subTest(case=case):
                numpy_prediction = numpy_forward(self.features, self.weights, case)
                torch_prediction = torch_cpu_forward(self.features, torch_weights, case)
                np.testing.assert_allclose(numpy_prediction, torch_prediction, atol=1e-5, rtol=0)
                self.assertEqual(numpy_prediction.shape, (3,))
                self.assertTrue(np.isfinite(numpy_prediction).all())

    def test_common_pair_permutation_preserves_paired_covariance(self):
        original = numpy_forward(self.features, self.weights, "paired_cov")
        changed = dict(self.features)
        changed["audio_grid"] = np.roll(self.features["audio_grid"], 1, axis=1)
        changed["vision_grid"] = np.roll(self.features["vision_grid"], 1, axis=1)
        common = numpy_forward(changed, self.weights, "paired_cov")
        np.testing.assert_allclose(common, original, atol=1e-5, rtol=0)

    def test_fixed_vision_roll_breaks_covariance_but_preserves_marginal_variance(self):
        changed = dict(self.features)
        changed["vision_grid"] = np.roll(self.features["vision_grid"], 1, axis=1)
        paired = numpy_forward(self.features, self.weights, "paired_cov")
        shifted = numpy_forward(changed, self.weights, "paired_cov")
        self.assertGreater(float(np.max(np.abs(paired - shifted))), 1e-5)
        marginal = numpy_forward(self.features, self.weights, "marginal_variance")
        marginal_shifted = numpy_forward(changed, self.weights, "marginal_variance")
        np.testing.assert_allclose(marginal, marginal_shifted, atol=1e-5, rtol=0)

    def test_train_fit_statistics_are_independent_and_invalid_rows_are_zeroed(self):
        rng = np.random.default_rng(18)
        n = 6
        raw = {
            "text": rng.normal(size=(n, EXPECTED_DIMS["text"])).astype(np.float32),
            "audio_mean": rng.normal(size=(n, EXPECTED_DIMS["audio"])).astype(np.float32),
            "vision_mean": rng.normal(size=(n, EXPECTED_DIMS["vision"])).astype(np.float32),
            "audio_grid": rng.normal(size=(n, 4, EXPECTED_DIMS["audio"])).astype(np.float32),
            "vision_grid": rng.normal(size=(n, 4, EXPECTED_DIMS["vision"])).astype(np.float32),
        }
        presence = np.ones((n, 3), dtype=bool)
        presence[2, 0] = False
        pair_valid = np.asarray([True, True, False, True, False, True])
        stats = {}
        for modality, raw_name in (("text", "text"), ("audio", "audio_mean"), ("vision", "vision_mean")):
            values = raw[raw_name].astype(np.float64)
            mean = values.mean(axis=0).astype(np.float32)
            scale = values.std(axis=0)
            scale[scale < 1e-6] = 1
            stats[modality] = {"mean": mean.tolist(), "scale": scale.astype(np.float32).tolist(),
                               "fit_indices": list(range(n))}
        rec = Recorder()
        normalized = _validate_stats_and_normalize(raw, presence, pair_valid, stats,
                                                   rec=rec, context="fixture", verify_fit=True)
        self.assertFalse(rec.findings)
        self.assertEqual(float(np.abs(normalized["text"][2]).sum()), 0.0)
        self.assertEqual(float(np.abs(normalized["audio_grid"][2]).sum()), 0.0)
        self.assertEqual(float(np.abs(normalized["vision_grid"][4]).sum()), 0.0)
        self.assertTrue(np.isfinite(normalized["audio_grid"]).all())

    def test_native_binary_metric_keeps_neutral_on_negative_side(self):
        metrics = native_metrics(np.asarray([0.0, 0.0]), np.asarray([1.0, -1.0]))
        self.assertEqual(metrics["Acc2_positive_strict"], 0.5)

    def test_paired_source_bootstrap_is_reproducible_and_clustered(self):
        ids = ["video_a/1", "video_a/2", "video_b/1", "video_c/1"]
        target = np.asarray([0.0, 0.2, -0.1, 0.5], np.float32)
        candidate = np.asarray([0.1, 0.1, -0.2, 0.3], np.float32)
        reference = np.asarray([0.3, 0.0, 0.2, 0.5], np.float32)
        source_names, draws = _source_draws(ids, seed=19, replicates=250)
        self.assertEqual(len(source_names), 3)
        one = paired_source_delta(candidate, reference, target, ids, source_names, draws)
        two = paired_source_delta(candidate, reference, target, ids, source_names, draws)
        self.assertEqual(one, two)
        self.assertEqual(one["replicates"], 250)
        self.assertEqual(one["source_count"], 3)
        self.assertEqual(len(one["paired_source_bootstrap_95_ci"]), 2)


if __name__ == "__main__":
    unittest.main()
