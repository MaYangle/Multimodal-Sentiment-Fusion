"""Tests for fixed-five JevAffect metrics, implemented with NumPy only."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevaffect.metrics import (  # noqa: E402
    CLASS_NAMES,
    MetricsError,
    accuracy,
    brier_score,
    classification_metrics,
    confusion_matrix,
    fit_temperature_on_validation,
    language_accuracy_summary,
    macro_f1,
    negative_log_likelihood,
    per_class_metrics,
    temperature_scaled_probabilities,
    top_label_ece_15,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_known_confusion_accuracy_and_per_class_values(self):
        y_true = np.array([0, 1, 2, 3, 4, 4], dtype=np.int64)
        y_pred = np.array([0, 2, 2, 3, 3, 4], dtype=np.int64)
        expected = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=np.int64,
        )

        np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)
        self.assertAlmostEqual(accuracy(y_true, y_pred), 4.0 / 6.0)
        self.assertAlmostEqual(macro_f1(y_true, y_pred), 0.6)

        classes = per_class_metrics(y_true, y_pred)
        self.assertEqual(len(classes), 5)
        self.assertEqual(tuple(row["class_name"] for row in classes), CLASS_NAMES)
        self.assertEqual(tuple(row["support"] for row in classes), (1, 1, 1, 1, 2))
        self.assertAlmostEqual(classes[0]["f1"], 1.0)
        self.assertAlmostEqual(classes[1]["f1"], 0.0)
        self.assertAlmostEqual(classes[2]["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(classes[3]["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(classes[4]["f1"], 2.0 / 3.0)

        report = classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(report["accuracy"], 4.0 / 6.0)
        self.assertAlmostEqual(report["macro_f1"], 0.6)
        self.assertEqual(report["n_samples"], 6)
        np.testing.assert_array_equal(report["confusion_matrix"], expected)

    def test_macro_f1_always_averages_all_five_classes(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        self.assertAlmostEqual(macro_f1(y_true, y_pred), 0.2)
        classes = per_class_metrics(y_true, y_pred)
        self.assertEqual(len(classes), 5)
        self.assertEqual(classes[0]["support"], 3)
        self.assertEqual(classes[0]["f1"], 1.0)
        for row in classes[1:]:
            self.assertEqual(row["support"], 0)
            self.assertEqual(row["f1"], 0.0)

    def test_classification_inputs_are_strictly_validated(self):
        invalid_pairs = (
            ([], []),
            ([0, 1], [0]),
            ([[0, 1]], [0]),
            ([0.0, 1.0], [0, 1]),
            ([True], [0]),
            ([5], [0]),
            ([-1], [0]),
            ([0], [5]),
            ([0], [math.nan]),
            ([0], [math.inf]),
        )
        for y_true, y_pred in invalid_pairs:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                with self.assertRaises(MetricsError):
                    accuracy(y_true, y_pred)


class ProbabilityMetricTests(unittest.TestCase):
    def test_perfect_probabilities_have_zero_nll_brier_and_ece(self):
        y_true = np.arange(5, dtype=np.int64)
        probabilities = np.eye(5, dtype=np.float64)
        self.assertAlmostEqual(negative_log_likelihood(y_true, probabilities), 0.0)
        self.assertAlmostEqual(brier_score(y_true, probabilities), 0.0)
        # Confidence=1.0 must be assigned to the final bin, not dropped.
        self.assertAlmostEqual(top_label_ece_15(y_true, probabilities), 0.0)

    def test_uniform_probabilities_have_known_losses_and_zero_top_label_ece(self):
        y_true = np.arange(5, dtype=np.int64)
        probabilities = np.full((5, 5), 0.2, dtype=np.float64)
        self.assertAlmostEqual(
            negative_log_likelihood(y_true, probabilities), math.log(5.0)
        )
        self.assertAlmostEqual(brier_score(y_true, probabilities), 0.8)
        # np.argmax resolves the uniform tie to class 0; one of five labels is 0.
        self.assertAlmostEqual(top_label_ece_15(y_true, probabilities), 0.0)

    def test_ece_includes_confidence_one_in_the_last_bin(self):
        y_true = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        probabilities = np.zeros((5, 5), dtype=np.float64)
        probabilities[:, 0] = 1.0
        self.assertAlmostEqual(top_label_ece_15(y_true, probabilities), 1.0)

    def test_probability_metrics_reject_empty_nonfinite_or_malformed_inputs(self):
        invalid = (
            ([], np.empty((0, 5))),
            ([0], [0.2, 0.2, 0.2, 0.2, 0.2]),
            ([0], np.zeros((1, 4))),
            ([0], np.zeros((1, 5, 1))),
            ([0], np.array([[math.nan, 0.25, 0.25, 0.25, 0.25]])),
            ([0], np.array([[math.inf, 0.0, 0.0, 0.0, 0.0]])),
            ([0], np.array([[-0.1, 0.3, 0.3, 0.3, 0.2]])),
            ([0], np.array([[1.1, 0.0, 0.0, 0.0, 0.0]])),
            ([0], np.array([[0.1, 0.2, 0.2, 0.2, 0.2]])),
            ([0], np.array([["0.2", "0.2", "0.2", "0.2", "0.2"]])),
            ([5], np.full((1, 5), 0.2)),
        )
        for labels, probabilities in invalid:
            for metric in (
                negative_log_likelihood,
                brier_score,
                top_label_ece_15,
            ):
                with self.subTest(labels=labels, probabilities=probabilities, metric=metric.__name__):
                    with self.assertRaises(MetricsError):
                        metric(labels, probabilities)


class LanguageAggregationTests(unittest.TestCase):
    def test_reports_per_language_overall_and_equal_language_accuracy(self):
        y_true = [0, 0, 0, 1]
        y_pred = [0, 0, 1, 1]
        languages = ["en", "en", "en", "zh"]
        result = language_accuracy_summary(y_true, y_pred, languages)
        self.assertAlmostEqual(result["overall_accuracy"], 0.75)
        self.assertAlmostEqual(result["per_language"]["en"]["accuracy"], 2.0 / 3.0)
        self.assertEqual(result["per_language"]["en"]["count"], 3)
        self.assertAlmostEqual(result["per_language"]["zh"]["accuracy"], 1.0)
        self.assertEqual(result["per_language"]["zh"]["count"], 1)
        self.assertAlmostEqual(result["equal_language_accuracy"], 5.0 / 6.0)

    def test_rejects_empty_missing_unexpected_or_malformed_language_subsets(self):
        cases = (
            ([0], [0], ["en"]),
            ([0, 1], [0, 1], ["en"]),
            ([0, 1], [0, 1], ["en", "fr"]),
            ([0, 1], [0, 1], ["en", "zh", "zh"]),
            ([0, 1], [0, 1], ["en", ""]),
            ([0, 1], [0, 1], [["en", "zh"]]),
        )
        for y_true, y_pred, languages in cases:
            with self.subTest(languages=languages):
                with self.assertRaises(MetricsError):
                    language_accuracy_summary(y_true, y_pred, languages)
        with self.assertRaises(MetricsError):
            language_accuracy_summary([0, 1], [0, 1], ["en", "zh"], language_order=None)


class TemperatureScalingTests(unittest.TestCase):
    def test_validation_log_grid_fit_returns_nll_and_preserves_argmax(self):
        labels = np.arange(5, dtype=np.int64)
        logits = np.zeros((5, 5), dtype=np.float64)
        logits[np.arange(5), labels] = 0.4  # Correct but deliberately under-confident.

        fit = fit_temperature_on_validation(
            logits,
            labels,
            min_temperature=0.05,
            max_temperature=20.0,
            grid_size=81,
        )
        self.assertEqual(set(fit), {"temperature", "before_nll", "after_nll"})
        self.assertGreaterEqual(fit["temperature"], 0.05)
        self.assertLessEqual(fit["temperature"], 20.0)
        self.assertGreater(fit["before_nll"], fit["after_nll"])
        self.assertGreaterEqual(fit["before_nll"], 0.0)
        self.assertGreaterEqual(fit["after_nll"], 0.0)

        before = np.argmax(logits, axis=1)
        after_probs = temperature_scaled_probabilities(logits, fit["temperature"])
        after = np.argmax(after_probs, axis=1)
        np.testing.assert_array_equal(before, after)

    def test_temperature_scaling_is_stable_and_rejects_invalid_temperature(self):
        logits = np.array([[1000.0, 999.0, 0.0, -2.0, -4.0]])
        probabilities = temperature_scaled_probabilities(logits, 2.0)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        for temperature in (0.0, -1.0, math.nan, math.inf, True):
            with self.subTest(temperature=temperature):
                with self.assertRaises(MetricsError):
                    temperature_scaled_probabilities(logits, temperature)

    def test_temperature_fit_rejects_invalid_inputs_and_bounds(self):
        labels = np.array([0, 1], dtype=np.int64)
        logits = np.zeros((2, 5), dtype=np.float64)
        invalid_cases = (
            (np.zeros((0, 5)), np.array([], dtype=np.int64)),
            (np.zeros((2, 4)), labels),
            (np.array([[math.nan] * 5, [0.0] * 5]), labels),
            (logits, np.array([0.0, 1.0])),
        )
        for values, target in invalid_cases:
            with self.subTest(values=values, target=target):
                with self.assertRaises(MetricsError):
                    fit_temperature_on_validation(values, target)

        with self.assertRaises(MetricsError):
            fit_temperature_on_validation(logits, labels, min_temperature=2, max_temperature=1)
        for grid_size in (1, 2.5, True):
            with self.subTest(grid_size=grid_size):
                with self.assertRaises(MetricsError):
                    fit_temperature_on_validation(logits, labels, grid_size=grid_size)


if __name__ == "__main__":
    unittest.main()
