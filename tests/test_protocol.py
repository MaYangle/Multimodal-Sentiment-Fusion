"""Standard-library tests for JevAffect's competition protocol module."""

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevaffect.protocol import (  # noqa: E402
    CLASSES,
    NEUT,
    ProtocolError,
    SIMS_V2_SCORE_TO_CLASS,
    SPOS,
    SNEG,
    WNEG,
    WPOS,
    export_predictions,
    make_prediction_record,
    map_mosei_sentiment,
    map_simsv2_label,
    validate_probability_vector,
)


class MoseiMappingTests(unittest.TestCase):
    def test_all_boundary_points_follow_the_published_possible_mapping(self):
        cases = (
            (-3.0, SNEG),
            (-2.0, SNEG),
            (-1.999999, WNEG),
            (-0.5, WNEG),
            (-0.499999, NEUT),
            (0.0, NEUT),
            (0.499999, NEUT),
            (0.5, WPOS),
            (1.999999, WPOS),
            (2.0, SPOS),
            (3.0, SPOS),
        )
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(map_mosei_sentiment(score), expected)

    def test_rejects_out_of_range_and_non_finite_or_non_numeric_values(self):
        for score in (-3.000001, 3.000001, math.nan, math.inf, -math.inf, True, "-1.0"):
            with self.subTest(score=score):
                with self.assertRaises(ProtocolError):
                    map_mosei_sentiment(score)


class SimsV2MappingTests(unittest.TestCase):
    def test_every_allowed_discrete_label_is_mapped(self):
        for value, expected in SIMS_V2_SCORE_TO_CLASS.items():
            with self.subTest(value=value):
                self.assertEqual(map_simsv2_label(value), expected)

    def test_accepts_small_float_representation_drift(self):
        self.assertEqual(map_simsv2_label(-0.8000005), SNEG)
        self.assertEqual(map_simsv2_label(0.6000005), WPOS)
        self.assertEqual(map_simsv2_label(-0.0000005), NEUT)

    def test_rejects_values_not_in_the_competition_discrete_set(self):
        for value in (-1.1, -0.7, -0.1, 0.1, 0.7, 1.1, math.nan, math.inf, True, "0.2"):
            with self.subTest(value=value):
                with self.assertRaises(ProtocolError):
                    map_simsv2_label(value)
        with self.assertRaises(ProtocolError):
            map_simsv2_label(-0.80001)

    def test_tolerance_must_not_make_adjacent_bins_ambiguous(self):
        for tolerance in (-1.0, 0.1, math.inf, math.nan, True):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ProtocolError):
                    map_simsv2_label(0.0, tolerance=tolerance)


class ProbabilityAndRecordTests(unittest.TestCase):
    def test_valid_vector_is_normalized_only_for_small_sum_drift(self):
        probabilities = validate_probability_vector((0.1, 0.2, 0.3, 0.15, 0.2500005))
        self.assertAlmostEqual(math.fsum(probabilities), 1.0)
        self.assertAlmostEqual(probabilities[0], 0.1 / 1.0000005)

    def test_rejects_invalid_vectors(self):
        invalid = (
            (0.1, 0.2, 0.3, 0.2),
            (0.1, 0.2, 0.3, 0.2, 0.3, 0.1),
            (-0.1, 0.3, 0.3, 0.3, 0.2),
            (0.1, 0.2, 0.3, 0.4, 0.1),
            (math.nan, 0.2, 0.3, 0.2, 0.3),
            (math.inf, 0.0, 0.0, 0.0, 0.0),
            (True, 0.2, 0.3, 0.2, 0.3),
            ("0.1", 0.2, 0.3, 0.2, 0.2),
        )
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ProtocolError):
                    validate_probability_vector(probabilities)

    def test_record_contains_argmax_all_scores_and_named_choice_confidence(self):
        record = make_prediction_record(
            "segment_007.mp4", (0.05, 0.1, 0.15, 0.5, 0.2)
        )
        self.assertEqual(record["ID"], "segment_007.mp4")
        self.assertEqual(record["Label"], WPOS)
        self.assertEqual(record["argmax"], WPOS)
        self.assertEqual(tuple(record["probabilities"]), CLASSES)
        self.assertAlmostEqual(record["max_probability"], 0.5)
        self.assertAlmostEqual(record["choice_confidence"], (0.5 - 0.2) / 0.8)
        self.assertNotIn("accuracy", record)

    def test_ties_choose_the_first_class_in_documented_class_order(self):
        record = make_prediction_record("tie.mp4", (0.1, 0.4, 0.1, 0.4, 0.0))
        self.assertEqual(record["argmax"], WNEG)


class ExportTests(unittest.TestCase):
    def test_orders_by_expected_ids_preserves_full_filenames_and_keeps_low_scores(self):
        records = [
            make_prediction_record("clip_02.mp4", (0.05, 0.05, 0.05, 0.8, 0.05)),
            make_prediction_record("clip_01.mp4", (0.1, 0.3, 0.2, 0.2, 0.2)),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "prediction.csv"
            jsonl_path = Path(temp_dir) / "low_confidence.jsonl"
            result = export_predictions(
                records,
                ("clip_01.mp4", "clip_02.mp4"),
                csv_path,
                jsonl_path,
                low_confidence_threshold=0.2,
            )

            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["ID", "Label"])
                rows = list(reader)
            self.assertEqual([row["ID"] for row in rows], ["clip_01.mp4", "clip_02.mp4"])
            self.assertEqual([row["Label"] for row in rows], [WNEG, WPOS])
            self.assertEqual(result, {"predictions_written": 2, "low_confidence_written": 1})

            low_records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["ID"] for record in low_records], ["clip_01.mp4"])
            self.assertEqual(low_records[0]["Label"], WNEG)
            self.assertLess(low_records[0]["choice_confidence"], 0.2)

    def test_duplicate_prediction_ids_fail_before_files_are_created(self):
        record = make_prediction_record("duplicate.mp4", (0.2, 0.2, 0.2, 0.2, 0.2))
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "prediction.csv"
            jsonl_path = Path(temp_dir) / "low.jsonl"
            with self.assertRaises(ProtocolError):
                export_predictions(
                    (record, record),
                    ("duplicate.mp4",),
                    csv_path,
                    jsonl_path,
                    low_confidence_threshold=0.0,
                )
            self.assertFalse(csv_path.exists())
            self.assertFalse(jsonl_path.exists())

    def test_missing_or_unexpected_ids_fail_before_files_are_created(self):
        record = make_prediction_record("known.mp4", (0.2, 0.2, 0.2, 0.2, 0.2))
        cases = (
            ((record,), ("known.mp4", "missing.mp4")),
            ((record,), ("other.mp4",)),
        )
        for records, expected_ids in cases:
            with self.subTest(expected_ids=expected_ids), tempfile.TemporaryDirectory() as temp_dir:
                csv_path = Path(temp_dir) / "prediction.csv"
                jsonl_path = Path(temp_dir) / "low.jsonl"
                with self.assertRaises(ProtocolError):
                    export_predictions(
                        records,
                        expected_ids,
                        csv_path,
                        jsonl_path,
                        low_confidence_threshold=0.0,
                    )
                self.assertFalse(csv_path.exists())
                self.assertFalse(jsonl_path.exists())

    def test_duplicate_expected_ids_and_unknown_or_abstaining_labels_are_rejected(self):
        record = make_prediction_record("known.mp4", (0.2, 0.2, 0.2, 0.2, 0.2))
        invalid_records = []
        unknown = dict(record)
        unknown["Label"] = "UNKNOWN"
        unknown["argmax"] = "UNKNOWN"
        invalid_records.append(unknown)
        abstain = dict(record)
        abstain["Label"] = "ABSTAIN"
        abstain["argmax"] = "ABSTAIN"
        invalid_records.append(abstain)

        cases = [((record,), ("known.mp4", "known.mp4"))]
        cases.extend(((invalid_record,), ("known.mp4",)) for invalid_record in invalid_records)
        for records, expected_ids in cases:
            with self.subTest(expected_ids=expected_ids, records=records), tempfile.TemporaryDirectory() as temp_dir:
                with self.assertRaises(ProtocolError):
                    export_predictions(
                        records,
                        expected_ids,
                        Path(temp_dir) / "prediction.csv",
                        Path(temp_dir) / "low.jsonl",
                        low_confidence_threshold=0.0,
                    )


if __name__ == "__main__":
    unittest.main()
