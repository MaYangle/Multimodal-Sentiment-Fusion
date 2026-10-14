import unittest

import numpy as np

from scripts.audit_native_av_times import (
    interval_overlap,
    nearest_time_pairs,
    validate_time_array,
    wavlm_center_and_hop_seconds,
)


class NativeAVTimeFixtureTests(unittest.TestCase):
    def test_wavlm_grid_uses_receptive_field_center_and_20ms_hop(self):
        center, hop, receptive_field, stride = wavlm_center_and_hop_seconds()
        self.assertEqual(receptive_field, 400)
        self.assertEqual(stride, 320)
        self.assertAlmostEqual(center, 0.01246875, places=10)
        self.assertAlmostEqual(hop, 0.02, places=12)

    def test_verified_offset_aligns_pairs_one_to_one(self):
        audio = np.array([0.112, 0.612, 1.112])
        vision = np.array([0.0, 0.5, 1.0])
        pairs = nearest_time_pairs(
            audio,
            vision,
            audio_to_video_offset_seconds=-0.112,
            require_common_support=True,
        )
        self.assertEqual([(p["vision_index"], p["audio_index"]) for p in pairs], [(0, 0), (1, 1), (2, 2)])
        self.assertTrue(all(p["absolute_delta_seconds"] < 1e-12 for p in pairs))

    def test_missing_audio_grid_sample_is_not_interpolated(self):
        # The audio anchor nearest 0.5s was removed. The 0.512s neighbor is
        # outside the declared 10ms tolerance, so no fabricated pair is made.
        audio = np.array([0.01246875, 0.03246875, 0.51246875])
        vision = np.array([0.5])
        self.assertEqual(nearest_time_pairs(audio, vision), [])

    def test_monotonic_and_zero_padding_guards(self):
        valid = np.array([[0.0, 0.5, 0.0, 0.0]], dtype=np.float64)
        summary = validate_time_array(
            valid,
            np.array([2], dtype=np.int64),
            name="fixture vision",
            expected_step_seconds=0.5,
            expected_start_seconds=0.0,
        )
        self.assertEqual(summary["valid_steps_total"], 2)
        self.assertEqual(summary["zero_padded_time_values"], 2)
        self.assertEqual(summary["grid_check"]["violating_valid_values"], 0)

        with self.assertRaisesRegex(ValueError, "monotonic"):
            validate_time_array(
                np.array([[0.5, 0.0]], dtype=np.float64),
                np.array([2], dtype=np.int64),
                name="decreasing fixture",
            )
        with self.assertRaisesRegex(ValueError, "padding must be zero"):
            validate_time_array(
                np.array([[0.0, 0.5, 0.25]], dtype=np.float64),
                np.array([2], dtype=np.int64),
                name="bad pad fixture",
            )

    def test_partial_overlap_reports_unmatched_tails(self):
        coverage = interval_overlap(
            np.array([0.1, 0.5, 1.1]),
            np.array([0.0, 0.5, 1.0, 1.5]),
        )
        self.assertTrue(coverage["intersects"])
        self.assertAlmostEqual(coverage["start_seconds"], 0.1)
        self.assertAlmostEqual(coverage["end_seconds"], 1.1)
        self.assertAlmostEqual(coverage["seconds"], 1.0)
        self.assertEqual(coverage["audio_anchors_outside"], 0)
        self.assertEqual(coverage["vision_anchors_outside"], 2)
        self.assertAlmostEqual(coverage["vision_support_seconds_outside"], 0.5)


if __name__ == "__main__":
    unittest.main()
